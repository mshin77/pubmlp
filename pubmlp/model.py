import torch
import torch.nn as nn
from transformers import AutoModel
from sentence_transformers import SentenceTransformer

from .config import sentence_transformer_models


class PubMLP(nn.Module):
    """Transformer encoder + MLP classifier for text classification."""

    def __init__(self, categorical_cols_num=0, numeric_cols_num=0, mlp_hidden_size=64,
                 output_size=1, dropout_rate=0.2, embedding_model='bert', model_name=None,
                 n_hidden_layers=1, pooling_strategy='auto', categorical_vocab_sizes=None):
        super().__init__()

        self.embedding_model = embedding_model
        self.numeric_cols_num = numeric_cols_num
        self.pooling_strategy = pooling_strategy
        self._use_sentence_transformer = embedding_model in sentence_transformer_models

        # Categorical embedding layers
        self.categorical_vocab_sizes = categorical_vocab_sizes or []
        self.cat_embeddings = nn.ModuleList()
        cat_embed_total = 0
        for vocab_size in self.categorical_vocab_sizes:
            embed_dim = min(50, (vocab_size + 1) // 2)
            self.cat_embeddings.append(nn.Embedding(vocab_size, embed_dim))
            cat_embed_total += embed_dim

        # Backward compat: if no vocab sizes given, fall back to scalar categorical input
        self.categorical_cols_num = categorical_cols_num if not self.categorical_vocab_sizes else 0
        if self._use_sentence_transformer:
            self.model_name = model_name or 'all-MiniLM-L6-v2'
            self.encoder = SentenceTransformer(self.model_name)
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False
            embedding_size = self.encoder.get_sentence_embedding_dimension()
        else:
            self.model_name = model_name or 'bert-base-uncased'
            self.encoder = AutoModel.from_pretrained(self.model_name)
            embedding_size = self.encoder.config.hidden_size

            if self.pooling_strategy == 'auto':
                self.pooling_strategy = self._detect_pooling_strategy()

        self.dropout = nn.Dropout(dropout_rate)

        input_size = embedding_size + cat_embed_total + self.categorical_cols_num + numeric_cols_num

        layers = [nn.Linear(input_size, mlp_hidden_size), nn.ReLU(), nn.Dropout(dropout_rate)]
        for _ in range(n_hidden_layers - 1):
            layers.extend([nn.Linear(mlp_hidden_size, mlp_hidden_size), nn.ReLU(), nn.Dropout(dropout_rate)])
        layers.append(nn.Linear(mlp_hidden_size, output_size))
        self.classifier = nn.Sequential(*layers)

    def _detect_pooling_strategy(self):
        """Detect whether the model has a pooler layer."""
        if hasattr(self.encoder, 'pooler') and self.encoder.pooler is not None:
            return 'pooler'
        return 'mean'

    def _mean_pooling(self, last_hidden_state, attention_mask):
        """Mean pooling over token embeddings, masked by attention."""
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        return (last_hidden_state * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)

    def forward(self, input_ids, attention_mask, categorical_tensor=None, numeric_tensor=None, texts=None):
        if self._use_sentence_transformer:
            if texts is None:
                raise ValueError("texts must be provided when using sentence-transformer")
            with torch.no_grad():
                sentence_embedding = self.encoder.encode(texts, convert_to_tensor=True, show_progress_bar=False)
                if categorical_tensor is not None and categorical_tensor.numel() > 0:
                    sentence_embedding = sentence_embedding.to(categorical_tensor.device)
                elif numeric_tensor is not None and numeric_tensor.numel() > 0:
                    sentence_embedding = sentence_embedding.to(numeric_tensor.device)
        else:
            outputs = self.encoder(input_ids, attention_mask)
            if self.pooling_strategy == 'pooler' and hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                sentence_embedding = outputs.pooler_output
            else:
                sentence_embedding = self._mean_pooling(outputs.last_hidden_state, attention_mask)

        features_concat = [sentence_embedding]

        # Learned categorical embeddings
        if self.categorical_vocab_sizes and categorical_tensor is not None and categorical_tensor.numel() > 0:
            for i, emb_layer in enumerate(self.cat_embeddings):
                features_concat.append(emb_layer(categorical_tensor[:, i]))
        elif self.categorical_cols_num > 0 and categorical_tensor is not None:
            features_concat.append(categorical_tensor)

        if self.numeric_cols_num > 0 and numeric_tensor is not None:
            features_concat.append(numeric_tensor)

        concat_features = torch.cat(features_concat, dim=1)
        return self.classifier(self.dropout(concat_features))
