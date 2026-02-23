"""Tests for pubmlp.model module."""

import pytest
import torch

from pubmlp.model import PubMLP


class TestPubMLPForward:
    def test_forward_text_only(self):
        model = PubMLP(
            categorical_cols_num=0, numeric_cols_num=0,
            mlp_hidden_size=16, embedding_model='bert', model_name='bert-base-uncased',
        )
        model.eval()
        input_ids = torch.zeros(2, 8, dtype=torch.long)
        attention_mask = torch.ones(2, 8, dtype=torch.long)
        out = model(input_ids, attention_mask)
        assert out.shape == (2, 1)

    def test_forward_with_scalar_features(self):
        model = PubMLP(
            categorical_cols_num=2, numeric_cols_num=1,
            mlp_hidden_size=16, embedding_model='bert', model_name='bert-base-uncased',
        )
        model.eval()
        input_ids = torch.zeros(3, 8, dtype=torch.long)
        attention_mask = torch.ones(3, 8, dtype=torch.long)
        cat = torch.randn(3, 2)
        num = torch.randn(3, 1)
        out = model(input_ids, attention_mask, cat, num)
        assert out.shape == (3, 1)

    def test_forward_with_embedding_categoricals(self):
        model = PubMLP(
            categorical_vocab_sizes=[10, 5],
            numeric_cols_num=1,
            mlp_hidden_size=16,
            embedding_model='bert',
            model_name='bert-base-uncased',
        )
        model.eval()
        input_ids = torch.zeros(3, 8, dtype=torch.long)
        attention_mask = torch.ones(3, 8, dtype=torch.long)
        cat = torch.tensor([[2, 1], [5, 3], [0, 4]], dtype=torch.long)
        num = torch.randn(3, 1)
        out = model(input_ids, attention_mask, cat, num)
        assert out.shape == (3, 1)

    def test_multi_label_output(self):
        model = PubMLP(
            mlp_hidden_size=16, output_size=3,
            embedding_model='bert', model_name='bert-base-uncased',
        )
        model.eval()
        out = model(torch.zeros(2, 8, dtype=torch.long), torch.ones(2, 8, dtype=torch.long))
        assert out.shape == (2, 3)

    def test_multiple_hidden_layers(self):
        model = PubMLP(
            mlp_hidden_size=16, n_hidden_layers=3,
            embedding_model='bert', model_name='bert-base-uncased',
        )
        model.eval()
        out = model(torch.zeros(1, 4, dtype=torch.long), torch.ones(1, 4, dtype=torch.long))
        assert out.shape == (1, 1)

    def test_embedding_dims(self):
        model = PubMLP(
            categorical_vocab_sizes=[10, 100],
            mlp_hidden_size=16,
            embedding_model='bert',
            model_name='bert-base-uncased',
        )
        # vocab=10 -> embed_dim=min(50, 5)=5
        assert model.cat_embeddings[0].embedding_dim == 5
        # vocab=100 -> embed_dim=min(50, 50)=50
        assert model.cat_embeddings[1].embedding_dim == 50


class TestMultipleColumnsForward:
    """Test model forward with multiple categorical embeddings + multiple numeric columns."""

    def test_three_categoricals_two_numerics(self):
        model = PubMLP(
            categorical_vocab_sizes=[15, 8, 5],
            numeric_cols_num=2,
            mlp_hidden_size=16,
            output_size=1,
            embedding_model='bert',
            model_name='bert-base-uncased',
        )
        model.eval()
        n = 4
        input_ids = torch.zeros(n, 8, dtype=torch.long)
        attention_mask = torch.ones(n, 8, dtype=torch.long)
        cat = torch.randint(0, 5, (n, 3))
        num = torch.randn(n, 2)
        out = model(input_ids, attention_mask, cat, num)
        assert out.shape == (n, 1)

    def test_three_categoricals_two_numerics_multi_label(self):
        model = PubMLP(
            categorical_vocab_sizes=[15, 8, 5],
            numeric_cols_num=2,
            mlp_hidden_size=16,
            output_size=3,
            embedding_model='bert',
            model_name='bert-base-uncased',
        )
        model.eval()
        n = 4
        input_ids = torch.zeros(n, 8, dtype=torch.long)
        attention_mask = torch.ones(n, 8, dtype=torch.long)
        cat = torch.randint(0, 5, (n, 3))
        num = torch.randn(n, 2)
        out = model(input_ids, attention_mask, cat, num)
        assert out.shape == (n, 3)

    def test_embedding_count_matches_vocab_sizes(self):
        model = PubMLP(
            categorical_vocab_sizes=[20, 10, 6, 4],
            mlp_hidden_size=16,
            embedding_model='bert',
            model_name='bert-base-uncased',
        )
        assert len(model.cat_embeddings) == 4
        assert model.cat_embeddings[0].num_embeddings == 20
        assert model.cat_embeddings[1].num_embeddings == 10
        assert model.cat_embeddings[2].num_embeddings == 6
        assert model.cat_embeddings[3].num_embeddings == 4


class TestPoolingDetection:
    def test_bert_uses_pooler(self):
        model = PubMLP(embedding_model='bert', model_name='bert-base-uncased', pooling_strategy='auto')
        assert model.pooling_strategy == 'pooler'

    def test_explicit_mean_pooling(self):
        model = PubMLP(embedding_model='bert', model_name='bert-base-uncased', pooling_strategy='mean')
        assert model.pooling_strategy == 'mean'


class TestSentenceTransformerPath:
    def test_sentence_transformer_flag(self):
        model = PubMLP(embedding_model='sentence-transformer', model_name='all-MiniLM-L6-v2')
        assert model._use_sentence_transformer is True

    def test_sentence_transformer_requires_texts(self):
        model = PubMLP(embedding_model='sentence-transformer', model_name='all-MiniLM-L6-v2')
        model.eval()
        with pytest.raises(ValueError, match="texts must be provided"):
            model(torch.zeros(1, 4, dtype=torch.long), torch.ones(1, 4, dtype=torch.long))

    def test_sentence_transformer_forward(self):
        model = PubMLP(
            embedding_model='sentence-transformer', model_name='all-MiniLM-L6-v2',
            mlp_hidden_size=16,
        )
        model.eval()
        out = model(
            torch.zeros(2, 4, dtype=torch.long),
            torch.ones(2, 4, dtype=torch.long),
            texts=["hello world", "test sentence"],
        )
        assert out.shape == (2, 1)
