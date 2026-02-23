import random
import numpy as np
import torch


# Models that use the SentenceTransformer encoder (frozen, no fine-tuning)
sentence_transformer_models = {'sentence-transformer', 'bge-small'}


class Config:
    """Configuration for PubMLP training and inference."""

    def __init__(self, **kwargs):
        # Random seed
        self.random_seed = kwargs.get('random_seed', 42)

        # Training hyperparameters
        self.batch_size = kwargs.get('batch_size', 16)
        self.eval_batch_size = kwargs.get('eval_batch_size', 32)
        self.epochs = kwargs.get('epochs', 10)
        self.learning_rate = kwargs.get('learning_rate', 2e-5)
        self.early_stopping_patience = kwargs.get('early_stopping_patience', 3)

        # Model architecture
        self.dropout_rate = kwargs.get('dropout_rate', 0.2)
        self.mlp_hidden_size = kwargs.get('mlp_hidden_size', 64)
        self.n_hidden_layers = kwargs.get('n_hidden_layers', 1)
        self.max_length = kwargs.get('max_length', 512)

        # Optimization
        self.gradient_clip_norm = kwargs.get('gradient_clip_norm', 1.0)
        self.warmup_steps = kwargs.get('warmup_steps', 0)

        # Embedding model
        self.embedding_model = kwargs.get('embedding_model', 'bert')
        self.model_name = kwargs.get('model_name', None)
        self.pooling_strategy = kwargs.get('pooling_strategy', 'auto')

        # Uncertainty thresholds
        self.uncertainty_low = kwargs.get('uncertainty_low', 0.3)
        self.uncertainty_high = kwargs.get('uncertainty_high', 0.7)

        # Cross-validation
        self.n_folds = kwargs.get('n_folds', 5)

        # Calibration
        self.calibration_method = kwargs.get('calibration_method', 'temperature')

        # Active learning
        self.al_query_strategy = kwargs.get('al_query_strategy', 'uncertainty')
        self.al_batch_size = kwargs.get('al_batch_size', 20)
        self.al_initial_sample_pct = kwargs.get('al_initial_sample_pct', 0.1)

        # Categorical encoding
        self.rare_threshold = kwargs.get('rare_threshold', 5)

        # Class weighting
        self.pos_weight = kwargs.get('pos_weight', 'auto')

        # SAFE stopping
        self.safe_consecutive_irrelevant = kwargs.get('safe_consecutive_irrelevant', 50)
        self.safe_min_screened_pct = kwargs.get('safe_min_screened_pct', 0.5)
        self.safe_random_sample_pct = kwargs.get('safe_random_sample_pct', 0.1)
        self.safe_switch_model = kwargs.get('safe_switch_model', False)

        if self.model_name is None:
            self.model_name = self._get_default_model_name()

    def _get_default_model_name(self):
        defaults = {
            'bert': 'bert-base-uncased',
            'modernbert': 'answerdotai/ModernBERT-base',
            'scibert': 'allenai/scibert_scivocab_uncased',
            'pubmedbert': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
            'sentence-transformer': 'all-MiniLM-L6-v2',
            'bge-small': 'BAAI/bge-small-en-v1.5',
        }
        return defaults.get(self.embedding_model, 'bert-base-uncased')

    def set_random_seeds(self):
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)
            torch.cuda.manual_seed_all(self.random_seed)

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def __repr__(self):
        items = [f"{k}={repr(v)}" for k, v in self.to_dict().items()]
        return f"Config({', '.join(items)})"


default_config = Config()

fast_config = Config(
    epochs=5,
    batch_size=32,
    embedding_model='sentence-transformer',
    model_name='all-MiniLM-L6-v2'
)

robust_config = Config(
    epochs=20,
    early_stopping_patience=5,
    dropout_rate=0.3,
    mlp_hidden_size=128
)

hitl_config = Config(
    al_query_strategy='uncertainty',
    al_batch_size=20,
    safe_consecutive_irrelevant=50,
    safe_min_screened_pct=0.5,
)

domain_configs = {
    'science': Config(embedding_model='scibert'),
    'medicine': Config(embedding_model='pubmedbert'),
    'general': Config(embedding_model='bert'),
    'modernbert': Config(embedding_model='modernbert'),
}
