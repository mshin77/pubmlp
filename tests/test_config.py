"""Tests for pubmlp.config module."""

import pytest

from pubmlp.config import Config, default_config, fast_config, robust_config, domain_configs


class TestConfigDefaults:
    def test_default_values(self):
        config = Config()
        assert config.random_seed == 42
        assert config.batch_size == 16
        assert config.epochs == 10
        assert config.learning_rate == 2e-5
        assert config.dropout_rate == 0.2
        assert config.mlp_hidden_size == 64
        assert config.n_hidden_layers == 1
        assert config.max_length == 512
        assert config.embedding_model == 'bert'
        assert config.model_name == 'bert-base-uncased'
        assert config.pooling_strategy == 'auto'

    def test_default_uncertainty_thresholds(self):
        config = Config()
        assert config.uncertainty_low == 0.3
        assert config.uncertainty_high == 0.7
        assert config.n_folds == 5


class TestConfigOverride:
    def test_kwargs_override(self):
        config = Config(epochs=20, batch_size=32, learning_rate=1e-5)
        assert config.epochs == 20
        assert config.batch_size == 32
        assert config.learning_rate == 1e-5

    def test_embedding_model_sets_model_name(self):
        config = Config(embedding_model='scibert')
        assert config.model_name == 'allenai/scibert_scivocab_uncased'

    def test_modernbert_model_name(self):
        config = Config(embedding_model='modernbert')
        assert config.model_name == 'answerdotai/ModernBERT-base'

    def test_bge_small_model_name(self):
        config = Config(embedding_model='bge-small')
        assert config.model_name == 'BAAI/bge-small-en-v1.5'

    def test_explicit_model_name_overrides_default(self):
        config = Config(embedding_model='bert', model_name='bert-large-uncased')
        assert config.model_name == 'bert-large-uncased'

    def test_n_hidden_layers(self):
        config = Config(n_hidden_layers=2)
        assert config.n_hidden_layers == 2


class TestConfigMethods:
    def test_to_dict(self):
        config = Config()
        d = config.to_dict()
        assert isinstance(d, dict)
        assert 'epochs' in d
        assert 'model_name' in d

    def test_repr(self):
        config = Config()
        r = repr(config)
        assert r.startswith('Config(')
        assert 'epochs' in r


class TestPresets:
    def test_default_config(self):
        assert default_config.embedding_model == 'bert'

    def test_fast_config(self):
        assert fast_config.epochs == 5
        assert fast_config.embedding_model == 'sentence-transformer'

    def test_robust_config(self):
        assert robust_config.epochs == 20
        assert robust_config.mlp_hidden_size == 128

    def test_domain_configs(self):
        assert 'science' in domain_configs
        assert 'medicine' in domain_configs
        assert 'modernbert' in domain_configs
        assert domain_configs['science'].model_name == 'allenai/scibert_scivocab_uncased'
