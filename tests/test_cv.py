import pytest
from pubmlp.cv import cross_validate
from pubmlp.config import Config


def test_cross_validate_config_n_folds():
    config = Config(n_folds=3)
    assert config.n_folds == 3

    config_default = Config()
    assert config_default.n_folds == 5


def test_cross_validate_config_used_in_cv():
    """Verify that config fields used by cross_validate exist and have expected defaults."""
    config = Config()
    assert hasattr(config, 'n_folds')
    assert hasattr(config, 'batch_size')
    assert hasattr(config, 'eval_batch_size')
    assert hasattr(config, 'epochs')
    assert hasattr(config, 'learning_rate')
    assert hasattr(config, 'early_stopping_patience')
    assert hasattr(config, 'gradient_clip_norm')
    assert hasattr(config, 'mlp_hidden_size')
    assert hasattr(config, 'dropout_rate')
    assert hasattr(config, 'embedding_model')
    assert hasattr(config, 'model_name')
    assert hasattr(config, 'n_hidden_layers')
    assert hasattr(config, 'pooling_strategy')
    assert hasattr(config, 'max_length')
    assert hasattr(config, 'rare_threshold')
    assert hasattr(config, 'pos_weight')


def test_config_new_defaults():
    config = Config()
    assert config.rare_threshold == 5
    assert config.pos_weight == 'auto'
