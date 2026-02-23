"""Tests for pubmlp.metrics module."""

import pytest
import numpy as np

from pubmlp.metrics import calculate_evaluation_metrics, _is_multi_label


class TestSingleLabelMetrics:
    def test_perfect_predictions(self):
        true = [0, 0, 1, 1]
        preds = [0, 0, 1, 1]
        probs = [0.1, 0.2, 0.9, 0.8]
        metrics = calculate_evaluation_metrics(true, preds, probs, save_figures=False)
        assert metrics['accuracy'] == 1.0
        assert metrics['f1_score'] == 1.0

    def test_all_wrong(self):
        true = [0, 0, 1, 1]
        preds = [1, 1, 0, 0]
        probs = [0.9, 0.8, 0.1, 0.2]
        metrics = calculate_evaluation_metrics(true, preds, probs, save_figures=False)
        assert metrics['accuracy'] == 0.0

    def test_returns_expected_keys(self):
        true = [0, 1, 0, 1]
        preds = [0, 1, 1, 0]
        probs = [0.2, 0.8, 0.6, 0.4]
        metrics = calculate_evaluation_metrics(true, preds, probs, save_figures=False)
        for key in ['accuracy', 'precision', 'recall', 'specificity', 'f1_score', 'roc_auc']:
            assert key in metrics


class TestMultiLabelMetrics:
    def test_perfect_multi_label(self):
        true = [[0, 1], [1, 0], [1, 1]]
        preds = [[0, 1], [1, 0], [1, 1]]
        probs = [[0.1, 0.9], [0.9, 0.1], [0.8, 0.8]]
        metrics = calculate_evaluation_metrics(
            true, preds, probs, save_figures=False,
            label_names=['label_a', 'label_b'],
        )
        assert 'per_label' in metrics
        assert metrics['macro_f1'] == 1.0
        assert metrics['hamming_loss'] == 0.0

    def test_multi_label_keys(self):
        true = [[0, 1], [1, 0]]
        preds = [[0, 0], [1, 1]]
        probs = [[0.2, 0.4], [0.8, 0.6]]
        metrics = calculate_evaluation_metrics(true, preds, probs, save_figures=False)
        assert 'per_label' in metrics
        assert 'macro_f1' in metrics
        assert 'hamming_loss' in metrics

    def test_default_label_names(self):
        true = [[0, 1], [1, 0]]
        preds = [[0, 1], [1, 0]]
        probs = [[0.1, 0.9], [0.9, 0.1]]
        metrics = calculate_evaluation_metrics(true, preds, probs, save_figures=False)
        assert 'label_0' in metrics['per_label']
        assert 'label_1' in metrics['per_label']


class TestIsMultiLabel:
    def test_single_label(self):
        assert not _is_multi_label([0, 1, 0])

    def test_multi_label_lists(self):
        assert _is_multi_label([[0, 1], [1, 0]])

    def test_multi_label_numpy(self):
        assert _is_multi_label([np.array([0, 1]), np.array([1, 0])])
