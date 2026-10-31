"""Tests for pubmlp.metrics module."""

import pytest
import numpy as np

from pubmlp.metrics import (
    calculate_evaluation_metrics, _is_multi_label,
    calculate_wss_at_recall, calculate_ndcg,
    calculate_ece, calculate_brier,
)


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


class TestWSSAtRecall:
    def test_perfect_ranking(self):
        # Relevant items ranked first — maximum WSS
        true = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
        probs = [0.99, 0.95, 0.90, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        result = calculate_wss_at_recall(true, probs, target_recall=0.95)
        assert result['wss'] > 0.5
        assert result['recall_achieved'] >= 0.95

    def test_random_ranking(self):
        # Random ranking — WSS near 0
        np.random.seed(42)
        true = [1] * 10 + [0] * 90
        probs = np.random.rand(100).tolist()
        result = calculate_wss_at_recall(true, probs, target_recall=0.95)
        assert result['wss'] >= 0.0
        assert result['screened_pct'] <= 1.0

    def test_no_relevant(self):
        true = [0, 0, 0, 0]
        probs = [0.5, 0.3, 0.7, 0.1]
        result = calculate_wss_at_recall(true, probs)
        assert np.isnan(result['wss'])
        assert np.isnan(result['screened_pct'])
        assert np.isnan(result['recall_achieved'])

    def test_returns_expected_keys(self):
        true = [1, 0, 1, 0]
        probs = [0.9, 0.1, 0.8, 0.2]
        result = calculate_wss_at_recall(true, probs)
        assert 'wss' in result
        assert 'screened_pct' in result
        assert 'recall_achieved' in result


class TestNDCG:
    def test_perfect_ranking(self):
        true = [1, 1, 1, 0, 0, 0]
        probs = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]
        score = calculate_ndcg(true, probs)
        assert score == pytest.approx(1.0)

    def test_worst_ranking(self):
        true = [1, 1, 0, 0]
        probs = [0.1, 0.2, 0.9, 0.8]
        score = calculate_ndcg(true, probs)
        assert score < 1.0

    def test_range_zero_to_one(self):
        np.random.seed(42)
        true = [1] * 10 + [0] * 90
        probs = np.random.rand(100).tolist()
        score = calculate_ndcg(true, probs)
        assert 0.0 <= score <= 1.0


class TestECE:
    def test_calibrated_within_bin(self):
        true = [0, 1, 0, 1]
        probs = [0.5, 0.5, 0.5, 0.5]
        assert calculate_ece(true, probs) == pytest.approx(0.0)

    def test_known_miscalibration(self):
        true = [0, 0, 1, 1]
        probs = [0.2, 0.2, 0.8, 0.8]
        assert calculate_ece(true, probs) == pytest.approx(0.2)

    def test_multi_label_macro(self):
        true = [[0, 0], [1, 0], [0, 1], [1, 1]]
        probs = [[0.5, 0.2], [0.5, 0.2], [0.5, 0.8], [0.5, 0.8]]
        per_label = [calculate_ece([r[j] for r in true], [p[j] for p in probs])
                     for j in range(2)]
        assert calculate_ece(true, probs) == pytest.approx(np.mean(per_label))

    def test_range_zero_to_one(self):
        np.random.seed(42)
        true = np.random.randint(0, 2, 100)
        probs = np.random.rand(100)
        assert 0.0 <= calculate_ece(true, probs) <= 1.0


class TestBrier:
    def test_perfect_probabilities(self):
        true = [0, 1, 0, 1]
        probs = [0.0, 1.0, 0.0, 1.0]
        assert calculate_brier(true, probs) == pytest.approx(0.0)

    def test_uninformative_probabilities(self):
        true = [0, 1]
        probs = [0.5, 0.5]
        assert calculate_brier(true, probs) == pytest.approx(0.25)

    def test_multi_label_macro(self):
        true = [[0, 1], [1, 0]]
        probs = [[0.0, 1.0], [0.5, 0.5]]
        assert calculate_brier(true, probs) == pytest.approx(0.125)
