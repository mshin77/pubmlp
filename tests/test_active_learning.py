import numpy as np
import pandas as pd
import pytest

from pubmlp.active_learning import (
    ALState, rank_by_uncertainty, rank_by_max_relevance,
    rank_by_hybrid_max_uncertainty, rank_by_hybrid_max_random,
    select_query_batch, create_review_batch, compare_reviewers,
    simulate_al, stratified_split,
)


class TestRanking:
    def test_rank_by_uncertainty(self):
        probs = np.array([0.1, 0.5, 0.9, 0.48])
        ranked = rank_by_uncertainty(probs)
        # 0.5 and 0.48 are closest to 0.5 — should be first two
        assert set(ranked[:2]) == {1, 3}

    def test_select_query_batch_size(self):
        probs = np.random.rand(100)
        batch = select_query_batch(probs, strategy='uncertainty', batch_size=10)
        assert len(batch) == 10

    def test_select_query_batch_random(self):
        batch = select_query_batch(np.random.rand(50), strategy='random', batch_size=5)
        assert len(batch) == 5


class TestReviewBatch:
    def test_create_review_batch_columns(self):
        df = pd.DataFrame({'title': ['a', 'b', 'c', 'd']})
        probs = np.array([0.1, 0.8, 0.5, 0.3])
        batch = create_review_batch(df, np.array([1, 2]), probs)
        assert 'model_probability' in batch.columns
        assert 'model_prediction' in batch.columns
        assert len(batch) == 2

    def test_compare_reviewers(self):
        model = [1, 1, 0, 0, 1]
        human = [1, 0, 0, 0, 1]
        result = compare_reviewers(model, human)
        assert result['agreement_rate'] == 0.8
        assert 1 in result['disagreement_indices']


class TestHybridStrategies:
    def test_hybrid_max_uncertainty_length(self):
        probs = np.random.rand(100)
        ranked = rank_by_hybrid_max_uncertainty(probs)
        assert len(ranked) == 100

    def test_hybrid_max_uncertainty_exploit_first(self):
        probs = np.array([0.9, 0.1, 0.5, 0.8, 0.2])
        ranked = rank_by_hybrid_max_uncertainty(probs)
        # Top exploit portion should be max-relevance ordered
        assert ranked[0] == 0  # 0.9 is highest

    def test_hybrid_max_random_length(self):
        probs = np.random.rand(100)
        ranked = rank_by_hybrid_max_random(probs)
        assert len(ranked) == 100

    def test_hybrid_max_random_no_duplicates(self):
        probs = np.random.rand(50)
        ranked = rank_by_hybrid_max_random(probs)
        assert len(set(ranked)) == 50

    def test_select_query_batch_hybrid(self):
        probs = np.random.rand(100)
        batch = select_query_batch(probs, strategy='hybrid_max_uncertainty', batch_size=10)
        assert len(batch) == 10

    def test_select_query_batch_hybrid_random(self):
        probs = np.random.rand(100)
        batch = select_query_batch(probs, strategy='hybrid_max_random', batch_size=10)
        assert len(batch) == 10


class TestSimulateAL:
    def test_returns_history(self):
        df = pd.DataFrame({
            'text': [f'text_{i}' for i in range(100)],
            'label': [1 if i < 20 else 0 for i in range(100)],
        })

        def dummy_model_fn(train_df, unlabeled_df):
            return np.random.rand(len(unlabeled_df))

        history = simulate_al(df, 'label', dummy_model_fn, batch_size=10, max_iterations=3)
        assert len(history) > 0
        assert 'iteration' in history[0]
        assert 'recall' in history[0]
        assert 'n_labeled' in history[0]

    def test_stops_when_all_found(self):
        df = pd.DataFrame({
            'text': [f'text_{i}' for i in range(50)],
            'label': [1 if i < 5 else 0 for i in range(50)],
        })

        def perfect_model_fn(train_df, unlabeled_df):
            # Return high probs for relevant, low for irrelevant
            return np.array([0.9 if unlabeled_df.iloc[i]['label'] == 1 else 0.1
                             for i in range(len(unlabeled_df))])

        history = simulate_al(df, 'label', perfect_model_fn, batch_size=10,
                              initial_pct=0.1, max_iterations=20)
        assert history[-1]['recall'] > 0

    def test_exhausts_pool(self):
        df = pd.DataFrame({
            'text': [f'text_{i}' for i in range(60)],
            'label': [1 if i < 10 else 0 for i in range(60)],
        })

        def dummy_fn(train_df, unlabeled_df):
            return np.random.rand(len(unlabeled_df))

        history = simulate_al(df, 'label', dummy_fn, batch_size=20,
                              initial_pct=0.1, max_iterations=50)
        assert len(history) > 0
        assert history[-1]['screened_pct'] <= 1.0


class TestALState:
    def test_serialization_roundtrip(self):
        state = ALState(labeled_indices=[0, 1], unlabeled_indices=[2, 3], iteration=1)
        restored = ALState.from_dict(state.to_dict())
        assert restored.labeled_indices == [0, 1]
        assert restored.iteration == 1


class TestSafeStratifiedSplit:
    def test_normal_stratified(self):
        X = pd.DataFrame({'x': range(100)})
        y = pd.Series([0] * 50 + [1] * 50)
        train_idx, val_idx = stratified_split(X, y, test_size=0.2, random_state=42)
        assert len(train_idx) == 80
        assert len(val_idx) == 20
        assert set(train_idx) & set(val_idx) == set()

    def test_rare_class_fallback(self):
        X = pd.DataFrame({'x': range(20)})
        y = pd.Series([0] * 19 + [1])
        train_idx, val_idx = stratified_split(X, y, test_size=0.2, random_state=42)
        assert len(train_idx) + len(val_idx) == 20

    def test_index_coverage(self):
        X = pd.DataFrame({'x': range(50)})
        y = pd.Series([0] * 25 + [1] * 25)
        train_idx, val_idx = stratified_split(X, y, test_size=0.3, random_state=7)
        all_idx = set(train_idx.tolist()) | set(val_idx.tolist())
        assert all_idx == set(range(50))
