import numpy as np
import pandas as pd
import pytest

from pubmlp.active_learning import (
    ALState, rank_by_uncertainty, select_query_batch,
    create_review_batch, compare_reviewers,
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


class TestALState:
    def test_serialization_roundtrip(self):
        state = ALState(labeled_indices=[0, 1], unlabeled_indices=[2, 3], iteration=1)
        restored = ALState.from_dict(state.to_dict())
        assert restored.labeled_indices == [0, 1]
        assert restored.iteration == 1
