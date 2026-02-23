"""Tests for pubmlp.preprocess module."""

import pytest
import pandas as pd
import numpy as np
import torch
from unittest.mock import MagicMock

from pubmlp.preprocess import (
    split_data, CustomDataset, collate_fn, FittedTransforms,
    _build_categorical_vocab, _encode_categorical, _fit_numeric, _apply_numeric,
    preprocess_dataset,
)


class TestSplitData:
    def test_split_proportions(self):
        df = pd.DataFrame({'x': range(100), 'y': range(100)})
        train, val, test = split_data(df)
        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10

    def test_no_overlap(self):
        df = pd.DataFrame({'x': range(100)})
        train, val, test = split_data(df)
        all_indices = set(train.index) | set(val.index) | set(test.index)
        assert len(all_indices) == 100

    def test_deterministic(self):
        df = pd.DataFrame({'x': range(50)})
        train1, _, _ = split_data(df, random_state=42)
        train2, _, _ = split_data(df, random_state=42)
        assert list(train1.index) == list(train2.index)


class TestCustomDataset:
    def test_len(self):
        n = 10
        ds = CustomDataset(
            input_ids=torch.zeros(n, 16, dtype=torch.long),
            attention_mask=torch.ones(n, 16, dtype=torch.long),
            labels=torch.zeros(n, 1, dtype=torch.float),
        )
        assert len(ds) == n

    def test_getitem_keys(self):
        ds = CustomDataset(
            input_ids=torch.zeros(5, 8, dtype=torch.long),
            attention_mask=torch.ones(5, 8, dtype=torch.long),
            labels=torch.zeros(5, 1, dtype=torch.float),
            categorical_tensor=torch.zeros(5, 2, dtype=torch.long),
            numeric_tensor=torch.zeros(5, 1),
        )
        item = ds[0]
        assert 'input_ids' in item
        assert 'attention_mask' in item
        assert 'labels' in item
        assert 'categorical_tensor' in item
        assert 'numeric_tensor' in item

    def test_getitem_with_texts(self):
        ds = CustomDataset(
            input_ids=torch.zeros(3, 4, dtype=torch.long),
            attention_mask=torch.ones(3, 4, dtype=torch.long),
            labels=torch.zeros(3, 1, dtype=torch.float),
            texts=['text a', 'text b', 'text c'],
        )
        assert ds[1]['texts'] == 'text b'

    def test_multi_label_shape(self):
        ds = CustomDataset(
            input_ids=torch.zeros(4, 8, dtype=torch.long),
            attention_mask=torch.ones(4, 8, dtype=torch.long),
            labels=torch.zeros(4, 3, dtype=torch.float),
        )
        assert ds[0]['labels'].shape == (3,)


class TestCollateFn:
    def test_batches_correctly(self):
        batch = [
            {
                'input_ids': torch.zeros(4, dtype=torch.long),
                'attention_mask': torch.ones(4, dtype=torch.long),
                'labels': torch.tensor([1.0]),
                'categorical_tensor': torch.tensor([2], dtype=torch.long),
                'numeric_tensor': torch.tensor([0.3]),
                'texts': 'hello',
            },
            {
                'input_ids': torch.ones(4, dtype=torch.long),
                'attention_mask': torch.ones(4, dtype=torch.long),
                'labels': torch.tensor([0.0]),
                'categorical_tensor': torch.tensor([1], dtype=torch.long),
                'numeric_tensor': torch.tensor([0.7]),
                'texts': 'world',
            },
        ]
        result = collate_fn(batch)
        assert result['input_ids'].shape == (2, 4)
        assert result['labels'].shape == (2, 1)
        assert result['texts'] == ['hello', 'world']

    def test_none_texts(self):
        batch = [
            {
                'input_ids': torch.zeros(4, dtype=torch.long),
                'attention_mask': torch.ones(4, dtype=torch.long),
                'labels': torch.tensor([1.0]),
                'categorical_tensor': torch.tensor([0], dtype=torch.long),
                'numeric_tensor': torch.tensor([0.3]),
                'texts': None,
            },
        ]
        result = collate_fn(batch)
        assert result['texts'] is None


class TestBuildCategoricalVocab:
    def test_basic_vocab(self):
        s = pd.Series(['A'] * 10 + ['B'] * 8 + ['C'] * 2)
        vocab = _build_categorical_vocab(s, rare_threshold=5)
        assert vocab['<UNK>'] == 0
        assert vocab['<RARE>'] == 1
        assert 'A' in vocab
        assert 'B' in vocab
        assert 'C' not in vocab  # count=2 < threshold=5

    def test_all_rare(self):
        s = pd.Series(['X', 'Y', 'Z'])
        vocab = _build_categorical_vocab(s, rare_threshold=5)
        assert len(vocab) == 2  # only UNK + RARE

    def test_threshold_one(self):
        s = pd.Series(['A', 'B', 'C'])
        vocab = _build_categorical_vocab(s, rare_threshold=1)
        assert 'A' in vocab and 'B' in vocab and 'C' in vocab


class TestEncodeCategorical:
    def test_known_values(self):
        vocab = {'<UNK>': 0, '<RARE>': 1, 'A': 2, 'B': 3}
        s = pd.Series(['A', 'B', 'A'])
        encoded = _encode_categorical(s, vocab)
        assert list(encoded) == [2, 3, 2]

    def test_unknown_value_maps_to_rare(self):
        vocab = {'<UNK>': 0, '<RARE>': 1, 'A': 2}
        s = pd.Series(['A', 'UNKNOWN'])
        encoded = _encode_categorical(s, vocab)
        assert list(encoded) == [2, 1]

    def test_nan_maps_to_unk(self):
        vocab = {'<UNK>': 0, '<RARE>': 1, 'A': 2}
        s = pd.Series(['A', None])
        encoded = _encode_categorical(s, vocab)
        assert list(encoded) == [2, 0]


class TestNumericTransforms:
    def test_min_transform(self):
        s = pd.Series([10.0, 20.0, 30.0])
        params = _fit_numeric(s, 'min')
        result = _apply_numeric(s, params)
        assert list(result) == [0.0, 10.0, 20.0]

    def test_max_transform(self):
        s = pd.Series([10.0, 20.0, 30.0])
        params = _fit_numeric(s, 'max')
        result = _apply_numeric(s, params)
        assert abs(result.iloc[2] - 1.0) < 1e-6

    def test_mean_transform(self):
        s = pd.Series([10.0, 20.0, 30.0])
        params = _fit_numeric(s, 'mean')
        result = _apply_numeric(s, params)
        assert abs(result.iloc[1]) < 1e-6  # mean should be ~0

    def test_quantile_transform(self):
        np.random.seed(0)
        s = pd.Series(np.random.randn(100))
        params = _fit_numeric(s, 'quantile')
        result = _apply_numeric(s, params)
        assert len(result) == 100

    def test_robust_transform(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0])
        params = _fit_numeric(s, 'robust')
        result = _apply_numeric(s, params)
        assert len(result) == 5

    def test_log1p_transform(self):
        s = pd.Series([0.0, 1.0, 10.0, 100.0, 1000.0])
        params = _fit_numeric(s, 'log1p')
        result = _apply_numeric(s, params)
        assert len(result) == 5

    def test_nan_filled_with_median(self):
        s = pd.Series([1.0, 2.0, np.nan, 4.0])
        params = _fit_numeric(s, 'min')
        result = _apply_numeric(s, params)
        # NaN should be filled with median=2.0, then min subtracted (1.0) -> 1.0
        assert abs(result.iloc[2] - 1.0) < 1e-6

    def test_invalid_transform_raises(self):
        s = pd.Series([1.0])
        with pytest.raises(ValueError, match="Invalid transform"):
            _fit_numeric(s, 'invalid_option')


class TestFittedTransforms:
    def test_categorical_vocab_sizes(self):
        ft = FittedTransforms(
            categorical_vocabs={'A': {'<UNK>': 0, '<RARE>': 1, 'x': 2, 'y': 3}}
        )
        assert ft.categorical_vocab_sizes == [4]

    def test_empty_defaults(self):
        ft = FittedTransforms()
        assert ft.categorical_vocab_sizes == []
        assert ft.categorical_vocabs == {}
        assert ft.numeric_params == {}

    def test_serialization_roundtrip_min(self):
        ft = FittedTransforms(
            categorical_vocabs={'col': {'<UNK>': 0, '<RARE>': 1, 'val': 2}},
            numeric_params={'PY': {'transform': 'min', 'median': 2020.0, 'min_val': 2000.0}},
        )
        restored = FittedTransforms.from_dict(ft.to_dict())
        assert restored.categorical_vocabs == ft.categorical_vocabs
        assert restored.numeric_params['PY']['min_val'] == 2000.0

    def test_serialization_roundtrip_robust(self):
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        scaler.fit(np.array([[1], [2], [3], [4]]))
        ft = FittedTransforms(
            numeric_params={'x': {'transform': 'robust', 'median': 2.5, 'scaler': scaler}},
        )
        d = ft.to_dict()
        restored = FittedTransforms.from_dict(d)
        assert abs(restored.numeric_params['x']['scaler'].center_[0] - scaler.center_[0]) < 1e-6

    def test_multiple_categorical_vocab_sizes(self):
        ft = FittedTransforms(
            categorical_vocabs={
                'SO': {'<UNK>': 0, '<RARE>': 1, 'JournalA': 2, 'JournalB': 3, 'JournalC': 4},
                'DT': {'<UNK>': 0, '<RARE>': 1, 'Article': 2, 'Review': 3},
                'LA': {'<UNK>': 0, '<RARE>': 1, 'English': 2},
            }
        )
        assert ft.categorical_vocab_sizes == [5, 4, 3]


def _make_mock_tokenizer(max_length=16):
    """Create a mock tokenizer that returns fixed-size tensors."""
    tokenizer = MagicMock()
    def mock_call(texts, max_length=16, truncation=True, padding='max_length', return_tensors='pt'):
        n = len(texts)
        return {
            'input_ids': torch.zeros(n, max_length, dtype=torch.long),
            'attention_mask': torch.ones(n, max_length, dtype=torch.long),
        }
    tokenizer.side_effect = mock_call
    return tokenizer


class TestPreprocessMultipleCategoricalColumns:
    """Test preprocess_dataset with multiple categorical columns."""

    def _make_df(self, n=30):
        np.random.seed(42)
        return pd.DataFrame({
            'TI': [f'Title {i}' for i in range(n)],
            'AB': [f'Abstract {i}' for i in range(n)],
            'SO': np.random.choice(['JournalA', 'JournalB', 'JournalC', 'JournalD'], n),
            'DT': np.random.choice(['Article', 'Review', 'Conference'], n),
            'LA': np.random.choice(['English', 'Spanish', 'French', 'German', 'Rare1'], n),
            'PY': np.random.randint(2000, 2025, n).astype(float),
            'label': np.random.randint(0, 2, n),
        })

    def test_fit_produces_vocab_per_column(self):
        df = self._make_df()
        tokenizer = _make_mock_tokenizer()
        col_spec = {
            'text_cols': ['TI', 'AB'],
            'categorical_cols': ['SO', 'DT', 'LA'],
            'numeric_cols': ['PY'],
            'label_col': 'label',
        }
        dataset, fitted = preprocess_dataset(
            df, tokenizer, torch.device('cpu'), col_spec, {'PY': 'min'},
            max_length=16, rare_threshold=3,
        )
        assert 'SO' in fitted.categorical_vocabs
        assert 'DT' in fitted.categorical_vocabs
        assert 'LA' in fitted.categorical_vocabs
        assert len(fitted.categorical_vocab_sizes) == 3

    def test_categorical_tensor_shape(self):
        df = self._make_df()
        tokenizer = _make_mock_tokenizer()
        col_spec = {
            'text_cols': ['TI', 'AB'],
            'categorical_cols': ['SO', 'DT', 'LA'],
            'numeric_cols': [],
            'label_col': 'label',
        }
        dataset, fitted = preprocess_dataset(
            df, tokenizer, torch.device('cpu'), col_spec, {},
            max_length=16,
        )
        # 3 categorical columns -> tensor shape (n, 3)
        assert dataset.categorical_tensor.shape == (30, 3)
        assert dataset.categorical_tensor.dtype == torch.long

    def test_fitted_transforms_reused_on_val(self):
        df = self._make_df(40)
        train_df = df.iloc[:30].reset_index(drop=True)
        val_df = df.iloc[30:].reset_index(drop=True)
        tokenizer = _make_mock_tokenizer()
        col_spec = {
            'text_cols': ['TI', 'AB'],
            'categorical_cols': ['SO', 'DT'],
            'numeric_cols': ['PY'],
            'label_col': 'label',
        }
        _, fitted = preprocess_dataset(
            train_df, tokenizer, torch.device('cpu'), col_spec, {'PY': 'min'},
            max_length=16,
        )
        val_dataset, none_result = preprocess_dataset(
            val_df, tokenizer, torch.device('cpu'), col_spec, {'PY': 'min'},
            max_length=16, fitted_transforms=fitted,
        )
        assert none_result is None
        assert val_dataset.categorical_tensor.shape[1] == 2

    def test_unseen_category_maps_to_rare(self):
        train_df = pd.DataFrame({
            'TI': ['a'] * 20,
            'AB': ['b'] * 20,
            'SO': ['JournalA'] * 10 + ['JournalB'] * 10,
            'label': [0] * 10 + [1] * 10,
        })
        val_df = pd.DataFrame({
            'TI': ['a'] * 5,
            'AB': ['b'] * 5,
            'SO': ['JournalA', 'JournalB', 'UNSEEN', 'UNSEEN', 'JournalA'],
            'label': [0, 1, 0, 1, 0],
        })
        tokenizer = _make_mock_tokenizer()
        col_spec = {
            'text_cols': ['TI', 'AB'],
            'categorical_cols': ['SO'],
            'numeric_cols': [],
            'label_col': 'label',
        }
        _, fitted = preprocess_dataset(
            train_df, tokenizer, torch.device('cpu'), col_spec, {},
            max_length=16, rare_threshold=5,
        )
        val_dataset, _ = preprocess_dataset(
            val_df, tokenizer, torch.device('cpu'), col_spec, {},
            max_length=16, fitted_transforms=fitted,
        )
        rare_idx = fitted.categorical_vocabs['SO']['<RARE>']
        # Rows 2, 3 have 'UNSEEN' -> should map to RARE
        assert val_dataset.categorical_tensor[2, 0].item() == rare_idx
        assert val_dataset.categorical_tensor[3, 0].item() == rare_idx


class TestPreprocessMultipleNumericColumns:
    """Test preprocess_dataset with multiple numeric columns and different transforms."""

    def _make_df(self, n=30):
        np.random.seed(42)
        return pd.DataFrame({
            'TI': [f'Title {i}' for i in range(n)],
            'AB': [f'Abstract {i}' for i in range(n)],
            'PY': np.random.randint(2000, 2025, n).astype(float),
            'TC': np.random.exponential(10, n),
            'NR': np.random.normal(30, 10, n),
            'label': np.random.randint(0, 2, n),
        })

    def test_multiple_numeric_tensor_shape(self):
        df = self._make_df()
        tokenizer = _make_mock_tokenizer()
        col_spec = {
            'text_cols': ['TI', 'AB'],
            'categorical_cols': [],
            'numeric_cols': ['PY', 'TC', 'NR'],
            'label_col': 'label',
        }
        numeric_transform = {'PY': 'min', 'TC': 'log1p', 'NR': 'robust'}
        dataset, fitted = preprocess_dataset(
            df, tokenizer, torch.device('cpu'), col_spec, numeric_transform,
            max_length=16,
        )
        assert dataset.numeric_tensor.shape == (30, 3)
        assert 'PY' in fitted.numeric_params
        assert 'TC' in fitted.numeric_params
        assert 'NR' in fitted.numeric_params

    def test_each_column_gets_own_transform(self):
        df = self._make_df()
        tokenizer = _make_mock_tokenizer()
        col_spec = {
            'text_cols': ['TI', 'AB'],
            'categorical_cols': [],
            'numeric_cols': ['PY', 'TC', 'NR'],
            'label_col': 'label',
        }
        numeric_transform = {'PY': 'min', 'TC': 'max', 'NR': 'mean'}
        _, fitted = preprocess_dataset(
            df, tokenizer, torch.device('cpu'), col_spec, numeric_transform,
            max_length=16,
        )
        assert fitted.numeric_params['PY']['transform'] == 'min'
        assert fitted.numeric_params['TC']['transform'] == 'max'
        assert fitted.numeric_params['NR']['transform'] == 'mean'

    def test_val_uses_train_numeric_params(self):
        np.random.seed(42)
        train_df = pd.DataFrame({
            'TI': ['a'] * 20, 'AB': ['b'] * 20,
            'PY': np.arange(2000, 2020, dtype=float),
            'TC': np.arange(1, 21, dtype=float),
            'label': [0] * 10 + [1] * 10,
        })
        val_df = pd.DataFrame({
            'TI': ['a'] * 5, 'AB': ['b'] * 5,
            'PY': [2025.0, 2026.0, 2027.0, 2028.0, 2029.0],
            'TC': [100.0, 200.0, 300.0, 400.0, 500.0],
            'label': [0, 1, 0, 1, 0],
        })
        tokenizer = _make_mock_tokenizer()
        col_spec = {
            'text_cols': ['TI', 'AB'],
            'categorical_cols': [],
            'numeric_cols': ['PY', 'TC'],
            'label_col': 'label',
        }
        _, fitted = preprocess_dataset(
            train_df, tokenizer, torch.device('cpu'), col_spec,
            {'PY': 'min', 'TC': 'max'}, max_length=16,
        )
        # PY min fitted on train: min=2000
        assert fitted.numeric_params['PY']['min_val'] == 2000.0
        # TC max fitted on train: max=20
        assert fitted.numeric_params['TC']['max_val'] == 20.0

        val_dataset, _ = preprocess_dataset(
            val_df, tokenizer, torch.device('cpu'), col_spec,
            {'PY': 'min', 'TC': 'max'}, max_length=16, fitted_transforms=fitted,
        )
        # Val PY[0] = 2025 - 2000 = 25
        assert abs(val_dataset.numeric_tensor[0, 0].item() - 25.0) < 1e-4
        # Val TC[0] = 100 / 20 = 5.0 (uses train max, not val max)
        assert abs(val_dataset.numeric_tensor[0, 1].item() - 5.0) < 1e-4


class TestPreprocessMultipleColumnsEndToEnd:
    """End-to-end tests with multiple categorical + numeric columns together."""

    def test_combined_multiple_columns_single_label(self):
        np.random.seed(42)
        n = 20
        df = pd.DataFrame({
            'TI': [f'Title {i}' for i in range(n)],
            'AB': [f'Abstract {i}' for i in range(n)],
            'SO': np.random.choice(['J1', 'J2', 'J3'], n),
            'DT': np.random.choice(['Article', 'Review'], n),
            'PY': np.random.randint(2000, 2025, n).astype(float),
            'TC': np.random.exponential(5, n),
            'label': np.random.randint(0, 2, n),
        })
        tokenizer = _make_mock_tokenizer()
        col_spec = {
            'text_cols': ['TI', 'AB'],
            'categorical_cols': ['SO', 'DT'],
            'numeric_cols': ['PY', 'TC'],
            'label_col': 'label',
        }
        dataset, fitted = preprocess_dataset(
            df, tokenizer, torch.device('cpu'), col_spec,
            {'PY': 'min', 'TC': 'robust'}, max_length=16, rare_threshold=2,
        )
        assert dataset.categorical_tensor.shape == (n, 2)
        assert dataset.categorical_tensor.dtype == torch.long
        assert dataset.numeric_tensor.shape == (n, 2)
        assert dataset.labels.shape == (n, 1)
        assert len(fitted.categorical_vocab_sizes) == 2

    def test_combined_multiple_columns_multi_label(self):
        np.random.seed(42)
        n = 20
        df = pd.DataFrame({
            'TI': [f'Title {i}' for i in range(n)],
            'AB': [f'Abstract {i}' for i in range(n)],
            'SO': np.random.choice(['J1', 'J2', 'J3'], n),
            'DT': np.random.choice(['Article', 'Review'], n),
            'PY': np.random.randint(2000, 2025, n).astype(float),
            'TC': np.random.exponential(5, n),
            'single_case': np.random.randint(0, 2, n),
            'tech_use': np.random.randint(0, 2, n),
        })
        tokenizer = _make_mock_tokenizer()
        col_spec = {
            'text_cols': ['TI', 'AB'],
            'categorical_cols': ['SO', 'DT'],
            'numeric_cols': ['PY', 'TC'],
            'label_col': ['single_case', 'tech_use'],
        }
        dataset, fitted = preprocess_dataset(
            df, tokenizer, torch.device('cpu'), col_spec,
            {'PY': 'min', 'TC': 'robust'}, max_length=16, rare_threshold=2,
        )
        assert dataset.categorical_tensor.shape == (n, 2)
        assert dataset.numeric_tensor.shape == (n, 2)
        assert dataset.labels.shape == (n, 2)
        assert dataset.labels.dtype == torch.float
