"""Tests for pubmlp.utils module."""

import os
import tempfile
import pytest
import torch
import pandas as pd

from pubmlp.utils import get_device, auto_batch_size, load_data, unpack_batch


class TestGetDevice:
    def test_returns_torch_device(self):
        device = get_device()
        assert isinstance(device, torch.device)

    def test_device_type(self):
        device = get_device()
        assert device.type in ('cpu', 'cuda')


class TestAutoBatchSize:
    def test_cpu_returns_8(self):
        assert auto_batch_size(torch.device('cpu')) == 8

    def test_returns_int(self):
        device = get_device()
        bs = auto_batch_size(device)
        assert isinstance(bs, int) and bs > 0


class TestLoadData:
    def test_load_csv(self, tmp_path):
        csv_file = tmp_path / "test.csv"
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(csv_file, index=False)
        df = load_data(csv_file)
        assert len(df) == 2
        assert list(df.columns) == ["a", "b"]

    def test_load_excel(self, tmp_path):
        xlsx_file = tmp_path / "test.xlsx"
        pd.DataFrame({"x": [10]}).to_excel(xlsx_file, index=False)
        df = load_data(xlsx_file)
        assert len(df) == 1
        assert "x" in df.columns


class TestUnpackBatch:
    def test_unpacks_all_fields(self):
        batch = {
            'input_ids': torch.zeros(2, 4, dtype=torch.long),
            'attention_mask': torch.ones(2, 4, dtype=torch.long),
            'categorical_tensor': torch.tensor([[2], [1]], dtype=torch.long),
            'numeric_tensor': torch.tensor([[1.0], [2.0]]),
            'labels': torch.tensor([[1.0], [0.0]]),
            'texts': ['a', 'b'],
        }
        input_ids, attn, cat, num, labels, texts = unpack_batch(batch, torch.device('cpu'))
        assert input_ids.shape == (2, 4)
        assert labels.dtype == torch.float32
        assert texts == ['a', 'b']

    def test_none_texts(self):
        batch = {
            'input_ids': torch.zeros(1, 2, dtype=torch.long),
            'attention_mask': torch.ones(1, 2, dtype=torch.long),
            'categorical_tensor': torch.tensor([[0.1]]),
            'numeric_tensor': torch.tensor([[0.2]]),
            'labels': torch.tensor([0]),
        }
        *_, texts = unpack_batch(batch, torch.device('cpu'))
        assert texts is None
