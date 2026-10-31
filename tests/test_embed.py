"""Tests for the cached-embedding fast path."""

import pytest
import torch
from torch.utils.data import DataLoader

from pubmlp.embed import compute_cls_embeddings
from pubmlp.model import PubMLP
from pubmlp.preprocess import CachedEmbeddingDataset, CustomDataset, collate_fn
from pubmlp.train import train_evaluate_model
from pubmlp.utils import cached_forward_fn


def _make_bert_model(**kwargs):
    defaults = dict(
        mlp_hidden_size=16,
        embedding_model='bert',
        model_name='bert-base-uncased',
    )
    defaults.update(kwargs)
    m = PubMLP(**defaults)
    m.eval()
    return m


class TestComputeClsEmbeddings:
    def test_shape_and_cpu(self):
        model = _make_bert_model()
        hidden = model.encoder.config.hidden_size
        ds = CustomDataset(
            input_ids=torch.zeros(4, 8, dtype=torch.long),
            attention_mask=torch.ones(4, 8, dtype=torch.long),
            labels=torch.zeros(4, 1),
            texts=["a", "b", "c", "d"],
        )
        loader = DataLoader(ds, batch_size=2, collate_fn=collate_fn)
        emb = compute_cls_embeddings(model, loader, torch.device('cpu'), show_progress=False)
        assert emb.shape == (4, hidden)
        assert emb.device.type == 'cpu'
        assert not emb.requires_grad

    def test_preserves_training_mode(self):
        model = _make_bert_model()
        model.train()
        ds = CustomDataset(
            input_ids=torch.zeros(2, 8, dtype=torch.long),
            attention_mask=torch.ones(2, 8, dtype=torch.long),
            labels=torch.zeros(2, 1),
            texts=["a", "b"],
        )
        loader = DataLoader(ds, batch_size=2, collate_fn=collate_fn)
        compute_cls_embeddings(model, loader, torch.device('cpu'), show_progress=False)
        assert model.training is True


class TestForwardFromEmbedding:
    def test_matches_full_forward_bert(self):
        model = _make_bert_model(
            categorical_vocab_sizes=[5, 3],
            numeric_cols_num=2,
        )
        input_ids = torch.zeros(3, 8, dtype=torch.long)
        attention_mask = torch.ones(3, 8, dtype=torch.long)
        cat = torch.tensor([[1, 2], [3, 0], [0, 1]], dtype=torch.long)
        num = torch.randn(3, 2)

        with torch.no_grad():
            emb = model._encode(input_ids, attention_mask)
            logits_cached = model.forward_from_embedding(emb, cat, num)
            logits_full = model(input_ids, attention_mask, cat, num)

        assert torch.allclose(logits_full, logits_cached, atol=1e-5)

    def test_matches_full_forward_text_only(self):
        model = _make_bert_model()
        input_ids = torch.zeros(2, 8, dtype=torch.long)
        attention_mask = torch.ones(2, 8, dtype=torch.long)

        with torch.no_grad():
            emb = model._encode(input_ids, attention_mask)
            logits_cached = model.forward_from_embedding(emb)
            logits_full = model(input_ids, attention_mask)

        assert torch.allclose(logits_full, logits_cached, atol=1e-5)

    def test_multi_label_output(self):
        model = _make_bert_model(output_size=3)
        input_ids = torch.zeros(2, 8, dtype=torch.long)
        attention_mask = torch.ones(2, 8, dtype=torch.long)
        with torch.no_grad():
            emb = model._encode(input_ids, attention_mask)
            logits_cached = model.forward_from_embedding(emb)
        assert logits_cached.shape == (2, 3)


class TestCachedTrainingRoundtrip:
    def test_one_epoch_cached_path(self):
        model = _make_bert_model(
            categorical_vocab_sizes=[4],
            numeric_cols_num=1,
        )
        device = torch.device('cpu')

        hidden = model.encoder.config.hidden_size
        n = 8
        emb = torch.randn(n, hidden)
        labels = torch.tensor([[1.0]] * 4 + [[0.0]] * 4)
        cat = torch.tensor([[0], [1], [2], [3], [0], [1], [2], [3]], dtype=torch.long)
        num = torch.randn(n, 1)

        def make_loader(indices, batch_size=4):
            ds = CachedEmbeddingDataset(emb[indices], labels[indices], cat[indices], num[indices])
            return DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn)

        train_loader = make_loader(list(range(6)), batch_size=3)
        val_loader = make_loader([6, 7], batch_size=2)

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-3,
        )
        criterion = torch.nn.BCEWithLogitsLoss()
        result = train_evaluate_model(
            model, train_loader, val_loader, None,
            optimizer, criterion, device, epochs=1,
            use_warmup=False, pos_weight=None,
            forward_fn=cached_forward_fn,
            early_stopping_patience=5,
        )
        train_losses = result[0]
        assert len(train_losses) == 1
        assert torch.isfinite(torch.tensor(train_losses[0]))
