"""Tests for pubmlp.benchmark module."""

import os

import numpy as np
import pandas as pd
import pytest
import torch

from conftest import needs_weights

from pubmlp.benchmark import _CachedHead, _journal_vocab, _tabular_tensors, _joined_text


def synthetic_frame(n=60, seed=0):
    rng = np.random.RandomState(seed)
    labels = np.array([1] * (n // 3) + [0] * (n - n // 3))
    return pd.DataFrame({
        'title': [f'record {i}' for i in range(n)],
        'abstract': ['relevant screening study' if l else 'unrelated topic entirely'
                     for l in labels],
        'year': rng.randint(2000, 2020, n).astype(float),
        'journal': rng.choice(['J1', 'J2', 'J3'], n),
        'label_included': labels,
    })


FULL_SPECS = {'text_cols': ['title', 'abstract'], 'categorical_cols': ['journal'],
              'numeric_cols': ['year'], 'label_col': 'label_included'}
TEXT_SPECS = {'text_cols': ['title', 'abstract'], 'categorical_cols': [],
              'numeric_cols': [], 'label_col': 'label_included'}


class TestCachedHead:
    def test_text_only_forward_shape(self):
        head = _CachedHead(embedding_size=16, mlp_hidden_size=8)
        out = head.forward_from_embedding(torch.randn(5, 16))
        assert out.shape == (5, 1)

    def test_fused_forward_shape(self):
        head = _CachedHead(embedding_size=16, journal_vocab_size=4, use_year=True,
                           mlp_hidden_size=8)
        out = head.forward_from_embedding(
            torch.randn(5, 16),
            torch.tensor([[1], [2], [3], [0], [1]]),
            torch.randn(5, 1),
        )
        assert out.shape == (5, 1)


class TestTabularTensors:
    def test_vocab_reserves_zero_for_unseen(self):
        df = synthetic_frame(n=12)
        vocab = _journal_vocab(df)
        assert 0 not in vocab.values()
        assert set(vocab.keys()) == set(df['journal'].unique())

    def test_tensor_shapes_and_unseen_mapping(self):
        df = synthetic_frame(n=12)
        vocab = _journal_vocab(df.iloc[:6])
        holdout = df.copy()
        holdout.loc[6:, 'journal'] = 'NEVER_SEEN'
        categorical, numeric = _tabular_tensors(holdout, FULL_SPECS, vocab, 2000.0)
        assert categorical.shape == (12, 1)
        assert numeric.shape == (12, 1)
        assert (categorical[6:] == 0).all()

    def test_text_only_returns_empty(self):
        df = synthetic_frame(n=5)
        categorical, numeric = _tabular_tensors(df, TEXT_SPECS, {}, 0.0)
        assert categorical.numel() == 0
        assert numeric.numel() == 0


def test_joined_text():
    df = pd.DataFrame({'title': ['a'], 'abstract': ['b']})
    assert _joined_text(df).iloc[0] == 'a b'


from pubmlp.benchmark import embed_dataset, _fit_cached, _metric_row
from pubmlp.config import Config


def small_config():
    return Config(random_seed=42, batch_size=16, eval_batch_size=32, epochs=2,
                  mlp_hidden_size=8, n_hidden_layers=1, early_stopping_patience=2,
                  al_batch_size=10, al_initial_sample_pct=0.2)


def synthetic_embeddings(df, dim=16, seed=0):
    rng = np.random.RandomState(seed)
    base = rng.randn(len(df), dim).astype('float32')
    base[df['label_included'].values == 1, 0] += 2.0
    return torch.tensor(base)


class TestMetricRow:
    def test_keys_with_probs(self):
        y = np.array([0, 1, 0, 1])
        row = _metric_row(y, np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.2, 0.8]))
        for key in ['f1', 'precision', 'recall', 'roc_auc', 'ece', 'brier', 'ndcg', 'wss95']:
            assert key in row
        assert row['f1'] == 1.0

    def test_none_probs(self):
        y = np.array([0, 1])
        row = _metric_row(y, np.array([0, 1]), None)
        assert row['roc_auc'] is None
        assert row['wss95'] is None

    def test_single_class_y_true_roc_auc_none(self):
        y = np.array([0, 0, 0, 0])
        row = _metric_row(y, np.array([0, 1, 0, 0]), np.array([0.1, 0.9, 0.2, 0.3]))
        assert row['roc_auc'] is None
        assert row['f1'] == 0.0
        assert row['precision'] == 0.0
        assert row['recall'] == 0.0


class TestFitCached:
    def test_shapes_and_predict_indices(self):
        df = synthetic_frame(n=60)
        embeddings = synthetic_embeddings(df)
        specs = FULL_SPECS
        idx = np.arange(60)
        preds, probs, predict_indices = _fit_cached(
            embeddings, df, specs, idx[:40], idx[40:50], idx[50:],
            small_config(), torch.device('cpu'), seed=1,
        )
        assert preds.shape == (10,)
        assert probs.shape == (10,)
        assert set(np.unique(preds)).issubset({0, 1})
        more_preds, more_probs = predict_indices(idx[:5])
        assert more_preds.shape == (5,)
        assert more_probs.shape == (5,)

    def test_text_only_specs(self):
        df = synthetic_frame(n=60)
        embeddings = synthetic_embeddings(df)
        specs = TEXT_SPECS
        idx = np.arange(60)
        preds, probs, predict_indices = _fit_cached(
            embeddings, df, specs, idx[:40], idx[40:50], idx[50:],
            small_config(), torch.device('cpu'), seed=1,
        )
        assert preds.shape == (10,)
        assert probs.shape == (10,)
        assert np.all((probs >= 0) & (probs <= 1))


class TestEmbedCache:
    def test_cache_hit_skips_embed_fn(self, tmp_path):
        df = synthetic_frame(n=8)
        calls = []

        def stub(texts):
            calls.append(len(texts))
            return np.random.RandomState(0).randn(len(texts), 4).astype('float32')

        first = embed_dataset(df, cache_dir=tmp_path, embed_fn=stub)
        second = embed_dataset(df, cache_dir=tmp_path, embed_fn=stub)
        assert calls == [8]
        assert first.shape == (8, 4)
        assert torch.equal(first, second)

    def test_different_data_different_cache(self, tmp_path):
        calls = []

        def stub(texts):
            calls.append(len(texts))
            return np.zeros((len(texts), 4), dtype='float32')

        embed_dataset(synthetic_frame(n=8), cache_dir=tmp_path, embed_fn=stub)
        embed_dataset(synthetic_frame(n=9), cache_dir=tmp_path, embed_fn=stub)
        assert calls == [8, 9]

    def test_model_name_in_cache_key(self, tmp_path):
        from pubmlp.config import Config
        calls = []

        def stub(texts):
            calls.append(len(texts))
            return np.random.RandomState(0).randn(len(texts), 4).astype('float32')

        df = synthetic_frame(n=8)
        config_m1 = Config(random_seed=42, embedding_model='sentence-transformer', model_name='m1')
        config_m2 = Config(random_seed=42, embedding_model='sentence-transformer', model_name='m2')

        embed_dataset(df, config=config_m1, cache_dir=tmp_path, embed_fn=stub)
        embed_dataset(df, config=config_m2, cache_dir=tmp_path, embed_fn=stub)
        assert calls == [8, 8]


from pubmlp.benchmark import _run_random, _run_tfidf_nb, _fit_cached


class TestStrategyRunners:
    def _splits(self, n=60):
        pool_idx = np.r_[0:13, 20:47]
        val_idx = np.r_[13:16, 47:54]
        test_idx = np.r_[16:20, 54:60]
        return pool_idx, val_idx, test_idx

    def test_run_tfidf_nb_single_row(self):
        df = synthetic_frame(n=60)
        pool_idx, _, test_idx = self._splits()
        rows = _run_tfidf_nb(df, pool_idx, test_idx, seed=1, sample_pct=0.5)
        assert len(rows) == 1
        assert rows[0]['strategy'] == 'tfidf_nb'
        assert rows[0]['n_coded'] == 20
        assert not np.isnan(rows[0]['roc_auc'])

    def test_run_random_one_row_per_pct(self):
        df = synthetic_frame(n=60)
        embeddings = synthetic_embeddings(df)
        pool_idx, val_idx, test_idx = self._splits()
        rows = _run_random(df, embeddings, FULL_SPECS, pool_idx, val_idx, test_idx,
                           small_config(), torch.device('cpu'), seed=1,
                           fit=_fit_cached, random_pcts=[0.25, 0.5])
        assert [r['strategy'] for r in rows] == ['random', 'random']
        assert rows[0]['n_coded'] < rows[1]['n_coded']

    def test_run_random_skips_tiny_pcts(self):
        df = synthetic_frame(n=60)
        embeddings = synthetic_embeddings(df)
        pool_idx, val_idx, test_idx = self._splits()
        rows = _run_random(df, embeddings, FULL_SPECS, pool_idx, val_idx, test_idx,
                           small_config(), torch.device('cpu'), seed=1,
                           fit=_fit_cached, random_pcts=None)
        assert len(rows) > 0
        assert all(r['n_coded'] >= 2 for r in rows)


from pubmlp.benchmark import _run_active_learning, _run_semantic


class TestActiveLearning:
    def test_rows_carry_stopping_fields(self):
        df = synthetic_frame(n=60)
        embeddings = synthetic_embeddings(df)
        idx = np.arange(60)
        rows = _run_active_learning(df, embeddings, FULL_SPECS, idx[:40], idx[40:50],
                                    idx[50:], small_config(), torch.device('cpu'),
                                    seed=1, fit=_fit_cached)
        assert 1 <= len(rows) <= 5
        for row in rows:
            assert row['strategy'] == 'active_learning'
            assert 0.0 <= row['recall_lb'] <= 1.0
            assert isinstance(row['stop_stat'], bool)
        n_coded = [row['n_coded'] for row in rows]
        assert n_coded == sorted(n_coded)

    def test_small_pool_raises(self):
        df = synthetic_frame(n=60)
        embeddings = synthetic_embeddings(df)
        idx = np.arange(60)
        with pytest.raises(ValueError, match='pool too small'):
            _run_active_learning(df, embeddings, FULL_SPECS, idx[:15], idx[40:50],
                                idx[50:], small_config(), torch.device('cpu'),
                                seed=1, fit=_fit_cached)


class TestSemantic:
    def test_semantic_row(self):
        df = synthetic_frame(n=60)
        embeddings = synthetic_embeddings(df).numpy()
        description = np.zeros(16, dtype='float32')
        description[0] = 1.0
        idx = np.arange(60)
        test_idx = idx[15:25]
        pool_idx = np.setdiff1d(idx, test_idx)
        rows = _run_semantic(df, embeddings, description, pool_idx, test_idx)
        assert rows[0]['strategy'] == 'semantic'
        assert rows[0]['n_coded'] == 0
        assert rows[0]['roc_auc'] is not None


from pubmlp.benchmark import _fit_finetune


@needs_weights('bert-base-uncased')
class TestFitFinetune:
    def test_contract(self):
        df = synthetic_frame(n=60)
        idx = np.arange(60)
        config = small_config()
        config.max_length = 64
        config.epochs = 1
        preds, probs, predict_indices = _fit_finetune(
            None, df, FULL_SPECS, idx[:40], idx[40:50], idx[50:],
            config, torch.device('cpu'), seed=1,
        )
        assert np.asarray(preds).shape == (10,)
        assert np.asarray(probs).shape == (10,)
        assert ((probs >= 0) & (probs <= 1)).all()
        more_preds, more_probs = predict_indices(idx[:5])
        assert np.asarray(more_preds).shape == (5,)
        assert np.asarray(more_probs).shape == (5,)


def test_fit_finetune_importable():
    assert callable(_fit_finetune)


from pubmlp.benchmark import run_simulation, summarize_runs

RUN_COLUMNS = ['dataset', 'seed', 'strategy', 'fusion', 'n_coded', 'f1', 'precision',
               'recall', 'roc_auc', 'ece', 'brier', 'ndcg', 'wss95', 'recall_lb', 'stop_stat']


class TestRunSimulation:
    def test_schema_and_fusion(self):
        df = synthetic_frame(n=60)
        runs = run_simulation(df, strategies=['tfidf_nb', 'active_learning'],
                              seeds=[1, 2], config=small_config(),
                              embeddings=synthetic_embeddings(df),
                              dataset_name='synthetic')
        assert list(runs.columns) == RUN_COLUMNS
        assert set(runs['seed']) == {1, 2}
        assert (runs['fusion'] == 'full').all()
        al = runs[runs['strategy'] == 'active_learning']
        assert al['recall_lb'].notna().all()
        assert runs[runs['strategy'] == 'tfidf_nb']['recall_lb'].isna().all()

    def test_text_only_downgrade(self):
        df = synthetic_frame(n=60)
        df['journal'] = np.nan
        df['year'] = np.nan
        runs = run_simulation(df, strategies=['tfidf_nb'], seeds=[1],
                              config=small_config(), embeddings=synthetic_embeddings(df))
        assert (runs['fusion'] == 'text_only').all()

    def test_missing_embeddings_raises(self):
        with pytest.raises(ValueError, match='embeddings'):
            run_simulation(synthetic_frame(), strategies=['random'], seeds=[1])

    def test_unknown_strategy_raises(self):
        df = synthetic_frame(n=60)
        with pytest.raises(ValueError, match='unknown strategy'):
            run_simulation(df, strategies=['bogus'], seeds=[1],
                           embeddings=synthetic_embeddings(df))

    def test_semantic_without_embeddings_raises(self):
        with pytest.raises(ValueError, match='semantic'):
            run_simulation(synthetic_frame(n=60), strategies=['semantic'], seeds=[1],
                           engine='finetune')

    def test_unknown_strategy_raises_with_no_seeds(self):
        df = synthetic_frame(n=60)
        with pytest.raises(ValueError, match='unknown strategy'):
            run_simulation(df, strategies=['bogus'], seeds=[],
                           embeddings=synthetic_embeddings(df))

    def test_semantic_finetune_engine_numpy_embeddings(self):
        df = synthetic_frame(n=60)
        embeddings = synthetic_embeddings(df).numpy()
        description = np.zeros(16, dtype='float32')
        description[0] = 1.0
        runs = run_simulation(df, strategies=['semantic'], seeds=[1], engine='finetune',
                              embeddings=embeddings, description_embedding=description)
        assert list(runs.columns) == RUN_COLUMNS
        assert (runs['strategy'] == 'semantic').any()


class TestSummarize:
    def test_summary_and_curves(self):
        df = synthetic_frame(n=60)
        runs = run_simulation(df, strategies=['tfidf_nb'], seeds=[1, 2],
                              config=small_config(), embeddings=synthetic_embeddings(df))
        result = summarize_runs(runs)
        assert set(result) == {'summary', 'curves'}
        assert len(result['summary']) == 1

    def test_curves_include_f1_recall_wss95(self):
        df = synthetic_frame(n=60)
        runs = run_simulation(df, strategies=['random'], seeds=[1, 2],
                              config=small_config(), embeddings=synthetic_embeddings(df))
        curves = summarize_runs(runs)['curves']
        top_level = set(curves.columns.get_level_values(0))
        assert {'f1', 'recall', 'wss95'}.issubset(top_level)


import pubmlp.benchmark as benchmark_module


class TestCli:
    def test_main_writes_output(self, tmp_path, monkeypatch):
        df = synthetic_frame(n=60)
        data_path = tmp_path / 'records.csv'
        df.to_csv(data_path, index=False)
        out_path = tmp_path / 'runs.csv'
        monkeypatch.setattr(benchmark_module, 'embed_dataset',
                            lambda frame, **kwargs: synthetic_embeddings(frame))
        code = benchmark_module.main([
            '--dataset', str(data_path), '--strategies', 'tfidf_nb',
            '--seeds', '1', '--out', str(out_path),
        ])
        assert code == 0
        written = pd.read_csv(out_path)
        assert list(written.columns) == benchmark_module.RUN_COLUMNS
        assert len(written) == 1

    def test_main_honors_model_flag(self, tmp_path, monkeypatch):
        df = synthetic_frame(n=60)
        data_path = tmp_path / 'records.csv'
        df.to_csv(data_path, index=False)
        out_path = tmp_path / 'runs.csv'
        seen_configs = []

        def fake_embed_dataset(frame, config=None, **kwargs):
            seen_configs.append(config)
            return synthetic_embeddings(frame)

        monkeypatch.setattr(benchmark_module, 'embed_dataset', fake_embed_dataset)
        code = benchmark_module.main([
            '--dataset', str(data_path), '--strategies', 'tfidf_nb',
            '--seeds', '1', '--out', str(out_path),
            '--model', 'm-custom', '--embedding-model', 'sentence-transformer',
        ])
        assert code == 0
        assert seen_configs[0].model_name == 'm-custom'
        written = pd.read_csv(out_path)
        assert list(written.columns) == benchmark_module.RUN_COLUMNS

    def test_missing_dataset_file_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(benchmark_module, 'embed_dataset',
                            lambda frame, **kwargs: synthetic_embeddings(frame))
        with pytest.raises(ValueError, match='not found'):
            benchmark_module.main([
                '--dataset', str(tmp_path / 'nope.csv'),
                '--strategies', 'tfidf_nb', '--seeds', '1',
                '--out', str(tmp_path / 'out.csv'),
            ])


def test_package_exports():
    import pubmlp
    for name in ['list_benchmarks', 'load_benchmark', 'load_manifest_corpus',
                 'normalize_benchmark_frame', 'build_column_specs',
                 'embed_dataset', 'run_simulation', 'summarize_runs']:
        assert hasattr(pubmlp, name)
        assert name in pubmlp.__all__
