"""Tests for pubmlp.datasets module."""

import numpy as np
import pandas as pd
import pytest
import sys

from pubmlp.datasets import (
    BENCHMARK_COLUMNS, normalize_benchmark_frame, metadata_fusion, build_column_specs,
    load_manifest_corpus, list_benchmarks, load_benchmark,
)


def full_frame(n=10):
    return pd.DataFrame({
        'title': [f't{i}' for i in range(n)],
        'abstract': [f'a{i}' for i in range(n)],
        'year': [2000.0 + i for i in range(n)],
        'journal': ['J1'] * n,
        'label_included': [1, 0] * (n // 2),
    })


class TestNormalize:
    def test_renames_and_orders_columns(self):
        raw = pd.DataFrame({
            'title': ['t'], 'abstract': ['a'],
            'publication_year': [2001], 'journal_name': ['J'], 'included': [1],
        })
        out = normalize_benchmark_frame(raw)
        assert list(out.columns) == BENCHMARK_COLUMNS
        assert out.loc[0, 'year'] == 2001.0
        assert out.loc[0, 'journal'] == 'J'
        assert out.loc[0, 'label_included'] == 1

    def test_fills_missing_columns_and_text(self):
        raw = pd.DataFrame({'title': [None, 't'], 'abstract': ['a', None],
                            'label_included': [0, 1]})
        out = normalize_benchmark_frame(raw)
        assert out.loc[0, 'title'] == ''
        assert out.loc[1, 'abstract'] == ''
        assert out['year'].isna().all()
        assert out['journal'].isna().all()

    def test_drops_unlabeled_rows(self):
        raw = full_frame()
        raw.loc[0, 'label_included'] = np.nan
        out = normalize_benchmark_frame(raw)
        assert len(out) == 9
        assert out['label_included'].dtype == int

    def test_rename_collision_keeps_canonical(self):
        raw = pd.DataFrame({
            'title': ['t'], 'abstract': ['a'],
            'year': [1999], 'publication_year': [2001],
            'journal': ['Canonical'], 'journal_name': ['Alias'],
            'label_included': [1], 'included': [0],
        })
        out = normalize_benchmark_frame(raw)
        assert list(out.columns) == BENCHMARK_COLUMNS
        assert out.loc[0, 'year'] == 1999.0
        assert out.loc[0, 'journal'] == 'Canonical'
        assert out.loc[0, 'label_included'] == 1


class TestFusion:
    def test_full(self):
        assert metadata_fusion(full_frame()) == 'full'

    def test_partial(self):
        df = full_frame()
        df['journal'] = np.nan
        assert metadata_fusion(df) == 'partial'

    def test_text_only(self):
        df = full_frame()
        df['journal'] = np.nan
        df['year'] = np.nan
        assert metadata_fusion(df) == 'text_only'

    def test_low_coverage_column_excluded(self):
        df = full_frame(n=10)
        df.loc[:4, 'journal'] = np.nan
        assert metadata_fusion(df) == 'partial'

    def test_absent_columns_treated_as_text_only(self):
        df = pd.DataFrame({
            'title': ['t1', 't2'], 'abstract': ['a1', 'a2'], 'label_included': [1, 0],
        })
        assert metadata_fusion(df) == 'text_only'


class TestColumnSpecs:
    def test_full_specs(self):
        specs, numeric_transform, fusion = build_column_specs(full_frame())
        assert specs == {'text_cols': ['title', 'abstract'], 'categorical_cols': ['journal'],
                         'numeric_cols': ['year'], 'label_col': 'label_included'}
        assert numeric_transform == {'year': 'min'}
        assert fusion == 'full'

    def test_text_only_specs(self):
        df = full_frame()
        df['journal'] = np.nan
        df['year'] = np.nan
        specs, numeric_transform, fusion = build_column_specs(df)
        assert specs['categorical_cols'] == []
        assert specs['numeric_cols'] == []
        assert numeric_transform == {}
        assert fusion == 'text_only'

    def test_absent_columns_specs(self):
        df = pd.DataFrame({
            'title': ['t1', 't2'], 'abstract': ['a1', 'a2'], 'label_included': [1, 0],
        })
        specs, numeric_transform, fusion = build_column_specs(df)
        assert specs['categorical_cols'] == []
        assert specs['numeric_cols'] == []
        assert numeric_transform == {}
        assert fusion == 'text_only'


class TestManifestCorpus:
    def _write(self, tmp_path, manifest_rows, record_rows):
        manifest = tmp_path / 'manifest.csv'
        records = tmp_path / 'records.csv'
        pd.DataFrame(manifest_rows).to_csv(manifest, index=False)
        pd.DataFrame(record_rows).to_csv(records, index=False)
        return manifest, records

    def test_full_match(self, tmp_path):
        manifest, records = self._write(
            tmp_path,
            {'UT': ['A', 'B'], 'single_case': [1, 0]},
            {'UT': ['A', 'B'], 'title': ['t1', 't2'], 'abstract': ['a1', 'a2']},
        )
        out = load_manifest_corpus(manifest, records)
        assert len(out) == 2
        assert set(['single_case', 'title']).issubset(out.columns)

    def test_below_coverage_raises(self, tmp_path):
        manifest, records = self._write(
            tmp_path,
            {'UT': list('ABCDEFGHIJKLMNOPQRST'), 'label': [1] * 20},
            {'UT': list('ABCDEFGHIJKLMNOPQR'), 'title': ['t'] * 18},
        )
        with pytest.raises(ValueError, match='coverage'):
            load_manifest_corpus(manifest, records)

    def test_missing_id_column_raises(self, tmp_path):
        manifest, records = self._write(
            tmp_path, {'other': ['A'], 'label': [1]}, {'UT': ['A'], 'title': ['t']},
        )
        with pytest.raises(ValueError, match='UT'):
            load_manifest_corpus(manifest, records)

    def test_duplicate_record_ids_raises_coverage_error(self, tmp_path):
        manifest, records = self._write(
            tmp_path,
            {'UT': list('ABCDEFGHIJ'), 'label': [1] * 10},
            {'UT': list('ABCDEABCDE'), 'title': ['t'] * 10},
        )
        with pytest.raises(ValueError, match='coverage'):
            load_manifest_corpus(manifest, records)

    def test_duplicate_record_ids_no_duplicates_in_result(self, tmp_path):
        manifest, records = self._write(
            tmp_path,
            {'UT': ['A', 'B', 'C', 'D', 'E'], 'label': [1, 1, 1, 1, 1]},
            {'UT': ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D', 'E', 'E'],
             'title': [f't{i}' for i in range(10)]},
        )
        out = load_manifest_corpus(manifest, records)
        assert len(out) == 5
        assert out[out.columns[0]].nunique() == 5

    def test_empty_manifest_raises_coverage_error(self, tmp_path):
        manifest, records = self._write(
            tmp_path,
            {'UT': [], 'label': []},
            {'UT': ['A'], 'title': ['t']},
        )
        with pytest.raises(ValueError, match='coverage'):
            load_manifest_corpus(manifest, records)

    def _mock_load_data(self, monkeypatch, manifest_df, records_df):
        def fake_load_data(path):
            return manifest_df if 'manifest' in str(path) else records_df
        monkeypatch.setattr('pubmlp.utils.load_data', fake_load_data)

    def test_int_and_string_ids_still_match(self, monkeypatch):
        manifest_df = pd.DataFrame({'UT': [1001, 1002, 1003], 'label': [1, 0, 1]})
        records_df = pd.DataFrame({'UT': ['1001', '1002', '1003'], 'title': ['t1', 't2', 't3']})
        assert manifest_df['UT'].dtype != records_df['UT'].dtype
        self._mock_load_data(monkeypatch, manifest_df, records_df)
        out = load_manifest_corpus('manifest.csv', 'records.csv')
        assert len(out) == 3

    def test_disjoint_ids_raise_no_ids_matched(self, monkeypatch):
        manifest_df = pd.DataFrame({'UT': [1001, 1002, 1003], 'label': [1, 0, 1]})
        records_df = pd.DataFrame({'UT': ['X1', 'X2', 'X3'], 'title': ['t1', 't2', 't3']})
        self._mock_load_data(monkeypatch, manifest_df, records_df)
        with pytest.raises(ValueError, match='no IDs matched'):
            load_manifest_corpus('manifest.csv', 'records.csv')


class TestSynergyWrappers:
    def test_missing_extra_hint(self, monkeypatch):
        monkeypatch.setitem(sys.modules, 'synergy_dataset', None)
        with pytest.raises(ImportError, match=r'pip install pubmlp\[benchmark\]'):
            list_benchmarks()
        with pytest.raises(ImportError, match=r'pip install pubmlp\[benchmark\]'):
            load_benchmark('anything')

    def test_empty_dataset_raises(self, monkeypatch):
        import types

        class FakeDataset:
            def __init__(self, name, path=None):
                self.name = name

            def to_frame(self, *args, **kwargs):
                return pd.DataFrame(columns=['title', 'abstract', 'label_included'])

        fake = types.ModuleType('synergy_dataset')
        fake.Dataset = FakeDataset
        monkeypatch.setitem(sys.modules, 'synergy_dataset', fake)
        with pytest.raises(ValueError, match='no labeled records'):
            load_benchmark('Empty_2020')

    def test_variables_include_year_and_journal(self):
        from pubmlp.datasets import _SYNERGY_VARIABLES, _venue_name

        assert _SYNERGY_VARIABLES['year'] == 'publication_year'
        assert _SYNERGY_VARIABLES['journal'] is _venue_name
        work = {'primary_location': {'source': {'display_name': 'Journal of Testing'}}}
        assert _venue_name(work) == 'Journal of Testing'
        assert _venue_name({'host_venue': {'display_name': 'Fallback Venue'}}) == 'Fallback Venue'
        assert _venue_name({}) is None
