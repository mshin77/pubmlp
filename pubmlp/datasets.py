"""Benchmark dataset access and normalization."""

import numpy as np
import pandas as pd

BENCHMARK_COLUMNS = ['title', 'abstract', 'year', 'journal', 'label_included']
_METADATA_COVERAGE = 0.9
_MANIFEST_COVERAGE = 0.95
_SYNERGY_HINT = "Synergy access requires the benchmark extra: pip install pubmlp[benchmark]"


def normalize_benchmark_frame(df):
    """Map a raw benchmark frame onto the pubmlp column set."""
    out = df.copy()
    renames = {'publication_year': 'year', 'journal_name': 'journal', 'included': 'label_included'}
    active_renames = {k: v for k, v in renames.items() if k in out.columns and v not in out.columns}
    out = out.rename(columns=active_renames)
    out = out.drop(columns=[k for k in renames if k in out.columns and k not in active_renames])
    for col in BENCHMARK_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    out['title'] = out['title'].fillna('')
    out['abstract'] = out['abstract'].fillna('')
    out['year'] = pd.to_numeric(out['year'], errors='coerce')
    out = out.dropna(subset=['label_included']).reset_index(drop=True)
    out['label_included'] = out['label_included'].astype(int)
    return out[BENCHMARK_COLUMNS]


def _coverage(df, col):
    """Fraction of non-missing values for col, or 0.0 when col is absent."""
    return df[col].notna().mean() if col in df.columns else 0.0


def metadata_fusion(df):
    """Classify metadata availability: full, partial, or text_only."""
    year_ok = _coverage(df, 'year') >= _METADATA_COVERAGE
    journal_ok = _coverage(df, 'journal') >= _METADATA_COVERAGE
    if year_ok and journal_ok:
        return 'full'
    if year_ok or journal_ok:
        return 'partial'
    return 'text_only'


def build_column_specs(df):
    """Column specifications, numeric transform, and fusion level for a benchmark frame."""
    fusion = metadata_fusion(df)
    specs = {'text_cols': ['title', 'abstract'], 'categorical_cols': [], 'numeric_cols': [],
             'label_col': 'label_included'}
    numeric_transform = {}
    if _coverage(df, 'journal') >= _METADATA_COVERAGE:
        specs['categorical_cols'] = ['journal']
    if _coverage(df, 'year') >= _METADATA_COVERAGE:
        specs['numeric_cols'] = ['year']
        numeric_transform = {'year': 'min'}
    return specs, numeric_transform, fusion


def load_manifest_corpus(manifest_path, records_path, id_col='UT'):
    """Join a packaged IDs+labels manifest against a user-supplied database export."""
    from .utils import load_data
    manifest = load_data(manifest_path)
    records = load_data(records_path)
    if id_col not in manifest.columns or id_col not in records.columns:
        raise ValueError(f"Both files need an '{id_col}' column")
    if len(manifest) == 0:
        raise ValueError(f"Manifest is empty; cannot compute coverage")
    records = records.drop_duplicates(subset=[id_col], keep='first')
    manifest_dtype, records_dtype = manifest[id_col].dtype, records[id_col].dtype
    manifest_ids = manifest.copy()
    records_ids = records.copy()
    manifest_ids[id_col] = manifest_ids[id_col].astype(str).str.strip()
    records_ids[id_col] = records_ids[id_col].astype(str).str.strip()
    merged = manifest_ids.merge(records_ids, on=id_col, how='inner', suffixes=('', '_export'))
    if merged[id_col].nunique() == 0:
        raise ValueError(
            f"no IDs matched: manifest '{id_col}' dtype {manifest_dtype}, "
            f"records '{id_col}' dtype {records_dtype}")
    coverage = merged[id_col].nunique() / manifest_ids[id_col].nunique()
    if coverage < _MANIFEST_COVERAGE:
        raise ValueError(
            f"Manifest coverage {coverage:.1%} below {_MANIFEST_COVERAGE:.0%}: "
            f"{manifest_ids[id_col].nunique() - merged[id_col].nunique()} of {manifest_ids[id_col].nunique()} IDs unmatched")
    print(f"Manifest join: {merged[id_col].nunique()}/{manifest_ids[id_col].nunique()} IDs matched ({coverage:.1%})")
    return merged.reset_index(drop=True)


def list_benchmarks():
    """Names available from the Synergy collection."""
    try:
        from synergy_dataset import iter_datasets
    except ImportError as error:
        raise ImportError(_SYNERGY_HINT) from error
    return sorted(d.name for d in iter_datasets())


def _venue_name(work):
    """Journal name from an OpenAlex work record."""
    location = work.get('primary_location') or work.get('host_venue') or {}
    source = location.get('source') or location
    return (source or {}).get('display_name')


_SYNERGY_VARIABLES = {
    'doi': 'doi',
    'title': 'title',
    'abstract': 'abstract',
    'year': 'publication_year',
    'journal': _venue_name,
}


def load_benchmark(name, variables=None):
    """One Synergy dataset as a normalized pubmlp frame, with year and journal metadata."""
    try:
        from synergy_dataset import Dataset
    except ImportError as error:
        raise ImportError(_SYNERGY_HINT) from error
    frame = Dataset(name).to_frame(variables or _SYNERGY_VARIABLES).reset_index()
    normalized = normalize_benchmark_frame(frame)
    if normalized.empty:
        raise ValueError(
            f"benchmark '{name}' produced no labeled records; the raw dataset is probably "
            "not downloaded yet (see synergy_dataset.download_raw_dataset)")
    return normalized
