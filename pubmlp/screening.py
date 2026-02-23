"""
Regex-based screening with semantic similarity scoring.

Screens bibliometric records using configurable regex patterns,
extracts evidence (window or sentence), and scores semantic similarity.
"""

import pandas as pd
import numpy as np
import re
import logging
from pathlib import Path
from typing import List, Dict

logger = logging.getLogger(__name__)

# Graceful imports for optional dependencies
try:
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    from nltk.tokenize import sent_tokenize
except ImportError:
    def sent_tokenize(text):
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if s.strip()]

import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .utils import load_data


def extract_window_evidence(text: str, pattern: str, field_name: str, window_size: int = 5) -> List[Dict]:
    """Extract word windows around regex matches."""
    if pd.isna(text) or not isinstance(text, str) or not text.strip():
        return []

    evidence = []
    seen = set()

    for match in re.finditer(pattern, text, re.IGNORECASE):
        before_words = text[:match.start()].split()[-window_size:]
        after_words = text[match.end():].split()[:window_size]

        parts = []
        if before_words:
            parts.append(' '.join(before_words))
        parts.append(match.group())
        if after_words:
            parts.append(' '.join(after_words))

        evidence_text = ' '.join(parts)
        if evidence_text not in seen:
            seen.add(evidence_text)
            evidence.append({'text': evidence_text, 'field': field_name, 'matched_term': match.group()})

    return evidence


def extract_sentence_evidence(text: str, pattern: str, field_name: str) -> List[Dict]:
    """Extract complete sentences containing regex matches."""
    if pd.isna(text) or not isinstance(text, str) or not text.strip():
        return []

    try:
        sentences = sent_tokenize(text)
    except Exception:
        sentences = text.split('. ')

    evidence = []
    seen = set()

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        match = re.search(pattern, sentence, re.IGNORECASE)
        if match and sentence not in seen:
            seen.add(sentence)
            evidence.append({'text': sentence, 'field': field_name, 'matched_term': match.group()})

    return evidence


def extract_all_evidence(row: pd.Series, pattern: str, fields: List[str],
                         unit: str = 'sentence', window_size: int = 5) -> List[Dict]:
    """Extract evidence from specified fields in a DataFrame row."""
    all_evidence = []
    extract_func = extract_window_evidence if unit == 'window' else extract_sentence_evidence

    for field in fields:
        if field not in row.index:
            continue
        if unit == 'window':
            all_evidence.extend(extract_func(row[field], pattern, field, window_size))
        else:
            all_evidence.extend(extract_func(row[field], pattern, field))

    return all_evidence


def format_evidence_display(evidence_list: List[Dict]) -> str:
    """Format evidence list as 'field: text; field: text; ...'"""
    if not evidence_list:
        return ''
    return '; '.join(f"{item['field']}: {item['text']}" for item in evidence_list)


def calculate_semantic_scores(evidence_list: List[Dict], criterion_description: str, model) -> Dict:
    """Calculate cosine similarity between evidence texts and criterion description."""
    if not evidence_list or model is None:
        return {'individual_scores': [], 'mean_score': 0.0, 'max_score': 0.0, 'count': 0}

    texts = [item['text'] for item in evidence_list]
    try:
        criterion_emb = model.encode([criterion_description])
        text_embs = model.encode(texts)
        similarities = cosine_similarity(text_embs, criterion_emb)
        scores = similarities.flatten().tolist()
        return {
            'individual_scores': scores,
            'mean_score': float(np.mean(scores)) if scores else 0.0,
            'max_score': float(max(scores)) if scores else 0.0,
            'count': len(scores),
        }
    except Exception as e:
        logger.error(f"Semantic scoring failed: {e}")
        return {'individual_scores': [], 'mean_score': 0.0, 'max_score': 0.0, 'count': len(texts)}


def regex_screen(input_file: str, inclusion_patterns: Dict, output_file: str = None,
                 fields: List[str] = None, unit: str = 'sentence', window_size: int = 5,
                 model_name: str = 'all-MiniLM-L6-v2') -> pd.DataFrame:
    """
    Screen dataset using regex patterns with optional semantic similarity.

    Args:
        input_file: Path to CSV or Excel input.
        inclusion_patterns: Dict of {criterion_name: {'pattern': regex, 'description': text}}.
        output_file: Path to save results (defaults to screened.xlsx next to input).
        fields: Columns to search (defaults to ['abstract', 'keywords', 'title']).
        unit: Evidence extraction unit - 'sentence' or 'window'.
        window_size: Words before/after match if unit='window'.
        model_name: Sentence transformer model for semantic scoring.

    Returns:
        DataFrame with screening results.
    """
    if fields is None:
        fields = ['abstract', 'keywords', 'title']

    logger.info("REGEX SCREENING WITH SEMANTIC SIMILARITY")

    df = load_data(input_file)
    logger.info(f"Loaded {len(df)} records from {input_file}")

    available_fields = [f for f in fields if f in df.columns]
    missing = [f for f in fields if f not in df.columns]
    if missing:
        logger.warning(f"Missing fields: {missing}")

    # Load semantic model
    model = None
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = SentenceTransformer(model_name, device=device)
        logger.info(f"Semantic model loaded: {model_name} on {device}")
    except Exception as e:
        logger.error(f"Failed to load semantic model: {e}")

    # Screen each criterion
    for criterion_name, criterion_config in inclusion_patterns.items():
        logger.info(f"Screening: {criterion_name}")

        evidence_results = [
            extract_all_evidence(row, criterion_config['pattern'], available_fields, unit, window_size)
            for _, row in df.iterrows()
        ]

        df[criterion_name] = [1 if ev else 0 for ev in evidence_results]
        df[f'{criterion_name}_evidence'] = [format_evidence_display(ev) for ev in evidence_results]

        if model is not None:
            semantic_results = [
                calculate_semantic_scores(ev, criterion_config['description'], model) for ev in evidence_results
            ]
            df[f'{criterion_name}_semantic_scores'] = [
                str(r['individual_scores']) if r['individual_scores'] else '' for r in semantic_results
            ]
            df[f'{criterion_name}_semantic_mean'] = [r['mean_score'] for r in semantic_results]
            df[f'{criterion_name}_semantic_max'] = [r['max_score'] for r in semantic_results]
        else:
            df[f'{criterion_name}_semantic_scores'] = ''
            df[f'{criterion_name}_semantic_mean'] = None
            df[f'{criterion_name}_semantic_max'] = None

        n_matched = df[criterion_name].sum()
        logger.info(f"  {n_matched} matched ({n_matched/len(df)*100:.1f}%)")

    # Combined: all criteria met
    criterion_names = list(inclusion_patterns.keys())
    df['meets_all_criteria'] = (df[criterion_names] == 1).all(axis=1).astype(int)
    logger.info(f"Papers meeting ALL criteria: {df['meets_all_criteria'].sum()}")

    # Save
    if output_file is None:
        output_file = Path(input_file).parent / 'screened.xlsx'

    # Reorder: original columns, then screening columns
    criterion_cols = []
    for name in criterion_names:
        criterion_cols.extend([name, f'{name}_evidence', f'{name}_semantic_scores',
                               f'{name}_semantic_mean', f'{name}_semantic_max'])
    criterion_cols = [c for c in criterion_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in criterion_cols + ['meets_all_criteria']]
    df = df[other_cols + criterion_cols + ['meets_all_criteria']]

    try:
        from openpyxl.utils import get_column_letter
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Screening Results', index=False)
            ws = writer.sheets['Screening Results']
            for idx, col in enumerate(df.columns, 1):
                max_len = max(df[col].astype(str).apply(len).max(), len(col))
                ws.column_dimensions[get_column_letter(idx)].width = min(max_len + 2, 100)
        logger.info(f"Saved: {output_file}")
    except ImportError:
        df.to_csv(str(output_file).replace('.xlsx', '.csv'), index=False)
        logger.info(f"Saved as CSV (openpyxl not installed): {output_file}")

    return df
