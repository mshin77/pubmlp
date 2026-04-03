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
        before = ' '.join(text[:match.start()].split()[-window_size:])
        after = ' '.join(text[match.end():].split()[:window_size])
        evidence_text = ' '.join(filter(None, [before, match.group(), after]))

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
    except (LookupError, TypeError):
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
    extract_func = extract_window_evidence if unit == 'window' else extract_sentence_evidence
    kwargs = {'window_size': window_size} if unit == 'window' else {}

    all_evidence = []
    for field in fields:
        if field in row.index:
            all_evidence.extend(extract_func(row[field], pattern, field, **kwargs))
    return all_evidence


def format_evidence_display(evidence_list: List[Dict]) -> str:
    """Format evidence list as 'field: text; field: text; ...'"""
    if not evidence_list:
        return ''
    return '; '.join(f"{item['field']}: {item['text']}" for item in evidence_list)


def calculate_semantic_scores(evidence_list: List[Dict], criterion_description: str, model) -> Dict:
    """Calculate cosine similarity between evidence texts and criterion description."""
    if not evidence_list or model is None:
        return {'individual_scores': [], 'mean_score': np.nan, 'max_score': np.nan, 'count': 0}

    texts = [item['text'] for item in evidence_list]
    try:
        criterion_emb = model.encode([criterion_description])
        text_embs = model.encode(texts)
        similarities = cosine_similarity(text_embs, criterion_emb)
        scores = similarities.flatten().tolist()
        return {
            'individual_scores': scores,
            'mean_score': float(np.mean(scores)) if scores else np.nan,
            'max_score': float(max(scores)) if scores else np.nan,
            'count': len(scores),
        }
    except Exception as e:
        logger.error(f"Semantic scoring failed: {e}")
        return {'individual_scores': [], 'mean_score': np.nan, 'max_score': np.nan, 'count': 0}


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

    logger.info("Regex screening with semantic similarity")

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

        semantic_results = [
            calculate_semantic_scores(ev, criterion_config.get('description', ''), model)
            for ev in evidence_results
        ] if model is not None else []

        df[f'{criterion_name}_semantic_scores'] = [
            str(r['individual_scores']) if r['individual_scores'] else '' for r in semantic_results
        ] if semantic_results else ''
        df[f'{criterion_name}_semantic_mean'] = [r['mean_score'] for r in semantic_results] if semantic_results else None
        df[f'{criterion_name}_semantic_max'] = [r['max_score'] for r in semantic_results] if semantic_results else None

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
            for col_index, col in enumerate(df.columns, 1):
                max_len = max(df[col].astype(str).apply(len).max(), len(col))
                ws.column_dimensions[get_column_letter(col_index)].width = min(max_len + 2, 100)
        logger.info(f"Saved: {output_file}")
    except ImportError:
        df.to_csv(str(output_file).replace('.xlsx', '.csv'), index=False)
        logger.info(f"Saved as CSV (openpyxl not installed): {output_file}")

    return df


def generate_descriptions(inclusion_patterns: Dict, domain: str = '') -> Dict:
    """Generate draft criterion descriptions from regex pattern terms.

    Extracts literal terms from each regex pattern and composes a
    natural language description. The user should review and refine
    each description before use — description quality directly affects
    semantic scoring accuracy.

    Args:
        inclusion_patterns: Dict of {criterion: {'pattern': regex, ...}}.
            Existing 'description' values are preserved if present.
        domain: Optional domain context (e.g., 'special education',
            'medical research') prepended to generated descriptions.

    Returns:
        Dict of {criterion: {'pattern': str, 'description': str, 'source': str}}.
        'source' is 'generated' or 'user-provided'.

    Example:
        >>> patterns = {'math': {'pattern': r'\\b(algebra|calculus|geometry)\\w*\\b'}}
        >>> drafts = generate_descriptions(patterns, domain='K-12 education')
        >>> # Review and edit drafts['math']['description'] before passing to regex_screen()
    """
    result = {}
    domain_prefix = f"In {domain}, the " if domain else "The "

    for name, config in inclusion_patterns.items():
        if config.get('description', '').strip():
            result[name] = {**config, 'source': 'user-provided'}
            continue

        terms = _extract_terms(config['pattern'])
        term_list = ', '.join(terms[:15])
        ellipsis = ', among others' if len(terms) > 15 else ''

        description = (
            f"{domain_prefix}study addresses {name.replace('_', ' ')} "
            f"as indicated by terms such as {term_list}{ellipsis}."
        )

        result[name] = {**config, 'description': description, 'source': 'generated'}

    return result


def _extract_terms(pattern: str) -> List[str]:
    """Extract readable terms from a regex pattern."""
    # Remove regex syntax: \b, \w*, grouping, quantifiers
    cleaned = re.sub(r'\\[bwsd]\+?\*?', '', pattern)
    cleaned = re.sub(r'[\[\](){}|^$+*?.]', ' ', cleaned)
    cleaned = re.sub(r'\\', '', cleaned)

    terms = [t.strip().replace('-', ' ') for t in cleaned.split()]
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for t in terms:
        if t and t not in seen and len(t) > 1:
            seen.add(t)
            unique.append(t)
    return unique


def confirm_descriptions(descriptions: Dict, save_path: str = None) -> Dict:
    """Validate that all descriptions are user-reviewed and return confirmed patterns.

    Checks that every criterion has a non-empty description.
    Optionally saves the confirmed descriptions to JSON for reproducibility.

    Args:
        descriptions: Dict from generate_descriptions() after user edits.
        save_path: Optional path to save confirmed descriptions as JSON.

    Returns:
        Dict of {criterion: {'pattern': str, 'description': str}} ready
        for regex_screen() and score_full_text().

    Raises:
        ValueError: If any criterion has an empty description.
    """
    missing = [name for name, cfg in descriptions.items()
               if not cfg.get('description', '').strip()]
    if missing:
        raise ValueError(f"Empty descriptions for: {missing}. Review and provide descriptions before confirming.")

    confirmed = {
        name: {'pattern': cfg['pattern'], 'description': cfg['description']}
        for name, cfg in descriptions.items()
    }

    if save_path:
        import json
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(confirmed, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved confirmed descriptions: {save_path}")

    return confirmed


def score_full_text(df: pd.DataFrame, inclusion_patterns: Dict,
                    fields: List[str] = None,
                    model_name: str = 'all-MiniLM-L6-v2') -> pd.DataFrame:
    """Score full record text against each criterion description.

    Complements evidence-based scoring by computing cosine similarity
    on the entire concatenated text, including records with no regex match.
    Adds {criterion}_semantic_full column per criterion.

    Args:
        df: DataFrame with text columns.
        inclusion_patterns: Dict of {criterion: {'description': str, ...}}.
            The description should be a detailed natural language statement of
            what qualifies a record for inclusion. Specificity matters:
            include domain terms, grade levels, methodology keywords, and
            concrete examples of what counts.
        fields: Text columns to concatenate (defaults to ['title', 'abstract']).
        model_name: Sentence-transformer model for embeddings.

    Returns:
        DataFrame with {criterion}_semantic_full columns added.
    """
    fields = fields or ['title', 'abstract']
    available = [f for f in fields if f in df.columns]
    if not available:
        raise ValueError(f"None of {fields} found in columns: {list(df.columns)}")

    texts = df[available].fillna('').astype(str).agg(' '.join, axis=1).tolist()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(model_name, device=device)
    text_embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=len(texts) > 500)

    for criterion_name, criterion_config in inclusion_patterns.items():
        description = criterion_config.get('description', '')
        if not description:
            logger.warning(f"No description for '{criterion_name}', skipping full-text scoring")
            continue

        desc_emb = model.encode([description], convert_to_numpy=True)
        scores = cosine_similarity(text_embs, desc_emb).flatten()
        df[f'{criterion_name}_semantic_full'] = scores

    return df


def compare_screening_configs(input_file: str, configs: Dict[str, Dict],
                              fields: List[str] = None,
                              model_name: str = 'all-MiniLM-L6-v2') -> pd.DataFrame:
    """Compare multiple screening configurations side by side.

    Run regex_screen with different inclusion_patterns (e.g., different
    descriptions, patterns, window sizes) and return a summary comparing
    match counts, semantic score distributions, and overlap.

    Args:
        input_file: Path to CSV or Excel.
        configs: Dict of {config_name: {'inclusion_patterns': dict, 'unit': str,
                 'window_size': int}}. Each entry defines one screening run.
                 'unit' defaults to 'sentence', 'window_size' defaults to 5.
        fields: Text columns to search.
        model_name: Sentence-transformer model.

    Returns:
        DataFrame with one row per config per criterion, columns:
        config, criterion, n_matched, match_pct, semantic_mean_median,
        semantic_max_median, semantic_full_median.
    """
    rows = []

    for config_name, cfg in configs.items():
        patterns = cfg['inclusion_patterns']
        unit = cfg.get('unit', 'sentence')
        window_size = cfg.get('window_size', 5)

        screened = regex_screen(
            input_file, patterns, output_file=None, fields=fields,
            unit=unit, window_size=window_size, model_name=model_name,
        )
        screened = score_full_text(screened, patterns, fields=fields, model_name=model_name)

        for criterion in patterns:
            n_matched = int(screened[criterion].sum())
            mean_col = f'{criterion}_semantic_mean'
            max_col = f'{criterion}_semantic_max'
            full_col = f'{criterion}_semantic_full'

            rows.append({
                'config': config_name,
                'criterion': criterion,
                'n_matched': n_matched,
                'match_pct': round(n_matched / len(screened) * 100, 1),
                'semantic_mean_median': round(screened[mean_col].median(), 4) if mean_col in screened.columns else np.nan,
                'semantic_max_median': round(screened[max_col].median(), 4) if max_col in screened.columns else np.nan,
                'semantic_full_median': round(screened[full_col].median(), 4) if full_col in screened.columns else np.nan,
            })

    return pd.DataFrame(rows)
