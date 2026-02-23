"""
Stratified sample creation with regex highlighting for human coding.
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Dict, List


def count_pattern_matches(text, pattern):
    """Count regex matches in text (case-insensitive)."""
    if pd.isna(text) or not str(text).strip():
        return 0
    try:
        return len(re.findall(pattern, str(text), re.IGNORECASE))
    except Exception:
        return 0


def highlight_pattern_matches(text, pattern, max_length=200):
    """Return up to 3 matched snippets with context for visual inspection."""
    if pd.isna(text) or not str(text).strip():
        return ''
    try:
        matches = list(re.finditer(pattern, str(text), re.IGNORECASE))
        if not matches:
            return ''
        snippets = []
        for match in matches[:3]:
            start = max(0, match.start() - 20)
            end = min(len(text), match.end() + 20)
            snippets.append(f"...{text[start:end].strip()}...")
        result = ' | '.join(snippets)
        return result[:max_length] if len(result) > max_length else result
    except Exception:
        return ''


def create_stratified_sample(df: pd.DataFrame, patterns: Dict[str, str],
                             text_cols: List[str] = None,
                             coding_labels: List[str] = None,
                             sample_size: float = 0.2,
                             random_seed: int = 42,
                             n_strata: int = 5) -> pd.DataFrame:
    """
    Create a stratified random sample with regex pattern highlights.

    Args:
        df: Input DataFrame.
        patterns: Dict of {label: regex_pattern} for highlighting.
        text_cols: Columns to combine for pattern matching (default: title, abstract, keywords).
        coding_labels: Label columns to add for human coding (default: pattern keys).
        sample_size: Proportion to sample (default 0.2).
        random_seed: For reproducibility.
        n_strata: Number of stratification bins.

    Returns:
        DataFrame with pattern highlights and empty coding columns.
    """
    if text_cols is None:
        text_cols = ['title', 'abstract', 'keywords']
    if coding_labels is None:
        coding_labels = list(patterns.keys())

    available_text_cols = [c for c in text_cols if c in df.columns]
    df = df.copy()
    df['_combined_text'] = df[available_text_cols].fillna('').astype(str).agg(' '.join, axis=1)

    # Add pattern count and snippet columns
    for label, pattern in patterns.items():
        df[f'{label}_pattern_count'] = df['_combined_text'].apply(lambda x: count_pattern_matches(x, pattern))
        df[f'{label}_pattern_snippets'] = df['_combined_text'].apply(lambda x: highlight_pattern_matches(x, pattern))

    # Total score for stratification
    count_cols = [f'{label}_pattern_count' for label in patterns]
    df['_total_pattern_score'] = df[count_cols].sum(axis=1)

    # Stratified sampling - handle case where all values are identical
    all_strata_labels = ['very_low', 'low', 'medium', 'high', 'very_high']
    if df['_total_pattern_score'].nunique() <= 1:
        df['strata'] = 'uniform'
    else:
        df['strata'] = pd.qcut(df['_total_pattern_score'], q=n_strata, duplicates='drop')
        n_actual_strata = len(df['strata'].cat.categories)
        strata_labels = all_strata_labels[:n_actual_strata]
        df['strata'] = df['strata'].cat.rename_categories(strata_labels)

    sample_df = df.groupby('strata', group_keys=False, observed=True).apply(
        lambda x: x.sample(frac=sample_size, random_state=random_seed), include_groups=False
    ).reset_index(drop=True)

    # Print strata distribution before cleanup
    print(f"Created sample: {len(sample_df)} records ({sample_size*100:.0f}%)")
    if 'strata' in sample_df.columns:
        print(f"Strata distribution:\n{sample_df['strata'].value_counts().sort_index()}")

    # Add empty coding columns
    for label in coding_labels:
        sample_df[label] = ''
    sample_df['notes'] = ''
    sample_df['coder_id'] = ''
    sample_df['coding_date'] = ''

    # Clean up temp columns
    sample_df = sample_df.drop(columns=['_combined_text', '_total_pattern_score', 'strata'], errors='ignore')

    return sample_df


def apply_conditional_formatting(excel_file, patterns: Dict[str, str]):
    """
    Apply conditional formatting to Excel coding sheet.

    Highlights coding column headers green and pattern count cells > 0 yellow.
    """
    try:
        from openpyxl import load_workbook
        from openpyxl.styles import PatternFill, Font
        from openpyxl.utils import get_column_letter
    except ImportError:
        print("openpyxl required for conditional formatting. Install: pip install openpyxl")
        return

    wb = load_workbook(excel_file)
    ws = wb.active
    yellow_fill = PatternFill(start_color='FFFF00', end_color='FFFF00', fill_type='solid')
    green_fill = PatternFill(start_color='90EE90', end_color='90EE90', fill_type='solid')
    bold_font = Font(bold=True)

    header_row = {cell.value: cell.column for cell in ws[1]}

    # Green headers for coding columns
    for label in patterns:
        if label in header_row:
            col_letter = get_column_letter(header_row[label])
            ws[f'{col_letter}1'].fill = green_fill
            ws[f'{col_letter}1'].font = bold_font

    # Yellow highlight for count > 0
    for label in patterns:
        count_col = f'{label}_pattern_count'
        if count_col in header_row:
            col_letter = get_column_letter(header_row[count_col])
            for row in range(2, ws.max_row + 1):
                cell = ws[f'{col_letter}{row}']
                try:
                    if cell.value and int(cell.value) > 0:
                        cell.fill = yellow_fill
                        cell.font = bold_font
                except (ValueError, TypeError):
                    pass

    # Column widths
    width_hints = {'title': 50, 'abstract': 80, 'keywords': 40, 'notes': 40}
    for label in patterns:
        width_hints[f'{label}_pattern_snippets'] = 60

    for col_name, width in width_hints.items():
        if col_name in header_row:
            ws.column_dimensions[get_column_letter(header_row[col_name])].width = width

    ws.freeze_panes = 'D2'
    wb.save(excel_file)
    print(f"Conditional formatting applied: {excel_file}")


def save_sample_excel(sample_df: pd.DataFrame, output_file, patterns: Dict[str, str]):
    """Save sample to Excel with conditional formatting."""
    output_file = Path(output_file)
    sample_df.to_excel(output_file, index=False, sheet_name='Sample')
    apply_conditional_formatting(output_file, patterns)
    print(f"Sample saved: {output_file}")
