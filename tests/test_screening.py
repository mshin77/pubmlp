"""Tests for pubmlp.screening module."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from pubmlp.screening import (
    extract_window_evidence,
    extract_sentence_evidence,
    format_evidence_display,
    score_full_text,
    generate_descriptions,
    confirm_descriptions,
    _extract_terms,
)


SAMPLE_ABSTRACT = (
    "This study examines mathematical difficulties among elementary school students. "
    "We investigated how linguistic factors influence algebra performance. "
    "Results show significant correlations between vocabulary and math achievement."
)


class TestWindowEvidence:
    def test_match_found(self):
        evidence = extract_window_evidence(SAMPLE_ABSTRACT, r'\bmath\w*\b', 'abstract', window_size=3)
        assert len(evidence) > 0
        assert evidence[0]['field'] == 'abstract'
        assert 'matched_term' in evidence[0]

    def test_no_match(self):
        evidence = extract_window_evidence("no relevant terms here", r'\bmath\w*\b', 'abstract')
        assert evidence == []

    def test_na_input(self):
        assert extract_window_evidence(None, r'\bmath\w*\b', 'abstract') == []
        assert extract_window_evidence('', r'\bmath\w*\b', 'abstract') == []

    def test_deduplication(self):
        text = "math math math"
        evidence = extract_window_evidence(text, r'\bmath\b', 'title', window_size=0)
        texts = [e['text'] for e in evidence]
        assert len(texts) == len(set(texts))


class TestSentenceEvidence:
    def test_match_found(self):
        evidence = extract_sentence_evidence(SAMPLE_ABSTRACT, r'\balgebra\w*\b', 'abstract')
        assert len(evidence) == 1
        assert 'algebra' in evidence[0]['text'].lower()

    def test_no_match(self):
        evidence = extract_sentence_evidence("No relevant terms.", r'\bcalculus\b', 'abstract')
        assert evidence == []

    def test_multiple_sentences(self):
        evidence = extract_sentence_evidence(SAMPLE_ABSTRACT, r'\bmath\w*\b', 'abstract')
        assert len(evidence) >= 2


class TestFormatEvidence:
    def test_formatting(self):
        evidence = [
            {'text': 'some text', 'field': 'abstract', 'matched_term': 'math'},
            {'text': 'other text', 'field': 'title', 'matched_term': 'algebra'},
        ]
        result = format_evidence_display(evidence)
        assert 'abstract: some text' in result
        assert 'title: other text' in result
        assert '; ' in result

    def test_empty(self):
        assert format_evidence_display([]) == ''


class TestScoreFullText:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            'title': [
                'Single-case design in special education',
                'Cooking recipes for beginners',
                'Multiple baseline reading intervention',
            ],
            'abstract': [
                'A multiple baseline design evaluated reading interventions for students.',
                'A guide to cooking Italian pasta dishes at home.',
                'Alternating treatment design compared two math interventions.',
            ],
        })

    @patch('pubmlp.screening.SentenceTransformer')
    def test_adds_semantic_full_columns(self, mock_st_class, sample_df):
        mock_model = MagicMock()
        mock_st_class.return_value = mock_model
        mock_model.encode.side_effect = [
            np.array([[0.9, 0.1], [0.1, 0.9], [0.7, 0.3]]),  # texts
            np.array([[0.8, 0.2]]),  # description
        ]

        patterns = {'math': {'pattern': r'\bmath\w*\b', 'description': 'mathematics education'}}
        result = score_full_text(sample_df.copy(), patterns)

        assert 'math_semantic_full' in result.columns
        assert len(result) == 3
        assert not result['math_semantic_full'].isna().any()

    @patch('pubmlp.screening.SentenceTransformer')
    def test_skips_empty_description(self, mock_st_class, sample_df):
        mock_model = MagicMock()
        mock_st_class.return_value = mock_model
        mock_model.encode.return_value = np.array([[0.5, 0.5]] * 3)

        patterns = {'math': {'pattern': r'\bmath\w*\b', 'description': ''}}
        result = score_full_text(sample_df.copy(), patterns)

        assert 'math_semantic_full' not in result.columns

    def test_missing_fields_raises(self, sample_df):
        patterns = {'math': {'pattern': r'\bmath\w*\b', 'description': 'test'}}
        with pytest.raises(ValueError, match="None of"):
            with patch('pubmlp.screening.SentenceTransformer'):
                score_full_text(sample_df.copy(), patterns, fields=['nonexistent'])


class TestExtractTerms:
    def test_simple_alternation(self):
        terms = _extract_terms(r'\b(algebra|calculus|geometry)\w*\b')
        assert 'algebra' in terms
        assert 'calculus' in terms
        assert 'geometry' in terms

    def test_hyphenated_terms(self):
        terms = _extract_terms(r'\bsingle[\s-]?case\b')
        assert 'single' in terms
        assert 'case' in terms

    def test_deduplication(self):
        terms = _extract_terms(r'\b(math|math|algebra)\w*\b')
        assert terms.count('math') == 1

    def test_filters_short_tokens(self):
        terms = _extract_terms(r'\b(a|algebra)\w*\b')
        assert 'a' not in terms
        assert 'algebra' in terms


class TestGenerateDescriptions:
    def test_generates_from_pattern(self):
        patterns = {'math': {'pattern': r'\b(algebra|geometry)\w*\b'}}
        result = generate_descriptions(patterns)

        assert 'math' in result
        assert 'algebra' in result['math']['description']
        assert result['math']['source'] == 'generated'

    def test_preserves_existing_description(self):
        patterns = {'math': {'pattern': r'\bmath\w*\b', 'description': 'custom description'}}
        result = generate_descriptions(patterns)

        assert result['math']['description'] == 'custom description'
        assert result['math']['source'] == 'user-provided'

    def test_domain_prefix(self):
        patterns = {'math': {'pattern': r'\b(algebra)\w*\b'}}
        result = generate_descriptions(patterns, domain='K-12 education')

        assert result['math']['description'].startswith('In K-12 education')

    def test_no_domain(self):
        patterns = {'math': {'pattern': r'\b(algebra)\w*\b'}}
        result = generate_descriptions(patterns)

        assert result['math']['description'].startswith('The study')


class TestConfirmDescriptions:
    def test_returns_confirmed(self):
        drafts = {
            'math': {'pattern': r'\bmath\b', 'description': 'math studies', 'source': 'generated'},
            'lang': {'pattern': r'\blang\b', 'description': 'language studies', 'source': 'generated'},
        }
        confirmed = confirm_descriptions(drafts)

        assert 'math' in confirmed
        assert confirmed['math']['description'] == 'math studies'
        assert 'source' not in confirmed['math']

    def test_raises_on_empty_description(self):
        drafts = {
            'math': {'pattern': r'\bmath\b', 'description': ''},
        }
        with pytest.raises(ValueError, match="Empty descriptions"):
            confirm_descriptions(drafts)

    def test_saves_to_json(self, tmp_path):
        drafts = {'math': {'pattern': r'\bmath\b', 'description': 'test', 'source': 'generated'}}
        path = str(tmp_path / 'confirmed.json')
        confirm_descriptions(drafts, save_path=path)

        import json
        with open(path) as f:
            saved = json.load(f)
        assert saved['math']['description'] == 'test'


class TestPatternFromTerms:
    def build(self, terms):
        from pubmlp.screening import pattern_from_terms

        return pattern_from_terms(terms)

    def test_truncation_matches_the_family(self):
        import re

        pattern = self.build('dyslex*')
        assert re.search(pattern, 'a study of dyslexia', re.I)
        assert re.search(pattern, 'dyslexic readers', re.I)

    def test_truncation_still_respects_word_starts(self):
        import re

        assert not re.search(self.build('dyslex*'), 'predyslexial notions', re.I)

    def test_a_phrase_matches_across_any_whitespace(self):
        import re

        pattern = self.build('higher education')
        assert re.search(pattern, 'in higher education today', re.I)
        assert re.search(pattern, 'in higher  education today', re.I)
        assert re.search(pattern, 'in higher\neducation today', re.I)

    def test_terms_are_literal_not_patterns(self):
        import re

        pattern = self.build('e.g.')
        assert re.search(pattern, 'see e.g. the review', re.I)
        assert not re.search(pattern, 'see eXgY the review', re.I)

    def test_several_terms_become_alternatives(self):
        import re

        pattern = self.build('university, college, higher education')
        for text in ('at university', 'a college course', 'in higher education'):
            assert re.search(pattern, text, re.I), text

    def test_blank_terms_are_dropped(self):
        assert self.build('dyslexia, , ') == self.build('dyslexia')

    def test_no_terms_gives_no_pattern(self):
        assert self.build('') == ''
        assert self.build([]) == ''

    def test_a_sequence_works_as_well_as_a_string(self):
        assert self.build(['dyslexia', 'college']) == self.build('dyslexia, college')


    def test_a_star_inside_a_phrase_truncates_that_word(self):
        import re

        pattern = self.build('applied behavior* analys*')
        assert re.search(pattern, 'applied behavior analysis', re.I)
        assert re.search(pattern, 'applied behavioral analyses', re.I)

    def test_a_star_inside_a_word_truncates_there(self):
        import re

        pattern = self.build('speech*to*text')
        assert re.search(pattern, 'speech-to-text output', re.I)

    def test_a_hyphen_stands_in_for_a_space(self):
        import re

        pattern = self.build('single* case* design')
        assert re.search(pattern, 'a single-case design', re.I)
        assert re.search(pattern, 'a single case design', re.I)

    def test_a_star_never_escapes_as_a_literal(self):
        pattern = self.build('multiple* baseline* design')
        assert '\\*' not in pattern

    def test_the_result_drives_a_pandas_search(self):
        import pandas as pd

        titles = pd.Series(['Dyslexia at university', 'Unrelated work'])
        hits = titles.str.contains(self.build('dyslex*'), case=False, regex=True)
        assert hits.tolist() == [True, False]


class TestHighlightEvidence:
    def workbook(self, tmp_path, evidence='THE STUDY USED A MULTIPLE BASELINE DESIGN.'):
        from openpyxl import Workbook

        book = Workbook()
        sheet = book.active
        for column, name in enumerate(['title', 'abstract', 'design_evidence'], start=1):
            sheet.cell(1, column, name)
        sheet.cell(2, 1, 'A multiple baseline study')
        sheet.cell(2, 2, 'Reading gains under a multiple baseline design.')
        sheet.cell(2, 3, evidence)
        path = tmp_path / 'screened.xlsx'
        book.save(path)
        return path

    def patterns(self):
        return {'design': r'multiple\s+baseline'}

    def test_it_marks_the_matched_term(self, tmp_path):
        from openpyxl import load_workbook

        from pubmlp.screening import highlight_evidence

        path = self.workbook(tmp_path)
        marked = highlight_evidence(path, self.patterns(), fields=['title', 'abstract'])
        assert marked == 3

        sheet = load_workbook(path, rich_text=True).active
        assert type(sheet.cell(2, 1).value).__name__ == 'CellRichText'

    def test_text_survives_the_marking(self, tmp_path):
        from openpyxl import load_workbook

        from pubmlp.screening import highlight_evidence

        path = self.workbook(tmp_path)
        highlight_evidence(path, self.patterns(), fields=['title', 'abstract'])
        sheet = load_workbook(path, rich_text=True).active
        joined = ''.join(str(part) for part in sheet.cell(2, 2).value)
        assert joined == 'Reading gains under a multiple baseline design.'

    def test_a_cell_without_a_match_is_left_alone(self, tmp_path):
        from openpyxl import load_workbook

        from pubmlp.screening import highlight_evidence

        path = self.workbook(tmp_path, evidence='No design named here.')
        highlight_evidence(path, self.patterns(), fields=['title', 'abstract'])
        sheet = load_workbook(path, rich_text=True).active
        assert sheet.cell(2, 3).value == 'No design named here.'

    def test_marking_twice_changes_nothing(self, tmp_path):
        from openpyxl import load_workbook

        from pubmlp.screening import highlight_evidence

        path = self.workbook(tmp_path)
        first = highlight_evidence(path, self.patterns(), fields=['title', 'abstract'])
        second = highlight_evidence(path, self.patterns(), fields=['title', 'abstract'])
        assert first == second

        sheet = load_workbook(path, rich_text=True).active
        joined = ''.join(str(part) for part in sheet.cell(2, 1).value)
        assert joined == 'A multiple baseline study'


    def test_spacing_around_a_marked_term_is_kept(self, tmp_path):
        import re
        import zipfile

        from pubmlp.screening import highlight_evidence

        path = self.workbook(tmp_path)
        highlight_evidence(path, self.patterns(), fields=['title', 'abstract'])

        book = zipfile.ZipFile(path)
        sheet = [n for n in book.namelist() if n.startswith('xl/worksheets/')][0]
        xml = book.read(sheet).decode('utf-8')
        assert not re.findall(r'<t>(\s[^<]*|[^<]*\s)</t>', xml)

    def test_the_workbook_stays_readable(self, tmp_path):
        from openpyxl import load_workbook

        from pubmlp.screening import highlight_evidence

        path = self.workbook(tmp_path)
        highlight_evidence(path, self.patterns(), fields=['title', 'abstract'])

        sheet = load_workbook(path, rich_text=True).active
        joined = ''.join(str(part) for part in sheet.cell(2, 1).value)
        assert joined == 'A multiple baseline study'

    def test_a_missing_field_is_skipped(self, tmp_path):
        from pubmlp.screening import highlight_evidence

        path = self.workbook(tmp_path)
        marked = highlight_evidence(path, self.patterns(), fields=['title', 'nothing_here'])
        assert marked == 2
