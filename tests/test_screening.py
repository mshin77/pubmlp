"""Tests for pubmlp.screening module."""

import pytest
import pandas as pd

from pubmlp.screening import (
    extract_window_evidence,
    extract_sentence_evidence,
    format_evidence_display,
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
