"""Tests for pubmlp.sample module."""

import pytest
import pandas as pd

from pubmlp.sample import create_stratified_sample, count_pattern_matches, highlight_pattern_matches


class TestCountPatternMatches:
    def test_basic_count(self):
        assert count_pattern_matches("cat and dog and cat", r"cat") == 2

    def test_case_insensitive(self):
        assert count_pattern_matches("Cat CAT cat", r"cat") == 3

    def test_nan_returns_zero(self):
        assert count_pattern_matches(float('nan'), r"cat") == 0

    def test_empty_string(self):
        assert count_pattern_matches("", r"cat") == 0

    def test_no_match(self):
        assert count_pattern_matches("hello world", r"xyz") == 0


class TestHighlightPatternMatches:
    def test_basic_highlight(self):
        result = highlight_pattern_matches("The cat sat on the mat", r"cat")
        assert "cat" in result

    def test_no_match(self):
        assert highlight_pattern_matches("hello world", r"xyz") == ''

    def test_nan_returns_empty(self):
        assert highlight_pattern_matches(float('nan'), r"cat") == ''

    def test_max_length_truncation(self):
        text = "cat " * 100
        result = highlight_pattern_matches(text, r"cat", max_length=50)
        assert len(result) <= 50

    def test_multiple_snippets(self):
        text = "First cat here. Then another cat there. And one more cat."
        result = highlight_pattern_matches(text, r"cat")
        assert '|' in result


class TestCreateStratifiedSample:
    def test_returns_dataframe(self):
        df = pd.DataFrame({
            'title': [f"Title {i}" for i in range(50)],
            'abstract': [f"Abstract about topic {i % 5}" for i in range(50)],
            'keywords': [f"keyword{i % 3}" for i in range(50)],
        })
        patterns = {'topic': r'topic'}
        sample = create_stratified_sample(df, patterns, sample_size=0.5)
        assert isinstance(sample, pd.DataFrame)
        assert len(sample) > 0
        assert len(sample) < len(df)

    def test_adds_coding_columns(self):
        df = pd.DataFrame({
            'title': ['a', 'b', 'c'] * 10,
            'abstract': ['x', 'y', 'z'] * 10,
            'keywords': ['k1', 'k2', 'k3'] * 10,
        })
        patterns = {'label1': r'a', 'label2': r'x'}
        sample = create_stratified_sample(df, patterns, sample_size=0.5)
        assert 'label1' in sample.columns
        assert 'label2' in sample.columns
        assert 'notes' in sample.columns
        assert 'coder_id' in sample.columns

    def test_pattern_count_columns(self):
        df = pd.DataFrame({
            'title': ['cat dog'] * 20,
            'abstract': ['bird fish'] * 20,
            'keywords': [''] * 20,
        })
        patterns = {'animal': r'cat|dog'}
        sample = create_stratified_sample(df, patterns, sample_size=0.5)
        assert 'animal_pattern_count' in sample.columns
        assert 'animal_pattern_snippets' in sample.columns
