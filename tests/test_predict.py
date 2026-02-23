"""Tests for pubmlp.predict module."""

import pytest

from pubmlp.predict import flag_uncertain


class TestFlagUncertain:
    def test_confident_predictions(self):
        probs = [0.1, 0.9, 0.05, 0.95]
        flags = flag_uncertain(probs)
        assert flags == [False, False, False, False]

    def test_uncertain_predictions(self):
        probs = [0.4, 0.5, 0.6, 0.35]
        flags = flag_uncertain(probs)
        assert all(flags)

    def test_boundary_values(self):
        flags = flag_uncertain([0.3, 0.7])
        assert flags == [False, False]

    def test_custom_thresholds(self):
        probs = [0.45]
        assert flag_uncertain(probs, low=0.4, high=0.6) == [True]
        assert flag_uncertain(probs, low=0.5, high=0.6) == [False]

    def test_empty(self):
        assert flag_uncertain([]) == []


class TestFlagUncertainMultiLabel:
    def test_multi_label_flags(self):
        probs = [[0.1, 0.5], [0.9, 0.4]]
        flags = flag_uncertain(probs)
        assert flags == [[False, True], [False, True]]

    def test_multi_label_all_confident(self):
        probs = [[0.1, 0.9], [0.05, 0.95]]
        flags = flag_uncertain(probs)
        assert flags == [[False, False], [False, False]]

    def test_multi_label_empty(self):
        assert flag_uncertain([]) == []
