import pytest

from pubmlp.stopping import (
    StoppingState, should_stop, update_stopping_state,
    transition_phase, calculate_wss, generate_stopping_report,
)
from pubmlp.config import Config


class TestShouldStop:
    def test_false_initially(self):
        state = StoppingState()
        config = Config(safe_consecutive_irrelevant=50, safe_min_screened_pct=0.5)
        assert should_stop(state, config, total_records=100) is False

    def test_true_after_consecutive(self):
        state = StoppingState(consecutive_irrelevant=50, total_screened=60)
        config = Config(safe_consecutive_irrelevant=50, safe_min_screened_pct=0.5)
        assert should_stop(state, config, total_records=100) is True

    def test_false_when_not_enough_screened(self):
        state = StoppingState(consecutive_irrelevant=50, total_screened=30)
        config = Config(safe_consecutive_irrelevant=50, safe_min_screened_pct=0.5)
        assert should_stop(state, config, total_records=100) is False


class TestUpdateState:
    def test_relevant_resets_consecutive(self):
        state = StoppingState(consecutive_irrelevant=10, total_screened=20)
        update_stopping_state(state, label=1)
        assert state.consecutive_irrelevant == 0
        assert state.total_relevant == 1
        assert state.total_screened == 21

    def test_irrelevant_increments(self):
        state = StoppingState()
        update_stopping_state(state, label=0)
        assert state.consecutive_irrelevant == 1
        assert state.total_relevant == 0


class TestPhaseTransitions:
    def test_random_to_active(self):
        state = StoppingState(phase='random', total_screened=15)
        config = Config(safe_random_sample_pct=0.1)
        transition_phase(state, config, total_records=100)
        assert state.phase == 'active'

    def test_active_to_switch(self):
        state = StoppingState(phase='active')
        config = Config(safe_switch_model=True)
        transition_phase(state, config, total_records=100)
        assert state.phase == 'switch'

    def test_switch_to_quality_check(self):
        state = StoppingState(phase='switch')
        config = Config()
        transition_phase(state, config, total_records=100)
        assert state.phase == 'quality_check'

    def test_active_to_quality_check_without_switch(self):
        state = StoppingState(phase='active')
        config = Config(safe_switch_model=False)
        transition_phase(state, config, total_records=100)
        assert state.phase == 'quality_check'


class TestWSS:
    def test_known_values(self):
        # Screened 60 of 100, recall = 0.95
        wss = calculate_wss(total_records=100, total_screened=60, recall=0.95)
        assert abs(wss - (0.4 - 0.05)) < 1e-6  # 0.35


class TestStoppingReport:
    def test_report_keys(self):
        state = StoppingState(total_screened=50, total_relevant=10)
        config = Config(safe_consecutive_irrelevant=50, safe_min_screened_pct=0.5)
        report = generate_stopping_report(state, total_records=100, config=config)
        assert set(report.keys()) == {
            'phase', 'total_screened', 'total_relevant', 'screened_pct',
            'estimated_recall', 'wss', 'consecutive_irrelevant', 'recommendation',
        }
        assert report['recommendation'] == 'continue'


class TestSerialization:
    def test_roundtrip(self):
        state = StoppingState(phase='active', total_screened=25, total_relevant=5)
        restored = StoppingState.from_dict(state.to_dict())
        assert restored.phase == 'active'
        assert restored.total_screened == 25
