import pytest

from pubmlp.stopping import (
    StoppingState, should_stop, update_stopping_state,
    transition_phase, calculate_wss, generate_stopping_report,
    expected_relevant, estimate_recall,
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

    def test_switch_resets_consecutive(self):
        state = StoppingState(phase='active', consecutive_irrelevant=30)
        config = Config(safe_switch_model=True)
        transition_phase(state, config, total_records=100)
        assert state.consecutive_irrelevant == 0

class TestPhase3Stopping:
    def test_switch_stops_on_consecutive_alone(self):
        state = StoppingState(phase='switch', consecutive_irrelevant=50, total_screened=10)
        config = Config(safe_consecutive_irrelevant=50, safe_min_screened_pct=0.5)
        assert should_stop(state, config, total_records=100) is True

    def test_switch_below_threshold_continues(self):
        state = StoppingState(phase='switch', consecutive_irrelevant=49, total_screened=90)
        config = Config(safe_consecutive_irrelevant=50, safe_min_screened_pct=0.5)
        assert should_stop(state, config, total_records=100) is False

    def test_switch_ignores_phase_2_criteria(self):
        state = StoppingState(phase='switch', consecutive_irrelevant=50, total_screened=10)
        config = Config(safe_consecutive_irrelevant=50)
        assert should_stop(state, config, total_records=100,
                           all_known_relevant_found=False) is True


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
            'estimated_recall', 'wss', 'consecutive_irrelevant',
            'criteria_evaluated', 'recommendation',
        }
        assert report['criteria_evaluated'] == 2
        assert report['recommendation'] == 'continue'


class TestSerialization:
    def test_roundtrip(self):
        state = StoppingState(phase='active', total_screened=25, total_relevant=5)
        restored = StoppingState.from_dict(state.to_dict())
        assert restored.phase == 'active'
        assert restored.total_screened == 25

class TestSafeDefaults:
    def test_defaults_follow_specification(self):
        config = Config()
        assert config.safe_min_screened_pct == 0.10
        assert config.safe_random_sample_pct == 0.01
        assert config.al_initial_sample_pct == 0.01
        assert config.safe_consecutive_irrelevant == 200


class TestExpectedRelevant:
    def test_projection(self):
        assert expected_relevant(8, 83, 8310) == pytest.approx(801.2, abs=0.5)

    def test_empty_sample_is_nan(self):
        value = expected_relevant(0, 0, 8310)
        assert value != value


class TestFourCriteria:
    def test_criterion_two_blocks_when_too_few_screened(self):
        state = StoppingState(consecutive_irrelevant=200, total_screened=900)
        config = Config()
        assert should_stop(state, config, 8310) is True
        assert should_stop(state, config, 8310, n_expected_relevant=801.2) is False

    def test_passes_when_all_four_hold(self):
        state = StoppingState(consecutive_irrelevant=200, total_screened=1700)
        config = Config()
        assert should_stop(state, config, 8310, n_expected_relevant=801.2,
                           all_known_relevant_found=True) is True

    def test_criterion_one_blocks(self):
        state = StoppingState(consecutive_irrelevant=200, total_screened=1700)
        config = Config()
        assert should_stop(state, config, 8310, n_expected_relevant=801.2,
                           all_known_relevant_found=False) is False

    def test_nan_expectation_blocks(self):
        state = StoppingState(consecutive_irrelevant=200, total_screened=1700)
        config = Config()
        assert should_stop(state, config, 8310,
                           n_expected_relevant=float('nan')) is False


class TestMetricFixes:
    def test_wss_is_tie_order_independent(self):
        from pubmlp.metrics import calculate_wss_at_recall
        import numpy as np
        flat = np.full(100, 0.5)
        front = np.zeros(100, dtype=int); front[:10] = 1
        back = np.zeros(100, dtype=int); back[90:] = 1
        assert (calculate_wss_at_recall(front, flat, 0.95)['screened_pct']
                == calculate_wss_at_recall(back, flat, 0.95)['screened_pct'])

    def test_wss_reports_below_random(self):
        from pubmlp.metrics import calculate_wss_at_recall
        import numpy as np
        truth = np.zeros(100, dtype=int); truth[:10] = 1
        inverted = np.linspace(0.1, 0.9, 100)
        assert calculate_wss_at_recall(truth, inverted, 0.95)['wss'] < 0

    def test_ndcg_single_record_is_nan(self):
        from pubmlp.metrics import calculate_ndcg
        value = calculate_ndcg([1], [0.9])
        assert value != value

    def test_report_records_four_criteria(self):
        state = StoppingState(consecutive_irrelevant=200, total_screened=1700)
        report = generate_stopping_report(state, 8310, Config(),
                                          n_expected_relevant=801.2,
                                          all_known_relevant_found=True)
        assert report['criteria_evaluated'] == 4
        assert report['recommendation'] == 'stop'

    def test_recall_uses_sample_prevalence_when_given(self):
        state = StoppingState(total_screened=900, total_relevant=700)
        enriched = estimate_recall(state, 8310)
        from_sample = estimate_recall(state, 8310, n_sample_relevant=8, n_sample=83)
        assert from_sample != enriched
        assert 0 < from_sample <= 1.0


class TestHowManyCriteriaWereTested:
    def test_two_when_neither_argument_is_given(self):
        from pubmlp.config import Config
        from pubmlp.stopping import StoppingState, generate_stopping_report

        report = generate_stopping_report(StoppingState(), 100, Config())
        assert report['criteria_evaluated'] == 2

    def test_three_when_one_is(self):
        from pubmlp.config import Config
        from pubmlp.stopping import StoppingState, generate_stopping_report

        report = generate_stopping_report(StoppingState(), 100, Config(),
                                          n_expected_relevant=12)
        assert report['criteria_evaluated'] == 3

    def test_four_when_both_are(self):
        from pubmlp.config import Config
        from pubmlp.stopping import StoppingState, generate_stopping_report

        report = generate_stopping_report(StoppingState(), 100, Config(),
                                          n_expected_relevant=12,
                                          all_known_relevant_found=True)
        assert report['criteria_evaluated'] == 4
