import math
import numpy as np
from dataclasses import dataclass, field, asdict


@dataclass
class StoppingState:
    phase: str = 'random'  # random | active | switch | quality_check
    consecutive_irrelevant: int = 0
    total_screened: int = 0
    total_relevant: int = 0
    history: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


def expected_relevant(n_sample_relevant, n_sample, total_records):
    """Project the corpus relevant count from a random sample.

    Valid only for a probability sample. Under active learning that is the
    prior-knowledge draw, since every later batch is model-selected.
    """
    if n_sample <= 0:
        return float('nan')
    return total_records * (n_sample_relevant / n_sample)


def should_stop(state, config, total_records, n_expected_relevant=None,
                all_known_relevant_found=None):
    """SAFE Phase 2 stopping test (Boetje & van de Schoot, 2024).

    The specification defines four criteria, all of which must hold:

    1. every predefined relevant record has been identified;
    2. records screened are at least twice the expected relevant count,
       projected from the prior-knowledge sample;
    3. at least ``config.safe_min_screened_pct`` of the corpus is screened;
    4. no relevant record appeared in the last
       ``config.safe_consecutive_irrelevant`` records screened.

    Criteria 3 and 4 are derived from ``state`` and ``config``. Criteria 1 and 2
    need information the state does not carry, so each is evaluated only when its
    argument is given and skipped when it is ``None``. Supplying neither tests
    two of the four criteria, which is weaker than the specification; supply both
    for the full test.

    Args:
        state: StoppingState holding the screening counters.
        config: Config supplying the SAFE thresholds.
        total_records: Size of the full corpus.
        n_expected_relevant: Projected relevant count, for example from
            ``expected_relevant``. Enables criterion 2.
        all_known_relevant_found: Whether every predefined relevant record has
            been retrieved. Enables criterion 1.

    Returns:
        bool: True when every evaluated criterion holds.
    """
    if all_known_relevant_found is False:
        return False

    if n_expected_relevant is not None:
        if n_expected_relevant != n_expected_relevant:
            return False
        if state.total_screened < 2 * n_expected_relevant:
            return False

    min_screened = config.safe_min_screened_pct * total_records
    return (state.consecutive_irrelevant >= config.safe_consecutive_irrelevant
            and state.total_screened >= min_screened)


def update_stopping_state(state, label):
    """Update counters after a human screening decision."""
    state.total_screened += 1
    if label == 1:
        state.total_relevant += 1
        state.consecutive_irrelevant = 0
    else:
        state.consecutive_irrelevant += 1
    state.history.append({
        'screened': state.total_screened,
        'relevant': state.total_relevant,
        'consecutive_irrelevant': state.consecutive_irrelevant,
    })
    return state


def transition_phase(state, config, total_records):
    """Advance phase based on screening progress."""
    screened_pct = state.total_screened / total_records if total_records > 0 else 0
    if state.phase == 'random' and screened_pct >= config.safe_random_sample_pct:
        state.phase = 'active'
    elif state.phase == 'active' and config.safe_switch_model:
        state.phase = 'switch'
    elif state.phase == 'active' and not config.safe_switch_model:
        state.phase = 'quality_check'
    elif state.phase == 'switch':
        state.phase = 'quality_check'
    return state


def estimate_recall(state, total_records, n_sample_relevant=None, n_sample=None):
    """Lower bound on recall of relevant records.

    Recall needs the corpus relevant count, which the screening state cannot
    supply. Passing the prior-knowledge sample counts gives the correct estimate:
    the corpus total is projected from that probability sample using the Wilson
    upper bound on its prevalence, so the resulting recall is conservative.

    Without those counts the prevalence of the screened set stands in for corpus
    prevalence. Under active learning the screened set is deliberately enriched,
    so that substitution overstates prevalence and the returned value is not a
    recall estimate. It is retained for callers that screen at random, and should
    not be reported for a model-directed screen.

    Args:
        state: StoppingState holding the screening counters.
        total_records: Size of the full corpus.
        n_sample_relevant: Relevant records in the probability sample.
        n_sample: Size of the probability sample.

    Returns:
        float: Recall lower bound, or NaN when it cannot be computed.
    """
    if state.total_screened == 0 or state.total_relevant == 0:
        return np.nan

    if n_sample_relevant is not None and n_sample:
        p = n_sample_relevant / n_sample
        n = n_sample
    else:
        p = state.total_relevant / state.total_screened
        n = state.total_screened

    z = 1.96
    denominator = 1 + z ** 2 / n
    centre = p + z ** 2 / (2 * n)
    spread = z * math.sqrt((p * (1 - p) + z ** 2 / (4 * n)) / n)

    if n_sample_relevant is not None and n_sample:
        bound = (centre + spread) / denominator
    else:
        bound = (centre - spread) / denominator

    estimated_total = bound * total_records
    if estimated_total <= 0:
        return np.nan
    return min(state.total_relevant / estimated_total, 1.0)


def calculate_wss(total_records, total_screened, recall):
    """Work Saved over Sampling at given recall level."""
    if total_records == 0 or (isinstance(recall, float) and np.isnan(recall)):
        return np.nan
    return (total_records - total_screened) / total_records - (1 - recall)


def recall_target_test(n_screened, n_relevant, N, target_recall=0.95, confidence=0.95):
    """Hypergeometric stopping test for recall-based screening."""
    from scipy.stats import hypergeom

    alpha = 1 - confidence

    if n_relevant == 0:
        return {'stop': False, 'recall_lower_bound': 0.0, 'K_max': N - n_screened}

    K_max = 0
    for m in range(1, N - n_screened + 1):
        R = n_relevant + m
        if R > N:
            break
        p = hypergeom.cdf(n_relevant, N, R, n_screened)
        if p < alpha:
            break
        K_max = m

    recall_lb = n_relevant / (n_relevant + K_max) if (n_relevant + K_max) > 0 else 1.0
    return {
        'stop': recall_lb >= target_recall,
        'recall_lower_bound': round(recall_lb, 6),
        'K_max': K_max,
    }


def generate_stopping_report(state, total_records, config=None,
                             n_expected_relevant=None, all_known_relevant_found=None,
                             n_sample_relevant=None, n_sample=None):
    """Report for human reviewer to decide whether to stop screening."""
    screened_pct = state.total_screened / total_records if total_records > 0 else 0
    recall = (estimate_recall(state, total_records, n_sample_relevant, n_sample)
              if n_sample else np.nan)
    wss = calculate_wss(total_records, state.total_screened, recall)
    stop = should_stop(state, config, total_records, n_expected_relevant,
                       all_known_relevant_found) if config else False
    return {
        'phase': state.phase,
        'total_screened': state.total_screened,
        'total_relevant': state.total_relevant,
        'screened_pct': round(screened_pct, 4),
        'estimated_recall': round(recall, 4),
        'wss': round(wss, 4),
        'consecutive_irrelevant': state.consecutive_irrelevant,
        'criteria_evaluated': (2 + (n_expected_relevant is not None)
                               + (all_known_relevant_found is not None)),
        'recommendation': 'stop' if stop else 'continue',
    }
