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


def should_stop(state, config, total_records):
    """SAFE criterion: stop when consecutive irrelevant threshold met AND minimum screened."""
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


def estimate_recall(state, total_records):
    """Wilson score lower bound estimate of recall."""
    if state.total_screened == 0:
        return np.nan
    if state.total_relevant == 0:
        return np.nan
    # Proportion of relevant found so far
    p = state.total_relevant / state.total_screened
    n = state.total_screened
    z = 1.96  # 95% confidence
    # Wilson lower bound
    denominator = 1 + z ** 2 / n
    centre = p + z ** 2 / (2 * n)
    spread = z * math.sqrt((p * (1 - p) + z ** 2 / (4 * n)) / n)
    lower = (centre - spread) / denominator
    # Estimated total relevant in full corpus
    estimated_total = lower * total_records
    if estimated_total == 0:
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


def generate_stopping_report(state, total_records, config=None):
    """Report for human reviewer to decide whether to stop screening."""
    screened_pct = state.total_screened / total_records if total_records > 0 else 0
    recall = estimate_recall(state, total_records)
    wss = calculate_wss(total_records, state.total_screened, recall)
    stop = should_stop(state, config, total_records) if config else False
    return {
        'phase': state.phase,
        'total_screened': state.total_screened,
        'total_relevant': state.total_relevant,
        'screened_pct': round(screened_pct, 4),
        'estimated_recall': round(recall, 4),
        'wss': round(wss, 4),
        'consecutive_irrelevant': state.consecutive_irrelevant,
        'recommendation': 'stop' if stop else 'continue',
    }
