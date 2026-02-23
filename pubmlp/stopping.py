import math
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
    if state.total_screened == 0 or state.total_relevant == 0:
        return 0.0
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
        return 0.0
    return min(state.total_relevant / estimated_total, 1.0)


def calculate_wss(total_records, total_screened, recall):
    """Work Saved over Sampling at given recall level."""
    if total_records == 0:
        return 0.0
    return (total_records - total_screened) / total_records - (1 - recall)


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
