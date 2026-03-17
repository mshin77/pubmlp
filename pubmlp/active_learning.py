import numpy as np
from dataclasses import dataclass, field, asdict
from sklearn.metrics import cohen_kappa_score


@dataclass
class ALState:
    labeled_indices: list = field(default_factory=list)
    unlabeled_indices: list = field(default_factory=list)
    iteration: int = 0
    history: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


def rank_by_uncertainty(probabilities):
    """Most uncertain (closest to 0.5) first."""
    probs = np.asarray(probabilities)
    return np.argsort(np.abs(probs - 0.5))


def rank_by_random(n, seed=42):
    rng = np.random.RandomState(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    return indices


def rank_by_max_relevance(probabilities):
    """Highest probability (most likely relevant) first."""
    return np.argsort(-np.asarray(probabilities))


def rank_by_hybrid_max_uncertainty(probabilities, exploit_ratio=0.95, seed=42):
    """95% max-relevance + 5% uncertainty."""
    probs = np.asarray(probabilities)
    n_exploit = int(len(probs) * exploit_ratio)
    max_ranked = rank_by_max_relevance(probs)
    unc_ranked = rank_by_uncertainty(probs)
    exploit_set = set(max_ranked[:n_exploit].tolist())
    explore = [i for i in unc_ranked if i not in exploit_set]
    return np.concatenate([max_ranked[:n_exploit], np.array(explore)])


def rank_by_hybrid_max_random(probabilities, exploit_ratio=0.95, seed=42):
    """95% max-relevance + 5% random."""
    probs = np.asarray(probabilities)
    n_exploit = int(len(probs) * exploit_ratio)
    max_ranked = rank_by_max_relevance(probs)
    rand_ranked = rank_by_random(len(probs), seed)
    exploit_set = set(max_ranked[:n_exploit].tolist())
    explore = [i for i in rand_ranked if i not in exploit_set]
    return np.concatenate([max_ranked[:n_exploit], np.array(explore)])


def select_query_batch(probabilities, strategy='uncertainty', batch_size=20, seed=42):
    probs = np.asarray(probabilities)
    ranked = {
        'uncertainty': lambda: rank_by_uncertainty(probs),
        'random': lambda: rank_by_random(len(probs), seed),
        'max_relevance': lambda: rank_by_max_relevance(probs),
        'hybrid_max_uncertainty': lambda: rank_by_hybrid_max_uncertainty(probs, seed=seed),
        'hybrid_max_random': lambda: rank_by_hybrid_max_random(probs, seed=seed),
    }[strategy]()
    return ranked[:batch_size]


def create_review_batch(df, indices, probabilities):
    """Subset df for human review with model probability and prediction."""
    probs = np.asarray(probabilities)
    batch = df.iloc[indices].copy()
    batch['model_probability'] = probs[indices]
    batch['model_prediction'] = (probs[indices] >= 0.5).astype(int)
    return batch


def merge_human_labels(df, review_batch, label_col='human_label'):
    """Merge human decisions from review batch back into main df."""
    df = df.copy()
    if label_col not in df.columns:
        df[label_col] = np.nan
    df.loc[review_batch.index, label_col] = review_batch[label_col]
    return df


def simulate_al(df, label_col, model_fn, strategy='uncertainty', batch_size=20,
                 initial_pct=0.1, max_iterations=50, seed=42):
    """Offline AL simulation using ground truth labels. model_fn(train_df, unlabeled_df) -> probabilities."""
    rng = np.random.RandomState(seed)
    n = len(df)
    initial_n = max(int(n * initial_pct), 2 * batch_size)
    all_indices = np.arange(n)
    rng.shuffle(all_indices)

    state = ALState(
        labeled_indices=all_indices[:initial_n].tolist(),
        unlabeled_indices=all_indices[initial_n:].tolist(),
    )
    history = []
    total_relevant = df[label_col].sum()

    for iteration in range(max_iterations):
        if not state.unlabeled_indices:
            break

        train_df = df.iloc[state.labeled_indices]
        unlabeled_df = df.iloc[state.unlabeled_indices]
        probs = model_fn(train_df, unlabeled_df)

        query_idx = select_query_batch(probs, strategy=strategy, batch_size=batch_size, seed=seed + iteration)
        actual_indices = [state.unlabeled_indices[i] for i in query_idx if i < len(state.unlabeled_indices)]

        state.labeled_indices.extend(actual_indices)
        state.unlabeled_indices = [i for i in state.unlabeled_indices if i not in set(actual_indices)]
        state.iteration = iteration + 1

        relevant_found = df.iloc[state.labeled_indices][label_col].sum()
        history.append({
            'iteration': iteration + 1,
            'n_labeled': len(state.labeled_indices),
            'n_relevant_found': int(relevant_found),
            'recall': relevant_found / total_relevant if total_relevant > 0 else 0.0,
            'screened_pct': len(state.labeled_indices) / n,
        })

        if relevant_found >= total_relevant:
            break

    return history


def compare_reviewers(model_predictions, human_labels):
    """Agreement rate, Cohen's kappa, and disagreement indices."""
    model_preds = np.asarray(model_predictions)
    human = np.asarray(human_labels)
    agreed = np.sum(model_preds == human)
    kappa = cohen_kappa_score(model_preds, human) if len(set(model_preds) | set(human)) > 1 else 1.0
    disagreement_indices = np.where(model_preds != human)[0]
    return {
        'agreement_rate': agreed / len(human),
        'kappa': kappa,
        'disagreement_indices': disagreement_indices.tolist(),
    }
