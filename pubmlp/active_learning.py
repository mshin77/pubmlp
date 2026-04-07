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
    """Most uncertain (closest to 0.5) first. For multi-label, averages uncertainty across labels."""
    probability_array = np.asarray(probabilities)
    uncertainty = np.abs(probability_array - 0.5)
    if uncertainty.ndim > 1:
        uncertainty = uncertainty.mean(axis=1)
    return np.argsort(uncertainty)


def rank_by_random(n, seed=42):
    rng = np.random.RandomState(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    return indices


def rank_by_max_relevance(probabilities):
    """Highest probability (most likely relevant) first. For multi-label, uses max across labels."""
    probability_array = np.asarray(probabilities)
    score = probability_array.max(axis=1) if probability_array.ndim > 1 else probability_array
    return np.argsort(-score)


def rank_by_hybrid_max_uncertainty(probabilities, exploit_ratio=0.95, seed=42):
    """95% max-relevance + 5% uncertainty."""
    probability_array = np.asarray(probabilities)
    n_exploit_samples = int(len(probability_array) * exploit_ratio)
    relevance_ranked = rank_by_max_relevance(probability_array)
    uncertainty_ranked = rank_by_uncertainty(probability_array)
    exploit_set = set(relevance_ranked[:n_exploit_samples].tolist())
    explore = [i for i in uncertainty_ranked if i not in exploit_set]
    return np.concatenate([relevance_ranked[:n_exploit_samples], np.array(explore)])


def rank_by_hybrid_max_random(probabilities, exploit_ratio=0.95, seed=42):
    """95% max-relevance + 5% random."""
    probability_array = np.asarray(probabilities)
    n_exploit_samples = int(len(probability_array) * exploit_ratio)
    relevance_ranked = rank_by_max_relevance(probability_array)
    random_ranked = rank_by_random(len(probability_array), seed)
    exploit_set = set(relevance_ranked[:n_exploit_samples].tolist())
    explore = [i for i in random_ranked if i not in exploit_set]
    return np.concatenate([relevance_ranked[:n_exploit_samples], np.array(explore)])


def select_query_batch(probabilities, strategy='uncertainty', batch_size=20, seed=42):
    probability_array = np.asarray(probabilities)
    ranked = {
        'uncertainty': lambda: rank_by_uncertainty(probability_array),
        'random': lambda: rank_by_random(len(probability_array), seed),
        'max_relevance': lambda: rank_by_max_relevance(probability_array),
        'hybrid_max_uncertainty': lambda: rank_by_hybrid_max_uncertainty(probability_array, seed=seed),
        'hybrid_max_random': lambda: rank_by_hybrid_max_random(probability_array, seed=seed),
    }[strategy]()
    return ranked[:batch_size]


def create_review_batch(df, indices, probabilities):
    """Subset df for human review with model probability and prediction."""
    probability_array = np.asarray(probabilities)
    batch = df.iloc[indices].copy()
    probs_subset = probability_array[indices]
    if probs_subset.ndim > 1:
        batch['model_probability'] = probs_subset.max(axis=1)
        batch['model_prediction'] = (probs_subset >= 0.5).astype(int).tolist()
    else:
        batch['model_probability'] = probs_subset
        batch['model_prediction'] = (probs_subset >= 0.5).astype(int)
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
    """Offline active learning simulation using ground truth labels. model_fn(train_df, unlabeled_df) -> probabilities."""
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

        training_df = df.iloc[state.labeled_indices]
        unlabeled_df = df.iloc[state.unlabeled_indices]
        predicted_probabilities = model_fn(training_df, unlabeled_df)

        query_index = select_query_batch(predicted_probabilities, strategy=strategy, batch_size=batch_size, seed=seed + iteration)
        n_unlabeled = len(state.unlabeled_indices)
        actual_indices = [state.unlabeled_indices[i] for i in query_index if i < n_unlabeled]

        state.labeled_indices.extend(actual_indices)
        state.unlabeled_indices = [i for i in state.unlabeled_indices if i not in set(actual_indices)]
        state.iteration = iteration + 1

        relevant_found = df.iloc[state.labeled_indices][label_col].sum()
        history.append({
            'iteration': iteration + 1,
            'n_labeled': len(state.labeled_indices),
            'n_relevant_found': int(relevant_found),
            'recall': relevant_found / total_relevant if total_relevant > 0 else np.nan,
            'screened_pct': len(state.labeled_indices) / n,
        })

        if relevant_found >= total_relevant:
            break

    return history


def compare_reviewers(model_predictions, human_labels):
    """Agreement rate, Cohen's kappa, and disagreement indices."""
    model_prediction_array = np.asarray(model_predictions)
    human_label_array = np.asarray(human_labels)
    agreed = np.sum(model_prediction_array == human_label_array)
    kappa = cohen_kappa_score(model_prediction_array, human_label_array) if len(set(model_prediction_array) | set(human_label_array)) > 1 else 1.0
    disagreement_indices = np.where(model_prediction_array != human_label_array)[0]
    return {
        'agreement_rate': agreed / len(human_label_array),
        'kappa': kappa,
        'disagreement_indices': disagreement_indices.tolist(),
    }


def safe_stratified_split(X, y, test_size=0.2, random_state=42):
    """Stratified train/val split with ShuffleSplit fallback for rare classes."""
    from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, val_idx = next(sss.split(X, y))
    except ValueError:
        sss = ShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, val_idx = next(sss.split(X))
    return train_idx, val_idx
