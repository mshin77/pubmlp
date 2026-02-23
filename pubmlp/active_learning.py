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


def select_query_batch(probabilities, strategy='uncertainty', batch_size=20, seed=42):
    probs = np.asarray(probabilities)
    ranked = {
        'uncertainty': lambda: rank_by_uncertainty(probs),
        'random': lambda: rank_by_random(len(probs), seed),
        'max_relevance': lambda: rank_by_max_relevance(probs),
    }[strategy]()
    return ranked[:batch_size]


def create_review_batch(df, indices, probabilities):
    """Subset df for human review, adding model probability and prediction columns."""
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


def compare_reviewers(model_predictions, human_labels):
    """Compare model vs human reviewer decisions."""
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
