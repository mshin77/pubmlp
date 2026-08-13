from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score, recall_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve
)


def _single_label_metrics(true_labels, predictions, probabilities,
                          output_dir=None, label_name='model', save_figures=True):
    """Compute metrics for a single binary label."""
    metrics = {
        'accuracy': sum(t == p for t, p in zip(true_labels, predictions)) / len(true_labels),
        'precision': precision_score(true_labels, predictions, zero_division=0),
        'recall': recall_score(true_labels, predictions, zero_division=0),
        'specificity': recall_score(true_labels, predictions, pos_label=0, zero_division=0),
        'f1_score': f1_score(true_labels, predictions, zero_division=0),
        'roc_auc': roc_auc_score(true_labels, probabilities) if len(set(true_labels)) > 1 else None,
    }

    print(f"Evaluation Metrics: {label_name}")
    print(classification_report(true_labels, predictions, labels=[0, 1],
                                target_names=['Exclude', 'Include'], digits=3, zero_division=0))
    print("Key Metrics:")
    for metric, value in metrics.items():
        if value is not None:
            print(f"  {metric}: {value:.3f}")

    if save_figures and output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        confusion = confusion_matrix(true_labels, predictions)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Exclude', 'Include'], yticklabels=['Exclude', 'Include'],
                    annot_kws={'size': 18}, ax=ax)
        ax.set_title(f'Confusion Matrix - {label_name}', fontsize=16, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=14)
        ax.set_xlabel('Predicted Label', fontsize=14)
        ax.tick_params(labelsize=13)
        confusion_path = output_dir / f'confusion_matrix_{label_name}.png'
        fig.savefig(confusion_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"\nConfusion matrix saved: {confusion_path}")

        if metrics['roc_auc'] is not None:
            false_positive_rate, true_positive_rate, _ = roc_curve(true_labels, probabilities)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(false_positive_rate, true_positive_rate, linewidth=2.5, label=f'ROC curve (AUC = {metrics["roc_auc"]:.3f})')
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random')
            ax.set_xlabel('False Positive Rate', fontsize=14)
            ax.set_ylabel('True Positive Rate', fontsize=14)
            ax.set_title(f'ROC Curve - {label_name}', fontsize=16, fontweight='bold')
            ax.legend(fontsize=13, loc='lower right')
            ax.tick_params(labelsize=13)
            ax.grid(alpha=0.3)
            roc_path = output_dir / f'roc_curve_{label_name}.png'
            fig.savefig(roc_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"ROC curve saved: {roc_path}")

    return metrics


def _is_multi_label(true_labels):
    """Detect multi-label from input shape."""
    return bool(true_labels) and isinstance(true_labels[0], (list, tuple, np.ndarray))


def calculate_evaluation_metrics(true_labels, predictions, probabilities,
                                 output_dir=None, label_name='model', save_figures=True,
                                 label_names=None):
    """
    Calculate evaluation metrics. Handles both single-label and multi-label.

    For multi-label: computes per-label metrics + macro-averaged F1 + hamming loss.

    Args:
        true_labels: list of ints (single) or list of lists (multi-label).
        predictions: same shape as true_labels.
        probabilities: same shape as true_labels.
        output_dir: path to save figures.
        label_name: prefix for figure filenames.
        save_figures: whether to save confusion matrix / ROC figures.
        label_names: list of label name strings for multi-label display.

    Returns:
        dict: metrics (single-label) or dict with 'per_label', 'macro_f1', 'hamming_loss' (multi-label).
    """
    if not _is_multi_label(true_labels):
        return _single_label_metrics(true_labels, predictions, probabilities,
                                     output_dir, label_name, save_figures)

    # Multi-label
    true_array = np.array(true_labels)
    prediction_array = np.array(predictions)
    probability_array = np.array(probabilities)
    num_labels = true_array.shape[1]

    if label_names is None:
        label_names = [f'label_{i}' for i in range(num_labels)]

    per_label = {}
    for i, label in enumerate(label_names):
        per_label[label] = _single_label_metrics(
            true_array[:, i].tolist(),
            prediction_array[:, i].tolist(),
            probability_array[:, i].tolist(),
            output_dir=output_dir,
            label_name=f'{label_name}_{label}',
            save_figures=save_figures,
        )

    f1_scores = [m['f1_score'] for m in per_label.values()]
    macro_f1 = np.mean(f1_scores)
    hamming = (true_array != prediction_array).mean()

    print(f"\nMacro F1: {macro_f1:.3f} | Hamming Loss: {hamming:.3f}")

    return {
        'per_label': per_label,
        'macro_f1': macro_f1,
        'hamming_loss': hamming,
    }


def _wss_single(true_array, probability_array, target_recall):
    """WSS@recall for a single 1D label vector."""
    n_total = len(true_array)
    n_relevant = true_array.sum()
    if n_relevant == 0:
        return {'wss': np.nan, 'screened_pct': np.nan, 'recall_achieved': np.nan}

    ranked = np.argsort(-probability_array, kind='stable')
    ranked_true = true_array[ranked].astype(float)

    # tied scores carry no ranking information, so each tie group contributes its
    # mean relevance: the expectation under random tie-breaking
    ranked_probability = probability_array[ranked]
    boundaries = np.flatnonzero(np.diff(ranked_probability)) + 1
    for group in np.split(np.arange(n_total), boundaries):
        if len(group) > 1:
            ranked_true[group] = ranked_true[group].mean()

    cumulative_relevant = np.cumsum(ranked_true)
    target_count = int(np.ceil(target_recall * n_relevant))
    screened_to_target = min(np.searchsorted(cumulative_relevant, target_count) + 1, len(cumulative_relevant))
    screened_pct = screened_to_target / n_total
    wss = (1 - screened_pct) - (1 - target_recall)
    return {
        'wss': wss,
        'screened_pct': screened_pct,
        'recall_achieved': cumulative_relevant[screened_to_target - 1] / n_relevant,
    }


def calculate_wss_at_recall(true_labels, probabilities, target_recall=0.95):
    """WSS@recall: fraction of screening effort saved at target recall.

    For multi-label (2D) inputs, computes per-label WSS and returns the macro average.
    """
    true_array = np.asarray(true_labels)
    probability_array = np.asarray(probabilities)

    if true_array.ndim == 1:
        return _wss_single(true_array, probability_array, target_recall)

    results = [_wss_single(true_array[:, j], probability_array[:, j], target_recall)
               for j in range(true_array.shape[1])]
    return {k: float(np.nanmean([r[k] for r in results])) for k in results[0]}


def calculate_ndcg(true_labels, probabilities):
    """Normalized discounted cumulative gain.

    For multi-label (2D) inputs, computes per-label NDCG and returns the mean.
    """
    from sklearn.metrics import ndcg_score
    true_array = np.asarray(true_labels, dtype=float)
    probability_array = np.asarray(probabilities, dtype=float)

    if len(true_array) < 2:
        return float('nan')

    if true_array.ndim == 1:
        return float(ndcg_score(true_array.reshape(1, -1), probability_array.reshape(1, -1)))

    scores = [float(ndcg_score(true_array[:, j].reshape(1, -1), probability_array[:, j].reshape(1, -1)))
              for j in range(true_array.shape[1])]
    return float(np.mean(scores))


def _ece_single(true_array, probability_array, n_bins):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_index = np.clip(np.digitize(probability_array, bins[1:-1]), 0, n_bins - 1)
    n_total = len(true_array)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_index == b
        if not mask.any():
            continue
        ece += mask.sum() / n_total * abs(true_array[mask].mean() - probability_array[mask].mean())
    return float(ece)


def calculate_ece(true_labels, probabilities, n_bins=10):
    """Expected calibration error with equal-width probability bins.

    For multi-label (2D) inputs, computes per-label ECE and returns the macro average.
    """
    true_array = np.asarray(true_labels, dtype=float)
    probability_array = np.asarray(probabilities, dtype=float)

    if true_array.ndim == 1:
        return _ece_single(true_array, probability_array, n_bins)

    scores = [_ece_single(true_array[:, j], probability_array[:, j], n_bins)
              for j in range(true_array.shape[1])]
    return float(np.mean(scores))


def calculate_brier(true_labels, probabilities):
    """Brier score: mean squared error of predicted probabilities.

    For multi-label (2D) inputs, computes per-label Brier and returns the macro average.
    """
    true_array = np.asarray(true_labels, dtype=float)
    probability_array = np.asarray(probabilities, dtype=float)

    if true_array.ndim == 1:
        return float(np.mean((probability_array - true_array) ** 2))

    scores = [float(np.mean((probability_array[:, j] - true_array[:, j]) ** 2))
              for j in range(true_array.shape[1])]
    return float(np.mean(scores))
