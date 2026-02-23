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

    print(f"EVALUATION METRICS: {label_name.upper()}")
    print(classification_report(true_labels, predictions, target_names=['Exclude', 'Include'], digits=3))
    print("Key Metrics:")
    for metric, value in metrics.items():
        if value is not None:
            print(f"  {metric}: {value:.3f}")

    if save_figures and output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cm = confusion_matrix(true_labels, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Exclude', 'Include'], yticklabels=['Exclude', 'Include'])
        plt.title(f'Confusion Matrix - {label_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        cm_path = output_dir / f'confusion_matrix_{label_name}.png'
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nConfusion matrix saved: {cm_path}")

        if metrics['roc_auc'] is not None:
            fpr, tpr, _ = roc_curve(true_labels, probabilities)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC curve (AUC = {metrics["roc_auc"]:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {label_name}')
            plt.legend()
            plt.grid(alpha=0.3)
            roc_path = output_dir / f'roc_curve_{label_name}.png'
            plt.savefig(roc_path, dpi=300, bbox_inches='tight')
            plt.close()
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
    true_arr = np.array(true_labels)
    pred_arr = np.array(predictions)
    prob_arr = np.array(probabilities)
    num_labels = true_arr.shape[1]

    if label_names is None:
        label_names = [f'label_{i}' for i in range(num_labels)]

    per_label = {}
    for i, lname in enumerate(label_names):
        per_label[lname] = _single_label_metrics(
            true_arr[:, i].tolist(),
            pred_arr[:, i].tolist(),
            prob_arr[:, i].tolist(),
            output_dir=output_dir,
            label_name=f'{label_name}_{lname}',
            save_figures=save_figures,
        )

    # Macro-averaged F1
    f1_scores = [m['f1_score'] for m in per_label.values()]
    macro_f1 = np.mean(f1_scores)

    # Hamming loss: fraction of labels that are incorrectly predicted
    hamming = (true_arr != pred_arr).mean()

    print(f"\nMACRO F1: {macro_f1:.3f} | Hamming Loss: {hamming:.3f}")

    return {
        'per_label': per_label,
        'macro_f1': macro_f1,
        'hamming_loss': hamming,
    }
