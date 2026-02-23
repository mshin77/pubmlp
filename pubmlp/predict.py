import torch
from tqdm import tqdm

from .utils import unpack_batch


def _run_inference(model, dataloader, device, threshold=0.5, calibration=None,
                   collect_labels=False, desc="Predicting"):
    """Shared inference loop for prediction functions."""
    model.eval()
    thresh = torch.tensor(threshold) if not isinstance(threshold, torch.Tensor) else threshold
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            input_ids, attention_mask, categorical_tensor, numeric_tensor, labels, texts = unpack_batch(batch, device)
            outputs = model(input_ids, attention_mask, categorical_tensor, numeric_tensor, texts)
            if calibration is not None:
                outputs = calibration.transform(outputs)
            probs = torch.sigmoid(outputs)
            preds = (probs > thresh.to(probs.device)).long()

            if outputs.shape[-1] == 1:
                all_preds.extend(preds.squeeze(-1).tolist())
                all_probs.extend(probs.squeeze(-1).tolist())
                if collect_labels:
                    all_labels.extend(labels.squeeze(-1).tolist())
            else:
                all_preds.extend(preds.tolist())
                all_probs.extend(probs.tolist())
                if collect_labels:
                    all_labels.extend(labels.tolist())

    return all_preds, all_probs, all_labels


def predict_model(model, unlabeled_dataloader, device, return_probs=True,
                   threshold=0.5, calibration=None):
    """
    Predict labels for unlabeled data.

    Single label: returns flat lists.
    Multi-label: returns list of lists.

    Args:
        threshold: Decision threshold (float or list of floats for per-label).
        calibration: Optional TemperatureScaling object applied to logits before sigmoid.

    Returns:
        tuple or list: (predictions, probabilities) if return_probs else predictions only.
    """
    preds, probs, _ = _run_inference(model, unlabeled_dataloader, device,
                                     threshold, calibration, desc="Predicting")
    return (preds, probs) if return_probs else preds


def get_predictions_and_labels(model, dataloader, device, threshold=0.5,
                                calibration=None):
    """
    Get predictions, probabilities, and true labels from a labeled dataloader.

    Args:
        threshold: Decision threshold (float or list of floats for per-label).
        calibration: Optional TemperatureScaling object applied to logits before sigmoid.

    Returns:
        tuple: (predictions, probabilities, true_labels)
    """
    return _run_inference(model, dataloader, device, threshold, calibration,
                          collect_labels=True, desc="Evaluating")


def flag_uncertain(probabilities, low=0.3, high=0.7):
    """
    Flag predictions with probability between low and high as uncertain.

    Single label: probabilities is list of floats, returns list of bools.
    Multi-label: probabilities is list of lists, returns list of lists of bools.
    """
    if not probabilities:
        return []
    if isinstance(probabilities[0], (list, tuple)):
        return [[low < p < high for p in row] for row in probabilities]
    return [low < p < high for p in probabilities]
