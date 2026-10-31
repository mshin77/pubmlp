import torch
from tqdm import tqdm

from .utils import default_forward_fn


def _run_inference(model, dataloader, device, threshold=0.5, calibration=None,
                   collect_labels=False, desc="Predicting", forward_fn=None):
    """Shared inference loop for prediction functions."""
    if forward_fn is None:
        forward_fn = default_forward_fn
    model.eval()
    threshold_tensor = torch.tensor(threshold) if not isinstance(threshold, torch.Tensor) else threshold
    all_predictions, all_probabilities, all_labels = [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            outputs = forward_fn(model, batch, device)
            labels = batch['labels'].to(device) if collect_labels else None
            if calibration is not None:
                outputs = calibration.transform(outputs)
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities > threshold_tensor.to(probabilities.device)).long()

            if outputs.shape[-1] == 1:
                all_predictions.extend(predictions.squeeze(-1).tolist())
                all_probabilities.extend(probabilities.squeeze(-1).tolist())
                if collect_labels:
                    all_labels.extend(labels.squeeze(-1).tolist())
            else:
                all_predictions.extend(predictions.tolist())
                all_probabilities.extend(probabilities.tolist())
                if collect_labels:
                    all_labels.extend(labels.tolist())

    return all_predictions, all_probabilities, all_labels


def predict_model(model, unlabeled_dataloader, device, return_probs=True,
                   threshold=0.5, calibration=None, forward_fn=None):
    """
    Predict labels for unlabeled data.

    Single label: returns flat lists.
    Multi-label: returns list of lists.

    Args:
        threshold: Decision threshold (float or list of floats for per-label).
        calibration: Optional TemperatureScaling object applied to logits before sigmoid.
        forward_fn: Forward override; pass ``cached_forward_fn`` for cached-embedding batches.

    Returns:
        tuple or list: (predictions, probabilities) if return_probs else predictions only.
    """
    predictions, probabilities, _ = _run_inference(model, unlabeled_dataloader, device,
                                                    threshold, calibration, desc="Predicting",
                                                    forward_fn=forward_fn)
    return (predictions, probabilities) if return_probs else predictions


def get_predictions_and_labels(model, dataloader, device, threshold=0.5,
                                calibration=None, forward_fn=None):
    """
    Get predictions, probabilities, and true labels from a labeled dataloader.

    Args:
        threshold: Decision threshold (float or list of floats for per-label).
        calibration: Optional TemperatureScaling object applied to logits before sigmoid.
        forward_fn: Forward override; pass ``cached_forward_fn`` for cached-embedding batches.

    Returns:
        tuple: (predictions, probabilities, true_labels)
    """
    return _run_inference(model, dataloader, device, threshold, calibration,
                          collect_labels=True, desc="Evaluating", forward_fn=forward_fn)


def flag_uncertain(probabilities, low=0.3, high=0.7):
    """
    Flag predictions with probability between low and high as uncertain.

    Single label: probabilities is list of floats, returns list of bools.
    Multi-label: probabilities is list of lists, returns list of lists of bools.
    """
    if not probabilities:
        return []
    if isinstance(probabilities[0], (list, tuple)):
        return [[low < probability < high for probability in row] for row in probabilities]
    return [low < probability < high for probability in probabilities]
