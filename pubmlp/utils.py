import torch
import logging

logger = logging.getLogger(__name__)


def get_device():
    """Detect optimal device (CUDA GPU or CPU)."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
    return device


def auto_batch_size(device):
    """Suggest batch size based on available GPU memory."""
    if not torch.cuda.is_available():
        return 8
    mem_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
    if mem_gb >= 40:
        return 32
    elif mem_gb >= 15:
        return 16
    return 8


def unpack_batch(batch, device):
    """Unpack tokenized batch to device."""
    return (
        batch['input_ids'].to(device),
        batch['attention_mask'].to(device),
        batch['categorical_tensor'].to(device),
        batch['numeric_tensor'].to(device),
        batch['labels'].to(device),
        batch.get('texts', None),
    )


def default_forward_fn(model, batch, device):
    """Tokenized forward through encoder + head."""
    input_ids, attention_mask, categorical_tensor, numeric_tensor, _, texts = unpack_batch(batch, device)
    return model(input_ids, attention_mask, categorical_tensor, numeric_tensor, texts)


def cached_forward_fn(model, batch, device):
    """Cached-embedding forward through head only."""
    return model.forward_from_embedding(
        batch['sentence_embedding'].to(device),
        batch['categorical_tensor'].to(device),
        batch['numeric_tensor'].to(device),
    )


def load_data(file_path):
    """Load CSV or Excel file into DataFrame."""
    import pandas as pd
    path = str(file_path)
    if path.endswith('.csv'):
        return pd.read_csv(path)
    return pd.read_excel(path)
