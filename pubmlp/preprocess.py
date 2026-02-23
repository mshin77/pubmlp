from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, Sampler, DataLoader
import torch
from sklearn.preprocessing import QuantileTransformer, RobustScaler


def split_data(data, random_state=42):
    """Split data into 80% train, 10% validation, 10% test."""
    shuffled = data.sample(frac=1, random_state=random_state)
    n = len(shuffled)
    train_end, val_end = int(0.8 * n), int(0.9 * n)
    return shuffled.iloc[:train_end], shuffled.iloc[train_end:val_end], shuffled.iloc[val_end:]


@dataclass
class FittedTransforms:
    """Stores fitted parameters from training data for reuse on val/test."""
    categorical_vocabs: dict = field(default_factory=dict)
    numeric_params: dict = field(default_factory=dict)

    @property
    def categorical_vocab_sizes(self):
        return [len(vocab) for vocab in self.categorical_vocabs.values()]

    def to_dict(self):
        numeric_serialized = {}
        for col, params in self.numeric_params.items():
            entry = {'transform': params['transform']}
            if 'median' in params:
                entry['median'] = float(params['median'])
            if params['transform'] == 'min':
                entry['min_val'] = float(params['min_val'])
            elif params['transform'] == 'max':
                entry['max_val'] = float(params['max_val'])
            elif params['transform'] == 'mean':
                entry['mean_val'] = float(params['mean_val'])
            elif params['transform'] in ('quantile', 'log1p'):
                entry['qt_params'] = {
                    'n_quantiles': params['qt'].n_quantiles,
                    'output_distribution': params['qt'].output_distribution,
                    'references_': params['qt'].references_.tolist(),
                    'quantiles_': params['qt'].quantiles_.tolist(),
                }
            elif params['transform'] == 'robust':
                entry['center'] = float(params['scaler'].center_[0])
                entry['scale'] = float(params['scaler'].scale_[0])
            numeric_serialized[col] = entry
        return {
            'categorical_vocabs': self.categorical_vocabs,
            'numeric_params': numeric_serialized,
        }

    @classmethod
    def from_dict(cls, d):
        obj = cls()
        obj.categorical_vocabs = d.get('categorical_vocabs', {})
        raw_numeric = d.get('numeric_params', {})
        for col, entry in raw_numeric.items():
            params = {'transform': entry['transform']}
            if 'median' in entry:
                params['median'] = entry['median']
            if entry['transform'] == 'min':
                params['min_val'] = entry['min_val']
            elif entry['transform'] == 'max':
                params['max_val'] = entry['max_val']
            elif entry['transform'] == 'mean':
                params['mean_val'] = entry['mean_val']
            elif entry['transform'] in ('quantile', 'log1p'):
                qt = QuantileTransformer(
                    n_quantiles=entry['qt_params']['n_quantiles'],
                    output_distribution=entry['qt_params']['output_distribution'],
                    random_state=0,
                )
                qt.references_ = np.array(entry['qt_params']['references_'])
                qt.quantiles_ = np.array(entry['qt_params']['quantiles_'])
                qt.n_features_in_ = 1
                params['qt'] = qt
            elif entry['transform'] == 'robust':
                scaler = RobustScaler()
                scaler.center_ = np.array([entry['center']])
                scaler.scale_ = np.array([entry['scale']])
                scaler.n_features_in_ = 1
                params['scaler'] = scaler
            obj.numeric_params[col] = params
        return obj


class CustomDataset(Dataset):
    """Dataset holding tokenized text, categorical/numeric features, and labels."""

    def __init__(self, input_ids, attention_mask, labels, categorical_tensor=None,
                 numeric_tensor=None, texts=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
        self.categorical_tensor = categorical_tensor if categorical_tensor is not None else torch.tensor([], dtype=torch.long)
        self.numeric_tensor = numeric_tensor if numeric_tensor is not None else torch.tensor([], dtype=torch.float)
        self.texts = texts

    def __getitem__(self, idx):
        item = {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx],
            'categorical_tensor': self.categorical_tensor[idx] if self.categorical_tensor.numel() > 0 else self.categorical_tensor,
            'numeric_tensor': self.numeric_tensor[idx] if self.numeric_tensor.numel() > 0 else self.numeric_tensor,
        }
        if self.texts is not None:
            item['texts'] = self.texts[idx]
        return item

    def __len__(self):
        return len(self.input_ids)


def _build_categorical_vocab(series, rare_threshold=5):
    """Build value->index mapping. Index 0=<UNK>, 1=<RARE>, 2+=actual values."""
    counts = series.value_counts()
    vocab = {'<UNK>': 0, '<RARE>': 1}
    idx = 2
    for val, count in counts.items():
        if count >= rare_threshold:
            vocab[val] = idx
            idx += 1
    return vocab


def _encode_categorical(series, vocab):
    """Map series values to integer indices using vocab."""
    rare_idx = vocab['<RARE>']
    unk_idx = vocab['<UNK>']
    return series.map(lambda v: vocab.get(v, rare_idx) if pd.notna(v) else unk_idx)


def _fit_numeric(series, transform):
    """Fit numeric transform on training data, return params dict."""
    clean = series.dropna()
    params = {'transform': transform, 'median': float(clean.median()) if len(clean) > 0 else 0.0}
    if transform == 'min':
        params['min_val'] = float(clean.min()) if len(clean) > 0 else 0.0
    elif transform == 'max':
        params['max_val'] = float(clean.max()) if len(clean) > 0 else 1.0
    elif transform == 'mean':
        params['mean_val'] = float(clean.mean()) if len(clean) > 0 else 0.0
    elif transform in ('quantile', 'log1p'):
        values = clean.values.reshape(-1, 1)
        if transform == 'log1p':
            values = np.log1p(values)
        qt = QuantileTransformer(output_distribution='normal', random_state=0)
        qt.fit(values)
        params['qt'] = qt
    elif transform == 'robust':
        scaler = RobustScaler()
        scaler.fit(clean.values.reshape(-1, 1))
        params['scaler'] = scaler
    else:
        raise ValueError(f"Invalid transform: {transform}")
    return params


def _apply_numeric(series, params):
    """Apply fitted numeric transform to a series."""
    filled = series.fillna(params['median'])
    transform = params['transform']
    if transform == 'min':
        return filled - params['min_val']
    elif transform == 'max':
        max_val = params['max_val']
        return filled / max_val if max_val != 0 else filled
    elif transform == 'mean':
        return filled - params['mean_val']
    elif transform == 'quantile':
        return pd.Series(
            params['qt'].transform(filled.values.reshape(-1, 1)).flatten(),
            index=series.index,
        )
    elif transform == 'log1p':
        log_vals = np.log1p(filled.values.reshape(-1, 1))
        return pd.Series(
            params['qt'].transform(log_vals).flatten(),
            index=series.index,
        )
    elif transform == 'robust':
        return pd.Series(
            params['scaler'].transform(filled.values.reshape(-1, 1)).flatten(),
            index=series.index,
        )
    raise ValueError(f"Unknown transform: {transform}")


def preprocess_dataset(data, tokenizer, device, column_specifications, numeric_transform,
                       max_length=512, fitted_transforms=None, rare_threshold=5):
    """
    Preprocess DataFrame into a CustomDataset.

    Args:
        data: DataFrame with text, categorical, numeric, and label columns.
        tokenizer: HuggingFace tokenizer.
        device: torch device.
        column_specifications: dict with 'text_cols', 'categorical_cols', 'numeric_cols', 'label_col'.
            label_col can be a string (single label) or list of strings (multi-label).
        numeric_transform: dict mapping numeric column names to transform type.
        max_length: Max token length for tokenization.
        fitted_transforms: FittedTransforms from training data. None = fit from this data (training).
        rare_threshold: Minimum count for a categorical value to get its own embedding index.

    Returns:
        tuple: (CustomDataset, FittedTransforms) when fitting (fitted_transforms=None)
        tuple: (CustomDataset, None) when applying existing transforms
    """
    fitting = fitted_transforms is None
    if fitting:
        fitted_transforms = FittedTransforms()

    # Concatenate text columns with separator (tokenizer adds [CLS]/[SEP] automatically)
    sep = tokenizer.sep_token or " "
    texts = [
        f" {sep} ".join(str(row[col]) for col in column_specifications["text_cols"])
        for _, row in data.iterrows()
    ]

    encoding = tokenizer(texts, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt")
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Categorical encoding — integer indices for nn.Embedding
    categorical_cols = column_specifications.get("categorical_cols", [])
    if categorical_cols:
        cat_indices = []
        for cat_col in categorical_cols:
            if fitting:
                vocab = _build_categorical_vocab(data[cat_col], rare_threshold)
                fitted_transforms.categorical_vocabs[cat_col] = vocab
            else:
                vocab = fitted_transforms.categorical_vocabs[cat_col]
            cat_indices.append(_encode_categorical(data[cat_col], vocab).values)
        categorical_tensor = torch.tensor(np.column_stack(cat_indices), dtype=torch.long).to(device)
    else:
        categorical_tensor = torch.tensor([], dtype=torch.long).to(device)

    # Numeric normalization — fit on train, apply stored params on val/test
    numeric_cols = column_specifications.get("numeric_cols", [])
    if numeric_cols:
        numeric_values = []
        for num_col in numeric_cols:
            if num_col not in numeric_transform:
                raise ValueError(f"Column {num_col} not found in numeric_transform.")
            if fitting:
                params = _fit_numeric(data[num_col], numeric_transform[num_col])
                fitted_transforms.numeric_params[num_col] = params
            else:
                params = fitted_transforms.numeric_params[num_col]
            numeric_values.append(_apply_numeric(data[num_col], params).values)
        numeric_arr = np.column_stack(numeric_values).astype(np.float32)
        numeric_tensor = torch.tensor(numeric_arr, dtype=torch.float).to(device)
    else:
        numeric_tensor = torch.tensor([], dtype=torch.float).to(device)

    # Labels — single (string) or multi-label (list)
    label_col = column_specifications["label_col"]
    if isinstance(label_col, list):
        label_values = data[label_col].values.astype(np.float32)
        labels = torch.tensor(label_values, dtype=torch.float).to(device)
    else:
        label_values = data[label_col].values.astype(np.float32)
        labels = torch.tensor(label_values, dtype=torch.float).unsqueeze(1).to(device)

    dataset = CustomDataset(input_ids, attention_mask, labels, categorical_tensor, numeric_tensor, texts)
    return (dataset, fitted_transforms) if fitting else (dataset, None)


def collate_fn(batch):
    """Custom collate to handle text lists in batches."""
    result = {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'labels': torch.stack([item['labels'] for item in batch]),
        'categorical_tensor': torch.stack([item['categorical_tensor'] for item in batch]),
        'numeric_tensor': torch.stack([item['numeric_tensor'] for item in batch]),
        'texts': None,
    }
    if 'texts' in batch[0] and batch[0]['texts'] is not None:
        result['texts'] = [item['texts'] for item in batch]
    return result


def create_dataloader(dataset: Dataset, sampler: Sampler, batch_size: int) -> DataLoader:
    """Create DataLoader with custom collate function."""
    return DataLoader(dataset, sampler=sampler(dataset), batch_size=batch_size, collate_fn=collate_fn)
