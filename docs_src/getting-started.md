# Getting Started

## Installation

```bash
pip install pubmlp
```

With optional dependencies:

```bash
# Screening tools (openpyxl, nltk)
pip install pubmlp[screening]
```

From GitHub:

```bash
pip install git+https://github.com/mshin77/pubmlp.git
```

## Configuration

```python
from pubmlp import Config

config = Config(
    random_seed=2025,
    embedding_model='bert',
    batch_size=8,
    epochs=3,
    learning_rate=1e-5,
    mlp_hidden_size=16,
    dropout_rate=0.2,
    early_stopping_patience=3,
)
```

Preset configurations are also available:

```python
from pubmlp import default_config, fast_config, robust_config, hitl_config
```

## Preprocessing

```python
import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from pubmlp import preprocess_dataset, create_dataloader, split_data

df = pd.read_excel("labeled_data.xlsx")
train_df, val_df, test_df = split_data(df, random_state=42)

tokenizer = AutoTokenizer.from_pretrained(config.model_name)

column_specifications = {
    "text_cols": ["TI", "AB"],
    "categorical_cols": ["SO"],
    "numeric_cols": ["PY"],
    "label_col": "label",
}

numeric_transform = {"PY": "min"}

# Fit transforms on training data
train_dataset, fitted = preprocess_dataset(
    train_df, tokenizer, device, column_specifications, numeric_transform
)
train_dataloader = create_dataloader(train_dataset, RandomSampler, config.batch_size)

# Apply fitted transforms to val/test (no data leakage)
val_dataset, _ = preprocess_dataset(
    val_df, tokenizer, device, column_specifications, numeric_transform,
    fitted_transforms=fitted
)
val_dataloader = create_dataloader(val_dataset, SequentialSampler, config.eval_batch_size)
```

## Training

```python
import torch.nn as nn
from torch.optim import AdamW
from pubmlp import PubMLP, train_evaluate_model, get_device

device = get_device()

model = PubMLP(
    categorical_vocab_sizes=fitted.categorical_vocab_sizes,
    numeric_cols_num=1,
    mlp_hidden_size=config.mlp_hidden_size,
    output_size=1,
    dropout_rate=config.dropout_rate,
    embedding_model=config.embedding_model,
    model_name=config.model_name,
).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = AdamW(model.parameters(), lr=config.learning_rate)

(train_losses, val_losses,
 train_accs, val_accs,
 test_acc, best_val_loss,
 best_model_state, best_epoch) = train_evaluate_model(
    model, train_dataloader, val_dataloader, test_dataloader,
    optimizer, criterion, device, config.epochs,
)
```

## Prediction

```python
from pubmlp import predict_model, flag_uncertain

predictions, probabilities = predict_model(model, test_dataloader, device)
uncertain = flag_uncertain(probabilities)
```

## Evaluation

```python
from pubmlp import (
    get_predictions_and_labels, calculate_evaluation_metrics, plot_results,
)

test_preds, test_probs, test_labels = get_predictions_and_labels(model, test_dataloader, device)
metrics = calculate_evaluation_metrics(test_labels, test_preds, test_probs)
plot_results(train_losses, val_losses, train_accs, val_accs, test_acc, best_val_loss)
```

## Multi-Label Classification

```python
column_specifications = {
    "text_cols": ["TI", "AB"],
    "categorical_cols": ["SO"],
    "numeric_cols": ["PY"],
    "label_col": ["single_case", "technology_use"],  # list for multi-label
}

train_dataset, fitted = preprocess_dataset(
    train_df, tokenizer, device, column_specifications, numeric_transform
)

model = PubMLP(
    categorical_vocab_sizes=fitted.categorical_vocab_sizes,
    numeric_cols_num=1,
    output_size=2,  # matches number of labels
    mlp_hidden_size=config.mlp_hidden_size,
    embedding_model=config.embedding_model,
    model_name=config.model_name,
).to(device)

# Per-label metrics
metrics = calculate_evaluation_metrics(
    test_labels, test_preds, test_probs,
    label_names=["single_case", "technology_use"],
)
```

## Supported Embedding Models

### Trainable encoders (fine-tuned during training)

| Key | Model | Params | Description |
|-----|-------|--------|-------------|
| `bert` | `bert-base-uncased` | 110M | General-purpose BERT |
| `modernbert` | `answerdotai/ModernBERT-base` | 150M | 2-4x faster than BERT, 8192 context |
| `scibert` | `allenai/scibert_scivocab_uncased` | 110M | Scientific text |
| `pubmedbert` | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | 110M | PubMed abstracts |

### Frozen encoders (fast, no GPU required)

| Key | Model | Params | Description |
|-----|-------|--------|-------------|
| `bge-small` | `BAAI/bge-small-en-v1.5` | 33M | Lightweight, CPU-friendly |
| `sentence-transformer` | `all-MiniLM-L6-v2` | 22M | Fast, lightweight |
