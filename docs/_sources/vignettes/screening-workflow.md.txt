# Screening Workflow

End-to-end example of using pubmlp to screen bibliometric records for a systematic review. Two approaches are shown:

- **Approach A — Sequential single-label**: Train one model per screening criterion. Best when criteria have very different class distributions or when you need separate tuning per label.
- **Approach B — Simultaneous multi-label**: Train one model that predicts all criteria at once. Best when criteria share signal from the same text, reducing total training time.

Both approaches use multiple categorical and numeric columns.

## Shared Setup

```python
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer
from torch.utils.data.sampler import RandomSampler, SequentialSampler

from pubmlp import (
    Config, PubMLP, FittedTransforms,
    preprocess_dataset, create_dataloader, split_data,
    train_evaluate_model, get_device,
    get_predictions_and_labels, calculate_evaluation_metrics,
    predict_model, flag_uncertain, plot_results,
)

df = pd.read_excel("labeled_data.xlsx")
df = df[["UT", "single_case", "technology_use", "SO", "DT", "PY", "TC", "AF", "TI", "AB", "DE"]].copy()
df["PY"] = pd.to_numeric(df["PY"], errors="coerce")
df["TC"] = pd.to_numeric(df["TC"], errors="coerce")
df.dropna(subset=["PY"], inplace=True)

for col in ["single_case", "technology_use"]:
    df[col] = df[col].map({"Yes": 1, "No": 0})

train_df, val_df, test_df = split_data(df, random_state=42)

config = Config(
    random_seed=2025,
    embedding_model="bert",
    batch_size=8,
    eval_batch_size=8,
    epochs=3,
    learning_rate=1e-5,
    mlp_hidden_size=16,
    dropout_rate=0.2,
    early_stopping_patience=3,
)

device = get_device()
tokenizer = AutoTokenizer.from_pretrained(config.model_name)

# Multiple categorical and numeric columns with different transforms
numeric_transform = {
    "PY": "min",       # publication year — shift to start at 0
    "TC": "log1p",     # citation counts — heavily skewed
}
```

---

## Approach A — Sequential Single-Label

Train a separate model for each screening criterion. This is the approach when each label needs independent tuning or has very different characteristics.

### A1. Single-Case Classification

```python
col_spec_sc = {
    "text_cols": ["AF", "TI", "AB", "DE"],
    "categorical_cols": ["SO", "DT"],
    "numeric_cols": ["PY", "TC"],
    "label_col": "single_case",             # single string = single-label
}

# Fit on training data
train_dataset_sc, fitted_sc = preprocess_dataset(
    train_df, tokenizer, device, col_spec_sc, numeric_transform
)
# Apply to val/test — no data leakage
val_dataset_sc, _ = preprocess_dataset(
    val_df, tokenizer, device, col_spec_sc, numeric_transform,
    fitted_transforms=fitted_sc
)
test_dataset_sc, _ = preprocess_dataset(
    test_df, tokenizer, device, col_spec_sc, numeric_transform,
    fitted_transforms=fitted_sc
)

train_loader_sc = create_dataloader(train_dataset_sc, RandomSampler, config.batch_size)
val_loader_sc = create_dataloader(val_dataset_sc, SequentialSampler, config.eval_batch_size)
test_loader_sc = create_dataloader(test_dataset_sc, SequentialSampler, config.eval_batch_size)

model_sc = PubMLP(
    categorical_vocab_sizes=fitted_sc.categorical_vocab_sizes,
    numeric_cols_num=2,
    mlp_hidden_size=config.mlp_hidden_size,
    output_size=1,
    dropout_rate=config.dropout_rate,
    embedding_model=config.embedding_model,
    model_name=config.model_name,
).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = AdamW(model_sc.parameters(), lr=config.learning_rate)

results_sc = train_evaluate_model(
    model_sc, train_loader_sc, val_loader_sc, test_loader_sc,
    optimizer, criterion, device, config.epochs,
    early_stopping_patience=config.early_stopping_patience,
)

preds, probs, labels = get_predictions_and_labels(model_sc, test_loader_sc, device)
metrics_sc = calculate_evaluation_metrics(labels, preds, probs, label_name="single_case")

torch.save(results_sc[6], "model_single_case.pth")
```

### A2. Technology-Use Classification

```python
col_spec_tu = {
    "text_cols": ["AF", "TI", "AB", "DE"],
    "categorical_cols": ["SO", "DT"],
    "numeric_cols": ["PY", "TC"],
    "label_col": "technology_use",           # different label, same features
}

train_dataset_tu, fitted_tu = preprocess_dataset(
    train_df, tokenizer, device, col_spec_tu, numeric_transform
)
val_dataset_tu, _ = preprocess_dataset(
    val_df, tokenizer, device, col_spec_tu, numeric_transform,
    fitted_transforms=fitted_tu
)
test_dataset_tu, _ = preprocess_dataset(
    test_df, tokenizer, device, col_spec_tu, numeric_transform,
    fitted_transforms=fitted_tu
)

train_loader_tu = create_dataloader(train_dataset_tu, RandomSampler, config.batch_size)
val_loader_tu = create_dataloader(val_dataset_tu, SequentialSampler, config.eval_batch_size)
test_loader_tu = create_dataloader(test_dataset_tu, SequentialSampler, config.eval_batch_size)

model_tu = PubMLP(
    categorical_vocab_sizes=fitted_tu.categorical_vocab_sizes,
    numeric_cols_num=2,
    mlp_hidden_size=config.mlp_hidden_size,
    output_size=1,
    dropout_rate=config.dropout_rate,
    embedding_model=config.embedding_model,
    model_name=config.model_name,
).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = AdamW(model_tu.parameters(), lr=config.learning_rate)

results_tu = train_evaluate_model(
    model_tu, train_loader_tu, val_loader_tu, test_loader_tu,
    optimizer, criterion, device, config.epochs,
    early_stopping_patience=config.early_stopping_patience,
)

preds, probs, labels = get_predictions_and_labels(model_tu, test_loader_tu, device)
metrics_tu = calculate_evaluation_metrics(labels, preds, probs, label_name="technology_use")

torch.save(results_tu[6], "model_technology_use.pth")
```

### A3. Predict Unlabeled (Sequential)

```python
unlabeled_df = pd.read_excel("unlabeled_data.xlsx")
unlabeled_df["PY"] = pd.to_numeric(unlabeled_df["PY"], errors="coerce")
unlabeled_df["TC"] = pd.to_numeric(unlabeled_df["TC"], errors="coerce")

# Predict single_case
unlabeled_ds_sc, _ = preprocess_dataset(
    unlabeled_df, tokenizer, device, col_spec_sc, numeric_transform,
    fitted_transforms=fitted_sc
)
loader_sc = create_dataloader(unlabeled_ds_sc, SequentialSampler, config.eval_batch_size)
model_sc.load_state_dict(torch.load("model_single_case.pth", map_location=device))
preds_sc, probs_sc = predict_model(model_sc, loader_sc, device)

# Predict technology_use
unlabeled_ds_tu, _ = preprocess_dataset(
    unlabeled_df, tokenizer, device, col_spec_tu, numeric_transform,
    fitted_transforms=fitted_tu
)
loader_tu = create_dataloader(unlabeled_ds_tu, SequentialSampler, config.eval_batch_size)
model_tu.load_state_dict(torch.load("model_technology_use.pth", map_location=device))
preds_tu, probs_tu = predict_model(model_tu, loader_tu, device)

unlabeled_df["single_case"] = ["Yes" if p == 1 else "No" for p in preds_sc]
unlabeled_df["single_case_prob"] = probs_sc
unlabeled_df["technology_use"] = ["Yes" if p == 1 else "No" for p in preds_tu]
unlabeled_df["technology_use_prob"] = probs_tu
unlabeled_df.to_excel("predicted_sequential.xlsx", index=False)
```

---

## Approach B — Simultaneous Multi-Label

Train one model that predicts all criteria at once. Shared encoder representations can improve performance when the labels are related.

### B1. Preprocess with Multi-Label

```python
col_spec_multi = {
    "text_cols": ["AF", "TI", "AB", "DE"],
    "categorical_cols": ["SO", "DT"],
    "numeric_cols": ["PY", "TC"],
    "label_col": ["single_case", "technology_use"],  # list = multi-label
}

train_dataset, fitted = preprocess_dataset(
    train_df, tokenizer, device, col_spec_multi, numeric_transform
)
val_dataset, _ = preprocess_dataset(
    val_df, tokenizer, device, col_spec_multi, numeric_transform,
    fitted_transforms=fitted
)
test_dataset, _ = preprocess_dataset(
    test_df, tokenizer, device, col_spec_multi, numeric_transform,
    fitted_transforms=fitted
)

train_loader = create_dataloader(train_dataset, RandomSampler, config.batch_size)
val_loader = create_dataloader(val_dataset, SequentialSampler, config.eval_batch_size)
test_loader = create_dataloader(test_dataset, SequentialSampler, config.eval_batch_size)
```

### B2. Train Multi-Label Model

```python
model = PubMLP(
    categorical_vocab_sizes=fitted.categorical_vocab_sizes,
    numeric_cols_num=2,
    mlp_hidden_size=config.mlp_hidden_size,
    output_size=2,                           # one output per label
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
    model, train_loader, val_loader, test_loader,
    optimizer, criterion, device, config.epochs,
    early_stopping_patience=config.early_stopping_patience,
)

torch.save(best_model_state, "model_multilabel.pth")
```

### B3. Evaluate Per-Label Metrics

```python
preds, probs, labels = get_predictions_and_labels(model, test_loader, device)

# Per-label P/R/F1/AUC + macro F1 + hamming loss
metrics = calculate_evaluation_metrics(
    labels, preds, probs,
    label_names=["single_case", "technology_use"],
    save_figures=True,
    output_dir="figures/",
)

# Access per-label results
for lname, lmetrics in metrics["per_label"].items():
    print(f'{lname}: F1={lmetrics["f1_score"]:.3f} AUC={lmetrics["roc_auc"]:.3f}')
print(f'Macro F1: {metrics["macro_f1"]:.3f}')
print(f'Hamming Loss: {metrics["hamming_loss"]:.3f}')
```

### B4. Predict Unlabeled (Multi-Label)

```python
unlabeled_df = pd.read_excel("unlabeled_data.xlsx")
unlabeled_df["PY"] = pd.to_numeric(unlabeled_df["PY"], errors="coerce")
unlabeled_df["TC"] = pd.to_numeric(unlabeled_df["TC"], errors="coerce")

# Dummy label columns for preprocessing (values ignored during prediction)
for col in ["single_case", "technology_use"]:
    unlabeled_df[col] = 0

unlabeled_dataset, _ = preprocess_dataset(
    unlabeled_df, tokenizer, device, col_spec_multi, numeric_transform,
    fitted_transforms=fitted
)
unlabeled_loader = create_dataloader(unlabeled_dataset, SequentialSampler, config.eval_batch_size)

model.load_state_dict(torch.load("model_multilabel.pth", map_location=device))
predictions, probabilities = predict_model(model, unlabeled_loader, device)

# Multi-label: predictions and probabilities are list of lists
uncertain = flag_uncertain(probabilities)
print(f"Records with any uncertain label: {sum(any(u) for u in uncertain)} / {len(uncertain)}")

label_names = ["single_case", "technology_use"]
for i, col in enumerate(label_names):
    unlabeled_df[col] = ["Yes" if p[i] == 1 else "No" for p in predictions]
    unlabeled_df[f"{col}_prob"] = [p[i] for p in probabilities]
    unlabeled_df[f"{col}_uncertain"] = [u[i] for u in uncertain]

unlabeled_df.to_excel("predicted_multilabel.xlsx", index=False)
```

---

## Which Approach to Choose?

| | Sequential (A) | Simultaneous (B) |
|---|---|---|
| **When to use** | Different class distributions per label, need separate hyperparameter tuning | Labels share text signal, want faster total training |
| **Models trained** | One per label | One for all labels |
| **Metrics** | Standard per-model metrics | Per-label + macro F1 + hamming loss |
| **Flexibility** | Can use different configs per label | Shared architecture and training |
| **Saving/loading** | Save one model + fitted_transforms per label | Save one model + one fitted_transforms |
