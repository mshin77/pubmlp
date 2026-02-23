# API Reference

## Configuration

#### `Config`

Dataclass holding all training and model hyperparameters.

| Parameter | Default | Description |
|---|---|---|
| `rare_threshold` | `5` | Minimum count for a categorical value to get its own embedding |
| `pos_weight` | `'auto'` | `'auto'` computes from training labels, `None` disables |

#### Preset Configurations

| Preset | Description |
|---|---|
| `default_config` | Balanced defaults for general use |
| `fast_config` | Fewer epochs, larger batch size for quick experiments |
| `robust_config` | More epochs, lower learning rate for production |
| `hitl_config` | Human-in-the-loop screening settings |
| `domain_configs` | Domain-specific presets (science, medicine, general, modernbert) |

## Data

#### `preprocess_dataset`

Tokenize text columns and encode categorical/numeric features into a dataset. Pass `fitted_transforms=None` to fit from training data, or a `FittedTransforms` object to reuse on val/test sets. Returns `(CustomDataset, FittedTransforms)`.

#### `column_specifications`

Dictionary specifying which DataFrame columns to use.

| Key | Value | Example |
|---|---|---|
| `text_cols` | List of text column names | `["TI", "AB"]` |
| `categorical_cols` | List of categorical column names | `["SO"]` |
| `numeric_cols` | List of numeric column names | `["PY"]` |
| `label_col` | String (single-label) or list (multi-label) | `"label"` or `["label_1", "label_2"]` |

#### Numeric Transforms

| Transform | Description |
|---|---|
| `min` | Subtract minimum value |
| `max` | Divide by maximum value |
| `mean` | Subtract mean |
| `quantile` | Quantile transform to normal distribution |
| `robust` | RobustScaler (median + IQR), better for outliers |
| `log1p` | log1p then quantile transform, for skewed features |

#### `load_data`

Load data from CSV or Excel files.

#### `split_data`

Split a DataFrame into train, validation, and test sets.

#### `FittedTransforms`

Stores fitted parameters from training data for reuse on val/test sets. Serialize with `to_dict()` and restore with `FittedTransforms.from_dict(d)`.

## Model

#### `PubMLP`

Multi-layer perceptron that combines transformer embeddings with categorical and numeric features.

| Parameter | Description |
|---|---|
| `categorical_vocab_sizes` | List of vocab sizes for nn.Embedding per categorical column |
| `output_size` | 1 for single-label, N for multi-label |

## Training & Evaluation

#### `train_evaluate_model`

Full training loop with validation, early stopping, and test evaluation. Returns `(train_losses, val_losses, train_accs, val_accs, test_acc, best_val_loss, best_model_state, best_epoch)`.

#### `calculate_evaluation_metrics`

Compute classification report, confusion matrix, and ROC-AUC. Single label: returns accuracy, precision, recall, specificity, f1_score, roc_auc. Multi-label: returns `per_label` metrics, `macro_f1`, `hamming_loss`.

#### `cross_validate`

Stratified K-fold cross-validation with per-fold metrics.

#### `plot_results`

Plot training/validation loss and accuracy curves.

#### `TemperatureScaling`

Post-hoc temperature scaling for model calibration.

#### `calibrate_model`

Fit temperature scaling on validation data.

#### `collect_logits`

Collect raw logits from a trained model.

## Prediction

#### `predict_model`

Run inference and return predictions and probabilities. Single label: flat lists. Multi-label: list of lists.

#### `flag_uncertain`

Flag predictions with probabilities in an uncertain range for human review. Multi-label: returns list of lists of bools.

## Screening

#### `regex_screen`

Screen a dataset using regex patterns with optional semantic similarity scoring.

```python
from pubmlp import regex_screen

results = regex_screen("records.csv", inclusion_patterns=["intervention", "randomized"])
```

#### `create_stratified_sample`

Create a stratified random sample with regex pattern highlights for human coding.

#### `save_sample_excel`

Save sample to Excel with conditional formatting for review.

## Active Learning

#### `select_query_batch`

Select the most uncertain samples for human review.

#### `create_review_batch`

Create a review batch DataFrame with model probability and prediction columns.

#### `compare_reviewers`

Compute inter-rater agreement (kappa + agreement rate) between model and human.

#### `merge_human_labels`

Merge human decisions from review batch back into the main DataFrame.

#### `ALState`

Dataclass tracking active learning iteration state.

## Stopping Rules

#### `should_stop`

Evaluate whether screening can stop based on SAFE criterion.

#### `update_stopping_state`

Update stopping state counters after a human screening decision.

#### `estimate_recall`

Wilson score lower bound estimate of recall.

#### `generate_stopping_report`

Generate a summary report of stopping criteria.

#### `calculate_wss`

Calculate Work Saved over Sampling at a given recall level.

#### `StoppingState`

Dataclass tracking stopping-rule state across iterations.

## Audit

#### `AuditTrail`

Record and persist screening decisions for reproducibility.

#### `AuditEntry`

Single audit log entry dataclass.

#### `summarize_human_decisions`

Summarize human reviewer decisions from an audit trail.

#### `generate_prisma_report`

Generate a PRISMA-style flow diagram report.

#### `interpret_kappa`

Interpret Cohen's kappa agreement level.

## Utilities

#### `get_device`

Return the best available PyTorch device (CUDA or CPU).

#### `auto_batch_size`

Suggest a batch size based on available GPU memory.
