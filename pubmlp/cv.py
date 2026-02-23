import copy
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import SequentialSampler, RandomSampler

from .model import PubMLP
from .preprocess import preprocess_dataset, create_dataloader
from .train import train_evaluate_model
from .predict import get_predictions_and_labels
from .metrics import calculate_evaluation_metrics


def cross_validate(data, tokenizer, device, column_specifications, numeric_transform, config,
                   categorical_cols_num=0, numeric_cols_num=0, output_dir=None,
                   categorical_vocab_sizes=None, output_size=1, label_names=None):
    """
    Stratified K-fold cross-validation for PubMLP.

    Args:
        data: DataFrame with text, categorical, numeric, and label columns.
        tokenizer: HuggingFace AutoTokenizer.
        device: torch device.
        column_specifications: dict with 'text_cols', 'categorical_cols', 'numeric_cols', 'label_col'.
        numeric_transform: dict mapping numeric columns to transform type.
        config: Config instance.
        categorical_cols_num: Number of categorical feature columns (legacy scalar mode).
        numeric_cols_num: Number of numeric feature columns.
        output_dir: Optional path to save per-fold figures.
        categorical_vocab_sizes: List of vocab sizes for embedding mode.
        output_size: Number of output labels.
        label_names: List of label name strings for multi-label display.

    Returns:
        dict with 'fold_metrics', 'mean_metrics', 'std_metrics', 'best_fold', 'best_model_state'.
    """
    config.set_random_seeds()
    label_col = column_specifications['label_col']
    # Stratification: composite key for multi-label, direct values for single
    if isinstance(label_col, list):
        strat_labels = data[label_col].astype(str).agg('_'.join, axis=1).values
    else:
        strat_labels = data[label_col].values
    skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.random_seed)

    fold_metrics = []
    best_fold_val_acc = 0.0
    best_fold_idx = 0
    best_model_state = None

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(data, strat_labels)):
        print(f'\n{"="*60}')
        print(f'Fold {fold_idx + 1}/{config.n_folds}')
        print(f'{"="*60}')
        print(f'Train: {len(train_idx)} | Val: {len(val_idx)}')

        train_data = data.iloc[train_idx].reset_index(drop=True)
        val_data = data.iloc[val_idx].reset_index(drop=True)

        # Fit transforms on train fold, apply to val fold
        train_dataset, fitted = preprocess_dataset(
            train_data, tokenizer, device, column_specifications,
            numeric_transform, max_length=config.max_length,
            rare_threshold=getattr(config, 'rare_threshold', 5),
        )
        val_dataset, _ = preprocess_dataset(
            val_data, tokenizer, device, column_specifications,
            numeric_transform, max_length=config.max_length,
            fitted_transforms=fitted,
        )

        train_loader = create_dataloader(train_dataset, RandomSampler, config.batch_size)
        val_loader = create_dataloader(val_dataset, SequentialSampler, config.eval_batch_size)

        # Use fitted vocab sizes if available from this fold's training data
        fold_vocab_sizes = fitted.categorical_vocab_sizes if fitted.categorical_vocabs else categorical_vocab_sizes

        model = PubMLP(
            categorical_cols_num=categorical_cols_num,
            numeric_cols_num=numeric_cols_num,
            mlp_hidden_size=config.mlp_hidden_size,
            dropout_rate=config.dropout_rate,
            embedding_model=config.embedding_model,
            model_name=config.model_name,
            n_hidden_layers=config.n_hidden_layers,
            pooling_strategy=config.pooling_strategy,
            categorical_vocab_sizes=fold_vocab_sizes,
            output_size=output_size,
        )
        model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        criterion = torch.nn.BCEWithLogitsLoss()

        results = train_evaluate_model(
            model, train_loader, val_loader, None,
            optimizer, criterion, device, config.epochs,
            early_stopping_patience=config.early_stopping_patience,
            gradient_clip_norm=config.gradient_clip_norm,
            pos_weight=getattr(config, 'pos_weight', None),
            warmup_steps=getattr(config, 'warmup_steps', 0),
        )

        train_losses, val_losses, train_accs, val_accs, test_acc, best_val_loss, fold_state, best_epoch = results

        if fold_state is not None:
            model.load_state_dict(fold_state)

        preds, probs, true_labels = get_predictions_and_labels(model, val_loader, device)

        fold_dir = f'{output_dir}/fold_{fold_idx + 1}' if output_dir else None
        metrics = calculate_evaluation_metrics(
            true_labels, preds, probs,
            output_dir=fold_dir,
            label_name=f'fold_{fold_idx + 1}',
            save_figures=output_dir is not None,
            label_names=label_names,
        )
        best_epoch_acc = val_accs[best_epoch - 1] if val_accs and best_epoch > 0 else 0.0
        metrics.update({
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'val_accuracy': best_epoch_acc,
        })
        fold_metrics.append(metrics)

        if best_epoch_acc > best_fold_val_acc:
            best_fold_val_acc = best_epoch_acc
            best_fold_idx = fold_idx
            best_model_state = copy.deepcopy(fold_state)

        # Print summary depends on single vs multi-label
        if 'f1_score' in metrics:
            print(f'Fold {fold_idx + 1} — F1: {metrics["f1_score"]:.3f} | '
                  f'Precision: {metrics["precision"]:.3f} | Recall: {metrics["recall"]:.3f}')
        else:
            print(f'Fold {fold_idx + 1} — Macro F1: {metrics["macro_f1"]:.3f} | '
                  f'Hamming Loss: {metrics["hamming_loss"]:.3f}')

    # Aggregate across folds
    metric_keys = [k for k in fold_metrics[0] if isinstance(fold_metrics[0][k], (int, float, np.integer, np.floating)) and fold_metrics[0][k] is not None]
    mean_metrics = {k: np.mean([fm[k] for fm in fold_metrics if fm.get(k) is not None]) for k in metric_keys}
    std_metrics = {k: np.std([fm[k] for fm in fold_metrics if fm.get(k) is not None]) for k in metric_keys}

    print(f'\n{"="*60}')
    print(f'Cross-Validation Summary ({config.n_folds} folds)')
    print(f'{"="*60}')
    summary_keys = [k for k in ['accuracy', 'precision', 'recall', 'specificity', 'f1_score',
                                 'roc_auc', 'macro_f1', 'hamming_loss'] if k in mean_metrics]
    for k in summary_keys:
        print(f'{k}: {mean_metrics[k]:.3f} ± {std_metrics[k]:.3f}')
    print(f'Best fold: {best_fold_idx + 1} (val acc: {best_fold_val_acc:.3f}%)')

    return {
        'fold_metrics': fold_metrics,
        'mean_metrics': mean_metrics,
        'std_metrics': std_metrics,
        'best_fold': best_fold_idx,
        'best_model_state': best_model_state,
    }
