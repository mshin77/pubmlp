"""Screening simulations over benchmark datasets."""

import argparse
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from .active_learning import safe_stratified_split, select_query_batch
from .calibration import calibrate_model
from .config import Config
from .datasets import build_column_specs, load_benchmark, normalize_benchmark_frame
from .metrics import calculate_brier, calculate_ece, calculate_ndcg, calculate_wss_at_recall
from .predict import predict_model
from .preprocess import CachedEmbeddingDataset, create_dataloader
from .stopping import recall_target_test
from .train import train_evaluate_model
from .utils import cached_forward_fn

RANDOM_PCTS = [0.01, 0.05, 0.10, 0.15, 0.20]
AL_MAX_ITERATIONS = 5
AL_MIN_RECORDS = 20
AL_PATIENCE = 2
AL_MIN_IMPROVEMENT = 0.005
AL_INITIAL_PCT = 0.01
RECALL_TARGET = 0.95
CACHED_HEAD_LR = 1e-3


class _CachedHead(nn.Module):
    """Classifier head over precomputed embeddings + tabular features."""

    def __init__(self, embedding_size, journal_vocab_size=0, use_year=False,
                 mlp_hidden_size=64, n_hidden_layers=1, dropout_rate=0.2, output_size=1):
        super().__init__()
        self.use_year = use_year
        self.journal_embedding = None
        journal_dim = 0
        if journal_vocab_size > 0:
            journal_dim = min(50, (journal_vocab_size + 1) // 2)
            self.journal_embedding = nn.Embedding(journal_vocab_size, journal_dim)
        input_size = embedding_size + journal_dim + (1 if use_year else 0)
        layers = [nn.Linear(input_size, mlp_hidden_size), nn.ReLU(), nn.Dropout(dropout_rate)]
        for _ in range(n_hidden_layers - 1):
            layers.extend([nn.Linear(mlp_hidden_size, mlp_hidden_size), nn.ReLU(),
                           nn.Dropout(dropout_rate)])
        layers.append(nn.Linear(mlp_hidden_size, output_size))
        self.classifier = nn.Sequential(*layers)

    def forward_from_embedding(self, sentence_embedding, categorical_tensor=None,
                               numeric_tensor=None):
        features = [sentence_embedding]
        if (self.journal_embedding is not None and categorical_tensor is not None
                and categorical_tensor.numel() > 0):
            features.append(self.journal_embedding(categorical_tensor[:, 0]))
        if self.use_year and numeric_tensor is not None and numeric_tensor.numel() > 0:
            features.append(numeric_tensor)
        return self.classifier(torch.cat(features, dim=1))


def _joined_text(frame):
    """Title and abstract concatenated into one text field per row."""
    return frame['title'].fillna('') + ' ' + frame['abstract'].fillna('')


def _journal_vocab(train_df):
    """Journal name to integer code, reserving 0 for unseen journals at predict time."""
    values = train_df['journal'].dropna().unique().tolist()
    return {value: index + 1 for index, value in enumerate(values)}


def _tabular_tensors(df, specs, journal_vocab, year_min):
    """Categorical and numeric tensors for the cached head, empty when specs omit them."""
    categorical = torch.tensor([], dtype=torch.long)
    numeric = torch.tensor([], dtype=torch.float)
    if specs['categorical_cols']:
        codes = df['journal'].map(journal_vocab).fillna(0).astype(int).values
        categorical = torch.tensor(codes, dtype=torch.long).unsqueeze(1)
    if specs['numeric_cols']:
        years = (df['year'].fillna(year_min) - year_min).values
        numeric = torch.tensor(years, dtype=torch.float).unsqueeze(1)
    return categorical, numeric


def embed_dataset(df, config=None, device=None, cache_dir='embeddings_cache', embed_fn=None):
    """One-time sentence embedding per dataset, cached by model name + data hash."""
    config = config or Config(random_seed=42, embedding_model='sentence-transformer', model_name='all-MiniLM-L6-v2')
    device = device or torch.device('cpu')
    texts = _joined_text(df).tolist()
    digest = hashlib.sha1('\x1f'.join(texts).encode('utf-8')).hexdigest()
    model_tag = str(config.model_name).replace('/', '_')
    cache_path = Path(cache_dir) / f'{model_tag}_{digest}.npz'
    if cache_path.exists():
        with np.load(cache_path) as data:
            embeddings = data['embeddings']
        return torch.tensor(embeddings, dtype=torch.float32)
    if embed_fn is None:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(config.model_name, device=str(device))
        embeddings = encoder.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    else:
        embeddings = np.asarray(embed_fn(texts))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, embeddings=embeddings.astype('float32'))
    return torch.tensor(embeddings, dtype=torch.float32)


def _metric_row(y_true, preds, probs):
    """Metric dict for one evaluation; probability metrics None when probs is None."""
    row = {
        'f1': f1_score(y_true, preds, zero_division=0),
        'precision': precision_score(y_true, preds, zero_division=0),
        'recall': recall_score(y_true, preds, zero_division=0),
    }
    if probs is None:
        row.update({'roc_auc': None, 'ece': None, 'brier': None, 'ndcg': None, 'wss95': None})
        return row
    roc_auc = roc_auc_score(y_true, probs) if len(np.unique(y_true)) >= 2 else None
    row.update({
        'roc_auc': roc_auc,
        'ece': calculate_ece(y_true, probs),
        'brier': calculate_brier(y_true, probs),
        'ndcg': calculate_ndcg(y_true, probs),
        'wss95': calculate_wss_at_recall(y_true, probs, target_recall=RECALL_TARGET)['wss'],
    })
    return row


def _fit_cached(embeddings, df, specs, train_idx, val_idx, test_idx, config, device, seed):
    """Train the cached head on one split; return test predictions and a subset predictor."""
    train_df = df.iloc[train_idx]
    journal_vocab = _journal_vocab(train_df) if specs['categorical_cols'] else {}
    year_min = float(train_df['year'].min()) if specs['numeric_cols'] else 0.0

    def subset(idx):
        frame = df.iloc[idx]
        categorical, numeric = _tabular_tensors(frame, specs, journal_vocab, year_min)
        labels = torch.tensor(frame['label_included'].values, dtype=torch.float).unsqueeze(1)
        return CachedEmbeddingDataset(embeddings[np.asarray(idx)], labels, categorical, numeric)

    train_dl = create_dataloader(subset(train_idx), RandomSampler, config.batch_size)
    val_dl = create_dataloader(subset(val_idx), SequentialSampler, config.eval_batch_size)
    test_dl = create_dataloader(subset(test_idx), SequentialSampler, config.eval_batch_size)

    torch.manual_seed(seed)
    head = _CachedHead(
        embedding_size=embeddings.shape[1],
        journal_vocab_size=len(journal_vocab) + 1 if journal_vocab else 0,
        use_year=bool(specs['numeric_cols']),
        mlp_hidden_size=config.mlp_hidden_size,
        n_hidden_layers=config.n_hidden_layers,
        dropout_rate=config.dropout_rate,
    ).to(device)
    optimizer = AdamW(head.parameters(), lr=CACHED_HEAD_LR, eps=1e-8)
    criterion = nn.BCEWithLogitsLoss()
    (_, _, _, _, _, _, state, _) = train_evaluate_model(
        head, train_dl, val_dl, test_dl, optimizer, criterion, device, config.epochs,
        early_stopping_patience=config.early_stopping_patience,
        gradient_clip_norm=config.gradient_clip_norm,
        use_warmup=False, pos_weight='auto', forward_fn=cached_forward_fn,
    )
    head.load_state_dict(state)
    calibration = calibrate_model(head, val_dl, device, forward_fn=cached_forward_fn)

    def predict_indices(idx):
        loader = create_dataloader(subset(idx), SequentialSampler, config.eval_batch_size)
        preds, probs = predict_model(head, loader, device, calibration=calibration,
                                     forward_fn=cached_forward_fn)
        return np.asarray(preds), np.asarray(probs)

    preds, probs = predict_indices(test_idx)
    return preds, probs, predict_indices


def _run_tfidf_nb(df, pool_idx, test_idx, seed, sample_pct=RANDOM_PCTS[-1]):
    """TF-IDF + naive-Bayes baseline trained on one fixed-size sample of the pool."""
    pool_df, test_df = df.iloc[pool_idx], df.iloc[test_idx]
    sample_idx, _ = safe_stratified_split(pool_df, pool_df['label_included'],
                                          test_size=1 - sample_pct, random_state=seed)
    train_df = pool_df.iloc[sample_idx]
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    model = MultinomialNB()
    model.fit(vectorizer.fit_transform(_joined_text(train_df)),
              train_df['label_included'].values)
    probs = model.predict_proba(vectorizer.transform(_joined_text(test_df)))[:, 1]
    preds = (probs >= 0.5).astype(int)
    y_true = test_df['label_included'].values
    return [{'strategy': 'tfidf_nb', 'n_coded': len(train_df),
             **_metric_row(y_true, preds, probs)}]


def _run_random(df, embeddings, specs, pool_idx, val_idx, test_idx, config, device, seed,
                fit, random_pcts=None):
    """Random-sampling baseline at each pct in random_pcts; skips pcts too small to split."""
    pool_idx = np.asarray(pool_idx)
    pool_df = df.iloc[pool_idx]
    y_true = df.iloc[test_idx]['label_included'].values
    rows = []
    for pct in (random_pcts or RANDOM_PCTS):
        n_sample = int(np.ceil(pct * len(pool_idx)))
        if n_sample < 2:
            continue
        sample_idx, _ = safe_stratified_split(pool_df, pool_df['label_included'],
                                              test_size=1 - pct, random_state=seed)
        train_idx = pool_idx[sample_idx]
        preds, probs, _ = fit(embeddings, df, specs, train_idx, val_idx, test_idx,
                              config, device, seed)
        rows.append({'strategy': 'random', 'n_coded': len(train_idx),
                     **_metric_row(y_true, preds, probs)})
    return rows


def _run_active_learning(df, embeddings, specs, pool_idx, val_idx, test_idx, config,
                         device, seed, fit):
    """Uncertainty-sampling active-learning loop, one row per iteration until it stops."""
    pool_idx = np.asarray(pool_idx)
    if len(pool_idx) <= AL_MIN_RECORDS:
        raise ValueError('pool too small for active learning')
    pool_df = df.iloc[pool_idx]
    y_true = df.iloc[test_idx]['label_included'].values
    val_y_true = df.iloc[val_idx]['label_included'].values
    initial_pct = max(AL_INITIAL_PCT, AL_MIN_RECORDS / len(pool_idx))
    seed_local, _ = safe_stratified_split(pool_df, pool_df['label_included'],
                                          test_size=1 - initial_pct, random_state=seed)
    coded = list(seed_local)
    uncoded = [i for i in range(len(pool_idx)) if i not in set(coded)]
    best_f1, no_improve = 0.0, 0
    rows = []
    for iteration in range(AL_MAX_ITERATIONS):
        train_idx = pool_idx[coded]
        preds, probs, predict_indices = fit(embeddings, df, specs, train_idx, val_idx,
                                            test_idx, config, device, seed + iteration)
        val_preds, _ = predict_indices(val_idx)
        val_f1 = f1_score(val_y_true, val_preds, zero_division=0)
        n_relevant = int(pool_df.iloc[coded]['label_included'].sum())
        stop_test = recall_target_test(n_screened=len(coded), n_relevant=n_relevant,
                                       N=len(pool_idx), target_recall=RECALL_TARGET)
        rows.append({'strategy': 'active_learning', 'n_coded': len(coded),
                     **_metric_row(y_true, preds, probs),
                     'recall_lb': stop_test['recall_lower_bound'],
                     'stop_stat': stop_test['stop']})
        if val_f1 > best_f1 + AL_MIN_IMPROVEMENT:
            best_f1, no_improve = val_f1, 0
        else:
            no_improve += 1
            if no_improve >= AL_PATIENCE:
                break
        if stop_test['stop'] or not uncoded:
            break
        _, uncoded_probs = predict_indices(pool_idx[uncoded])
        batch_size = min(config.al_batch_size, len(uncoded))
        query_local = select_query_batch(uncoded_probs, strategy='uncertainty',
                                         batch_size=batch_size)
        query = [uncoded[int(i)] for i in query_local]
        coded = coded + query
        uncoded = [i for i in uncoded if i not in set(query)]
    return rows


def _run_semantic(df, embeddings, description_embedding, pool_idx, test_idx):
    """Cosine-similarity-to-description baseline with a pool-tuned decision threshold."""
    embeddings = np.asarray(embeddings, dtype='float64')
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    scores = (embeddings / np.clip(norms, 1e-12, None)) @ (
        description_embedding / max(np.linalg.norm(description_embedding), 1e-12))
    pool_scores = scores[np.asarray(pool_idx)]
    pool_labels = df.iloc[pool_idx]['label_included'].values
    grid = np.quantile(pool_scores, np.linspace(0.05, 0.95, 40))
    threshold = grid[int(np.argmax([
        f1_score(pool_labels, (pool_scores >= t).astype(int), zero_division=0)
        for t in grid
    ]))]
    test_scores = scores[np.asarray(test_idx)]
    y_true = df.iloc[test_idx]['label_included'].values
    preds = (test_scores >= threshold).astype(int)
    span = scores.max() - scores.min()
    probs = ((test_scores - scores.min()) / span if span > 0
             else np.full(len(test_scores), 0.5))
    return [{'strategy': 'semantic', 'n_coded': 0, **_metric_row(y_true, preds, probs)}]


def _fit_finetune(embeddings, df, specs, train_idx, val_idx, test_idx, config, device, seed):
    """Fine-tune the full model on one split; return test predictions and a subset predictor."""
    from transformers import AutoTokenizer
    from .model import PubMLP
    from .preprocess import preprocess_dataset

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    numeric_transform = {'year': 'min'} if specs['numeric_cols'] else {}

    def build(idx, fitted=None):
        return preprocess_dataset(df.iloc[idx].reset_index(drop=True), tokenizer, device,
                                  specs, numeric_transform, max_length=config.max_length,
                                  fitted_transforms=fitted)

    train_ds, fitted = build(train_idx)
    val_ds, _ = build(val_idx, fitted)
    test_ds, _ = build(test_idx, fitted)
    train_dl = create_dataloader(train_ds, RandomSampler, config.batch_size)
    val_dl = create_dataloader(val_ds, SequentialSampler, config.eval_batch_size)
    test_dl = create_dataloader(test_ds, SequentialSampler, config.eval_batch_size)

    torch.manual_seed(seed)
    model = PubMLP(categorical_vocab_sizes=fitted.categorical_vocab_sizes,
                   numeric_cols_num=len(specs['numeric_cols']),
                   mlp_hidden_size=config.mlp_hidden_size, output_size=1,
                   dropout_rate=config.dropout_rate,
                   embedding_model=config.embedding_model, model_name=config.model_name,
                   n_hidden_layers=config.n_hidden_layers, pooling_strategy=config.pooling_strategy).to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, eps=1e-8)
    criterion = nn.BCEWithLogitsLoss()
    (_, _, _, _, _, _, state, _) = train_evaluate_model(
        model, train_dl, val_dl, test_dl, optimizer, criterion, device, config.epochs,
        early_stopping_patience=config.early_stopping_patience,
        gradient_clip_norm=config.gradient_clip_norm, use_warmup=False, pos_weight='auto',
    )
    model.load_state_dict(state)
    calibration = calibrate_model(model, val_dl, device)

    def predict_indices(idx):
        ds, _ = build(idx, fitted)
        loader = create_dataloader(ds, SequentialSampler, config.eval_batch_size)
        preds, probs = predict_model(model, loader, device, calibration=calibration)
        return np.asarray(preds), np.asarray(probs)

    preds, probs = predict_indices(test_idx)
    return preds, probs, predict_indices


RUN_COLUMNS = ['dataset', 'seed', 'strategy', 'fusion', 'n_coded', 'f1', 'precision',
               'recall', 'roc_auc', 'ece', 'brier', 'ndcg', 'wss95', 'recall_lb', 'stop_stat']


def run_simulation(df, strategies, seeds, config=None, engine='cached', embeddings=None,
                   description_embedding=None, device=None, dataset_name='dataset',
                   random_pcts=None):
    """Per-seed strategy simulation over one benchmark frame; one tidy row per evaluation."""
    config = config or Config(random_seed=42)
    device = device or torch.device('cpu')
    fits = {'cached': _fit_cached, 'finetune': _fit_finetune}
    if engine not in fits:
        raise ValueError(f"unknown engine: {engine}")
    fit = fits[engine]
    if engine == 'cached' and embeddings is None:
        raise ValueError("engine='cached' needs embeddings; call embed_dataset first "
                         "or pass embeddings=")
    if 'semantic' in strategies and embeddings is None:
        raise ValueError("strategy 'semantic' needs embeddings; call embed_dataset first "
                         "or pass embeddings=")
    if embeddings is not None:
        embeddings = torch.as_tensor(np.asarray(embeddings), dtype=torch.float32)
    known_strategies = {'random', 'active_learning', 'tfidf_nb', 'semantic'}
    for strategy in strategies:
        if strategy not in known_strategies:
            raise ValueError(f"unknown strategy: {strategy}")
    if 'semantic' in strategies and description_embedding is None:
        raise ValueError("strategy 'semantic' needs description_embedding")
    specs, _, fusion = build_column_specs(df)
    rows = []
    for seed in seeds:
        pool_val_idx, test_idx = safe_stratified_split(df, df['label_included'],
                                                       test_size=0.10, random_state=seed)
        pool_val_df = df.iloc[pool_val_idx]
        pool_local, val_local = safe_stratified_split(pool_val_df,
                                                      pool_val_df['label_included'],
                                                      test_size=0.11, random_state=seed)
        pool_idx = np.asarray(pool_val_idx)[pool_local]
        val_idx = np.asarray(pool_val_idx)[val_local]
        runners = {
            'random': lambda: _run_random(df, embeddings, specs, pool_idx, val_idx,
                                          test_idx, config, device, seed, fit, random_pcts),
            'active_learning': lambda: _run_active_learning(df, embeddings, specs,
                                                            pool_idx, val_idx, test_idx,
                                                            config, device, seed, fit),
            'tfidf_nb': lambda: _run_tfidf_nb(df, pool_idx, test_idx, seed),
            'semantic': lambda: _run_semantic(df, embeddings.cpu().numpy(),
                                              np.asarray(description_embedding).ravel(),
                                              pool_idx, test_idx),
        }
        for strategy in strategies:
            for row in runners[strategy]():
                row.update({'dataset': dataset_name, 'seed': seed, 'fusion': fusion})
                rows.append(row)
    return pd.DataFrame(rows).reindex(columns=RUN_COLUMNS)


def summarize_runs(runs_df):
    """Mean (SD) per dataset and strategy at final effort, plus WSS/recall/F1 curve frame."""
    final = (runs_df.sort_values('n_coded')
             .groupby(['dataset', 'seed', 'strategy'], dropna=False).tail(1))
    metric_cols = ['n_coded', 'f1', 'roc_auc', 'ece', 'brier', 'ndcg', 'wss95']
    summary = (final.groupby(['dataset', 'strategy'])[metric_cols]
               .agg(['mean', 'std']).round(3).reset_index())
    curves = (runs_df.dropna(subset=['n_coded'])
              .groupby(['dataset', 'strategy', 'n_coded'])[['f1', 'recall', 'wss95']]
              .agg(['mean', 'std']).reset_index())
    return {'summary': summary, 'curves': curves}


def main(argv=None):
    """Run benchmark simulations from the command line; return a process exit code."""
    parser = argparse.ArgumentParser(prog='pubmlp.benchmark',
                                     description='Screening simulations over benchmark datasets')
    parser.add_argument('--dataset', required=True, help='Synergy name or CSV/Excel path')
    parser.add_argument('--strategies', default='random,active_learning,tfidf_nb')
    parser.add_argument('--seeds', type=int, default=20)
    parser.add_argument('--out', default='runs.csv')
    parser.add_argument('--cache-dir', default='embeddings_cache')
    parser.add_argument('--engine', default='cached', choices=['cached', 'finetune'])
    parser.add_argument('--model', default=None,
                        help='model_name passed to Config (e.g. a HuggingFace checkpoint); '
                             'defaults to the embedding-model default')
    parser.add_argument('--embedding-model', default='sentence-transformer',
                        help="embedding_model passed to Config (e.g. 'bert', 'scibert', "
                             "'sentence-transformer'); default sentence-transformer")
    args = parser.parse_args(argv)

    from .utils import get_device, load_data
    strategies = args.strategies.split(',')
    path = Path(args.dataset)
    if path.exists():
        df = normalize_benchmark_frame(load_data(path))
        name = path.stem
    else:
        if (os.sep in args.dataset or '/' in args.dataset or
            args.dataset.endswith(('.csv', '.xlsx', '.xls'))):
            raise ValueError(f"dataset file not found: {args.dataset}")
        df = load_benchmark(args.dataset)
        name = args.dataset
    device = get_device()
    config = Config(random_seed=42, embedding_model=args.embedding_model, model_name=args.model)
    embeddings = None
    if args.engine == 'cached' or 'semantic' in strategies:
        embeddings = embed_dataset(df, config=config, device=device, cache_dir=args.cache_dir)
    runs = run_simulation(df, strategies=strategies,
                          seeds=range(1, args.seeds + 1), engine=args.engine, config=config,
                          embeddings=embeddings, device=device, dataset_name=name)
    runs.to_csv(args.out, index=False)
    result = summarize_runs(runs)
    print(result['summary'].to_string(index=False))
    print(f'Saved: {args.out} ({len(runs)} rows)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
