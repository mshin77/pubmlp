"""
PubMLP: Screening, full-text review, and coding for systematic reviews.

Fuses transformer embeddings with tabular features through a multilayer
perceptron (MLP) for human-in-the-loop screening workflows.
"""

__version__ = "0.6.0"
__author__ = "Mikyung Shin"
__license__ = "MIT"

from .config import (
    Config,
    default_config,
    fast_config,
    robust_config,
    hitl_config,
    domain_configs,
    sentence_transformer_models,
)
from .model import PubMLP
from .train import (
    train_evaluate_model,
    calculate_loss,
    calculate_accuracy,
    calculate_pos_weight,
)
from .predict import (
    predict_model,
    get_predictions_and_labels,
    flag_uncertain,
)
from .metrics import (
    calculate_evaluation_metrics,
    calculate_wss_at_recall,
    calculate_ndcg,
    calculate_ece,
    calculate_brier,
)
from .preprocess import (
    preprocess_dataset,
    create_dataloader,
    split_data,
    CustomDataset,
    CachedEmbeddingDataset,
    collate_fn,
    FittedTransforms,
)
from .plotting import plot_results, plot_al_progress
from .utils import (
    get_device,
    auto_batch_size,
    load_data,
    unpack_batch,
    default_forward_fn,
    cached_forward_fn,
)
from .cv import cross_validate
from .calibration import (
    TemperatureScaling,
    collect_logits,
    calibrate_model,
)
from .embed import compute_cls_embeddings
from .audit import (
    AuditTrail,
    AuditEntry,
    interpret_kappa,
    summarize_human_decisions,
    generate_prisma_report,
)
from .active_learning import (
    ALState,
    select_query_batch,
    create_review_batch,
    compare_reviewers,
    merge_human_labels,
    simulate_al,
    rank_by_hybrid_max_uncertainty,
    rank_by_hybrid_max_random,
    safe_stratified_split,
)
from .stopping import (
    StoppingState,
    should_stop,
    expected_relevant,
    update_stopping_state,
    generate_stopping_report,
    calculate_wss,
    transition_phase,
    estimate_recall,
    recall_target_test,
)
from .llm import llm_screen, build_prompt, parse_response
from .fulltext import (read_pdf, detect_sections, detect_page_labels,
                       extract_fulltext_evidence, format_anchor)
from .evidence import (find_keyword_spans, search_document, highlight_markdown,
                       format_evidence)
from .confidence import (score_answer, interpret_confidence, needs_escalation,
                         score_extractions, confidence_report)
from .rag import chunk_text, build_index, retrieve, extract_with_rag
from .provenance import (ProvenanceTracker, load_provenance,
                         compare_provenances)
from .screening import (
    pattern_from_terms,
    regex_screen,
    extract_window_evidence,
    extract_sentence_evidence,
    extract_all_evidence,
    format_evidence_display,
    calculate_semantic_scores,
    score_full_text,
    compare_screening_configs,
    generate_descriptions,
    confirm_descriptions,
)
from .sample import (
    create_stratified_sample,
    save_sample_excel,
    apply_conditional_formatting,
    count_pattern_matches,
    highlight_pattern_matches,
)
from .datasets import (
    list_benchmarks,
    load_benchmark,
    load_manifest_corpus,
    normalize_benchmark_frame,
    build_column_specs,
)
from .benchmark import (
    embed_dataset,
    run_simulation,
    summarize_runs,
)

__all__ = [
    'Config', 'default_config', 'fast_config', 'robust_config', 'hitl_config', 'domain_configs', 'sentence_transformer_models',
    'PubMLP',
    'train_evaluate_model', 'calculate_loss', 'calculate_accuracy', 'calculate_pos_weight',
    'predict_model', 'get_predictions_and_labels', 'flag_uncertain',
    'calculate_evaluation_metrics', 'calculate_wss_at_recall', 'calculate_ndcg',
    'calculate_ece', 'calculate_brier',
    'preprocess_dataset', 'create_dataloader', 'split_data', 'CustomDataset', 'CachedEmbeddingDataset', 'collate_fn', 'FittedTransforms',
    'plot_results', 'plot_al_progress',
    'get_device', 'auto_batch_size', 'load_data', 'unpack_batch', 'default_forward_fn', 'cached_forward_fn',
    'compute_cls_embeddings',
    'cross_validate',
    'TemperatureScaling', 'collect_logits', 'calibrate_model',
    'AuditTrail', 'AuditEntry', 'interpret_kappa', 'summarize_human_decisions', 'generate_prisma_report',
    'ALState', 'select_query_batch', 'create_review_batch', 'compare_reviewers', 'merge_human_labels',
    'simulate_al', 'rank_by_hybrid_max_uncertainty', 'rank_by_hybrid_max_random', 'safe_stratified_split',
    'StoppingState', 'should_stop', 'expected_relevant',
    'llm_screen', 'build_prompt', 'parse_response',
    'find_keyword_spans', 'search_document', 'highlight_markdown', 'format_evidence',
    'score_answer', 'interpret_confidence', 'needs_escalation',
    'score_extractions', 'confidence_report',
    'chunk_text', 'build_index', 'retrieve', 'extract_with_rag',
    'ProvenanceTracker', 'load_provenance', 'compare_provenances',
    'read_pdf', 'detect_sections', 'detect_page_labels',
    'extract_fulltext_evidence', 'format_anchor', 'update_stopping_state', 'generate_stopping_report', 'calculate_wss',
    'transition_phase', 'estimate_recall', 'recall_target_test',
    'regex_screen', 'pattern_from_terms', 'extract_window_evidence', 'extract_sentence_evidence', 'extract_all_evidence',
    'format_evidence_display', 'calculate_semantic_scores',
    'create_stratified_sample', 'save_sample_excel', 'apply_conditional_formatting',
    'count_pattern_matches', 'highlight_pattern_matches',
    'score_full_text', 'compare_screening_configs',
    'generate_descriptions', 'confirm_descriptions',
    'list_benchmarks', 'load_benchmark', 'load_manifest_corpus',
    'normalize_benchmark_frame', 'build_column_specs',
    'embed_dataset', 'run_simulation', 'summarize_runs',
]
