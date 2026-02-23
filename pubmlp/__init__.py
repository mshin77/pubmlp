"""
PubMLP: Multimodal publication classifier with LLM and deep learning.

Fuses transformer embeddings with tabular features through a multilayer
perceptron (MLP) for human-in-the-loop screening workflows.
"""

__version__ = "0.1.0"
__author__ = "Mikyung Shin"
__license__ = "MIT"

from .config import Config, default_config, fast_config, robust_config, hitl_config, domain_configs, sentence_transformer_models
from .model import PubMLP
from .train import train_evaluate_model, calculate_loss, calculate_accuracy, calculate_pos_weight
from .predict import predict_model, get_predictions_and_labels, flag_uncertain
from .metrics import calculate_evaluation_metrics
from .preprocess import preprocess_dataset, create_dataloader, split_data, CustomDataset, collate_fn, FittedTransforms
from .plotting import plot_results
from .utils import get_device, auto_batch_size, load_data, unpack_batch
from .cv import cross_validate
from .calibration import TemperatureScaling, collect_logits, calibrate_model
from .audit import AuditTrail, AuditEntry, interpret_kappa, summarize_human_decisions, generate_prisma_report
from .active_learning import ALState, select_query_batch, create_review_batch, compare_reviewers, merge_human_labels
from .stopping import StoppingState, should_stop, update_stopping_state, generate_stopping_report, calculate_wss, transition_phase, estimate_recall
from .screening import regex_screen, extract_window_evidence, extract_sentence_evidence, extract_all_evidence, format_evidence_display, calculate_semantic_scores
from .sample import create_stratified_sample, save_sample_excel, apply_conditional_formatting, count_pattern_matches, highlight_pattern_matches

__all__ = [
    'Config', 'default_config', 'fast_config', 'robust_config', 'hitl_config', 'domain_configs', 'sentence_transformer_models',
    'PubMLP',
    'train_evaluate_model', 'calculate_loss', 'calculate_accuracy', 'calculate_pos_weight',
    'predict_model', 'get_predictions_and_labels', 'flag_uncertain',
    'calculate_evaluation_metrics',
    'preprocess_dataset', 'create_dataloader', 'split_data', 'CustomDataset', 'collate_fn', 'FittedTransforms',
    'plot_results',
    'get_device', 'auto_batch_size', 'load_data', 'unpack_batch',
    'cross_validate',
    'TemperatureScaling', 'collect_logits', 'calibrate_model',
    'AuditTrail', 'AuditEntry', 'interpret_kappa', 'summarize_human_decisions', 'generate_prisma_report',
    'ALState', 'select_query_batch', 'create_review_batch', 'compare_reviewers', 'merge_human_labels',
    'StoppingState', 'should_stop', 'update_stopping_state', 'generate_stopping_report', 'calculate_wss',
    'transition_phase', 'estimate_recall',
    'regex_screen', 'extract_window_evidence', 'extract_sentence_evidence', 'extract_all_evidence',
    'format_evidence_display', 'calculate_semantic_scores',
    'create_stratified_sample', 'save_sample_excel', 'apply_conditional_formatting',
    'count_pattern_matches', 'highlight_pattern_matches',
]
