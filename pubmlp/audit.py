from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

import pandas as pd

from .active_learning import compare_reviewers


@dataclass
class AuditEntry:
    record_id: str
    model_prediction: int
    model_probability: float
    human_label: int = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    reviewer_id: str = None
    phase: str = 'screening'
    notes: str = ''


class AuditTrail:
    def __init__(self):
        self.entries = []

    def log_decision(self, record_id, prediction, probability, phase='screening',
                     reviewer_id=None):
        self.entries.append(AuditEntry(
            record_id=str(record_id),
            model_prediction=int(prediction),
            model_probability=float(probability),
            phase=phase,
            reviewer_id=reviewer_id,
        ))

    def log_batch(self, record_ids, predictions, probabilities, phase='screening',
                  reviewer_id=None):
        for rid, pred, prob in zip(record_ids, predictions, probabilities):
            self.log_decision(rid, pred, prob, phase, reviewer_id)

    def update_human_label(self, record_id, human_label, reviewer_id=None, notes=''):
        for entry in self.entries:
            if entry.record_id == str(record_id):
                entry.human_label = int(human_label)
                entry.reviewer_id = reviewer_id
                entry.notes = notes
                entry.timestamp = datetime.now(timezone.utc).isoformat()
                return
        raise KeyError(f"Record {record_id} not found in audit trail")

    def get_disagreements(self):
        return [e for e in self.entries
                if e.human_label is not None and e.model_prediction != e.human_label]

    def calculate_agreement(self):
        reviewed = [e for e in self.entries if e.human_label is not None]
        if not reviewed:
            return {'total': 0, 'agreed': 0, 'disagreed': 0, 'kappa': None}
        model_preds = [e.model_prediction for e in reviewed]
        human_labels = [e.human_label for e in reviewed]
        result = compare_reviewers(model_preds, human_labels)
        agreed = int(result['agreement_rate'] * len(reviewed))
        return {
            'total': len(reviewed),
            'agreed': agreed,
            'disagreed': len(reviewed) - agreed,
            'kappa': result['kappa'],
        }

    def to_dataframe(self):
        return pd.DataFrame([asdict(e) for e in self.entries])

    def export_csv(self, path):
        self.to_dataframe().to_csv(path, index=False)

    def to_dict(self):
        return {'entries': [asdict(e) for e in self.entries]}

    @classmethod
    def from_dict(cls, d):
        obj = cls()
        obj.entries = [AuditEntry(**e) for e in d['entries']]
        return obj


def interpret_kappa(kappa):
    if kappa < 0:
        return 'poor'
    if kappa <= 0.20:
        return 'slight'
    if kappa <= 0.40:
        return 'fair'
    if kappa <= 0.60:
        return 'moderate'
    if kappa <= 0.80:
        return 'substantial'
    return 'almost perfect'


def summarize_human_decisions(audit_trail, uncertainty_low=0.3, uncertainty_high=0.7):
    """Summarize human reviewer decisions against model predictions."""
    entries = audit_trail.entries
    reviewed = [e for e in entries if e.human_label is not None]
    overrides = [e for e in reviewed if e.model_prediction != e.human_label]
    return {
        'total': len(entries),
        'included': sum(1 for e in entries if e.model_prediction == 1),
        'excluded': sum(1 for e in entries if e.model_prediction == 0),
        'uncertain': sum(1 for e in entries if uncertainty_low < e.model_probability < uncertainty_high),
        'human_reviewed': len(reviewed),
        'human_overrides': len(overrides),
    }


# PRISMA 2020 Item 8 + trAIce M3/M8/M9/R1/R2 (screening-scoped)
prisma_screening_items = {
    'item_8': 'Selection process: automation tools used',
    'M3': 'Purpose/Stage: AI applied at title/abstract screening',
    'M8': 'Human-AI Interaction: human reviewer validation process',
    'M9': 'Performance Evaluation: screening model metrics',
    'R1': 'Study Selection: AI vs human exclusion counts in flow',
    'R2': 'Performance Metrics: AI screening performance results',
}


def generate_prisma_report(audit_trail, config=None):
    """Populate PRISMA Item 8 + screening-relevant trAIce items from audit data."""
    uncertainty_low = getattr(config, 'uncertainty_low', 0.3) if config else 0.3
    uncertainty_high = getattr(config, 'uncertainty_high', 0.7) if config else 0.7
    summary = summarize_human_decisions(audit_trail, uncertainty_low, uncertainty_high)
    agreement = audit_trail.calculate_agreement()

    report = {
        'item_8': {
            'description': prisma_screening_items['item_8'],
            'tool': 'pubmlp',
            'stage': 'title/abstract screening',
            'model': getattr(config, 'embedding_model', None) if config else None,
            'calibration': getattr(config, 'calibration_method', None) if config else None,
        },
        'M3': {
            'description': prisma_screening_items['M3'],
            'stage': 'title/abstract screening',
            'strategy': getattr(config, 'al_query_strategy', None) if config else None,
        },
        'M8': {
            'description': prisma_screening_items['M8'],
            'human_reviewed': summary['human_reviewed'],
            'human_overrides': summary['human_overrides'],
            'agreement_kappa': agreement['kappa'],
            'kappa_interpretation': interpret_kappa(agreement['kappa']) if agreement['kappa'] is not None else None,
        },
        'M9': {
            'description': prisma_screening_items['M9'],
            'total_screened': summary['total'],
            'uncertain_flagged': summary['uncertain'],
        },
        'R1': {
            'description': prisma_screening_items['R1'],
            'model_included': summary['included'],
            'model_excluded': summary['excluded'],
            'human_overrides': summary['human_overrides'],
        },
        'R2': {
            'description': prisma_screening_items['R2'],
            'agreement': agreement,
        },
    }
    return report
