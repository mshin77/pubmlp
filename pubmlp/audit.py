from dataclasses import dataclass, field, asdict, fields as fields_of
from datetime import datetime, timezone

import pandas as pd

from .active_learning import compare_reviewers


@dataclass
class AuditEntry:
    record_id: str
    model_prediction: int = None
    model_probability: float = None
    human_label: int = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    reviewer_id: str = None
    phase: str = 'screening'
    notes: str = ''
    rater_type: str = 'human'


class AuditTrail:
    def __init__(self):
        self.entries = []

    def log_decision(self, record_id, prediction, probability, phase='screening',
                     reviewer_id=None):
        """Log what the model said about a record, before any human sees it.

        An unscored record carries None rather than a stand-in class: writing a
        placeholder here and a human decision later would make the two agree by
        construction, and the agreement figures are read as a validation of the
        model.
        """
        self.entries.append(AuditEntry(
            record_id=str(record_id),
            model_prediction=None if prediction is None else int(prediction),
            model_probability=None if probability is None else float(probability),
            phase=phase,
            reviewer_id=reviewer_id,
        ))

    def log_batch(self, record_ids, predictions, probabilities, phase='screening',
                  reviewer_id=None):
        for record_id, prediction, probability in zip(record_ids, predictions, probabilities):
            self.log_decision(record_id, prediction, probability, phase, reviewer_id)

    def update_human_label(self, record_id, human_label, reviewer_id=None, notes='',
                           rater_type='human'):
        """Attach a human decision to a logged record.

        One entry is kept per reviewer, so a second independent reviewer on the
        same record appends rather than overwrites and dual screening stays
        visible to ``generate_prisma_report``.
        """
        matched = [e for e in self.entries if e.record_id == str(record_id)]
        if not matched:
            raise KeyError(f"Record {record_id} not found in audit trail")

        mine = [e for e in matched if e.reviewer_id == reviewer_id]
        if not mine:
            mine = [e for e in matched if e.reviewer_id is None and e.human_label is None]
        if not mine:
            template = matched[0]
            entry = AuditEntry(
                record_id=template.record_id,
                model_prediction=template.model_prediction,
                model_probability=template.model_probability,
                phase=template.phase,
                reviewer_id=reviewer_id,
                rater_type=rater_type,
            )
            self.entries.append(entry)
            mine = [entry]

        for entry in mine:
            entry.human_label = int(human_label)
            entry.notes = notes or entry.notes
            entry.rater_type = rater_type
            if reviewer_id is not None:
                entry.reviewer_id = reviewer_id
        return len(mine)

    def clear_human_label(self, record_id, reviewer_id=None, rater_type='human'):
        """Withdraw a decision, leaving the record open for screening again.

        Only the matching rater's own entries are cleared, so withdrawing a human
        decision does not erase a second reviewer's or a model's.
        """
        cleared = 0
        for entry in self.entries:
            if entry.record_id != str(record_id) or entry.human_label is None:
                continue
            if entry.rater_type != rater_type:
                continue
            if reviewer_id is not None and entry.reviewer_id != reviewer_id:
                continue
            entry.human_label = None
            cleared += 1
        return cleared

    def decided_ids(self, rater_type='human'):
        """Records carrying a decision from this kind of rater."""
        return {e.record_id for e in self.entries if e.human_label is not None
                and (rater_type is None or e.rater_type == rater_type)}

    def get_disagreements(self):
        return [e for e in self.entries
                if e.human_label is not None and e.model_prediction is not None
                and e.model_prediction != e.human_label]

    def calculate_agreement(self, rater_type='human'):
        """Model against human, over records the model actually scored.

        An unscored record has nothing to agree with, so it is left out rather
        than counted as a match.
        """
        reviewed = [e for e in self.entries if e.human_label is not None
                    and e.model_prediction is not None
                    and (rater_type is None or e.rater_type == rater_type)]
        if not reviewed:
            return {'total': 0, 'agreed': 0, 'disagreed': 0, 'kappa': None}
        model_preds = [e.model_prediction for e in reviewed]
        human_labels = [e.human_label for e in reviewed]
        result = compare_reviewers(model_preds, human_labels)
        agreed = sum(1 for p, h in zip(model_preds, human_labels) if p == h)
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

    @classmethod
    def from_csv(cls, path):
        """Reload a trail exported by ``export_csv`` so it accumulates across runs."""
        frame = pd.read_csv(path)
        fields = {f.name for f in fields_of(AuditEntry)}
        records = [{k: v for k, v in row.items() if k in fields}
                   for row in frame.to_dict('records')]
        for row in records:
            row['record_id'] = str(row['record_id'])
            row['human_label'] = None if pd.isna(row.get('human_label')) else int(row['human_label'])
            row['model_prediction'] = (None if pd.isna(row.get('model_prediction'))
                                       else int(row['model_prediction']))
            row['model_probability'] = (None if pd.isna(row.get('model_probability'))
                                        else float(row['model_probability']))
            row['reviewer_id'] = None if pd.isna(row.get('reviewer_id')) else row['reviewer_id']
            row['notes'] = '' if pd.isna(row.get('notes')) else row['notes']
            if pd.isna(row.get('rater_type', None)):
                row['rater_type'] = 'human'
        obj = cls()
        obj.entries = [AuditEntry(**row) for row in records]
        return obj

    def record_ids(self):
        return {e.record_id for e in self.entries}


def interpret_kappa(kappa):
    if kappa is None:
        return None
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
    """Summarize human reviewer decisions against model predictions.

    A record the model never scored counts towards neither its included nor its
    excluded tally, and is not an override: there is no prediction to overturn.
    ``scored`` says how many of the entries the model actually saw, so a flow
    diagram cannot read a screening done without a model as a model result.
    """
    entries = audit_trail.entries
    scored = [e for e in entries if e.model_prediction is not None]
    reviewed = [e for e in entries if e.human_label is not None
                and e.rater_type == 'human']
    overrides = [e for e in reviewed if e.model_prediction is not None
                 and e.model_prediction != e.human_label]
    return {
        'total': len(entries),
        'scored': len(scored),
        'included': sum(1 for e in scored if e.model_prediction == 1),
        'excluded': sum(1 for e in scored if e.model_prediction == 0),
        'uncertain': sum(1 for e in entries if e.model_probability is not None
                         and uncertainty_low < e.model_probability < uncertainty_high),
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

    entries = audit_trail.entries
    reviewed = [e for e in entries
                if e.human_label is not None and e.rater_type == 'human']
    model_raters = sorted({e.reviewer_id for e in entries
                           if e.human_label is not None and e.rater_type != 'human'})
    reviewers_per_record = {}
    for entry in reviewed:
        reviewers_per_record.setdefault(entry.record_id, set()).add(entry.reviewer_id)
    reviewers = {r for ids in reviewers_per_record.values() for r in ids if r is not None}
    records = audit_trail.record_ids()

    from . import __version__

    report = {
        'item_8': {
            'description': prisma_screening_items['item_8'],
            'tool': 'pubmlp',
            'tool_version': __version__,
            'stage': 'title/abstract screening',
            'model_checkpoint': getattr(config, 'model_name', None) if config else None,
            'embedding_model': getattr(config, 'embedding_model', None) if config else None,
            'calibration': getattr(config, 'calibration_method', None) if config else None,
            'records_logged': len(records),
            'records_human_reviewed': len(reviewers_per_record),
            'proportion_human_reviewed': (round(len(reviewers_per_record) / len(records), 4)
                                          if records else None),
            'reviewers': sorted(reviewers),
            'reviewers_per_record_max': (max(len(ids) for ids in reviewers_per_record.values())
                                         if reviewers_per_record else 0),
            'independent_screening': (max((len(ids) for ids in reviewers_per_record.values()),
                                          default=0) > 1),
            'model_raters': model_raters,
            'phases': sorted({e.phase for e in entries}),
            'automation_role': 'ranking and pre-annotation; every logged record carries a human decision'
                               if reviewers_per_record and len(reviewers_per_record) == len(records)
                               else 'ranking and pre-annotation; human review incomplete',
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
            # without this a screen the model never scored is indistinguishable
            # from one where it excluded everything
            'records_model_scored': summary['scored'],
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
