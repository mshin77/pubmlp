import pytest

from pubmlp.config import Config
from pubmlp.audit import (
    AuditTrail, interpret_kappa, summarize_human_decisions,
    generate_coding_report, coding_report_fields,
)


class TestAuditTrail:
    def test_log_and_retrieve(self):
        trail = AuditTrail()
        trail.log_decision('r1', 1, 0.9)
        trail.log_decision('r2', 0, 0.2)
        df = trail.to_dataframe()
        assert len(df) == 2
        assert set(df.columns) >= {'record_id', 'model_prediction', 'model_probability'}

    def test_log_batch(self):
        trail = AuditTrail()
        trail.log_batch(['a', 'b', 'c'], [1, 0, 1], [0.8, 0.3, 0.7])
        assert len(trail.entries) == 3

    def test_disagreements(self):
        trail = AuditTrail()
        trail.log_decision('r1', 1, 0.9)
        trail.log_decision('r2', 0, 0.2)
        trail.update_human_label('r1', 0, reviewer_id='rev1', notes='wrong')
        trail.update_human_label('r2', 0)
        disagreements = trail.get_disagreements()
        assert len(disagreements) == 1
        assert disagreements[0].record_id == 'r1'

    def test_kappa_undefined_when_one_class_observed(self):
        trail = AuditTrail()
        for i in range(10):
            trail.log_decision(str(i), 1, 0.9)
            trail.update_human_label(str(i), 1)
        agreement = trail.calculate_agreement()
        assert agreement['total'] == 10
        assert agreement['agreed'] == 10
        assert agreement['kappa'] is None

    def test_kappa_computed_when_both_classes_observed(self):
        trail = AuditTrail()
        for i, (pred, human) in enumerate([(1, 1), (1, 0), (0, 0), (0, 1)]):
            trail.log_decision(str(i), pred, 0.5)
            trail.update_human_label(str(i), human)
        assert trail.calculate_agreement()['kappa'] == 0.0

    def test_model_rater_excluded_from_human_counts(self):
        trail = AuditTrail()
        trail.log_batch(['r1', 'r2'], [1, 1], [0.9, 0.8])
        trail.update_human_label('r1', 1, reviewer_id='coder_1')
        trail.update_human_label('r1', 1, reviewer_id='llm', rater_type='model')
        trail.update_human_label('r2', 0, reviewer_id='llm', rater_type='model')
        selection = generate_coding_report(trail, config=Config())['process']
        assert selection['records_human_reviewed'] == 1
        assert selection['reviewers'] == ['coder_1']
        assert selection['model_raters'] == ['llm']
        assert selection['independent_screening'] is False
        assert summarize_human_decisions(trail)['human_reviewed'] == 1

    def test_serialization_roundtrip(self):
        trail = AuditTrail()
        trail.log_decision('r1', 1, 0.8)
        trail.update_human_label('r1', 0)
        restored = AuditTrail.from_dict(trail.to_dict())
        assert len(restored.entries) == 1
        assert restored.entries[0].human_label == 0


class TestInterpretKappa:
    @pytest.mark.parametrize('kappa,expected', [
        (-0.1, 'poor'), (0.1, 'slight'), (0.3, 'fair'),
        (0.5, 'moderate'), (0.7, 'substantial'), (0.9, 'almost perfect'),
    ])
    def test_scale(self, kappa, expected):
        assert interpret_kappa(kappa) == expected


class TestSummarizeHumanDecisions:
    def test_counts(self):
        trail = AuditTrail()
        trail.log_batch(['a', 'b', 'c'], [1, 0, 1], [0.8, 0.2, 0.5])
        trail.update_human_label('a', 0)
        summary = summarize_human_decisions(trail)
        assert summary['total'] == 3
        assert summary['included'] == 2
        assert summary['excluded'] == 1
        assert summary['human_reviewed'] == 1
        assert summary['human_overrides'] == 1


class TestPrismaReport:
    def test_report_keys(self):
        trail = AuditTrail()
        trail.log_decision('r1', 1, 0.9)
        report = generate_coding_report(trail)
        assert set(report.keys()) == {'process', 'stage', 'human_oversight',
                                      'coded_volume', 'decision_counts', 'agreement'}

    def test_all_prisma_items_documented(self):
        assert set(coding_report_fields.keys()) == {'process', 'stage', 'human_oversight',
                                      'coded_volume', 'decision_counts', 'agreement'}


class TestDualScreening:
    def test_second_reviewer_appends(self):
        trail = AuditTrail()
        trail.log_batch(['r1'], [1], [0.9], phase='active')
        trail.update_human_label('r1', 1, reviewer_id='a')
        trail.update_human_label('r1', 0, reviewer_id='b')
        assert len(trail.entries) == 2
        assert {e.reviewer_id for e in trail.entries} == {'a', 'b'}

    def test_process_is_audit_derived(self):
        trail = AuditTrail()
        trail.log_batch(['r1', 'r2'], [1, 0], [0.9, 0.1], phase='active')
        trail.update_human_label('r1', 1, reviewer_id='a')
        trail.update_human_label('r1', 1, reviewer_id='b')
        selection = generate_coding_report(trail, config=Config())['process']
        assert selection['records_logged'] == 2
        assert selection['records_human_reviewed'] == 1
        assert selection['proportion_human_reviewed'] == 0.5
        assert selection['reviewers_per_record_max'] == 2
        assert selection['independent_screening'] is True
        assert selection['tool_version']
        assert selection['model_checkpoint']

    def test_reload_preserves_labels_and_timestamps(self, tmp_path):
        trail = AuditTrail()
        trail.log_batch(['r1', 'r2'], [1, 0], [0.9, 0.1])
        trail.update_human_label('r1', 0, reviewer_id='a')
        path = tmp_path / 'trail.csv'
        trail.export_csv(path)
        back = AuditTrail.from_csv(path)
        assert len(back.entries) == 2
        assert back.entries[0].human_label == 0
        assert back.entries[1].human_label is None
        assert back.entries[0].timestamp == trail.entries[0].timestamp

    def test_agreement_count_is_exact(self):
        trail = AuditTrail()
        trail.log_batch(['a', 'b', 'c'], [1, 1, 1], [0.9, 0.9, 0.9])
        for rid, label in (('a', 1), ('b', 1), ('c', 0)):
            trail.update_human_label(rid, label, reviewer_id='r')
        result = trail.calculate_agreement()
        assert result == {'total': 3, 'agreed': 2, 'disagreed': 1, 'kappa': result['kappa']}
