import pytest

from pubmlp.audit import (
    AuditTrail, interpret_kappa, summarize_human_decisions,
    generate_prisma_report, prisma_screening_items,
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

    def test_kappa_calculation(self):
        trail = AuditTrail()
        for i in range(10):
            trail.log_decision(str(i), 1, 0.9)
            trail.update_human_label(str(i), 1)
        agreement = trail.calculate_agreement()
        assert agreement['total'] == 10
        assert agreement['agreed'] == 10
        assert agreement['kappa'] == 1.0

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
        report = generate_prisma_report(trail)
        assert set(report.keys()) == {'item_8', 'M3', 'M8', 'M9', 'R1', 'R2'}

    def test_all_prisma_items_documented(self):
        assert set(prisma_screening_items.keys()) == {'item_8', 'M3', 'M8', 'M9', 'R1', 'R2'}
