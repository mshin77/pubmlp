import pandas as pd
import pytest

from pubmlp.audit import AuditTrail
from pubmlp.llm import build_prompt, parse_response, llm_screen

PATTERNS = {
    'dyslexia': {'description': 'The record addresses dyslexia.'},
    'higher_education': {'description': 'The record involves a postsecondary context.'},
}


def frame():
    return pd.DataFrame([
        {'an': 'r1', 'title': 'Dyslexia at university', 'abstract': 'College students.'},
        {'an': 'r2', 'title': 'Reading in primary school', 'abstract': 'Children.'},
    ])


def responder(text):
    return lambda prompt: text


class TestPrompt:
    def test_carries_criterion_descriptions(self):
        prompt = build_prompt(frame().iloc[0], PATTERNS, ['title', 'abstract'])
        assert 'postsecondary' in prompt
        assert 'dyslexia' in prompt
        assert 'JSON' in prompt

    def test_omits_absent_fields(self):
        prompt = build_prompt(frame().iloc[0], PATTERNS, ['title'])
        assert 'abstract:' not in prompt


class TestParse:
    def test_plain_json(self):
        reply = '{"decisions": {"dyslexia": {"label": 1, "confidence": 0.8, "rationale": "x"}}}'
        parsed = parse_response(reply, ['dyslexia'])
        assert parsed['dyslexia'] == {'label': 1, 'confidence': 0.8, 'rationale': 'x'}

    def test_fenced_json(self):
        reply = '```json\n{"decisions": {"dyslexia": {"label": 0}}}\n```'
        assert parse_response(reply, ['dyslexia'])['dyslexia']['label'] == 0

    def test_unparseable_gives_none_not_zero(self):
        for reply in ('', 'I cannot help', '{broken'):
            parsed = parse_response(reply, ['dyslexia'])
            assert parsed['dyslexia']['label'] is None

    def test_missing_criterion_gives_none(self):
        reply = '{"decisions": {"dyslexia": {"label": 1}}}'
        parsed = parse_response(reply, ['dyslexia', 'higher_education'])
        assert parsed['higher_education']['label'] is None


class TestScreen:
    def test_columns_are_llm_suffixed(self):
        out, _ = llm_screen(frame(), PATTERNS,
                            responder('{"decisions": {"dyslexia": {"label": 1}, "higher_education": {"label": 1}}}'))
        assert 'dyslexia' not in out.columns or out['dyslexia'].isna().all()
        for criterion in PATTERNS:
            assert f'{criterion}_llm' in out.columns
            assert f'{criterion}_llm_confidence' in out.columns
            assert f'{criterion}_llm_rationale' in out.columns
        assert list(out['llm_meets_all_criteria']) == [1, 1]

    def test_failed_call_leaves_na_not_exclude(self):
        out, provenance = llm_screen(frame(), PATTERNS, responder('no'))
        assert out['llm_meets_all_criteria'].isna().all()
        assert provenance['failed_calls'] == 2

    def test_provider_exception_is_contained(self):
        def raising(prompt):
            raise RuntimeError('provider down')

        out, provenance = llm_screen(frame(), PATTERNS, raising)
        assert provenance['failed_calls'] == 2
        assert out['llm_meets_all_criteria'].isna().all()

    def test_provenance_records_reporting_fields(self):
        _, provenance = llm_screen(frame(), PATTERNS, responder('no'),
                                   model_name='stub-1', parameters={'temperature': 0})
        assert provenance['model'] == 'stub-1'
        assert provenance['parameters'] == {'temperature': 0}
        assert provenance['prompt_template']
        assert provenance['criteria']['dyslexia']
        assert 'sole screener' in provenance['role']
        assert provenance['run_at']

    def test_logs_as_separate_reviewer(self):
        trail = AuditTrail()
        trail.log_batch(['r1', 'r2'], [1, 1], [0.9, 0.6])
        trail.update_human_label('r1', 1, reviewer_id='coder_1')
        reply = '{"decisions": {"dyslexia": {"label": 1}, "higher_education": {"label": 1}}}'
        _, provenance = llm_screen(frame(), PATTERNS, responder(reply),
                                   reviewer_id='llm_screener', audit_trail=trail)
        assert provenance['audit_entries'] == 2
        reviewers = {e.reviewer_id for e in trail.entries if e.human_label is not None}
        assert reviewers == {'coder_1', 'llm_screener'}
