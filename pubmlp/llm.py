import json
import logging
import re
from datetime import datetime, timezone

import pandas as pd

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """You are screening a record for a systematic review at the title and abstract stage.

Criteria:
{criteria_block}

Record:
{record_block}

Decide each criterion independently on the record text alone. Where the text is
insufficient, favour inclusion: a false inclusion costs one full-text read, a
false exclusion loses the study.

Reply with JSON only, no other text:
{{"decisions": {{{decision_keys}}}}}

Each decision is an object with "label" (1 or 0), "confidence" (0.0 to 1.0), and
"rationale" (one sentence quoting the deciding text)."""


def build_prompt(record, inclusion_patterns, fields):
    criteria_block = '\n'.join(
        f"- {name}: {spec.get('description', '')}" for name, spec in inclusion_patterns.items())
    record_block = '\n'.join(
        f"{field}: {record.get(field, '')}" for field in fields if record.get(field) is not None)
    decision_keys = ', '.join(f'"{name}": {{...}}' for name in inclusion_patterns)
    return PROMPT_TEMPLATE.format(criteria_block=criteria_block, record_block=record_block,
                                  decision_keys=decision_keys)


def parse_response(text, criteria):
    """Pull per-criterion decisions out of a model reply.

    Missing or unparseable decisions come back as None rather than a guess, so a
    failed call is distinguishable from a negative decision.
    """
    empty = {c: {'label': None, 'confidence': None, 'rationale': ''} for c in criteria}
    if not text:
        return empty

    match = re.search(r'\{.*\}', text, re.S)
    if not match:
        return empty
    try:
        payload = json.loads(match.group())
    except json.JSONDecodeError:
        logger.warning("Unparseable model reply")
        return empty

    decisions = payload.get('decisions', payload)
    parsed = {}
    for criterion in criteria:
        entry = decisions.get(criterion) or {}
        label = entry.get('label')
        parsed[criterion] = {
            'label': int(label) if label in (0, 1, '0', '1') else None,
            'confidence': entry.get('confidence'),
            'rationale': str(entry.get('rationale', ''))[:500],
        }
    return parsed


def llm_screen(df, inclusion_patterns, respond, fields=None, model_name='',
               parameters=None, reviewer_id='llm', audit_trail=None, phase='screening'):
    """Screen records with a language model as an additional independent screener.

    Columns carry an ``_llm`` suffix; a record still advances on the human
    decision. Self-reported confidence is not calibrated.

    Args:
        df: Records to screen.
        inclusion_patterns: ``{criterion: {'description': text}}``, the same
            structure ``regex_screen`` takes.
        respond: Callable mapping a prompt string to reply text.
        fields: Record columns shown to the model (default title, abstract).
        model_name: Provider and version, recorded for reporting.
        parameters: Decoding parameters, recorded for reporting.
        reviewer_id: Identity written to the audit trail.
        audit_trail: AuditTrail to log decisions into, when the records were
            already logged by the classifier.
        phase: Screening phase recorded on new audit entries.

    Returns:
        tuple: (DataFrame with ``{criterion}_llm``, ``{criterion}_llm_confidence``
        and ``{criterion}_llm_rationale`` columns plus ``llm_meets_all_criteria``,
        provenance dict for PRISMA reporting).
    """
    if fields is None:
        fields = ['title', 'abstract']

    criteria = list(inclusion_patterns)
    available = [f for f in fields if f in df.columns]
    missing = [f for f in fields if f not in df.columns]
    if missing:
        logger.warning(f"Missing fields: {missing}")

    frame = df.copy()
    results, failures = [], 0
    for _, record in frame.iterrows():
        prompt = build_prompt(record, inclusion_patterns, available)
        try:
            reply = respond(prompt)
        except Exception:
            logger.exception("Model call failed")
            reply = ''
        parsed = parse_response(reply, criteria)
        if all(parsed[c]['label'] is None for c in criteria):
            failures += 1
        results.append(parsed)

    for criterion in criteria:
        frame[f'{criterion}_llm'] = pd.array([r[criterion]['label'] for r in results],
                                             dtype='Int64')
        frame[f'{criterion}_llm_confidence'] = [r[criterion]['confidence'] for r in results]
        frame[f'{criterion}_llm_rationale'] = [r[criterion]['rationale'] for r in results]

    decided = frame[[f'{c}_llm' for c in criteria]]
    frame['llm_meets_all_criteria'] = pd.array(
        [None if row.isna().any() else int((row == 1).all()) for _, row in decided.iterrows()],
        dtype='Int64')

    provenance = {
        'role': 'additional independent screener; not a sole screener',
        'model': model_name,
        'parameters': parameters or {},
        'prompt_template': PROMPT_TEMPLATE,
        'criteria': {name: spec.get('description', '') for name, spec in inclusion_patterns.items()},
        'fields': available,
        'records': len(frame),
        'failed_calls': failures,
        'reviewer_id': reviewer_id,
        'run_at': datetime.now(timezone.utc).isoformat(),
    }

    if audit_trail is not None:
        logged = 0
        for record_id, decision in zip(frame.get('an', frame.index), frame['llm_meets_all_criteria']):
            if pd.isna(decision):
                continue
            try:
                logged += audit_trail.update_human_label(
                    record_id, int(decision), reviewer_id=reviewer_id,
                    rater_type='model')
            except KeyError:
                continue
        provenance['audit_entries'] = logged

    logger.info(f"Screened {len(frame)} records, {failures} failed calls")
    return frame, provenance
