import re

NEGATIVE_MARKERS = (
    ('not found', 0.40),
    ('not mentioned', 0.30),
    ('not specified', 0.30),
    ('unclear', 0.20),
    ('uncertain', 0.20),
)


def score_answer(answer, passages=(), query=''):
    """Score how well an extracted answer is grounded in its source passages.

    A heuristic for triage, not a probability. Not comparable to the calibrated
    probability from ``predict_model`` or the self-report from ``llm_screen``.

    Args:
        answer: The extracted answer text.
        passages: Source passages the answer should be grounded in.
        query: The extraction query, unused in scoring and kept for provenance.

    Returns:
        float: Grounding score from 0.0 to 1.0.
    """
    if not answer or not isinstance(answer, str):
        return 0.0
    if answer.startswith('ERROR'):
        return 0.0

    lowered = answer.lower()
    score = 0.5

    for marker, penalty in NEGATIVE_MARKERS:
        if marker in lowered:
            score -= penalty

    quotes = len(re.findall(r'"[^"]{10,}"', answer))
    score += min(0.2, quotes * 0.1)

    if re.search(r'passage\s+\d+', lowered):
        score += 0.1
    if re.search(r'(p\.|page)\s*\d+', lowered):
        score += 0.1
    if len(answer) > 50:
        score += 0.05
    if len(answer) > 150:
        score += 0.05
    if re.search(r'\b\d+\b', answer):
        score += 0.05

    if passages:
        answer_words = set(lowered.split())
        passage_words = set(' '.join(passages).lower().split())
        overlap = len(answer_words & passage_words)
        if overlap > 5:
            score += min(0.15, overlap * 0.01)

    return max(0.0, min(1.0, score))


def interpret_confidence(score):
    """Band label for a grounding score."""
    if score is None:
        return None
    if score >= 0.8:
        return 'high'
    if score >= 0.6:
        return 'moderate'
    if score >= 0.4:
        return 'low'
    return 'very low'


def needs_escalation(score, threshold=0.7):
    """Whether an extraction should go to a second reviewer.

    Returning True is a request for human attention, not a verdict on the answer.
    """
    return score is None or score < threshold


def score_extractions(extractions, passages_list=None, answer_key='answer',
                      threshold=0.7):
    """Score a list of extractions and flag the ones needing a second reviewer.

    Returns:
        list: each extraction with ``confidence``, ``confidence_band``, and
        ``escalate`` added.
    """
    passages_list = passages_list or [()] * len(extractions)
    scored = []
    for extraction, passages in zip(extractions, passages_list):
        score = score_answer(extraction.get(answer_key, ''), passages)
        scored.append({
            **extraction,
            'confidence': round(score, 4),
            'confidence_band': interpret_confidence(score),
            'escalate': needs_escalation(score, threshold),
        })
    return scored


def confidence_report(scored, threshold=0.7):
    """Counts by band and how many extractions need a second reviewer."""
    if not scored:
        return {'total': 0, 'mean_confidence': None, 'escalate': 0, 'bands': {}}
    scores = [item['confidence'] for item in scored]
    bands = {}
    for item in scored:
        bands[item['confidence_band']] = bands.get(item['confidence_band'], 0) + 1
    return {
        'total': len(scored),
        'mean_confidence': round(sum(scores) / len(scores), 4),
        'escalate': sum(1 for item in scored if item['escalate']),
        'threshold': threshold,
        'bands': bands,
    }
