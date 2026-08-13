import logging

import numpy as np

from .confidence import interpret_confidence, needs_escalation, score_answer

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """Answer the question using only the passages below.

Question: {query}

Passages:
{passages}

Quote the sentence the answer rests on and give its passage number. If the
passages do not answer the question, reply exactly: not found."""


def chunk_text(text, chunk_size=1000, overlap=200):
    """Split text into overlapping character chunks, breaking at whitespace."""
    if chunk_size <= overlap:
        raise ValueError("chunk_size must exceed overlap")

    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            space = text.rfind(' ', start, end)
            # breaking at a space that sits inside the overlap would put the next
            # chunk behind this one, and the loop would never reach the end
            if space > start + overlap:
                end = space
        body = text[start:end].strip()
        if body:
            chunks.append({'index': len(chunks), 'start': start, 'text': body})
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def retrieve(query, chunks, embeddings, model, top_k=5):
    """Rank chunks against a query by cosine similarity.

    Retrieval runs against the embeddings in memory, so no vector database and no
    external service is involved.

    Returns:
        list: top chunks, each with a ``similarity`` value, highest first.
    """
    query_vector = np.asarray(model.encode([query]))
    matrix = np.asarray(embeddings)
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vector)
    norms[norms == 0] = 1e-12
    scores = (matrix @ query_vector.T).ravel() / norms

    order = np.argsort(-scores)[:top_k]
    return [{**chunks[i], 'similarity': float(scores[i])} for i in order]


def build_index(chunks, model, batch_size=32):
    """Embed chunks once for reuse across queries."""
    texts = [chunk['text'] for chunk in chunks]
    return np.asarray(model.encode(texts, batch_size=batch_size, show_progress_bar=False))


def extract_with_rag(text, queries, respond, model, top_k=5, chunk_size=1000,
                     overlap=200, template=None, tracker=None, threshold=0.7):
    """Answer questions about a document from retrieved passages.

    Retrieval is local; generation goes through ``respond``. Each answer carries
    its passages and a grounding score; answers below ``threshold`` are flagged.

    Args:
        text: Document text.
        queries: ``{name: query}`` or ``{name: {'query': str}}``.
        respond: Callable mapping a prompt to reply text.
        model: Sentence encoder with an ``encode`` method.
        top_k: Passages retrieved per query.
        chunk_size: Characters per chunk.
        overlap: Characters shared between neighbouring chunks.
        template: Prompt template overriding the default. Must accept ``query``
            and ``passages``.
        tracker: ProvenanceTracker to record the retrieval configuration and the
            prompts actually sent.
        threshold: Grounding score below which an answer is escalated.

    Returns:
        dict: ``{name: {'answer', 'passages', 'confidence', 'confidence_band',
        'escalate'}}``.
    """
    prompt_template = template or PROMPT_TEMPLATE
    chunks = chunk_text(text, chunk_size, overlap)
    embeddings = build_index(chunks, model)

    if tracker is not None:
        tracker.log_retrieval(
            embedding_model=getattr(model, 'name_or_path', str(type(model).__name__)),
            chunk_size=chunk_size, chunk_overlap=overlap, top_k=top_k)

    results = {}
    for name, spec in queries.items():
        query = spec['query'] if isinstance(spec, dict) else spec
        passages = retrieve(query, chunks, embeddings, model, top_k)
        numbered = '\n\n'.join(
            f"[{i + 1}] (p. chunk {p['index']}) {p['text']}" for i, p in enumerate(passages))
        prompt = prompt_template.format(query=query, passages=numbered)

        try:
            answer = respond(prompt)
        except Exception:
            logger.exception(f"Model call failed for {name}")
            answer = 'ERROR: model call failed'

        if tracker is not None:
            tracker.log_prompt(prompt)

        score = score_answer(answer, [p['text'] for p in passages], query)
        results[name] = {
            'answer': answer,
            'passages': passages,
            'confidence': round(score, 4),
            'confidence_band': interpret_confidence(score),
            'escalate': needs_escalation(score, threshold),
        }

    escalated = sum(1 for r in results.values() if r['escalate'])
    logger.info(f"{len(results)} queries answered, {escalated} flagged for review")
    return results
