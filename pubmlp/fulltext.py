import logging
import re

logger = logging.getLogger(__name__)

SECTION_NAMES = (
    'abstract', 'introduction', 'background', 'literature review', 'method',
    'methods', 'methodology', 'materials and methods', 'participants',
    'measures', 'procedure', 'analysis', 'results', 'findings', 'discussion',
    'limitations', 'conclusion', 'conclusions', 'references', 'appendix',
)

_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')

# a heading often shares its line with the text that follows it
_HEADING_START = re.compile(r'(?i)^(' + '|'.join(SECTION_NAMES) + r')[\s.:]*')

_PAGE_OF = re.compile(r'(?i)\bpage\s+(\d{1,4})\s+of\s+(\d{1,4})\b')
_BARE_NUMBER = re.compile(r'\d{1,4}')


def read_pdf(path, max_pages=None):
    """Read a born-digital PDF into per-page text.

    Requires the ``fulltext`` extra. Pages that yield no text are kept with an
    empty string so page numbers stay aligned with the document.
    """
    try:
        import pdfplumber
    except ImportError:
        raise ImportError('Full-text reading requires the fulltext extra: '
                          'pip install "pubmlp[fulltext]"')

    pages = []
    with pdfplumber.open(path) as document:
        for number, page in enumerate(document.pages, start=1):
            if max_pages and number > max_pages:
                break
            pages.append({'page': number, 'text': page.extract_text() or ''})

    empty = sum(1 for p in pages if not p['text'].strip())
    if empty == len(pages):
        logger.warning(f"No extractable text in {path}; the file is likely scanned")
    elif empty:
        logger.info(f"{empty} of {len(pages)} pages had no extractable text")
    return pages


def detect_page_labels(pages):
    """Attach the page number printed in the document to each page.

    Reads ``Page N of M`` running heads and bare footer integers. A label is
    accepted only when the printed-to-file offset holds on every labelled page.

    Returns:
        list: pages with ``printed_page`` set to the printed number, or to None
        throughout when no consistent numbering was found.
    """
    candidates = {}
    for index, page in enumerate(pages):
        lines = [l.strip() for l in page['text'].splitlines() if l.strip()]
        if not lines:
            continue
        for line in lines[:2] + lines[-2:]:
            explicit = _PAGE_OF.search(line)
            if explicit:
                candidates[index] = int(explicit.group(1))
                break
            if _BARE_NUMBER.fullmatch(line):
                candidates[index] = int(line)
                break

    offsets = {value - index for index, value in candidates.items()}
    if len(offsets) != 1:
        if candidates:
            logger.warning("Printed page numbers were inconsistent; using file positions")
        return [{**page, 'printed_page': None} for page in pages]

    offset = offsets.pop()
    return [{**page, 'printed_page': index + offset} for index, page in enumerate(pages)]


def detect_sections(pages):
    """Label each page with the most recent section heading seen.

    A heading style the vocabulary misses leaves the previous label in place.
    """
    labelled, current = [], ''
    for page in pages:
        for line in page['text'].splitlines():
            stripped = re.sub(r'^[\d.\s]+', '', line.strip())
            if not stripped:
                continue
            match = _HEADING_START.match(stripped)
            if match:
                current = match.group(1).title()
        labelled.append({**page, 'section': current})
    return labelled


def extract_fulltext_evidence(path, inclusion_patterns, model=None, max_pages=None,
                              window_words=0):
    """Locate criterion evidence in a PDF, anchored to page and section.

    A criterion with no matching span returns an empty list, not a verdict.

    Args:
        path: PDF file path.
        inclusion_patterns: ``{criterion: {'pattern': regex, 'description': text}}``,
            the same structure ``regex_screen`` takes.
        model: Sentence encoder for similarity against the criterion description.
            Omitted, spans come back without similarity scores.
        max_pages: Stop after this many pages.
        window_words: Words of context either side of the match. Zero returns the
            whole sentence.

    Returns:
        dict: ``{criterion: [{'page', 'section', 'text', 'matched_term',
        'similarity'}]}``.
    """
    pages = detect_sections(detect_page_labels(read_pdf(path, max_pages=max_pages)))
    evidence = {name: [] for name in inclusion_patterns}

    for criterion, spec in inclusion_patterns.items():
        pattern = re.compile(spec['pattern'], re.IGNORECASE)
        for page in pages:
            for sentence in _SENTENCE_SPLIT.split(page['text'].replace('\n', ' ')):
                match = pattern.search(sentence)
                if not match:
                    continue
                text = sentence.strip()
                if window_words:
                    words = text.split()
                    hit = len(text[:match.start()].split())
                    lo = max(0, hit - window_words)
                    text = ' '.join(words[lo:hit + window_words + 1])
                evidence[criterion].append({
                    'page': page.get('printed_page') or page['page'],
                    'pdf_page': page['page'],
                    'printed_page': page.get('printed_page'),
                    'section': page['section'],
                    'text': text,
                    'matched_term': match.group(),
                    'similarity': None,
                })

    if model is not None:
        from .screening import calculate_semantic_scores
        for criterion, spans in evidence.items():
            if not spans:
                continue
            scores = calculate_semantic_scores(
                spans, inclusion_patterns[criterion].get('description', ''), model)
            for span, score in zip(spans, scores.get('individual_scores') or []):
                span['similarity'] = score

    found = {c: len(v) for c, v in evidence.items()}
    logger.info(f"{path}: spans per criterion {found}")
    return evidence


def format_anchor(span):
    """Citation anchor for a span, as the reviewer sees it.

    Falls back to the file position, marked as such, so a reader is never given
    a page number the document does not carry.
    """
    section = f" · {span['section']}" if span.get('section') else ''
    if span.get('printed_page'):
        return f"p. {span['printed_page']}{section}"
    return f"file p. {span['pdf_page']}{section}"
