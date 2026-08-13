"""Read a search export into the canonical record columns.

Databases disagree on column names for the same field, so the reviewer's file is
mapped onto one vocabulary rather than the pipeline learning each export format.
An unrecognised column is kept as it stands: a corpus is the reviewer's, and
dropping a column silently is worse than carrying one nobody reads.
"""

import io
import re

CANONICAL = {
    'an': ('an', 'accession number', 'accession', 'record_id', 'id', 'ut', 'uid'),
    'title': ('title', 'ti', 'document title', 'article title'),
    'abstract': ('abstract', 'ab', 'summary'),
    'author': ('author', 'authors', 'au', 'contributors', 'author full names'),
    'journal': ('journal', 'source', 'so', 'source title', 'publication', 'venue',
                'journal title'),
    'year': ('year', 'py', 'publication year', 'date', 'publication date'),
    'doi': ('doi', 'di'),
    'keywords': ('keywords', 'de', 'author keywords', 'keywords plus'),
}

RIS_TAGS = {
    'TI': 'title', 'T1': 'title', 'AB': 'abstract', 'N2': 'abstract',
    'AU': 'author', 'A1': 'author', 'JO': 'journal', 'JF': 'journal', 'T2': 'journal',
    'PY': 'year', 'Y1': 'year', 'DO': 'doi', 'KW': 'keywords', 'AN': 'an', 'ID': 'an',
}

# Web of Science plaintext and EndNote .ciw share the ISI field codes
ISI_TAGS = {
    'TI': 'title', 'AB': 'abstract', 'AU': 'author', 'AF': 'author',
    'SO': 'journal', 'JI': 'journal', 'PY': 'year', 'DI': 'doi',
    'DE': 'keywords', 'ID': 'keywords', 'UT': 'an',
}

BIBTEX_FIELDS = {
    'title': 'title', 'abstract': 'abstract', 'author': 'author',
    'journal': 'journal', 'journaltitle': 'journal', 'booktitle': 'journal',
    'year': 'year', 'date': 'year', 'doi': 'doi', 'keywords': 'keywords',
}


def _canonical_name(column):
    tidy = re.sub(r'\s+', ' ', str(column)).strip().lower()
    for canonical, aliases in CANONICAL.items():
        if tidy in aliases:
            return canonical
    return column


def _read_ris(text):
    """Parse RIS, joining repeated tags rather than keeping only the last one."""
    records, current = [], {}
    for line in text.splitlines():
        match = re.match(r'^([A-Z][A-Z0-9])\s+-\s?(.*)$', line)
        if not match:
            continue
        tag, value = match.group(1), match.group(2).strip()
        if tag == 'ER':
            if current:
                records.append(current)
            current = {}
            continue
        field = RIS_TAGS.get(tag)
        if not field or not value:
            continue
        current[field] = f'{current[field]}; {value}' if field in current else value
    if current:
        records.append(current)
    return records


def _is_record_end(line):
    """Whether a line is the end-of-record tag rather than text beginning with it."""
    return line[:2] == 'ER' and not line[2:].strip(' -	')


def _read_tagged(text, tags, separator='  - '):
    """Parse a line-tagged export, joining repeated tags and wrapped lines."""
    records, current, last = [], {}, None
    for line in text.splitlines():
        if _is_record_end(line.strip()):
            if current:
                records.append(current)
            current, last = {}, None
            continue
        # a tag with nothing after it still names the field, so an empty remainder
        # counts as a tag line rather than as a continuation of the field before
        if line[:2].strip() and line[:2].isalnum() and line[2:3] in (' ', '	', ''):
            tag, value = line[:2], line[2:].lstrip(' -	').strip()
            field = tags.get(tag)
            last = field
            if not field or not value:
                continue
            current[field] = f'{current[field]}; {value}' if field in current else value
        elif last and line.strip():
            # a tag line with no text of its own still names the field its
            # continuation lines belong to, so the field is filled from here
            joiner = '; ' if last in ('author', 'keywords') else ' '
            held = current.get(last, '')
            current[last] = f'{held}{joiner}{line.strip()}' if held else line.strip()
    if current:
        records.append(current)
    return records


def _read_bibtex(text):
    """Parse BibTeX entries, taking only the fields a screening corpus needs."""
    records = []
    for entry in re.finditer(r'@\w+\s*\{([^,]*),(.*?)\}\s*(?=@|\Z)', text, re.S):
        key, body = entry.group(1).strip(), entry.group(2)
        record = {'an': key} if key else {}
        for field in re.finditer(r'(\w+)\s*=\s*[{"]([^{}"]*)["}]', body, re.S):
            name = BIBTEX_FIELDS.get(field.group(1).strip().lower())
            if not name:
                continue
            value = re.sub(r'\s+', ' ', field.group(2)).strip(' {}')
            if value:
                record[name] = value
        if record.get('title'):
            records.append(record)
    return records


def _sniff(text):
    """Tell RIS from ISI-tagged text, since both arrive as .txt."""
    if re.search(r'^TY\s+-\s', text, re.M):
        return 'ris'
    if re.search(r'^(FN|VR|PT)\s', text, re.M):
        return 'isi'
    return 'ris'


READABLE = ('.csv', '.xlsx', '.xls', '.ris', '.nbib', '.txt', '.bib', '.ciw')


def find_exports(folder, recursive=True):
    """Search exports in a folder, newest first."""
    from pathlib import Path as _Path

    root = _Path(folder).expanduser()
    if not root.is_dir():
        return []
    found = (root.rglob('*') if recursive else root.glob('*'))
    files = [item for item in found
             if item.is_file() and item.suffix.lower() in READABLE]
    return sorted(files, key=lambda item: item.stat().st_mtime, reverse=True)


def read_records(filename, data):
    """Read an export into a DataFrame with canonical column names.

    Args:
        filename: Original name, used only to pick the reader.
        data: File contents as bytes.

    Returns:
        DataFrame with an ``an`` column guaranteed, so every record can be
        referred to even when the export carried no identifier.
    """
    import pandas as pd

    suffix = filename.lower().rsplit('.', 1)[-1] if '.' in filename else ''
    if suffix in ('xlsx', 'xls'):
        frame = pd.read_excel(io.BytesIO(data))
    elif suffix == 'bib':
        frame = pd.DataFrame(_read_bibtex(data.decode('utf-8', errors='replace')))
    elif suffix == 'ciw':
        frame = pd.DataFrame(_read_tagged(data.decode('utf-8', errors='replace'), ISI_TAGS))
    elif suffix in ('ris', 'nbib'):
        frame = pd.DataFrame(_read_ris(data.decode('utf-8', errors='replace')))
    elif suffix == 'txt':
        text = data.decode('utf-8', errors='replace')
        frame = pd.DataFrame(_read_ris(text) if _sniff(text) == 'ris'
                             else _read_tagged(text, ISI_TAGS))
    else:
        frame = pd.read_csv(io.BytesIO(data), encoding_errors='replace')

    if frame.empty:
        raise ValueError('no records found in this file')

    frame = frame.rename(columns={c: _canonical_name(c) for c in frame.columns})
    frame = frame.loc[:, ~frame.columns.duplicated()]
    if 'title' not in frame.columns:
        raise ValueError('no title column found; a screening corpus needs titles')
    if 'an' not in frame.columns:
        frame.insert(0, 'an', [f'record-{i + 1}' for i in range(len(frame))])
    frame['an'] = _unique_ids(frame['an'])
    return frame


BLANK = ('', 'nan', 'none', 'null', 'nat', '<na>', '-')


def _unique_ids(column):
    """One identifier per record, since a decision is applied by identifier.

    A blank takes a generated identifier and a repeat takes a suffix, so two
    records can never share a name and close together.
    """
    import pandas as pd

    ids = column.astype(str).str.strip()
    ids = ids.mask(ids.str.lower().isin(BLANK),
                   pd.Series([f'record-{i + 1}' for i in range(len(ids))], index=ids.index))
    repeat = ids.groupby(ids).cumcount()
    return ids.mask(repeat > 0, ids + '-' + (repeat + 1).astype(str))
