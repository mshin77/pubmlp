import pytest

from pubmlp.fulltext import (detect_page_labels, detect_sections, format_anchor,
                             SECTION_NAMES)


def page(number, text):
    return {'page': number, 'text': text}


class TestPageLabels:
    def test_reads_page_of_running_head(self):
        pages = [page(1, 'Author et al. Journal (2026) 15:185\nbody'),
                 page(2, 'Author et al. Journal (2026) 15:185 Page 2 of 14\nbody'),
                 page(3, 'Author et al. Journal (2026) 15:185 Page 3 of 14\nbody')]
        assert [p['printed_page'] for p in detect_page_labels(pages)] == [1, 2, 3]

    def test_reads_offset_numbering(self):
        pages = [page(1, 'body\n576'), page(2, 'body\n577'), page(3, 'body\n578')]
        labeled = detect_page_labels(pages)
        assert [p['printed_page'] for p in labeled] == [576, 577, 578]
        assert [p['page'] for p in labeled] == [1, 2, 3]

    def test_back_fills_unlabeled_first_page(self):
        pages = [page(1, 'title page, no number'),
                 page(2, 'body\n188'),
                 page(3, 'body\n189')]
        assert [p['printed_page'] for p in detect_page_labels(pages)] == [187, 188, 189]

    def test_inconsistent_numbers_fall_back(self):
        pages = [page(1, 'body\n5'), page(2, 'body\n99'), page(3, 'body\n7')]
        assert all(p['printed_page'] is None for p in detect_page_labels(pages))

    def test_no_numbers_fall_back(self):
        pages = [page(1, 'body only'), page(2, 'body only')]
        assert all(p['printed_page'] is None for p in detect_page_labels(pages))


class TestSections:
    def test_heading_sharing_a_line_with_body_text(self):
        pages = [page(1, 'Abstract\nBackground Systematic reviews are essential'),
                 page(2, 'Results The model recovered every include')]
        labeled = detect_sections(pages)
        assert labeled[0]['section'] == 'Background'
        assert labeled[1]['section'] == 'Results'

    def test_unmatched_heading_keeps_previous_label(self):
        pages = [page(1, 'Methods\nwe did things'), page(2, 'Some Idiosyncratic Heading')]
        assert [p['section'] for p in detect_sections(pages)] == ['Method', 'Method']

    def test_vocabulary_is_not_empty(self):
        assert 'discussion' in SECTION_NAMES


class TestAnchor:
    def test_prefers_printed_page(self):
        span = {'page': 188, 'pdf_page': 2, 'printed_page': 188, 'section': 'Results'}
        assert format_anchor(span) == 'p. 188 · Results'

    def test_marks_file_position_when_printed_unknown(self):
        span = {'page': 2, 'pdf_page': 2, 'printed_page': None, 'section': 'Results'}
        assert format_anchor(span) == 'file p. 2 · Results'

    def test_omits_absent_section(self):
        span = {'page': 5, 'pdf_page': 5, 'printed_page': 5, 'section': ''}
        assert format_anchor(span) == 'p. 5'
