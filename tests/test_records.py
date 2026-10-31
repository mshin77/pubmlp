import warnings

import pandas as pd
import pytest

warnings.filterwarnings('ignore')


from pubmlp.records import read_records


SCOPUS = (b'Authors,Title,Year,Source title,Abstract,DOI\n'
          b'"Shin, M.",Dyslexia support at university,2025,J Learn Disabil,'
          b'Students in higher education.,10.1000/a\n')

WOS = (b'UT,TI,AB,SO,PY,AU\n'
       b'WOS:001,Reading fluency in college,A study.,Reading Research,2024,"McKenna, J."\n')

RIS = (b'TY  - JOUR\nAU  - Shin, M.\nAU  - McKenna, J.\n'
       b'TI  - Screen readers in higher education\n'
       b'AB  - A review of adoption.\nJO  - J Learn Disabil\nPY  - 2025\n'
       b'AN  - EJ12345\nER  - \n')


BIBTEX = (b'@article{Shin2025,\n'
          b'  author = {Shin, Mikyung and McKenna, John},\n'
          b'  title = {Assistive technology for university students},\n'
          b'  journal = {Journal of Learning Disabilities},\n'
          b'  year = {2025},\n  doi = {10.1177/00222194}\n}\n')

WOS_PLAIN = ('FN Clarivate Analytics Web of Science\nVR 1.0\nPT J\n'
             'AU Shin, M\n   McKenna, J\n'
             'TI Reading fluency supports in postsecondary\n   settings\n'
             'SO READING RESEARCH QUARTERLY\nAB Fluency practice.\n'
             'PY 2024\nUT WOS:000123456\nER\n').encode()

CIW = (b'PT J\nAU Alvarez, R\nTI Screen reader adoption\nSO ASSIST TECH\n'
       b'AB A study.\nPY 2025\nUT WOS:000999\nER\n')

class TestColumnVocabulary:
    def test_scopus_headers_map_to_canonical_names(self):
        frame = read_records('scopus.csv', SCOPUS)
        assert {'title', 'abstract', 'author', 'journal', 'year', 'doi'} <= set(frame.columns)

    def test_web_of_science_codes_map_to_the_same_names(self):
        frame = read_records('wos.csv', WOS)
        assert frame['journal'][0] == 'Reading Research'
        assert frame['an'][0] == 'WOS:001'

    def test_ris_repeated_tags_are_joined_not_overwritten(self):
        frame = read_records('export.ris', RIS)
        assert frame['author'][0] == 'Shin, M.; McKenna, J.'
        assert frame['an'][0] == 'EJ12345'

    def test_records_without_an_identifier_get_one(self):
        frame = read_records('bare.csv', b'Title\nA study of reading\nAnother study\n')
        assert list(frame['an']) == ['record-1', 'record-2']

    def test_unrecognized_columns_are_kept(self):
        frame = read_records('extra.csv', b'Title,Funding note\nA study,Grant 42\n')
        assert 'Funding note' in frame.columns

    def test_a_file_without_titles_is_refused(self):
        with pytest.raises(ValueError, match='title'):
            read_records('wrong.csv', b'Author,Year\nShin,2025\n')

    def test_an_empty_file_is_refused(self):
        with pytest.raises(ValueError):
            read_records('empty.csv', b'Title\n')


class TestBibliometrixFormats:
    def test_bibtex_entries_read(self):
        frame = read_records('scopus.bib', BIBTEX)
        assert frame['an'][0] == 'Shin2025'
        assert frame['title'][0].startswith('Assistive technology')
        assert frame['year'][0] == '2025'

    def test_web_of_science_plaintext_reads(self):
        frame = read_records('savedrecs.txt', WOS_PLAIN)
        assert frame['an'][0] == 'WOS:000123456'
        assert frame['journal'][0] == 'READING RESEARCH QUARTERLY'

    def test_wrapped_lines_rejoin(self):
        frame = read_records('savedrecs.txt', WOS_PLAIN)
        assert frame['title'][0] == 'Reading fluency supports in postsecondary settings'
        assert frame['author'][0] == 'Shin, M; McKenna, J'

    def test_endnote_ciw_reads(self):
        assert read_records('export.ciw', CIW)['title'][0] == 'Screen reader adoption'

    def test_txt_is_sniffed_between_ris_and_isi(self):
        ris = b'TY  - JOUR\nTI  - Note-taking software\nPY  - 2024\nER  - \n'
        assert read_records('pubmed.txt', ris)['title'][0] == 'Note-taking software'
        assert read_records('wos.txt', WOS_PLAIN)['an'][0] == 'WOS:000123456'

    def test_every_bibliometrix_extension_is_accepted(self):
        for name, raw in [('a.csv', SCOPUS), ('a.bib', BIBTEX), ('a.ciw', CIW),
                          ('a.txt', WOS_PLAIN), ('a.ris', RIS)]:
            assert len(read_records(name, raw)) >= 1, name


class TestFindExports:
    def make(self, root, names):
        for name in names:
            (root / name).write_bytes(b'Title\nA study\n')

    def test_only_readable_extensions_come_back(self, tmp_path):
        from pubmlp.records import find_exports

        self.make(tmp_path, ['a.csv', 'b.ris', 'c.bib', 'notes.docx', 'image.png'])
        names = {p.name for p in find_exports(tmp_path)}
        assert names == {'a.csv', 'b.ris', 'c.bib'}

    def test_subfolders_are_searched_unless_refused(self, tmp_path):
        from pubmlp.records import find_exports

        nested = tmp_path / 'exports' / 'march'
        nested.mkdir(parents=True)
        self.make(nested, ['deep.csv'])
        self.make(tmp_path, ['top.csv'])
        assert len(find_exports(tmp_path)) == 2
        assert [p.name for p in find_exports(tmp_path, recursive=False)] == ['top.csv']

    def test_newest_first(self, tmp_path):
        import os
        import time

        from pubmlp.records import find_exports

        self.make(tmp_path, ['old.csv', 'new.csv'])
        os.utime(tmp_path / 'old.csv', (time.time() - 600, time.time() - 600))
        assert [p.name for p in find_exports(tmp_path)][0] == 'new.csv'

    def test_a_missing_folder_is_empty_not_an_error(self, tmp_path):
        from pubmlp.records import find_exports

        assert find_exports(tmp_path / 'nowhere') == []


class TestOneIdentifierPerRecord:
    """A decision is applied by identifier, so two records may not share one."""

    def test_a_blank_accession_number_gets_its_own(self):
        frame = read_records('export.csv', b'an,title\n,First study\n,Second study\n')
        assert list(frame['an']) == ['record-1', 'record-2']

    def test_a_repeated_one_is_suffixed(self):
        data = b'an,title\nWOS:001,First\nWOS:001,Second\nWOS:002,Third\n'
        assert list(read_records('export.csv', data)['an']) == [
            'WOS:001', 'WOS:001-2', 'WOS:002']

    def test_a_unique_one_is_left_alone(self):
        data = b'an,title\nWOS:001,First\nWOS:002,Second\n'
        assert list(read_records('export.csv', data)['an']) == ['WOS:001', 'WOS:002']


class TestTaggedExportEdges:
    def test_a_wrapped_line_starting_with_er_does_not_end_the_record(self):
        export = (b'FN Thomson\nPT J\nTI Reading support at university\n'
                  b'AB Students were recruited from four sites.\n'
                  b'   ERIC and PsycINFO were also searched.\n'
                  b'PY 2025\nER\n')
        frame = read_records('wos.txt', export)
        assert len(frame) == 1
        assert 'ERIC' in frame['abstract'][0]
        assert frame['year'][0] == '2025'

    def test_a_field_whose_text_starts_on_the_next_line_is_kept(self):
        export = (b'FN Thomson\nPT J\nTI Reading support at university\n'
                  b'AB\n   Students with dyslexia were interviewed.\n'
                  b'   A thematic analysis followed.\nER\n')
        frame = read_records('wos.txt', export)
        assert frame['abstract'][0].startswith('Students with dyslexia')
        assert 'thematic analysis' in frame['abstract'][0]
