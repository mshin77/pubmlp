import logging

import numpy as np
import pytest

from pubmlp.licensing import check_llm_use, is_local_endpoint
from pubmlp.provenance import ProvenanceTracker
from pubmlp.rag import chunk_pages, extract_with_rag


class Encoder:
    def encode(self, texts, **kwargs):
        return np.array([[float('baseline' in t), 1.0] for t in texts])


LOCAL = 'http://localhost:11434/v1'
PAGES = [{'page': 1, 'text': 'A multiple baseline design across four participants.'}]


def local_permission():
    return check_llm_use(access='subscription', ai_terms='silent', processing='local')


class TestEndpoint:
    @pytest.mark.parametrize('url', ['http://localhost:11434/v1', 'http://127.0.0.1:8000',
                                     'http://[::1]:11434'])
    def test_local_hosts(self, url):
        assert is_local_endpoint(url)

    @pytest.mark.parametrize('url', ['https://api.openai.com/v1', 'http://0.0.0.0:11434',
                                     'http://model.localhost:80',
                                     '', None, 'not a url'])
    def test_other_hosts(self, url):
        assert not is_local_endpoint(url)


class TestPermissionEnforcement:
    def test_local_verdict_without_endpoint_is_refused(self):
        sent = []
        with pytest.raises(PermissionError, match='endpoint'):
            extract_with_rag(PAGES, {'a': 'q'}, lambda p: sent.append(p) or '', Encoder(),
                             permission=local_permission())
        assert sent == []

    def test_local_verdict_with_remote_endpoint_is_refused(self):
        sent = []
        with pytest.raises(PermissionError, match='api.openai.com'):
            extract_with_rag(PAGES, {'a': 'q'}, lambda p: sent.append(p) or '', Encoder(),
                             permission=local_permission(),
                             endpoint='https://api.openai.com/v1')
        assert sent == []

    def test_local_verdict_with_local_endpoint_runs(self):
        results = extract_with_rag(PAGES, {'a': 'q'}, lambda p: 'not found', Encoder(),
                                   permission=local_permission(), endpoint=LOCAL)
        assert 'a' in results

    def test_api_verdict_accepts_remote_endpoint(self):
        permission = check_llm_use(access='open_access', license_type='cc-by', processing='api')
        results = extract_with_rag(PAGES, {'a': 'q'}, lambda p: 'not found', Encoder(),
                                   permission=permission, endpoint='https://api.openai.com/v1')
        assert 'a' in results

    def test_incomplete_permission_is_rejected(self):
        with pytest.raises(ValueError, match='permission'):
            extract_with_rag(PAGES, {'a': 'q'}, lambda p: '', Encoder(),
                             permission={'verdict': 'any_llm'})

    def test_unchecked_permission_warns(self, caplog):
        with caplog.at_level(logging.WARNING, logger='pubmlp.rag'):
            extract_with_rag(PAGES, {'a': 'q'}, lambda p: 'not found', Encoder(), permission='unchecked')
        assert 'unchecked' in caplog.text

    def test_endpoint_is_recorded(self):
        tracker = ProvenanceTracker('fulltext', 'extraction')
        extract_with_rag(PAGES, {'a': 'q'}, lambda p: 'not found', Encoder(),
                         tracker=tracker, permission=local_permission(), endpoint=LOCAL)
        assert tracker.to_dict()['permission']['endpoint'] == LOCAL


class TestEmptyInput:
    def test_document_without_text_raises(self):
        with pytest.raises(ValueError, match='text'):
            extract_with_rag([{'page': 1, 'text': ''}], {'a': 'q'}, lambda p: '', Encoder(), permission='unchecked')

    def test_pages_without_page_key_are_numbered(self):
        chunks = chunk_pages([{'text': 'first page'}, {'text': 'second page'}])
        assert [c['pdf_page'] for c in chunks] == [1, 2]

class TestLicenseSpelling:
    @pytest.mark.parametrize('license_type', ['CC BY 4.0', 'cc-by-4.0', ' cc-by ', 'CC_BY', 'CC-BY-4'])
    def test_spelling_variants_of_cc_by(self, license_type):
        result = check_llm_use(access='open_access', license_type=license_type, processing='api')
        assert result['verdict'] == 'any_llm'
        assert result['inputs']['license_type'] == 'cc-by'

    def test_missing_value_from_a_spreadsheet(self):
        result = check_llm_use(access='open_access', license_type=float('nan'))
        assert result['inputs']['license_type'] is None


class TestProvenanceEnvironment:
    def test_records_extraction_dependencies_without_importing_them(self):
        import sys

        before = 'torch' in sys.modules
        environment = ProvenanceTracker('s', 't').to_dict()['environment']
        assert {'torch', 'numpy', 'transformers', 'sentence-transformers'} <= set(environment)
        assert ('torch' in sys.modules) == before


class TestPrivateCloud:
    def permission(self):
        return check_llm_use(access='subscription', ai_terms='allows', processing='private_cloud')

    def test_private_cloud_needs_an_endpoint(self):
        with pytest.raises(PermissionError, match='endpoint'):
            extract_with_rag(PAGES, {'a': 'q'}, lambda p: '', Encoder(),
                             permission=self.permission())

    def test_private_cloud_with_endpoint_runs(self):
        results = extract_with_rag(PAGES, {'a': 'q'}, lambda p: 'not found', Encoder(),
                                   permission=self.permission(),
                                   endpoint='http://localhost:11434/v1')
        assert 'a' in results
