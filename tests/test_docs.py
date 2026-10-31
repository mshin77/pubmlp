import re
from pathlib import Path

SITE = Path(__file__).parents[1] / 'docs'
BASE = 'https://mshin77.github.io/pubmlp/'
PAGES = ['index.html', 'getting-started.html', 'api.html', 'changelog.html', 'support.html', 'vignettes/screening-workflow.html']


def test_docs_pages_load_assets_without_version_tags():
    pages = list(SITE.rglob('*.html'))
    assert pages
    assert [p.name for p in pages if re.search(r'\.(?:js|css)\?(?:v|digest)=', p.read_text(encoding='utf-8'))] == []


def test_docs_pages_carry_a_description_and_clean_canonical():
    for page in PAGES:
        html = (SITE / page).read_text(encoding='utf-8')
        assert re.search(r'<meta content="[^"]+" name="description" />', html), page
        clean = '' if page == 'index.html' else page
        assert f'<link rel="canonical" href="{BASE}{clean}" />' in html, page


def test_docs_headings_hold_only_their_text():
    for page in PAGES:
        html = (SITE / page).read_text(encoding='utf-8')
        assert re.findall(r'<a class="headerlink"[^>]*>[^<]+</a>', html) == [], page
