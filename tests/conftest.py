import os

# cached weights must not trigger a revision check: the hub client is closed by
# earlier tests, and a suite that reaches the network is not reproducible anyway
if not os.environ.get('PUBMLP_RUN_DOWNLOADS'):
    os.environ.setdefault('HF_HUB_OFFLINE', '1')

# numpy must be imported before torch: importing it through torch trips its
# BLAS probe against Anaconda's MKL and aborts the interpreter at collection
import numpy  # noqa: F401,E402
import torch  # noqa: F401,E402

from pathlib import Path  # noqa: E402

import pytest  # noqa: E402


def weights_cached(name):
    """True when Hugging Face already holds the weights, so no network is needed."""
    home = Path(os.environ.get('HF_HOME') or Path.home() / '.cache' / 'huggingface')
    hub = home / 'hub' if (home / 'hub').is_dir() else home
    leaf = name.split('/')[-1]
    return any(
        any(f.stat().st_size > 1_000_000 for f in cached.rglob('*') if f.is_file())
        for cached in hub.glob(f'models--*{leaf}'))


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    """Keep each test's default journal out of the real user directory."""
    monkeypatch.setenv('PUBMLP_DATA_DIR', str(tmp_path / 'pubmlp-data'))


def needs_weights(name):
    """Skip only when the weights are absent and fetching them is not permitted."""
    return pytest.mark.skipif(
        not (os.environ.get('PUBMLP_RUN_DOWNLOADS') or weights_cached(name)),
        reason=f'{name} not cached; set PUBMLP_RUN_DOWNLOADS=1 to fetch it')
