import importlib

import pytest


@pytest.fixture
def reload_config(monkeypatch):
    """Reload planet_md.config after PLANET_MD_DIR is changed via monkeypatch.

    Module-level assignments like PROJ_ROOT are computed once at import time,
    so tests that change the env var must force re-evaluation.
    """

    def _reload():
        from planet_md import config

        return importlib.reload(config)

    return _reload
