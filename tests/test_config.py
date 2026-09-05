from pathlib import Path


def test_planet_md_dir_unset_uses_default(monkeypatch, reload_config):
    monkeypatch.delenv("PLANET_MD_DIR", raising=False)
    config = reload_config()
    assert config.PROJ_ROOT == Path(config.__file__).resolve().parents[1]


def test_planet_md_dir_set_to_real_path(monkeypatch, tmp_path, reload_config):
    monkeypatch.setenv("PLANET_MD_DIR", str(tmp_path))
    config = reload_config()
    assert config.PROJ_ROOT == tmp_path


def test_planet_md_dir_empty_string_falls_back_to_default(monkeypatch, reload_config):
    monkeypatch.setenv("PLANET_MD_DIR", "")
    config = reload_config()
    assert config.PROJ_ROOT == Path(config.__file__).resolve().parents[1]


def test_planet_md_dir_unset_never_resolves_under_original_author_home(
    monkeypatch, reload_config
):
    monkeypatch.delenv("PLANET_MD_DIR", raising=False)
    config = reload_config()
    assert "ssledzieski" not in str(config.PROJ_ROOT)
