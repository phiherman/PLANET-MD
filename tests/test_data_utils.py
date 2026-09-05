import pytest

from planet_md.data.utils import MDDataset


def _forbid_network_calls(monkeypatch):
    def _raise(*args, **kwargs):
        raise AssertionError("should not be called")

    monkeypatch.setattr("planet_md.data.utils.get_model", _raise)
    monkeypatch.setattr("planet_md.data.utils.get_structure_vae", _raise)


def test_missing_h5_raises_clear_error(tmp_path, monkeypatch):
    _forbid_network_calls(monkeypatch)
    missing = tmp_path / "does_not_exist.h5"

    with pytest.raises(FileNotFoundError, match="not found"):
        MDDataset(processed_h5=missing)


def test_directory_instead_of_file_raises_clear_error(tmp_path, monkeypatch):
    _forbid_network_calls(monkeypatch)

    with pytest.raises(FileNotFoundError, match="not found"):
        MDDataset(processed_h5=tmp_path)
