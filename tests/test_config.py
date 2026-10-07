import pytest

from evcharge.config import ConfigError, Secret, get_settings


def test_secret_is_not_exposed_by_repr_or_str():
    secret = Secret("super-secret-token")
    assert "super-secret-token" not in repr(secret)
    assert "super-secret-token" not in str(secret)
    assert "super-secret-token" not in f"{secret}"
    assert secret.reveal() == "super-secret-token"


def test_missing_token_raises_actionable_error(monkeypatch, tmp_path):
    monkeypatch.setenv("ACN_API_TOKEN", "")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ConfigError, match="ACN_API_TOKEN"):
        get_settings()


def test_rejects_non_https_api_url(monkeypatch):
    monkeypatch.setenv("ACN_API_TOKEN", "t0ken")
    monkeypatch.setenv("ACN_API_URL", "http://ev.caltech.edu/api/v1/")
    with pytest.raises(ConfigError, match="https"):
        get_settings()


def test_normalises_url_and_resolves_data_dirs(monkeypatch):
    monkeypatch.setenv("ACN_API_TOKEN", "t0ken")
    monkeypatch.setenv("ACN_API_URL", "https://example.org/api/v1")
    settings = get_settings()
    assert settings.acn_api_url.endswith("/")
    assert settings.raw_dir.name == "raw"
    assert settings.raw_dir.parent == settings.data_dir
