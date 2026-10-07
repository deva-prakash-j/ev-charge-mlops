import pytest

from evcharge.config import ConfigError, Secret, get_settings


def test_secret_is_not_exposed_by_repr_or_str():
    secret = Secret("super-secret-token")
    assert "super-secret-token" not in repr(secret)
    assert "super-secret-token" not in str(secret)
    assert "super-secret-token" not in f"{secret}"
    assert secret.reveal() == "super-secret-token"


def test_settings_load_without_a_token(monkeypatch):
    """A fresh clone has no .env; the sample path must still work."""
    monkeypatch.setenv("ACN_API_TOKEN", "")
    settings = get_settings()
    assert settings.acn_api_token is None
    assert settings.raw_dir.name == "raw"


def test_require_token_raises_actionable_error(monkeypatch):
    monkeypatch.setenv("ACN_API_TOKEN", "")
    with pytest.raises(ConfigError, match="ACN_API_TOKEN"):
        get_settings().require_token()


def test_missing_token_message_points_at_the_sample_path(monkeypatch):
    """The error must tell an evaluator they can run without credentials."""
    monkeypatch.setenv("ACN_API_TOKEN", "")
    with pytest.raises(ConfigError) as excinfo:
        get_settings().require_token()
    assert "--source sample" in str(excinfo.value)


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
