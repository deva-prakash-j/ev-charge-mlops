"""Typed settings loaded from environment / .env, with secrets kept out of reprs."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"

DEFAULT_API_URL = "https://ev.caltech.edu/api/v1/"
SITES: tuple[str, ...] = ("caltech", "jpl", "office001")


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or malformed."""


class Secret:
    """Wraps a secret so it cannot leak through repr, str, or logging."""

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        self._value = value

    def reveal(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return "Secret('***')"

    __str__ = __repr__


@dataclass(frozen=True)
class Settings:
    acn_api_token: Secret
    acn_api_url: str = DEFAULT_API_URL
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    artifacts_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "artifacts")

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def interim_dir(self) -> Path:
        return self.data_dir / "interim"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def sample_dir(self) -> Path:
        return self.data_dir / "sample"


def load_config(name: str = "data") -> dict[str, Any]:
    """Load a YAML config from configs/ by stem name."""
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh)
    if not isinstance(loaded, dict):
        raise ConfigError(f"Config {path.name} must be a YAML mapping.")
    return loaded


def get_settings() -> Settings:
    """Load settings, preferring real environment variables over .env."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    token = (os.getenv("ACN_API_TOKEN") or "").strip()
    if not token:
        raise ConfigError(
            "ACN_API_TOKEN is not set.\n"
            "  1. Copy-Item .env.example .env\n"
            "  2. Paste your token from https://ev.caltech.edu/dataset into .env\n"
            "  3. Re-run. (.env is gitignored — never commit it.)"
        )

    url = (os.getenv("ACN_API_URL") or DEFAULT_API_URL).strip()
    if not url.startswith("https://"):
        raise ConfigError(f"ACN_API_URL must be an https:// URL, got: {url!r}")
    if not url.endswith("/"):
        url += "/"

    data_dir = Path(os.getenv("DATA_DIR") or "data")
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir

    artifacts_dir = Path(os.getenv("ARTIFACTS_DIR") or "artifacts")
    if not artifacts_dir.is_absolute():
        artifacts_dir = PROJECT_ROOT / artifacts_dir

    return Settings(
        acn_api_token=Secret(token),
        acn_api_url=url,
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
    )
