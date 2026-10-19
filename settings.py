"""Configuration and persistent settings for the ScoutX desktop launcher.

Manages OS-specific configuration directories, user preferences, and persistent
defaults while strictly preserving zero-dependency operation for the core library.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_RESULT_LIMIT: int = 8
DEFAULT_DEBOUNCE_MS: int = 150
MIN_DEBOUNCE_MS: int = 0
MAX_DEBOUNCE_MS: int = 2000
MIN_RESULT_LIMIT: int = 1
MAX_RESULT_LIMIT: int = 50


def get_default_config_dir(platform_name: str | None = None) -> Path:
    """Return the OS-standard directory for storing ScoutX configuration files."""
    plat = (platform_name or sys.platform).lower()
    home = Path.home()

    if plat == "darwin":
        return home / "Library" / "Application Support" / "ScoutX"
    elif plat.startswith("win"):
        appdata = os.environ.get("APPDATA")
        base = Path(appdata) if appdata else (home / "AppData" / "Roaming")
        return base / "ScoutX"
    else:
        # Linux and other Unix-like OS (XDG standard)
        xdg_config = os.environ.get("XDG_CONFIG_HOME")
        base = Path(xdg_config) if xdg_config else (home / ".config")
        return base / "scoutx"


def get_default_data_dir(platform_name: str | None = None) -> Path:
    """Return the OS-standard directory for storing persistent index and usage databases."""
    plat = (platform_name or sys.platform).lower()
    home = Path.home()

    if plat == "darwin":
        return home / "Library" / "Application Support" / "ScoutX"
    elif plat.startswith("win"):
        localappdata = os.environ.get("LOCALAPPDATA")
        if localappdata:
            return Path(localappdata) / "ScoutX"
        appdata = os.environ.get("APPDATA")
        base = Path(appdata) if appdata else (home / "AppData" / "Local")
        return base / "ScoutX"
    else:
        # Linux and other Unix-like OS (XDG data standard)
        xdg_data = os.environ.get("XDG_DATA_HOME")
        base = Path(xdg_data) if xdg_data else (home / ".local" / "share")
        return base / "scoutx"


@dataclass
class LauncherSettings:
    """Persisted user settings for the ScoutX desktop launcher."""

    search_root: str = field(default_factory=lambda: str(Path.home()))
    usage_db: str = ""
    index_db: str = ""
    result_limit: int = DEFAULT_RESULT_LIMIT
    debounce_ms: int = DEFAULT_DEBOUNCE_MS
    hide_on_focus_loss: bool = True
    llm_provider: str | None = None
    llm_model: str | None = None
    llm_api_key_env: str | None = None
    embedding_provider: str = "local_hash"
    embedding_api_key_env: str | None = None
    laya_shadow: bool = False
    laya_base_url: str = "http://127.0.0.1:8001"
    laya_timeout: float = 0.5

    def __post_init__(self) -> None:
        data_dir = get_default_data_dir()
        if not self.usage_db:
            self.usage_db = str(data_dir / "usage.db")
        if not self.index_db:
            self.index_db = str(data_dir / "index.db")

        # Validate bounds
        if not isinstance(self.result_limit, int) or isinstance(self.result_limit, bool):
            self.result_limit = DEFAULT_RESULT_LIMIT
        self.result_limit = max(MIN_RESULT_LIMIT, min(MAX_RESULT_LIMIT, self.result_limit))

        if not isinstance(self.debounce_ms, int) or isinstance(self.debounce_ms, bool):
            self.debounce_ms = DEFAULT_DEBOUNCE_MS
        self.debounce_ms = max(MIN_DEBOUNCE_MS, min(MAX_DEBOUNCE_MS, self.debounce_ms))

        if not isinstance(self.hide_on_focus_loss, bool):
            self.hide_on_focus_loss = True

        if not isinstance(self.laya_shadow, bool):
            self.laya_shadow = False

        if not isinstance(self.laya_timeout, (int, float)) or self.laya_timeout <= 0:
            self.laya_timeout = 0.5
        else:
            self.laya_timeout = float(self.laya_timeout)

        # Validate embedding provider
        if self.embedding_provider not in {"local_hash", "openai_compatible", "gemini_native"}:
            self.embedding_provider = "local_hash"

        # Validate LLM provider
        if self.llm_provider not in {None, "openai_compatible", "gemini_native"}:
            self.llm_provider = None

        # Redact/reject any raw secret values inadvertently passed as env variable names
        if self.llm_api_key_env and (" " in self.llm_api_key_env or len(self.llm_api_key_env) > 64):
            self.llm_api_key_env = None
        if self.embedding_api_key_env and (
            " " in self.embedding_api_key_env or len(self.embedding_api_key_env) > 64
        ):
            self.embedding_api_key_env = None


def load_settings(config_path: str | Path | None = None) -> tuple[LauncherSettings, list[str]]:
    """Load settings from JSON, returning (settings, list_of_diagnostics).

    If the configuration file is missing, returns safe defaults.
    If the file contains invalid JSON or schema errors, safely recovers to defaults
    and provides human-readable diagnostic messages.
    """
    diagnostics: list[str] = []
    target_path = Path(config_path) if config_path else (get_default_config_dir() / "config.json")

    if not target_path.exists():
        return LauncherSettings(), diagnostics

    try:
        content = target_path.read_text(encoding="utf-8")
        raw_data = json.loads(content)
        if not isinstance(raw_data, dict):
            diagnostics.append(
                f"Configuration at {target_path} is not a JSON object; falling back to defaults"
            )
            return LauncherSettings(), diagnostics
    except (json.JSONDecodeError, OSError) as exc:
        diagnostics.append(
            f"Failed to read configuration file at {target_path}: {exc}; falling back to defaults"
        )
        return LauncherSettings(), diagnostics

    # Safe mapping of validated fields
    kwargs: dict[str, Any] = {}
    valid_keys = {
        "search_root",
        "usage_db",
        "index_db",
        "result_limit",
        "debounce_ms",
        "hide_on_focus_loss",
        "llm_provider",
        "llm_model",
        "llm_api_key_env",
        "embedding_provider",
        "embedding_api_key_env",
        "laya_shadow",
        "laya_base_url",
        "laya_timeout",
    }

    for key, value in raw_data.items():
        if key in valid_keys:
            kwargs[key] = value

    try:
        settings = LauncherSettings(**kwargs)
    except Exception as exc:
        diagnostics.append(
            f"Malformed configuration fields in {target_path}: {exc}; recovering to safe defaults"
        )
        settings = LauncherSettings()

    return settings, diagnostics


def save_settings(settings: LauncherSettings, config_path: str | Path | None = None) -> Path:
    """Persist settings to disk as formatted JSON."""
    target_path = Path(config_path) if config_path else (get_default_config_dir() / "config.json")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(settings)
    target_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return target_path
