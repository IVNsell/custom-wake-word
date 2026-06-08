"""Load YAML configuration and resolve project paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Project root (parent of src/)
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "default.yaml"


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load config/default.yaml or a custom YAML file."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with cfg_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = str(ROOT)
    return cfg


def resolve_path(cfg: dict[str, Any], key: str) -> Path:
    """Resolve a path key from config paths section relative to project root."""
    return ROOT / cfg["paths"][key]
