"""Load pre-trained wake word models from catalog/manifest.json."""

from __future__ import annotations

import json
from pathlib import Path

from .config import ROOT


def load_catalog(manifest_path: Path | None = None) -> list[dict]:
    """Return catalog entries whose .onnx files exist on disk."""
    path = manifest_path or (ROOT / "catalog" / "manifest.json")
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    models = data.get("models", [])
    catalog_dir = path.parent / "models"
    result = []
    for m in models:
        if not m.get("enabled", True):
            continue
        onnx = catalog_dir / m["file"]
        if onnx.exists():
            m = {**m, "path": str(onnx.resolve())}
            result.append(m)
    return result


def list_catalog_for_api() -> list[dict]:
    """Catalog metadata for UI (includes availability without file paths)."""
    path = ROOT / "catalog" / "manifest.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for m in data.get("models", []):
        onnx = ROOT / "catalog" / "models" / m["file"]
        out.append(
            {
                "id": m["id"],
                "phrase": m["phrase"],
                "language": m.get("language", "unknown"),
                "available": onnx.exists(),
                "threshold": m.get("threshold", 0.5),
            }
        )
    return out
