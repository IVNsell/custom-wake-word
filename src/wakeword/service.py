"""
High-level API for integrating wake word training and inference into an assistant backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from .catalog import list_catalog_for_api, load_catalog
from .config import load_config, resolve_path
from .inference import WakeWordEngine
from .recordings import validate_phrase, validate_recordings_dir
from .train_pipeline import train_user_phrase


@dataclass
class TrainRequest:
    """Training job descriptor (for future HTTP API)."""
    user_id: str
    phrase: str
    recordings_dir: Path


@dataclass
class TrainResult:
    """Result of validation or training."""
    success: bool
    model_path: Path | None
    errors: list[str]
    warnings: list[str]


class WakeWordPlatform:
    """Single entry point for assistant backend: catalog, train, inference."""

    def __init__(self, config_path: Path | None = None):
        self.cfg = load_config(config_path)

    def get_catalog(self) -> list[dict]:
        """List catalog entries for UI (includes availability flag)."""
        return list_catalog_for_api()

    def get_catalog_models(self) -> list[dict]:
        """List only models that exist on disk."""
        return load_catalog()

    def validate_user_input(self, phrase: str, recordings_dir: Path) -> TrainResult:
        """Validate phrase text and recording files before training."""
        errors = list(validate_phrase(phrase, self.cfg))
        v = validate_recordings_dir(recordings_dir, self.cfg)
        errors.extend(v.errors)
        return TrainResult(
            success=len(errors) == 0,
            model_path=None,
            errors=errors,
            warnings=v.warnings,
        )

    def train_user_model(self, phrase: str, recordings_dir: Path) -> TrainResult:
        """Full pipeline: augment → features → train → export ONNX."""
        check = self.validate_user_input(phrase, recordings_dir)
        if not check.success:
            return check
        try:
            onnx = train_user_phrase(phrase, recordings_dir, self.cfg)
            return TrainResult(True, onnx, [], check.warnings)
        except Exception as e:
            return TrainResult(False, None, [str(e)], check.warnings)

    def create_engine(self, model_path: Path, threshold: float | None = None) -> WakeWordEngine:
        """Create inference engine with default or custom threshold."""
        inf = self.cfg["inference"]
        return WakeWordEngine(
            model_path,
            threshold=threshold or inf["threshold"],
            trigger_frames=inf["trigger_frames"],
            refractory_sec=inf["refractory_sec"],
        )

    def save_user_uploads(
        self,
        user_id: str,
        files: list[tuple[str, BinaryIO]],
    ) -> Path:
        """Save 3-10 uploaded WAV files to workspace/recordings/{user_id}/."""
        base = resolve_path(self.cfg, "user_recordings") / user_id
        base.mkdir(parents=True, exist_ok=True)
        for name, stream in files:
            dest = base / name
            dest.write_bytes(stream.read())
        return base
