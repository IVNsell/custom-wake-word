"""Validate user wake phrase recordings (3-10 WAV files)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass
class RecordingValidation:
    """Result of validating a recordings directory."""
    ok: bool
    files: list[Path]
    errors: list[str]
    warnings: list[str]


def _normalize_phrase(phrase: str) -> str:
    phrase = phrase.strip().lower()
    phrase = re.sub(r"\s+", " ", phrase)
    return phrase


def validate_phrase(phrase: str, cfg: dict) -> list[str]:
    """Check wake phrase length and word count."""
    errors: list[str] = []
    p = _normalize_phrase(phrase)
    if not p:
        errors.append("Phrase is empty.")
    words = p.split()
    max_words = cfg["phrase"]["max_words"]
    if len(words) > max_words:
        errors.append(f"Phrase too long: max {max_words} words, got {len(words)}.")
    if len(p) < 3:
        errors.append("Phrase too short (minimum ~3 characters).")
    return errors


def validate_recordings_dir(
    recordings_dir: Path,
    cfg: dict,
) -> RecordingValidation:
    """Validate count, duration, and format of user WAV files."""
    min_c = cfg["recordings"]["min_count"]
    max_c = cfg["recordings"]["max_count"]
    target_sr = cfg["recordings"]["sample_rate"]
    max_dur = cfg["phrase"]["max_duration_sec"]

    errors: list[str] = []
    warnings: list[str] = []

    if not recordings_dir.is_dir():
        return RecordingValidation(False, [], [f"Directory not found: {recordings_dir}"], [])

    files = sorted(recordings_dir.glob("*.wav"))
    if not files:
        files = sorted(recordings_dir.glob("*.flac"))

    if len(files) < min_c:
        errors.append(f"Need at least {min_c} recordings, found {len(files)}.")
    if len(files) > max_c:
        errors.append(f"Maximum {max_c} recordings, found {len(files)}. Keep the best ones.")

    rec = cfg["recordings"].get("recommended_count", 7)
    if min_c <= len(files) < rec:
        warnings.append(f"Recommended {rec} recordings for stability (currently {len(files)}).")

    for path in files:
        try:
            data, sr = sf.read(path, dtype="float32", always_2d=True)
        except Exception as e:
            errors.append(f"{path.name}: cannot read — {e}")
            continue
        if data.shape[1] > 1:
            warnings.append(f"{path.name}: stereo — will be converted to mono.")
        duration = len(data) / sr
        if duration > max_dur:
            warnings.append(f"{path.name}: longer than {max_dur}s — trim to a short phrase.")
        if duration < 0.3:
            errors.append(f"{path.name}: clip too short ({duration:.2f}s).")
        if sr != target_sr:
            warnings.append(f"{path.name}: {sr} Hz -> will resample to {target_sr} Hz.")

    ok = len(errors) == 0 and min_c <= len(files) <= max_c
    return RecordingValidation(ok, files, errors, warnings)


def load_mono_16k(path: Path, target_sr: int = 16000) -> np.ndarray:
    """Load audio file as normalized mono float32 at 16 kHz."""
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != target_sr:
        from scipy import signal

        num = int(len(mono) * target_sr / sr)
        mono = signal.resample(mono, num)
    peak = np.max(np.abs(mono)) or 1.0
    return (mono / peak * 0.95).astype(np.float32)
