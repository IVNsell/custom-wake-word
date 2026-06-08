"""Augment few user recordings into thousands of training positives."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import soundfile as sf
from audiomentations import Compose, Gain, PitchShift, TimeStretch

from .recordings import load_mono_16k


def _list_negative_clips(negatives_root: Path) -> list[Path]:
    """Recursively find audio files in the platform noise corpus."""
    if not negatives_root.exists():
        return []
    exts = {".wav", ".flac", ".ogg", ".mp3"}
    return [p for p in negatives_root.rglob("*") if p.suffix.lower() in exts]


def _load_clip_slice(path: Path, length: int, sr: int = 16000) -> np.ndarray:
    """Load a random slice of given length from an audio file."""
    data, file_sr = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if file_sr != sr:
        from scipy import signal

        mono = signal.resample(mono, int(len(mono) * sr / file_sr))
    if len(mono) < length:
        reps = int(np.ceil(length / len(mono)))
        mono = np.tile(mono, reps)
    start = random.randint(0, max(0, len(mono) - length))
    return mono[start : start + length].astype(np.float32)


def _mix_noise(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Mix background noise at a target signal-to-noise ratio."""
    clean_power = np.mean(clean**2) + 1e-10
    noise_power = np.mean(noise**2) + 1e-10
    target_noise = clean_power / (10 ** (snr_db / 10))
    scale = np.sqrt(target_noise / noise_power)
    mixed = clean + noise * scale
    peak = np.max(np.abs(mixed)) or 1.0
    return (mixed / peak * 0.95).astype(np.float32)


def build_augment_pipeline() -> Compose:
    """Build audiomentations pipeline: time stretch, pitch, gain."""
    return Compose(
        [
            TimeStretch(min_rate=0.85, max_rate=1.15, p=0.6, leave_length_unchanged=False),
            PitchShift(min_semitones=-2, max_semitones=2, p=0.6),
            Gain(min_gain_db=-6, max_gain_db=6, p=0.7),
        ],
        p=1.0,
    )


def augment_user_recordings(
    source_files: list[Path],
    output_dir: Path,
    negatives_root: Path,
    rounds: int,
    sample_rate: int = 16000,
) -> int:
    """
    Expand 3-10 recordings into hundreds/thousands of augmented positives.
    Each round applies random augmentation and optional noise mixing.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    augment = build_augment_pipeline()
    neg_paths = _list_negative_clips(negatives_root)
    written = 0

    for r in range(rounds):
        for i, src in enumerate(source_files):
            clean = load_mono_16k(src, sample_rate)
            # audiomentations expects shape (channels, samples)
            aug = augment(samples=clean[np.newaxis, :], sample_rate=sample_rate)[0]

            if neg_paths:
                snr = random.uniform(3, 18)
                noise = _load_clip_slice(random.choice(neg_paths), len(aug), sample_rate)
                aug = _mix_noise(aug, noise, snr)

            out = output_dir / f"pos_{i:02d}_r{r:04d}.wav"
            sf.write(out, aug, sample_rate, subtype="PCM_16")
            written += 1

    return written
