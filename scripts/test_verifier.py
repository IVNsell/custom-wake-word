#!/usr/bin/env python3
"""Offline test for wake model + phonetic verifier."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wakeword.inference import WakeWordEngine
from wakeword.verifier import PhraseVerifier


def load_int16_16k(path: Path) -> np.ndarray:
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != 16000:
        from scipy import signal

        mono = signal.resample(mono, int(len(mono) * 16000 / sr))
    return (mono * 32767).astype(np.int16)


def run_file(model: Path, verifier: PhraseVerifier, wav: Path, threshold: float, trigger_frames: int):
    engine = WakeWordEngine(
        model,
        threshold=threshold,
        trigger_frames=trigger_frames,
        verifier=verifier,
    )
    pcm = np.concatenate(
        [
            np.zeros(16000, dtype=np.int16),
            load_int16_16k(wav),
            np.zeros(16000, dtype=np.int16),
        ]
    )
    max_model_score = 0.0
    wakes = 0
    last_verification = None
    for start in range(0, len(pcm) - 1280 + 1, 1280):
        score, fired = engine.process_chunk(pcm[start : start + 1280])
        max_model_score = max(max_model_score, score)
        wakes += int(fired)
        last_verification = engine.last_verification or last_verification
    return max_model_score, wakes, last_verification


def main():
    p = argparse.ArgumentParser(description="Test wake model + MFCC/DTW verifier")
    p.add_argument("model", type=Path)
    p.add_argument("--positives", type=Path, default=ROOT / "workspace" / "recordings")
    p.add_argument(
        "--test-negatives",
        type=Path,
        default=ROOT / "workspace" / "hard_negatives",
        help="Unknown/non-wake words used only for offline testing",
    )
    p.add_argument(
        "--negative-references",
        type=Path,
        default=None,
        help="Optional anti-references used by the verifier",
    )
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--trigger-frames", type=int, default=2)
    p.add_argument("--verify-threshold", type=float, default=0.34)
    p.add_argument("--verify-segment-threshold", type=float, default=0.20)
    p.add_argument("--verify-negative-margin", type=float, default=0.04)
    args = p.parse_args()

    verifier = PhraseVerifier(
        args.positives,
        negative_dir=args.negative_references if args.negative_references and args.negative_references.exists() else None,
        threshold=args.verify_threshold,
        segment_threshold=args.verify_segment_threshold,
        negative_margin=args.verify_negative_margin,
    )

    for label, folder in [("POSITIVES", args.positives), ("TEST_NEGATIVES", args.test_negatives)]:
        if not folder.exists():
            continue
        total = 0
        files = sorted(folder.glob("*.wav"))
        print(f"\n{label}: {folder}")
        for wav in files:
            model_score, wakes, result = run_file(
                args.model,
                verifier,
                wav,
                args.threshold,
                args.trigger_frames,
            )
            total += wakes
            verify_score = result.score if result else 0.0
            negative_score = result.negative_score if result else 0.0
            print(
                f"{wav.name}: model={model_score:.3f} wakes={wakes} "
                f"verify={verify_score:.3f} neg={negative_score:.3f}"
            )
        print(f"TOTAL_WAKE={total}/{len(files)}")


if __name__ == "__main__":
    main()
