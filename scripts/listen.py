#!/usr/bin/env python3
"""Live microphone test for a trained wake word model."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wakeword.config import load_config
from wakeword.inference import WakeWordEngine
from wakeword.verifier import PhraseVerifier


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model", type=Path, help="Path to .onnx model")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument(
        "--trigger-frames",
        type=int,
        default=None,
        help="Consecutive frames above threshold to trigger (1=fast, 3=stable)",
    )
    p.add_argument(
        "--verify-recordings",
        type=Path,
        default=None,
        help="Reference recordings folder for MFCC/DTW phrase verification",
    )
    p.add_argument(
        "--verify-negatives",
        type=Path,
        default=None,
        help="Similar non-wake words used as anti-references",
    )
    p.add_argument("--verify-threshold", type=float, default=0.34)
    p.add_argument("--verify-segment-threshold", type=float, default=0.20)
    p.add_argument("--verify-negative-margin", type=float, default=0.04)
    p.add_argument(
        "--show-scores",
        action="store_true",
        help="Print near-threshold model scores for debugging",
    )
    args = p.parse_args()

    cfg = load_config()
    inf = cfg["inference"]
    verifier = None
    if args.verify_recordings:
        verifier = PhraseVerifier(
            args.verify_recordings,
            negative_dir=args.verify_negatives,
            threshold=args.verify_threshold,
            segment_threshold=args.verify_segment_threshold,
            negative_margin=args.verify_negative_margin,
        )
        print(f"Verifier: {args.verify_recordings}")
        if args.verify_negatives:
            print(f"Verifier negatives: {args.verify_negatives}")

    engine = WakeWordEngine(
        args.model,
        threshold=args.threshold or inf["threshold"],
        trigger_frames=args.trigger_frames if args.trigger_frames is not None else inf["trigger_frames"],
        refractory_sec=inf["refractory_sec"],
        verifier=verifier,
    )

    sr = 16000
    chunk = inf["frame_samples"]
    frame_ms = chunk / sr * 1000
    est_ms = engine.trigger_frames * frame_ms

    print(
        f"Listening… threshold={engine.threshold}, "
        f"trigger_frames={engine.trigger_frames} (~{est_ms:.0f} ms to WAKE). Ctrl+C to exit."
    )

    try:
        import sounddevice as sd
    except ImportError:
        print("Install: pip install sounddevice")
        sys.exit(1)

    last_score_print = 0.0

    def format_segments(values) -> str:
        return "/".join(f"{v:.3f}" for v in values)

    def callback(indata, _frames, _time, status):
        nonlocal last_score_print
        if status:
            print(status, file=sys.stderr)
        pcm = (indata[:, 0] * 32767).astype(np.int16)
        score, fired = engine.process_chunk(pcm)
        if fired:
            if engine.last_verification:
                v = engine.last_verification
                print(
                    f"WAKE  model={score:.3f} verify={v.score:.3f} "
                    f"neg={v.negative_score:.3f} "
                    f"segments={format_segments(v.segment_scores)} ref={v.reference}"
                )
            else:
                print(f"WAKE  model={score:.3f}")
        elif engine.last_verification:
            v = engine.last_verification
            print(
                f"REJECT model={score:.3f} verify={v.score:.3f} "
                f"neg={v.negative_score:.3f} "
                f"segments={format_segments(v.segment_scores)}"
            )
        elif args.show_scores and score > engine.threshold * 0.7:
            now = time.monotonic()
            if now - last_score_print >= 0.5:
                print(f"SCORE model={score:.3f}")
                last_score_print = now

    try:
        with sd.InputStream(samplerate=sr, channels=1, dtype="float32", blocksize=chunk, callback=callback):
            while True:
                sd.sleep(1000)
    except KeyboardInterrupt:
        print("\nExit.")
        sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExit.")
        sys.exit(0)
