#!/usr/bin/env python3
"""Live microphone test for a trained wake word model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wakeword.config import load_config
from wakeword.inference import WakeWordEngine


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
    args = p.parse_args()

    cfg = load_config()
    inf = cfg["inference"]
    engine = WakeWordEngine(
        args.model,
        threshold=args.threshold or inf["threshold"],
        trigger_frames=args.trigger_frames if args.trigger_frames is not None else inf["trigger_frames"],
        refractory_sec=inf["refractory_sec"],
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

    def callback(indata, _frames, _time, status):
        if status:
            print(status, file=sys.stderr)
        pcm = (indata[:, 0] * 32767).astype(np.int16)
        score, fired = engine.process_chunk(pcm)
        if fired:
            print(f"\n>>> WAKE! score={score:.3f}\n")
        elif score > engine.threshold * 0.7:
            print(f"\rscore={score:.3f}", end="", flush=True)

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
