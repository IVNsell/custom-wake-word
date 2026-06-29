#!/usr/bin/env python3
"""Live microphone test for a trained wake word model."""

from __future__ import annotations

import argparse
import queue
import sys
import threading
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
    p.add_argument("--verify-threshold", type=float, default=0.28)
    p.add_argument("--verify-segment-threshold", type=float, default=0.15)
    p.add_argument("--verify-negative-margin", type=float, default=0.04)
    p.add_argument("--verify-trust-model", type=float, default=0.95,
                   help="When model score is this high, use a slightly lower verify bar")
    p.add_argument("--verify-trust-threshold", type=float, default=0.24,
                   help="Minimum verify score when --verify-trust-model is met")
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
        defer_verification=verifier is not None,
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

    verify_jobs: queue.Queue[tuple[np.ndarray, float] | None] = queue.Queue(maxsize=2)
    verify_results: queue.Queue[tuple[float, object]] = queue.Queue()
    overflow_notice_at = 0.0

    def verify_worker():
        while True:
            job = verify_jobs.get()
            if job is None:
                break
            snapshot, model_score = job
            result = verifier.verify(snapshot, fast=False)
            verify_results.put((model_score, result))

    worker = None
    if verifier is not None:
        worker = threading.Thread(target=verify_worker, daemon=True)
        worker.start()

    last_score_print = 0.0
    last_reject_key = ""
    last_reject_at = 0.0

    def format_segments(values) -> str:
        return "/".join(f"{v:.3f}" for v in values)

    def should_accept(model_score: float, result) -> bool:
        if result.accepted:
            return True
        if model_score >= args.verify_trust_model and result.score >= args.verify_trust_threshold:
            return min(result.segment_scores) >= args.verify_segment_threshold
        return False

    def print_verification(score: float, result, accepted: bool) -> None:
        nonlocal last_reject_key, last_reject_at
        if accepted:
            print(
                f"WAKE  model={score:.3f} verify={result.score:.3f} "
                f"neg={result.negative_score:.3f} "
                f"segments={format_segments(result.segment_scores)} ref={result.reference}"
            )
            return

        reject_key = f"{result.score:.3f}|{'/'.join(f'{v:.3f}' for v in result.segment_scores)}"
        now = time.monotonic()
        if reject_key == last_reject_key and now - last_reject_at < 0.35:
            return
        last_reject_key = reject_key
        last_reject_at = now
        print(
            f"REJECT model={score:.3f} verify={result.score:.3f} "
            f"neg={result.negative_score:.3f} "
            f"segments={format_segments(result.segment_scores)}"
        )

    def callback(indata, _frames, _time, status):
        nonlocal last_score_print, overflow_notice_at
        if status:
            now = time.monotonic()
            if "overflow" in str(status).lower() and now - overflow_notice_at > 2.0:
                print("Mic buffer overflow — retrying with safer audio settings.", file=sys.stderr)
                overflow_notice_at = now
        pcm = (indata[:, 0] * 32767).astype(np.int16)
        score, fired = engine.process_chunk(pcm)

        pending = engine.consume_pending()
        if pending is not None and verifier is not None:
            snapshot, pending_score = pending
            while True:
                try:
                    verify_jobs.get_nowait()
                except queue.Empty:
                    break
            try:
                verify_jobs.put_nowait((snapshot, pending_score))
            except queue.Full:
                pass

        if fired:
            if engine.last_verification:
                print_verification(score, engine.last_verification, accepted=True)
            else:
                print(f"WAKE  model={score:.3f}")
        elif args.show_scores and score > engine.threshold * 0.7:
            now = time.monotonic()
            if now - last_score_print >= 0.5:
                print(f"SCORE model={score:.3f}")
                last_score_print = now

    try:
        stream_kwargs = dict(
            samplerate=sr,
            channels=1,
            dtype="float32",
            blocksize=chunk,
            callback=callback,
            latency="high",
        )
        try:
            stream = sd.InputStream(**stream_kwargs)
        except TypeError:
            stream_kwargs.pop("latency", None)
            stream = sd.InputStream(**stream_kwargs)

        with stream:
            while True:
                try:
                    score, result = verify_results.get(timeout=0.05)
                except queue.Empty:
                    sd.sleep(50)
                    continue
                accepted = should_accept(score, result)
                engine.apply_verification(result, accepted=accepted)
                print_verification(score, result, accepted=accepted)
    except KeyboardInterrupt:
        if worker is not None:
            verify_jobs.put(None)
            worker.join(timeout=1.0)
        print("\nExit.")
        sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExit.")
        sys.exit(0)
