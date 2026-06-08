#!/usr/bin/env python3
"""Measure wake word inference latency and estimate end-to-end delay."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wakeword.config import load_config
from wakeword.inference import WakeWordEngine


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def benchmark_raw_inference(engine: WakeWordEngine, chunk_samples: int, runs: int, warmup: int) -> list[float]:
    rng = np.random.default_rng(0)
    pcm = (rng.standard_normal(chunk_samples) * 8000).astype(np.int16)

    for _ in range(warmup):
        engine.process_chunk(pcm)

    times_ms: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        engine.process_chunk(pcm)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    return times_ms


def print_stats(label: str, times_ms: list[float]) -> None:
    s = sorted(times_ms)
    print(f"\n{label}")
    print(f"  n={len(s)}")
    print(f"  p50:  {_percentile(s, 50):.2f} ms")
    print(f"  p95:  {_percentile(s, 95):.2f} ms")
    print(f"  p99:  {_percentile(s, 99):.2f} ms")
    print(f"  mean: {statistics.mean(s):.2f} ms")
    print(f"  min:  {min(s):.2f} ms")
    print(f"  max:  {max(s):.2f} ms")


def main():
    p = argparse.ArgumentParser(description="Benchmark wake word latency")
    p.add_argument("model", type=Path, help="Path to .onnx model")
    p.add_argument("--runs", type=int, default=2000, help="Number of inference runs")
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--trigger-frames", type=int, default=None)
    args = p.parse_args()

    cfg = load_config()
    inf = cfg["inference"]
    chunk = inf["frame_samples"]
    trigger_frames = args.trigger_frames if args.trigger_frames is not None else inf["trigger_frames"]
    sr = inf["sample_rate"]
    frame_ms = chunk / sr * 1000.0

    engine = WakeWordEngine(
        args.model,
        threshold=args.threshold or inf["threshold"],
        trigger_frames=trigger_frames,
        refractory_sec=inf["refractory_sec"],
    )

    print(f"Model: {args.model}")
    print(f"Chunk: {chunk} samples ({frame_ms:.1f} ms @ {sr} Hz)")
    print(f"trigger_frames: {trigger_frames} (consecutive frames required for WAKE)")

    times = benchmark_raw_inference(engine, chunk, args.runs, args.warmup)
    print_stats("Inference (1 chunk, CPU ONNX)", times)

    p50 = _percentile(sorted(times), 50)
    e2e_min_ms = trigger_frames * frame_ms + trigger_frames * p50
    e2e_typical_ms = trigger_frames * frame_ms + trigger_frames * _percentile(sorted(times), 95)

    print("\nEnd-to-end estimate (phrase start → WAKE event):")
    print(f"  audio buffer: {trigger_frames} x {frame_ms:.1f} ms = {trigger_frames * frame_ms:.1f} ms")
    print(f"  + inference:  {trigger_frames} x p50/p95")
    print(f"  => optimistic ~{e2e_min_ms:.0f} ms")
    print(f"  => typical     ~{e2e_typical_ms:.0f} ms")
    print("\nNote: actual delay depends on chunk alignment when you start speaking.")


if __name__ == "__main__":
    main()
