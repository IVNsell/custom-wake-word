#!/usr/bin/env python3
"""Train a personal wake phrase from 3-10 user recordings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wakeword.config import load_config, resolve_path
from wakeword.train_pipeline import train_user_phrase


def main():
    p = argparse.ArgumentParser(description="Train personal wake word (3-10 recordings)")
    p.add_argument("phrase", help='Wake phrase, e.g. "aizek" or "hey nova"')
    p.add_argument(
        "--recordings",
        type=Path,
        default=None,
        help="Recordings folder (default: workspace/recordings)",
    )
    p.add_argument("--config", type=Path, default=None)
    p.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only augment + extract features, skip training",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    rec_dir = args.recordings or resolve_path(cfg, "user_recordings")
    rec_dir.mkdir(parents=True, exist_ok=True)

    print(f"Phrase: {args.phrase}")
    print(f"Recordings: {rec_dir}")
    print(f"Noise corpus: {resolve_path(cfg, 'negatives_root')}")

    onnx = train_user_phrase(
        args.phrase,
        rec_dir,
        cfg,
        skip_oww_train=args.prepare_only,
    )
    print(f"\nDone: {onnx}")


if __name__ == "__main__":
    main()
