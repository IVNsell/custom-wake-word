#!/usr/bin/env python3
"""Один раз: проиндексировать data/negatives/ → shared_negatives.npy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wakeword.config import load_config, resolve_path
from wakeword.features import build_negative_features


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--max-source-files", type=int, default=None, help="Лимит исходных файлов")
    p.add_argument(
        "--features-only",
        action="store_true",
        help="Только embeddings (staging уже готов, без повторного prepare)",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    neg_root = resolve_path(cfg, "negatives_root")
    out = resolve_path(cfg, "negatives_features")

    print(f"Корпус: {neg_root}")
    print(f"Выход: {out}")
    build_negative_features(
        neg_root,
        out,
        max_files=args.max_source_files,
        features_only=args.features_only,
    )
    print("OK: Готово")


if __name__ == "__main__":
    main()
