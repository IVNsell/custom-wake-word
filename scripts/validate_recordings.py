#!/usr/bin/env python3
"""Проверка папки с записями перед обучением."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wakeword.config import load_config, resolve_path
from wakeword.recordings import validate_phrase, validate_recordings_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("phrase")
    p.add_argument("--recordings", type=Path, default=None)
    args = p.parse_args()

    cfg = load_config()
    rec = args.recordings or resolve_path(cfg, "user_recordings")

    for e in validate_phrase(args.phrase, cfg):
        print("ERROR:", e)
    v = validate_recordings_dir(rec, cfg)
    for w in v.warnings:
        print("WARN:", w)
    for e in v.errors:
        print("ERROR:", e)
    if v.ok:
        print(f"OK: {len(v.files)} файлов")
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
