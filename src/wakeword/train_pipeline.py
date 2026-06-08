from __future__ import annotations

import re
import shutil
from pathlib import Path

import yaml

from .augment import augment_user_recordings
from .config import ROOT, load_config, resolve_path
from .features import extract_features_from_wavs
from .oww_trainer import train_and_export_onnx
from .recordings import validate_phrase, validate_recordings_dir


def _slug(phrase: str) -> str:
    s = phrase.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s, flags=re.UNICODE)
    s = re.sub(r"\s+", "_", s)
    return s[:48] or "wake_word"


def _openwakeword_train_script() -> Path | None:
    """Legacy: openWakeWord train.py (не используется — несовместимость speechbrain/torchaudio)."""
    try:
        import openwakeword

        return Path(openwakeword.__file__).parent / "train.py"
    except ImportError:
        return None


def generate_oww_training_yaml(
    model_name: str,
    work_dir: Path,
    pos_features: Path,
    neg_features: Path | None,
    cfg: dict,
) -> Path:
    """Конфиг для openWakeWord train.py — позитивы из записей пользователя."""
    feature_files: dict[str, str] = {"positive": str(pos_features.resolve())}
    batch: dict[str, int] = {"positive": 64, "adversarial_negative": 32}

    if neg_features and neg_features.exists():
        feature_files["platform_negatives"] = str(neg_features.resolve())
        batch["platform_negatives"] = 960

    # Базовый ACAV — скачайте один раз в data/features/ (см. README)
    acav = ROOT / "data" / "features" / "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
    if acav.exists():
        feature_files["ACAV100M_sample"] = str(acav.resolve())
        batch["ACAV100M_sample"] = 512

    train_cfg = {
        "model_name": model_name,
        "target_phrase": [cfg.get("_phrase", model_name)],
        "n_samples": 0,
        "n_samples_val": 0,
        "feature_data_files": feature_files,
        "batch_n_per_class": batch,
        "model_type": cfg["training"]["model_type"],
        "layer_size": cfg["training"]["layer_size"],
        "steps": cfg["training"]["steps"],
        "max_negative_weight": cfg["training"]["max_negative_weight"],
        "target_false_positives_per_hour": cfg["training"]["target_false_positives_per_hour"],
    }

    out = work_dir / f"{model_name}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        yaml.dump(train_cfg, f, allow_unicode=True, default_flow_style=False)
    return out


def train_user_phrase(
    phrase: str,
    recordings_dir: Path,
    cfg: dict | None = None,
    *,
    skip_oww_train: bool = False,
) -> Path:
    """
    Полный цикл: 3–10 WAV → аугментация → features → openWakeWord train → ONNX.
    Возвращает путь к .onnx.
    """
    cfg = cfg or load_config()
    cfg["_phrase"] = phrase

    phrase_errors = validate_phrase(phrase, cfg)
    if phrase_errors:
        raise ValueError("\n".join(phrase_errors))

    validation = validate_recordings_dir(recordings_dir, cfg)
    if not validation.ok:
        raise ValueError("\n".join(validation.errors))

    for w in validation.warnings:
        print(f"WARN: {w}")

    model_name = _slug(phrase)
    work = Path(cfg.get("_work_dir", resolve_path(cfg, "temp_training") / model_name))
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    aug_dir = work / "augmented_positives"
    rounds = cfg["training"]["augmentation_rounds"]
    neg_root = resolve_path(cfg, "negatives_root")

    n_aug = augment_user_recordings(
        validation.files,
        aug_dir,
        neg_root,
        rounds=rounds,
    )
    print(f"OK: Создано {n_aug} аугментированных позитивов из {len(validation.files)} записей")

    pos_feat = work / "positive_features.npy"
    extract_features_from_wavs(aug_dir, pos_feat)
    print(f"OK: Позитивные features: {pos_feat}")

    neg_feat_path = resolve_path(cfg, "negatives_features")
    if not neg_feat_path.exists():
        raise FileNotFoundError(
            f"Нет {neg_feat_path}. Сначала: python scripts/admin_build_negatives.py --features-only"
        )

    out_user = resolve_path(cfg, "user_output") / model_name
    out_user.mkdir(parents=True, exist_ok=True)
    onnx_dst = out_user / f"{model_name}.onnx"

    if skip_oww_train:
        yaml_path = generate_oww_training_yaml(model_name, work, pos_feat, neg_feat_path, cfg)
        shutil.copy2(yaml_path, out_user / "training_config.yaml")
        print(f"OK: Подготовлено (train пропущен): {out_user}")
        return out_user

    train_and_export_onnx(
        pos_feat,
        neg_feat_path,
        onnx_dst,
        layer_size=cfg["training"]["layer_size"],
        steps=cfg["training"]["steps"],
        max_negative_weight=cfg["training"]["max_negative_weight"],
        val_split=1.0 - cfg["training"]["train_split"],
    )

    meta = {
        "phrase": phrase,
        "model_name": model_name,
        "source_recordings": len(validation.files),
        "augmented_clips": n_aug,
        "onnx": str(onnx_dst.name),
    }
    (out_user / "meta.yaml").write_text(yaml.dump(meta, allow_unicode=True), encoding="utf-8")
    print(f"OK: Модель: {onnx_dst}")
    return onnx_dst
