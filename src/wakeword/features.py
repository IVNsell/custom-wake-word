"""Extract openWakeWord embeddings from WAV files into .npy feature arrays."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm


def _patch_torchaudio_for_speechbrain() -> None:
    """Compat shim: speechbrain 0.5.x expects set_audio_backend removed in torchaudio 2.4+."""
    import torchaudio

    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda _backend: None  # type: ignore[attr-defined]
    if not hasattr(torchaudio, "get_audio_backend"):
        torchaudio.get_audio_backend = lambda: "soundfile"  # type: ignore[attr-defined]
    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]  # type: ignore[attr-defined]


def _stack_clips(clips: list[np.ndarray], clip_size: int) -> np.ndarray:
    """Pad/truncate clips to fixed length (replaces openwakeword.data.stack_clips)."""
    out = np.zeros((len(clips), clip_size), dtype=np.int16)
    for i, clip in enumerate(clips):
        n = min(len(clip), clip_size)
        out[i, :n] = clip[:n]
    return out


def _require_openwakeword():
    try:
        import openwakeword  # noqa: F401
        from openwakeword import utils
    except ImportError as e:
        raise ImportError("Install dependencies: pip install -r requirements.txt") from e
    return utils


def extract_features_from_wavs(
    wav_dir: Path,
    output_npy: Path,
    clip_size_seconds: float = 2.0,
    batch_size: int = 32,
) -> tuple[int, tuple]:
    """
    Extract openWakeWord embeddings from a folder of WAV files.
    Downloads embedding models on first run.
    """
    _patch_torchaudio_for_speechbrain()
    utils = _require_openwakeword()
    import openwakeword

    wavs = sorted(wav_dir.glob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"No WAV files in {wav_dir}")

    oww_root = Path(openwakeword.__file__).parent
    utils.download_models(model_names=["embedding"], target_directory=str(oww_root))

    from openwakeword.utils import AudioFeatures

    F = AudioFeatures()
    clip_samples = int(16000 * clip_size_seconds)
    rows: list[np.ndarray] = []

    for i in tqdm(range(0, len(wavs), batch_size), desc="features"):
        batch_paths = wavs[i : i + batch_size]
        clips = []
        for p in batch_paths:
            import soundfile as sf

            data, sr = sf.read(p, dtype="int16")
            if data.ndim > 1:
                data = data.mean(axis=1).astype(np.int16)
            if sr != 16000:
                from scipy import signal

                data = signal.resample(data, int(len(data) * 16000 / sr)).astype(np.int16)
            clips.append(data)
        stacked = _stack_clips(clips, clip_size=clip_samples)
        feats = F.embed_clips(stacked, batch_size=batch_size)
        rows.append(feats)

    features = np.vstack(rows).astype(np.float32)
    output_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_npy, features)
    return features.shape[0], features.shape[1:]


def build_negative_features(
    negatives_root: Path,
    output_npy: Path,
    max_files: int | None = None,
    *,
    features_only: bool = False,
) -> None:
    """
    Index platform noise corpus into shared_negatives.npy.
    Slices long files into 2-second chunks, then extracts embeddings.
    """
    staging = output_npy.parent / "_staging_neg_wav"
    staging.mkdir(parents=True, exist_ok=True)

    existing = list(staging.glob("*.wav"))
    if features_only:
        if not existing:
            raise FileNotFoundError(
                f"--features-only: no WAV in {staging}. Run without this flag first."
            )
        print(f"Skipping prepare: using {len(existing)} staged chunks")
    else:
        exts = {".wav", ".flac", ".ogg", ".mp3"}
        files = [p for p in negatives_root.rglob("*") if p.suffix.lower() in exts]
        if not files:
            raise FileNotFoundError(f"Add audio to {negatives_root} (see README).")
        if max_files:
            files = files[:max_files]

        import soundfile as sf
        from scipy import signal

        idx = 0
        for src in tqdm(files, desc="prepare negatives"):
            try:
                data, sr = sf.read(src, dtype="float32", always_2d=True)
            except Exception:
                continue
            mono = data.mean(axis=1)
            if sr != 16000:
                mono = signal.resample(mono, int(len(mono) * 16000 / sr))
            chunk_len = 16000 * 2
            for start in range(0, max(1, len(mono) - chunk_len), chunk_len):
                chunk = mono[start : start + chunk_len]
                if len(chunk) < 8000:
                    continue
                out = staging / f"neg_{idx:06d}.wav"
                sf.write(out, chunk, 16000, subtype="PCM_16")
                idx += 1
                if max_files and idx >= max_files * 10:
                    break

    extract_features_from_wavs(staging, output_npy)
    # Remove temporary staging files
    for f in staging.glob("*.wav"):
        f.unlink()
    staging.rmdir()
