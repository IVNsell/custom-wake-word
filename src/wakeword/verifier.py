"""Lightweight phonetic verifier for wake word candidates.

The wake model is fast but can confuse short similar words. This verifier adds
an independent MFCC + DTW check against the user's own reference recordings.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.fftpack import dct
from scipy.spatial.distance import cdist

from .recordings import load_mono_16k


@dataclass
class VerificationResult:
    """Result of comparing candidate audio against reference phrases."""

    accepted: bool
    score: float
    negative_score: float
    best_cost: float
    reference: str
    negative_reference: str
    segment_scores: tuple[float, float, float]


def _trim_silence(audio: np.ndarray, sample_rate: int = 16000, threshold: float = 0.08) -> np.ndarray:
    """Trim leading/trailing silence using short-time RMS."""
    if audio.size == 0:
        return audio

    peak = float(np.max(np.abs(audio))) or 1.0
    audio = (audio / peak).astype(np.float32)
    frame = int(sample_rate * 0.025)
    hop = int(sample_rate * 0.010)
    if len(audio) < frame:
        return audio

    rms = []
    starts = []
    for start in range(0, len(audio) - frame + 1, hop):
        chunk = audio[start : start + frame]
        rms.append(float(np.sqrt(np.mean(chunk**2))))
        starts.append(start)

    rms_arr = np.asarray(rms, dtype=np.float32)
    active = np.where(rms_arr >= max(float(rms_arr.max()) * threshold, 1e-4))[0]
    if active.size == 0:
        return audio

    start = max(0, starts[int(active[0])] - frame)
    end = min(len(audio), starts[int(active[-1])] + frame * 2)
    return audio[start:end].astype(np.float32)


def _mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int = 40,
    f_min: float = 80.0,
    f_max: float = 7600.0,
) -> np.ndarray:
    """Create triangular mel filterbank."""

    def hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
        return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)

    def mel_to_hz(mel: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_points = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(1, n_mels + 1):
        left, center, right = bins[i - 1], bins[i], bins[i + 1]
        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1
        for j in range(left, center):
            if 0 <= j < filters.shape[1]:
                filters[i - 1, j] = (j - left) / (center - left)
        for j in range(center, right):
            if 0 <= j < filters.shape[1]:
                filters[i - 1, j] = (right - j) / (right - center)
    return filters


def _delta(features: np.ndarray) -> np.ndarray:
    """Simple temporal delta features."""
    if len(features) < 3:
        return np.zeros_like(features)
    padded = np.pad(features, ((1, 1), (0, 0)), mode="edge")
    return (padded[2:] - padded[:-2]) * 0.5


def mfcc_features(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """Compute normalized MFCC + delta features."""
    audio = _trim_silence(audio, sample_rate)
    if audio.size == 0:
        return np.zeros((1, 26), dtype=np.float32)

    audio = audio.astype(np.float32)
    audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])

    frame_len = int(sample_rate * 0.025)
    hop = int(sample_rate * 0.010)
    n_fft = 512
    if len(audio) < frame_len:
        audio = np.pad(audio, (0, frame_len - len(audio)))

    frames = []
    window = np.hamming(frame_len).astype(np.float32)
    for start in range(0, len(audio) - frame_len + 1, hop):
        frames.append(audio[start : start + frame_len] * window)
    framed = np.stack(frames).astype(np.float32)

    spectrum = np.abs(np.fft.rfft(framed, n=n_fft)) ** 2
    mel = spectrum @ _mel_filterbank(sample_rate, n_fft).T
    log_mel = np.log(np.maximum(mel, 1e-10))
    coeffs = dct(log_mel, type=2, axis=1, norm="ortho")[:, 1:14]
    feats = np.concatenate([coeffs, _delta(coeffs)], axis=1).astype(np.float32)

    mean = feats.mean(axis=0, keepdims=True)
    std = feats.std(axis=0, keepdims=True) + 1e-6
    return ((feats - mean) / std).astype(np.float32)


def _dtw_cost(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized DTW distance between two feature sequences."""
    if a.size == 0 or b.size == 0:
        return float("inf")
    dist = cdist(a, b, metric="cosine")
    dist = np.nan_to_num(dist, nan=1.0, posinf=1.0, neginf=1.0)
    n, m = dist.shape
    dp = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i, j] = dist[i - 1, j - 1] + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[n, m] / (n + m))


def _split_three(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split features into start/middle/end."""
    n = len(features)
    a = max(1, n // 3)
    b = max(a + 1, (2 * n) // 3)
    return features[:a], features[a:b], features[b:]


def _candidate_windows(
    audio: np.ndarray,
    sample_rate: int = 16000,
    *,
    window_secs: tuple[float, ...] | None = None,
) -> list[np.ndarray]:
    """Build tail-aligned windows from a live mic buffer.

    The wake model can fire while the rolling buffer still contains older
    silence or room noise. Trying several recent windows makes verification
    much more stable than scoring the full buffer at once.
    """
    if audio.size == 0:
        return [audio.astype(np.float32)]

    audio = audio.astype(np.float32)
    if window_secs is None:
        window_secs = (0.7, 0.9, 1.1, 1.4, 1.8)

    windows: list[np.ndarray] = []
    seen_lengths: set[int] = set()
    for sec in window_secs:
        keep = int(sample_rate * sec)
        chunk = audio[-keep:] if len(audio) > keep else audio
        trimmed = _trim_silence(chunk, sample_rate)
        if trimmed.size == 0:
            continue
        key = int(trimmed.size // (sample_rate * 0.05))
        if key in seen_lengths:
            continue
        seen_lengths.add(key)
        windows.append(trimmed)

    if not windows:
        fallback = audio[-int(sample_rate * 1.2) :] if len(audio) > 0 else audio
        windows.append(_trim_silence(fallback, sample_rate))
    return windows


class PhraseVerifier:
    """MFCC + DTW phrase verifier using user reference recordings."""

    def __init__(
        self,
        reference_dir: str | Path,
        *,
        negative_dir: str | Path | None = None,
        threshold: float = 0.28,
        segment_threshold: float = 0.15,
        negative_margin: float = 0.04,
        sample_rate: int = 16000,
    ):
        self.reference_dir = Path(reference_dir)
        self.threshold = threshold
        self.segment_threshold = segment_threshold
        self.negative_margin = negative_margin
        self.sample_rate = sample_rate
        self.references: list[tuple[str, np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]] = []
        self.negative_references: list[tuple[str, np.ndarray]] = []

        files = sorted(self.reference_dir.glob("*.wav"))
        if not files:
            raise FileNotFoundError(f"No reference WAV files in {self.reference_dir}")

        for path in files:
            audio = load_mono_16k(path, sample_rate)
            feats = mfcc_features(audio, sample_rate)
            self.references.append((path.name, feats, _split_three(feats)))

        ref_samples = []
        for path in files:
            audio = load_mono_16k(path, sample_rate).astype(np.float32) / 32768.0
            trimmed = _trim_silence(audio, sample_rate)
            ref_samples.append(max(trimmed.size, int(sample_rate * 0.4)))
        ref_sec = float(np.median(ref_samples)) / sample_rate
        self._window_secs = tuple(
            sorted(
                {
                    max(0.6, ref_sec * 0.85),
                    max(0.7, ref_sec),
                    max(0.8, ref_sec * 1.15),
                    max(0.9, ref_sec * 1.35),
                    1.4,
                    1.8,
                }
            )
        )
        self._live_window_secs = tuple(
            sorted(
                {
                    max(0.6, ref_sec * 0.85),
                    max(0.75, ref_sec * 1.0),
                    max(0.85, ref_sec * 1.15),
                    max(1.0, ref_sec * 1.35),
                }
            )
        )

        if negative_dir:
            for path in sorted(Path(negative_dir).glob("*.wav")):
                audio = load_mono_16k(path, sample_rate)
                self.negative_references.append((path.name, mfcc_features(audio, sample_rate)))

    @staticmethod
    def _score_from_cost(cost: float) -> float:
        return float(np.exp(-3.0 * cost))

    def _score_reference(
        self,
        feats: np.ndarray,
        cand_segments: tuple[np.ndarray, np.ndarray, np.ndarray],
        name: str,
        ref: np.ndarray,
        ref_segments: tuple[np.ndarray, np.ndarray, np.ndarray],
        *,
        best_negative_score: float,
    ) -> VerificationResult:
        full_cost = _dtw_cost(feats, ref)
        segment_costs = tuple(_dtw_cost(c, r) for c, r in zip(cand_segments, ref_segments))
        segment_scores = tuple(self._score_from_cost(c) for c in segment_costs)
        score = (
            self._score_from_cost(full_cost) * 0.40
            + segment_scores[0] * 0.18
            + segment_scores[1] * 0.22
            + segment_scores[2] * 0.20
        )
        segment_mean = float(np.mean(segment_scores))
        accepted = (
            score >= self.threshold
            and min(segment_scores) >= self.segment_threshold
            and score >= best_negative_score + self.negative_margin
        ) or (
            segment_mean >= self.threshold + 0.02
            and min(segment_scores) >= self.segment_threshold
            and score >= self.threshold - 0.03
            and score >= best_negative_score + self.negative_margin
        )
        return VerificationResult(
            accepted=accepted,
            score=float(score),
            negative_score=float(best_negative_score),
            best_cost=full_cost,
            reference=name,
            negative_reference="",
            segment_scores=segment_scores,
        )

    def verify(self, audio_int16_or_float: np.ndarray, *, fast: bool = False) -> VerificationResult:
        """Return whether candidate audio matches the reference phrase."""
        audio = np.asarray(audio_int16_or_float)
        if audio.dtype.kind in {"i", "u"}:
            audio = audio.astype(np.float32) / 32768.0
        else:
            audio = audio.astype(np.float32)

        window_secs = self._live_window_secs if fast else self._window_secs
        references = self.references
        candidate_windows = _candidate_windows(
            audio,
            self.sample_rate,
            window_secs=window_secs,
        )

        best_negative_score = 0.0
        best_negative_name = ""
        for window in candidate_windows:
            feats = mfcc_features(window, self.sample_rate)
            for name, neg in self.negative_references:
                score = self._score_from_cost(_dtw_cost(feats, neg))
                if score > best_negative_score:
                    best_negative_score = score
                    best_negative_name = name

        best: VerificationResult | None = None
        best_accepted: VerificationResult | None = None
        for window in candidate_windows:
            feats = mfcc_features(window, self.sample_rate)
            cand_segments = _split_three(feats)
            for name, ref, ref_segments in references:
                result = self._score_reference(
                    feats,
                    cand_segments,
                    name,
                    ref,
                    ref_segments,
                    best_negative_score=best_negative_score,
                )
                result.negative_reference = best_negative_name
                if best is None or result.score > best.score:
                    best = result
                if result.accepted and (
                    best_accepted is None or result.score >= best_accepted.score
                ):
                    best_accepted = result
                if (
                    fast
                    and best_accepted is not None
                    and best_accepted.score >= self.threshold + 0.06
                ):
                    assert best is not None
                    return best_accepted

        assert best is not None
        return best_accepted or best
