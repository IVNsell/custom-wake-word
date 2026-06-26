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


class PhraseVerifier:
    """MFCC + DTW phrase verifier using user reference recordings."""

    def __init__(
        self,
        reference_dir: str | Path,
        *,
        negative_dir: str | Path | None = None,
        threshold: float = 0.34,
        segment_threshold: float = 0.20,
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

        if negative_dir:
            for path in sorted(Path(negative_dir).glob("*.wav")):
                audio = load_mono_16k(path, sample_rate)
                self.negative_references.append((path.name, mfcc_features(audio, sample_rate)))

    @staticmethod
    def _score_from_cost(cost: float) -> float:
        return float(np.exp(-3.0 * cost))

    def verify(self, audio_int16_or_float: np.ndarray) -> VerificationResult:
        """Return whether candidate audio matches the reference phrase."""
        audio = np.asarray(audio_int16_or_float)
        if audio.dtype.kind in {"i", "u"}:
            audio = audio.astype(np.float32) / 32768.0
        else:
            audio = audio.astype(np.float32)

        feats = mfcc_features(audio, self.sample_rate)
        cand_segments = _split_three(feats)

        best_negative_score = 0.0
        best_negative_name = ""
        for name, neg in self.negative_references:
            score = self._score_from_cost(_dtw_cost(feats, neg))
            if score > best_negative_score:
                best_negative_score = score
                best_negative_name = name

        best: VerificationResult | None = None
        for name, ref, ref_segments in self.references:
            full_cost = _dtw_cost(feats, ref)
            segment_costs = tuple(_dtw_cost(c, r) for c, r in zip(cand_segments, ref_segments))
            segment_scores = tuple(self._score_from_cost(c) for c in segment_costs)
            # The end of the word is weighted because aizek/alex differ most there.
            score = (
                self._score_from_cost(full_cost) * 0.55
                + segment_scores[0] * 0.10
                + segment_scores[1] * 0.15
                + segment_scores[2] * 0.20
            )
            accepted = (
                score >= self.threshold
                and min(segment_scores) >= self.segment_threshold
                and score >= best_negative_score + self.negative_margin
            )
            result = VerificationResult(
                accepted=accepted,
                score=float(score),
                negative_score=float(best_negative_score),
                best_cost=full_cost,
                reference=name,
                negative_reference=best_negative_name,
                segment_scores=segment_scores,
            )
            if best is None or result.score > best.score:
                best = result

        assert best is not None
        return best
