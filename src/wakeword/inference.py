"""Real-time wake word inference via openWakeWord ONNX runtime."""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path

import numpy as np


class WakeWordEngine:
    """
    Lightweight inference wrapper for voice assistants.
    Uses openWakeWord Model with ONNX on CPU (~2 ms per 80 ms chunk).
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        threshold: float = 0.5,
        trigger_frames: int = 3,
        refractory_sec: float = 2.0,
        inference_framework: str = "onnx",
        verifier=None,
        verifier_window_sec: float = 2.5,
        defer_verification: bool = False,
        verify_capture_chunks: int = 5,
        verify_capture_cooldown_sec: float = 0.45,
        verify_snapshot_sec: float = 1.5,
    ):
        from openwakeword.model import Model

        self.model_path = Path(model_path)
        self.threshold = threshold
        self.trigger_frames = trigger_frames
        self.refractory_sec = refractory_sec
        self.verifier = verifier
        self.defer_verification = defer_verification
        self.verify_capture_chunks = verify_capture_chunks
        self.verify_capture_cooldown_sec = verify_capture_cooldown_sec
        self.verify_snapshot_sec = verify_snapshot_sec
        self.last_verification = None
        self.pending_candidate: np.ndarray | None = None
        self.pending_score: float = 0.0
        self._capture_active = False
        self._capture_remaining = 0
        self._capture_best_score = 0.0
        self._capture_snapshot: np.ndarray | None = None
        self._capture_cooldown_until = 0.0
        # Rolling window of recent scores for debounced triggering
        self._scores: deque[float] = deque(maxlen=trigger_frames)
        self._last_trigger = 0.0
        self._audio_buffer: deque[int] = deque(maxlen=int(16000 * verifier_window_sec))

        self._oww = Model(
            wakeword_models=[str(self.model_path)],
            inference_framework=inference_framework,
        )
        self._model_key = list(self._oww.models.keys())[0]

    def process_chunk(self, audio_int16) -> tuple[float, bool]:
        """
        Process one audio chunk.

        Args:
            audio_int16: numpy int16, 16 kHz, typically 1280 samples (~80 ms).

        Returns:
            (score, triggered) — triggered is True when wake word detected.
        """
        if not isinstance(audio_int16, np.ndarray):
            audio_int16 = np.asarray(audio_int16, dtype=np.int16)
        audio_int16 = audio_int16.astype(np.int16, copy=False)
        self._audio_buffer.extend(audio_int16.tolist())
        self.last_verification = None

        pred = self._oww.predict(audio_int16)
        score = float(pred.get(self._model_key, 0.0))
        self._scores.append(score)

        now = time.monotonic()
        # Refractory period prevents duplicate triggers for the same utterance
        if now - self._last_trigger < self.refractory_sec:
            return score, False

        if self._capture_active:
            if score >= self.threshold * 0.85:
                self._capture_snapshot = np.asarray(self._audio_buffer, dtype=np.int16).copy()
                self._capture_best_score = max(self._capture_best_score, score)
            self._capture_remaining -= 1
            if self._capture_remaining <= 0 or score < self.threshold * 0.55:
                self._finish_capture()
            return score, False

        if now < self._capture_cooldown_until:
            return score, False

        if len(self._scores) == self.trigger_frames and all(s >= self.threshold for s in self._scores):
            if self.verifier is not None:
                self._scores.clear()
                if self.defer_verification:
                    self._start_capture(score)
                    return score, False

                candidate = np.asarray(self._audio_buffer, dtype=np.int16).copy()
                self.last_verification = self.verifier.verify(candidate)
                if not self.last_verification.accepted:
                    return score, False

            self._last_trigger = now
            self._scores.clear()
            return score, True

        return score, False

    def _start_capture(self, score: float) -> None:
        """Wait a few frames so the full phrase lands in the audio buffer."""
        self._capture_active = True
        self._capture_remaining = self.verify_capture_chunks
        self._capture_best_score = score
        self._capture_snapshot = np.asarray(self._audio_buffer, dtype=np.int16).copy()

    def _finish_capture(self) -> None:
        """Release the best deferred snapshot for async verification."""
        if self.defer_verification and self._capture_snapshot is not None:
            keep = int(16000 * self.verify_snapshot_sec)
            snapshot = self._capture_snapshot[-keep:] if len(self._capture_snapshot) > keep else self._capture_snapshot
            self.pending_candidate = snapshot.copy()
            self.pending_score = self._capture_best_score
        self._capture_active = False
        self._capture_remaining = 0
        self._capture_snapshot = None
        self._capture_best_score = 0.0
        self._capture_cooldown_until = time.monotonic() + self.verify_capture_cooldown_sec

    def consume_pending(self) -> tuple[np.ndarray, float] | None:
        """Take a deferred verifier candidate, if any."""
        if self.pending_candidate is None:
            return None
        snapshot = self.pending_candidate
        score = self.pending_score
        self.pending_candidate = None
        self.pending_score = 0.0
        return snapshot, score

    def apply_verification(self, result, *, accepted: bool | None = None) -> bool:
        """Apply verifier result after async verification."""
        self.last_verification = result
        is_accepted = result.accepted if accepted is None else accepted
        if is_accepted:
            self._last_trigger = time.monotonic()
            return True
        return False

    @classmethod
    def from_catalog(cls, model_id: str, catalog_manifest: Path | None = None):
        """Load a pre-trained model from catalog/manifest.json."""
        import json

        from .config import ROOT

        manifest = catalog_manifest or (ROOT / "catalog" / "manifest.json")
        data = json.loads(manifest.read_text(encoding="utf-8"))
        entry = next((m for m in data["models"] if m["id"] == model_id), None)
        if not entry:
            raise KeyError(f"Model {model_id} not found in catalog")
        path = ROOT / "catalog" / "models" / entry["file"]
        return cls(path, threshold=entry.get("threshold", 0.5))
