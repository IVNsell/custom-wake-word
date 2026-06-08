from __future__ import annotations

import time
from collections import deque
from pathlib import Path


class WakeWordEngine:
    """
    Инференс для ассистента: ONNX через openWakeWord (быстрый CPU-путь).
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        threshold: float = 0.5,
        trigger_frames: int = 3,
        refractory_sec: float = 2.0,
        inference_framework: str = "onnx",
    ):
        from openwakeword.model import Model

        self.model_path = Path(model_path)
        self.threshold = threshold
        self.trigger_frames = trigger_frames
        self.refractory_sec = refractory_sec
        self._scores: deque[float] = deque(maxlen=trigger_frames)
        self._last_trigger = 0.0

        self._oww = Model(
            wakeword_models=[str(self.model_path)],
            inference_framework=inference_framework,
        )
        self._model_key = list(self._oww.models.keys())[0]

    def process_chunk(self, audio_int16) -> tuple[float, bool]:
        """
        audio_int16: numpy int16, 16 kHz, длина кратна 1280 (или как у openWakeWord).
        Возвращает (score, triggered).
        """
        import numpy as np

        if not isinstance(audio_int16, np.ndarray):
            audio_int16 = np.asarray(audio_int16, dtype=np.int16)

        pred = self._oww.predict(audio_int16)
        score = float(pred.get(self._model_key, 0.0))
        self._scores.append(score)

        now = time.monotonic()
        if now - self._last_trigger < self.refractory_sec:
            return score, False

        if len(self._scores) == self.trigger_frames and all(s >= self.threshold for s in self._scores):
            self._last_trigger = now
            self._scores.clear()
            return score, True

        return score, False

    @classmethod
    def from_catalog(cls, model_id: str, catalog_manifest: Path | None = None):
        import json

        from .config import ROOT

        manifest = catalog_manifest or (ROOT / "catalog" / "manifest.json")
        data = json.loads(manifest.read_text(encoding="utf-8"))
        entry = next((m for m in data["models"] if m["id"] == model_id), None)
        if not entry:
            raise KeyError(f"Модель {model_id} не в каталоге")
        path = ROOT / "catalog" / "models" / entry["file"]
        return cls(path, threshold=entry.get("threshold", 0.5))
