# Custom Wake Word

Train your own **short wake phrase** (like Porcupine) with **3–10 voice recordings**, run inference on **CPU via ONNX** (~2 ms per chunk), fully **offline**.

Built for voice assistants: users record a custom trigger word; you maintain a shared noise corpus once; everyone gets a personal `.onnx` model.

---

## What is this?

A Python toolkit that lets you:

1. **Record** a short phrase 3–10 times (e.g. "hey nova", "aizek")
2. **Train** a tiny wake-word model locally (~6–10 min)
3. **Run** always-on detection from the microphone (~80–250 ms latency)

No cloud. No subscription. Open source.

---

## Quick start

### 1. Install

```bash
git clone https://github.com/IVNsell/custom-wake-word.git
cd custom-wake-word

python -m venv .venv

# Windows
.\.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
pip install sounddevice   # microphone test only
```

### 2. Build the noise corpus (admin, once)

Drop background audio into `data/negatives/` — see [data/negatives/README.md](data/negatives/README.md).

```bash
python scripts/admin_build_negatives.py
```

This can take **hours** with a large corpus. If preparation finished but features failed:

```bash
python scripts/admin_build_negatives.py --features-only
```

Output: `data/features/shared_negatives.npy`

### 3. Record your wake phrase

Put **3–10** WAV files in `workspace/recordings/`:

| Rule | Value |
|------|-------|
| Clips | 3–10 files |
| Duration | 0.6–2.5 seconds each |
| Content | **Only** the wake phrase, no background noise |
| Format | WAV, 16 kHz mono preferred |

```bash
python scripts/validate_recordings.py "aizek"
```

### 4. Train

```bash
python scripts/train_user_phrase.py "aizek"
```

Model output:

```
output/user_models/aizek/aizek.onnx
```

### 5. Test with microphone

```bash
python scripts/listen.py "output/user_models/aizek/aizek.onnx" --trigger-frames 1 --threshold 0.6
```

Say your phrase — you should see `>>> WAKE!`.

### 6. Benchmark speed

```bash
python scripts/benchmark_latency.py "output/user_models/aizek/aizek.onnx"
```

Typical results: **~2 ms** inference (p50), **~80–250 ms** end-to-end depending on settings.

---

## How it works

```
User recordings (3–10 WAV)
        │
        ▼
Augmentation ×80 + mix with platform noise
        │
        ▼
openWakeWord embeddings → positive_features.npy
        │
        ▼
Train small DNN + shared_negatives.npy
        │
        ▼
phrase.onnx → CPU inference ~2 ms / 80 ms chunk
```

| Role | Responsibility |
|------|----------------|
| **End user** | Records only their short wake phrase |
| **Platform (you)** | Maintains `data/negatives/` — hours of noise & speech |
| **System** | Augments, trains, exports ONNX for the assistant |

**Detection latency:**

```
≈ trigger_frames × 80 ms + ~3 ms inference
```

Configure in [config/default.yaml](config/default.yaml).

---

## Project structure

```
custom-wake-word/
├── config/default.yaml       # limits, training, inference
├── data/
│   ├── negatives/            # noise corpus (audio not in git)
│   └── features/             # shared_negatives.npy (generated)
├── workspace/recordings/     # user WAV files
├── catalog/                  # pre-trained .onnx models
├── output/user_models/       # trained models
├── scripts/                  # CLI tools
└── src/wakeword/             # Python SDK
```

---

## CLI reference

| Command | Purpose |
|---------|---------|
| `admin_build_negatives.py` | Index `data/negatives/` → `.npy` |
| `admin_build_negatives.py --features-only` | Embeddings only (staging ready) |
| `validate_recordings.py "phrase"` | Check 3–10 WAV files |
| `train_user_phrase.py "phrase"` | Augment + train + export ONNX |
| `listen.py model.onnx` | Live microphone test |
| `benchmark_latency.py model.onnx` | Measure p50/p99 latency |

---

## Python API

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path("path/to/custom-wake-word/src")))

from wakeword.service import WakeWordPlatform
from wakeword.inference import WakeWordEngine

# Train
platform = WakeWordPlatform()
result = platform.train_user_model("aizek", Path("workspace/recordings"))
if result.success:
    print(result.model_path)

# Inference
engine = WakeWordEngine(
    "output/user_models/aizek/aizek.onnx",
    threshold=0.6,
    trigger_frames=1,
)
score, triggered = engine.process_chunk(audio_int16_16khz)
```

---

## Tuning speed vs accuracy

In `config/default.yaml`:

```yaml
inference:
  threshold: 0.6
  trigger_frames: 1   # 1 ≈ 80 ms, 2 ≈ 160 ms, 3 ≈ 240 ms
  refractory_sec: 2.0 # cooldown between detections
```

| `trigger_frames` | Latency | False activations |
|------------------|---------|-------------------|
| 1 | ~80 ms | Higher risk |
| 2 | ~160 ms | Balanced |
| 3 | ~240 ms | More stable |

---

## Requirements

| Component | Minimum |
|-----------|---------|
| Python | 3.10+ |
| OS | Windows / Linux |
| RAM | 8 GB (16 GB for large noise corpus) |
| GPU | Optional (speeds up feature extraction) |
| Microphone | For `listen.py` |

---

## Free datasets for `data/negatives/`

- [M-AILABS](https://github.com/i-celeste-aurora/m-ailabs-dataset) — speech in many languages
- [Mozilla Common Voice](https://github.com/common-voice/cv-dataset)

For noise: UrbanSound8K, ESC-50, or record your own environment.

---

## What gets committed to git

**Included:** code, docs, config, `LICENSE`

**Excluded** (via `.gitignore`): `data/negatives/*.wav`, `*.npy`, recordings, trained models, `.venv`

---

## Limitations

- openWakeWord embeddings work best on **English**; other languages need more recordings and a larger noise corpus
- Porcupine-level false-alarm rates require **8+ hours** of background testing
- Large noise indexing can take **hours**

---

## License

MIT — see [LICENSE](LICENSE).  
Uses [openWakeWord](https://github.com/dscripka/openWakeWord) (Apache 2.0) for embeddings and inference.
