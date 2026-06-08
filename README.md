# custom-wake-word

Train a custom wake phrase from a handful of recordings and run detection on CPU with ONNX.

I wanted something like Porcupine but without a paid API — record your phrase a few times, train locally, ship a small `.onnx` file. Uses [openWakeWord](https://github.com/dscripka/openWakeWord) for embeddings; the rest is augmentation + a tiny DNN on top.

Works offline. Tested on Windows with an RTX GPU for training; inference is CPU-only.

## Install

```bash
git clone https://github.com/IVNsell/custom-wake-word.git
cd custom-wake-word
python -m venv .venv

# Windows
.\.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
pip install sounddevice   # only for listen.py
```

## Usage

**1. Noise corpus (do once)**

Put background audio under `data/negatives/` — rain, traffic, speech, music, whatever you have. Details in [data/negatives/README.md](data/negatives/README.md).

```bash
python scripts/admin_build_negatives.py
```

Writes `data/features/shared_negatives.npy`. On a big corpus this takes a while. If wav prep already finished:

```bash
python scripts/admin_build_negatives.py --features-only
```

**2. Record your phrase**

Drop WAV files into `workspace/recordings/`:

| Rule | Value |
|------|-------|
| Clips | 3–10 files |
| Duration | 0.6–2.5 s each |
| Content | only the wake phrase, no background noise |
| Format | WAV, 16 kHz mono preferred (stereo 48 kHz ok) |

```bash
python scripts/validate_recordings.py "aizek"
```

**3. Train**

```bash
python scripts/train_user_phrase.py "aizek"
```

Output: `output/user_models/aizek/aizek.onnx`

**4. Listen**

```bash
python scripts/listen.py "output/user_models/aizek/aizek.onnx" --trigger-frames 1 --threshold 0.6
```

**5. Benchmark** (optional)

```bash
python scripts/benchmark_latency.py "output/user_models/aizek/aizek.onnx"
```

On my machine inference is ~2 ms per chunk; end-to-end trigger is roughly 80–250 ms depending on `trigger_frames`.

## Pipeline

```
recordings (3–10 wav)
    → augment (~×80) + mix with negatives
    → openWakeWord embeddings
    → train DNN on shared_negatives.npy
    → phrase.onnx
```

| who | does what |
|-----|-----------|
| user | records their wake phrase |
| platform | keeps `data/negatives/` (noise + speech corpus) |
| scripts | augment, train, export onnx |

## Layout

```
config/default.yaml
data/negatives/          # your noise files (not in git)
data/features/           # shared_negatives.npy
workspace/recordings/    # user wavs
output/user_models/      # trained models
catalog/                 # optional pre-trained onnx
scripts/                 # cli
src/wakeword/            # library
```

## Scripts

| script | what it does |
|--------|--------------|
| `admin_build_negatives.py` | index `data/negatives/` → `shared_negatives.npy` |
| `admin_build_negatives.py --features-only` | embeddings only (wav prep already done) |
| `validate_recordings.py "phrase"` | check recordings before train |
| `train_user_phrase.py "phrase"` | augment + train + export onnx |
| `listen.py model.onnx` | live mic test |
| `benchmark_latency.py model.onnx` | p50/p95 latency |

## Python

```python
import sys
from pathlib import Path

sys.path.insert(0, "src")

from wakeword.service import WakeWordPlatform
from wakeword.inference import WakeWordEngine

platform = WakeWordPlatform()
result = platform.train_user_model("aizek", Path("workspace/recordings"))

engine = WakeWordEngine("output/user_models/aizek/aizek.onnx", threshold=0.6, trigger_frames=1)
score, fired = engine.process_chunk(audio_int16_16khz)
```

## Inference tuning

`config/default.yaml`:

```yaml
inference:
  threshold: 0.6
  trigger_frames: 1   # 1 = fast (~80ms), 3 = fewer false triggers
  refractory_sec: 2.0
```

| `trigger_frames` | latency | false triggers |
|------------------|---------|----------------|
| 1 | ~80 ms | more |
| 2 | ~160 ms | medium |
| 3 | ~240 ms | fewer |

Rough formula: `trigger_frames × 80ms` audio buffer + a few ms onnx.

## Requirements

| Component | Minimum |
|-----------|---------|
| Python | 3.10+ |
| OS | Windows / Linux |
| RAM | 8 GB (16 GB if indexing a huge negative set) |
| GPU | optional, speeds up feature extraction |
| Mic | only for `listen.py` |

## License

MIT — see [LICENSE](LICENSE). openWakeWord is Apache 2.0.
