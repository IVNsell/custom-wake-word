# custom-wake-word

Train a custom wake phrase from a handful of recordings and run offline detection on CPU.

The detector uses two stages:

1. **Wake model** — openWakeWord embeddings + a small ONNX model for fast candidate detection.
2. **Phonetic verifier** — MFCC + DTW comparison against the user's own recordings to reject similar short words.

This helps with cases where the neural wake model thinks two short words are close, for example `aizek` vs `alex`.

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

| Step | Command | Result |
|------|---------|--------|
| Record | Put 3–10 WAV files in `workspace/recordings/` | User reference phrase |
| Validate | `python scripts/validate_recordings.py "aizek"` | Checks count, duration, format |
| Train | `python scripts/train_user_phrase.py "aizek"` | `output/user_models/aizek/aizek.onnx` |
| Offline test | `python scripts/test_verifier.py "output/user_models/aizek/aizek.onnx"` | Shows accepted/rejected WAVs |
| Live use | `python scripts/listen.py "output/user_models/aizek/aizek.onnx" --trigger-frames 2 --threshold 0.9 --verify-recordings "workspace/recordings"` | Microphone wake detection |

### 1. Noise corpus (do once)

Put background audio under `data/negatives/` — rain, traffic, speech, music, whatever you have. Details in [data/negatives/README.md](data/negatives/README.md).

```bash
python scripts/admin_build_negatives.py
```

Writes `data/features/shared_negatives.npy`. On a big corpus this takes a while. If wav prep already finished:

```bash
python scripts/admin_build_negatives.py --features-only
```

### 2. Record your phrase

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

### 3. Train

```bash
python scripts/train_user_phrase.py "aizek"
```

Output: `output/user_models/aizek/aizek.onnx`

Optional: put similar non-wake words in `workspace/hard_negatives/` and train with:

```bash
python scripts/train_user_phrase.py "aizek" --hard-negatives workspace/hard_negatives
```

Hard negatives are not required. They are only useful when you want extra separation from known confusing words.

### 4. Test offline

```bash
python scripts/test_verifier.py "output/user_models/aizek/aizek.onnx" --positives "workspace/recordings"
```

If you have test negatives (for example similar words that should not wake):

```bash
python scripts/test_verifier.py "output/user_models/aizek/aizek.onnx" --positives "workspace/recordings" --test-negatives "workspace/hard_negatives"
```

### 5. Listen

```bash
python scripts/listen.py "output/user_models/aizek/aizek.onnx" --trigger-frames 2 --threshold 0.9 --verify-recordings "workspace/recordings"
```

The verifier is positive-only by default. Optional anti-references:

```bash
python scripts/listen.py "output/user_models/aizek/aizek.onnx" --trigger-frames 2 --threshold 0.9 --verify-recordings "workspace/recordings" --verify-negatives "workspace/hard_negatives"
```

### 6. Benchmark

```bash
python scripts/benchmark_latency.py "output/user_models/aizek/aizek.onnx"
```

## Pipeline

```
recordings (3–10 wav)
    → augmentation + noise mixing
    → streaming-compatible openWakeWord features
    → DNN training against shared negatives
    → phrase.onnx
    → live wake score
    → MFCC/DTW phonetic verification
    → WAKE / reject
```

| who | does what |
|-----|-----------|
| user | records their wake phrase |
| platform | keeps `data/negatives/` (noise + speech corpus) |
| wake model | quickly finds candidate wake events |
| verifier | checks phrase shape against user recordings |

## Layout

```
config/default.yaml
data/negatives/          # your noise files (not in git)
data/features/           # shared_negatives.npy
workspace/recordings/    # user wavs
workspace/hard_negatives/ # optional confusing words for testing/training
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
| `test_verifier.py model.onnx` | offline positive/negative test |
| `listen.py model.onnx` | live mic test with optional verifier |
| `benchmark_latency.py model.onnx` | p50/p95 latency |

## Python

```python
import sys
from pathlib import Path

sys.path.insert(0, "src")

from wakeword.service import WakeWordPlatform
from wakeword.inference import WakeWordEngine
from wakeword.verifier import PhraseVerifier

platform = WakeWordPlatform()
result = platform.train_user_model("aizek", Path("workspace/recordings"))

verifier = PhraseVerifier("workspace/recordings")
engine = WakeWordEngine(
    "output/user_models/aizek/aizek.onnx",
    threshold=0.9,
    trigger_frames=2,
    verifier=verifier,
)
score, fired = engine.process_chunk(audio_int16_16khz)
```

## Inference tuning

`config/default.yaml`:

```yaml
inference:
  threshold: 0.9
  trigger_frames: 2   # 1 = fast (~80ms), 3 = fewer false triggers
  refractory_sec: 2.0
```

| `trigger_frames` | latency | false triggers |
|------------------|---------|----------------|
| 1 | ~80 ms | more |
| 2 | ~160 ms | medium |
| 3 | ~240 ms | fewer |

Rough formula: `trigger_frames × 80ms` audio buffer + a few ms onnx.

The phonetic verifier runs only after the wake model finds a candidate, so it adds a small cost only near possible wake events.

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
