# custom-wake-word

Train a custom wake phrase from a handful of recordings and run offline detection on CPU.

Works for **any** short phrase — no hard-coded word lists. Each user trains on their own recordings; the shared noise corpus (`data/negatives/`) provides general background negatives (traffic, speech, room noise, etc.).

## How detection works

**Default (recommended): wake model only**

```
microphone → openWakeWord embeddings → small ONNX DNN → WAKE
```

**Optional: two-stage mode**

```
microphone → wake model (fast candidate) → phonetic verifier (MFCC + DTW) → WAKE / reject
```

The verifier compares live audio against the user's own reference recordings. It is **optional** — use it when the wake model still confuses very similar short words. Most setups work with the model alone after training.

During training the platform **auto-mines confusers** from `shared_negatives.npy`: fragments in the general corpus that look acoustically close to your phrase. No need to record specific "bad" words like similar names.

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
| Validate | `python scripts/validate_recordings.py "your phrase"` | Checks count, duration, format |
| Train | `python scripts/train_user_phrase.py "your phrase"` | `output/user_models/<slug>/<slug>.onnx` |
| Live (model) | `python scripts/listen.py "output/user_models/aizek/aizek.onnx" --trigger-frames 2 --threshold 0.9` | Microphone wake detection |
| Live (+ verifier) | add `--verify-recordings workspace/recordings` to the command above | Extra phonetic check |
| Offline test | `python scripts/test_verifier.py "output/user_models/aizek/aizek.onnx"` | Test model ± verifier on WAV files |

Replace `aizek` with any phrase you trained.

### 1. Noise corpus (do once)

Put background audio under `data/negatives/` — rain, traffic, speech, music. Details in [data/negatives/README.md](data/negatives/README.md).

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

Output: `output/user_models/aizek/aizek.onnx` (overwrites previous model in that folder).

Training steps:

1. Augment your recordings (time stretch, pitch, gain, noise mixing)
2. Extract streaming-compatible openWakeWord features
3. Auto-mine confusers from the shared negative corpus
4. Train a small DNN and export ONNX

Retraining the same phrase does **not** require deleting the old model — the script overwrites it.

**Optional advanced:** extra WAVs of known confusing words in `workspace/hard_negatives/`:

```bash
python scripts/train_user_phrase.py "aizek" --hard-negatives workspace/hard_negatives
```

Not required for a universal setup. If you later train on a different phrase (e.g. `alex`), you do not need to change anything in the platform negatives.

### 4. Listen (live mic)

**Model only** — start here:

```bash
python scripts/listen.py "output/user_models/aizek/aizek.onnx" --trigger-frames 2 --threshold 0.9
```

**With optional phonetic verifier:**

```bash
python scripts/listen.py "output/user_models/aizek/aizek.onnx" \
  --trigger-frames 2 --threshold 0.9 \
  --verify-recordings "workspace/recordings"
```

The verifier runs in a background thread (does not block the mic). Verification uses only your positive reference recordings by default.

Optional anti-references (advanced):

```bash
python scripts/listen.py "output/user_models/aizek/aizek.onnx" \
  --trigger-frames 2 --threshold 0.9 \
  --verify-recordings "workspace/recordings" \
  --verify-negatives "workspace/hard_negatives"
```

### 5. Test offline

Model + verifier on saved WAVs:

```bash
python scripts/test_verifier.py "output/user_models/aizek/aizek.onnx" --positives "workspace/recordings"
```

With optional test negatives (evaluation only):

```bash
python scripts/test_verifier.py "output/user_models/aizek/aizek.onnx" \
  --positives "workspace/recordings" \
  --test-negatives "workspace/hard_negatives"
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
    → auto-mine confusers from shared_negatives.npy
    → DNN training (positives vs platform negatives + confusers)
    → phrase.onnx
    → live wake score → WAKE
    → (optional) MFCC/DTW phonetic verification → WAKE / reject
```

| who | does what |
|-----|-----------|
| user | records their wake phrase (any language / word) |
| platform | keeps `data/negatives/` (general noise + speech corpus) |
| training | mines confusers automatically — no named-word lists |
| wake model | fast candidate detection on CPU |
| verifier (optional) | second-stage check against user recordings |

## Layout

```
config/default.yaml
data/negatives/           # general noise/speech (not in git)
data/features/            # shared_negatives.npy
workspace/recordings/     # user wavs for training
workspace/hard_negatives/ # optional, advanced only
output/user_models/       # trained models
catalog/                  # optional pre-trained onnx
scripts/                  # cli
src/wakeword/             # library
```

## Scripts

| script | what it does |
|--------|--------------|
| `admin_build_negatives.py` | index `data/negatives/` → `shared_negatives.npy` |
| `admin_build_negatives.py --features-only` | embeddings only (wav prep already done) |
| `validate_recordings.py "phrase"` | check recordings before train |
| `train_user_phrase.py "phrase"` | augment + train + export onnx |
| `listen.py model.onnx` | live mic test (verifier optional) |
| `test_verifier.py model.onnx` | offline WAV test with optional verifier |
| `benchmark_latency.py model.onnx` | p50/p95 latency |

## Python API

```python
import sys
from pathlib import Path

sys.path.insert(0, "src")

from wakeword.service import WakeWordPlatform
from wakeword.inference import WakeWordEngine
from wakeword.verifier import PhraseVerifier  # optional

platform = WakeWordPlatform()
result = platform.train_user_model("aizek", Path("workspace/recordings"))

# Model only
engine = WakeWordEngine(
    "output/user_models/aizek/aizek.onnx",
    threshold=0.9,
    trigger_frames=2,
)
score, fired = engine.process_chunk(audio_int16_16khz)

# Optional verifier
verifier = PhraseVerifier("workspace/recordings")
engine = WakeWordEngine(
    "output/user_models/aizek/aizek.onnx",
    threshold=0.9,
    trigger_frames=2,
    verifier=verifier,
    defer_verification=True,
)
score, fired = engine.process_chunk(audio_int16_16khz)
```

## Tuning

### Wake model (`config/default.yaml` → `inference`)

```yaml
inference:
  threshold: 0.9        # use 0.9 in listen.py for trained models
  trigger_frames: 2       # 1 = fast (~80ms), 3 = fewer false triggers
  refractory_sec: 2.0     # cooldown after WAKE
```

| `trigger_frames` | latency | false triggers |
|------------------|---------|----------------|
| 1 | ~80 ms | more |
| 2 | ~160 ms | medium |
| 3 | ~240 ms | fewer |

Rough formula: `trigger_frames × 80 ms` + a few ms ONNX.

### Training (`config/default.yaml` → `training`)

| key | default | purpose |
|-----|---------|---------|
| `mine_confusers` | `true` | auto-find hard negatives in shared corpus |
| `confuser_top_k` | `2500` | how many confuser rows to keep |
| `confuser_weight` | `4000` | loss weight for confusers vs plain negatives |
| `confuser_batch` | `384` | confuser samples per training batch |
| `max_negative_weight` | `1500` | loss weight for general negatives |
| `steps` | `35000` | training steps |

### Verifier (listen.py flags, optional)

| flag | default | purpose |
|------|---------|---------|
| `--verify-threshold` | `0.28` | minimum phonetic match score |
| `--verify-segment-threshold` | `0.15` | minimum per-segment score |
| `--verify-trust-model` | `0.95` | when model is this confident, relax verify bar |
| `--verify-trust-threshold` | `0.24` | minimum verify score in trust mode |

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
