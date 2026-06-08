# Custom Wake Word

Open-source платформа wake word на Python: **своя короткая фраза** (3–10 записей), **общий корпус шума**, обучение локально, инференс через **ONNX на CPU** (~2 ms на чанк).

> **Имя папки и GitHub:** локально проект может лежать в `custom-wake-word`, а репозиторий на GitHub называться иначе (например `phrase-wake`) — это нормально. Подробнее: [GITHUB.md](GITHUB.md).

Подходит для голосовых ассистентов: пользователь записывает имя бота, вы один раз собираете фоновый аудио-корпус, модель обучается и работает офлайн.

---

## Возможности

- Обучение **любой короткой фразы** (1–3 слова) из **3–10 WAV** пользователя
- Общий **корпус негативов** (дождь, дорога, музыка, речь на разных языках) — один раз для всех моделей
- Аугментация записей (pitch, tempo, gain, микс с шумом)
- Экспорт **`.onnx`** — совместим с [openWakeWord](https://github.com/dscripka/openWakeWord) inference
- **Каталог** готовых бесплатных фраз (`catalog/`)
- Скорость inference **~2–3 ms** (p50 на CPU), end-to-end **~80–250 ms** (настраивается)
- Python API для встраивания в ассистент

---

## Требования

| Компонент | Минимум |
|-----------|---------|
| Python | **3.10+** |
| ОС | Windows / Linux |
| RAM | 8 GB+ (16 GB для большого корпуса) |
| GPU | Опционально (ускоряет `admin_build_negatives`, train ~6 мин и на CPU) |
| Микрофон | Для `listen.py` + `pip install sounddevice` |

---

## Установка

```bash
git clone https://github.com/YOUR_USERNAME/phrase-wake.git
cd phrase-wake
```

Имя папки после clone = имя репозитория на GitHub. Локально у вас может быть другое имя — на работу не влияет.

python -m venv .venv

# Windows
.\.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
pip install sounddevice   # для проверки с микрофона
```

Проверка GPU (опционально):

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

---

## Быстрый старт (полный цикл)

### Шаг 1. Корпус шума (админ, один раз)

Положите аудио в `data/negatives/` — см. [data/negatives/README.md](data/negatives/README.md).

Подпапки: `rain/`, `road/`, `music/`, `speech_ru/`, `speech_en/`, `speech_other/` и т.д.  
Форматы: `.wav`, `.flac`, `.mp3`, `.ogg`. Имена файлов любые.

```bash
python scripts/admin_build_negatives.py
```

Долго при большом корпусе (часы). Если `prepare` уже прошёл, а упал только `features`:

```bash
python scripts/admin_build_negatives.py --features-only
```

Результат: `data/features/shared_negatives.npy`

### Шаг 2. Записи пользователя (3–10 файлов)

Папка: `workspace/recordings/`

| Параметр | Значение |
|----------|----------|
| Файлов | 3–10 |
| Длина клипа | 0.6–2.5 сек |
| Содержимое | Только wake-фраза, без фона |
| Формат | WAV, лучше 16 kHz mono (48000 stereo тоже ок) |

Проверка:

```bash
python scripts/validate_recordings.py "айзек"
```

### Шаг 3. Обучение

```bash
python scripts/train_user_phrase.py "айзек"
```

~6–10 минут. Модель:

```
output/user_models/айзек/айзек.onnx
```

### Шаг 4. Проверка с микрофона

```bash
python scripts/listen.py "output/user_models/айзек/айзек.onnx"
```

Быстрый режим (~80 ms):

```bash
python scripts/listen.py "output/user_models/айзек/айзек.onnx" --trigger-frames 1 --threshold 0.6
```

### Шаг 5. Замер скорости

```bash
python scripts/benchmark_latency.py "output/user_models/айзек/айзек.onnx"
```

---

## Как это работает

```
Пользователь (3–10 WAV)
        │
        ▼
   Аугментация ×80 + микс с data/negatives/
        │
        ▼
   Embeddings (openWakeWord) → positive_features.npy
        │
        ▼
   Train маленькой DNN + shared_negatives.npy
        │
        ▼
   айзек.onnx  →  CPU inference ~2 ms / 80 ms чанк
```

| Роль | Что делает |
|------|------------|
| **Пользователь** | Записывает только свою короткую фразу |
| **Платформа (вы)** | Собирает `data/negatives/` — часы шума и речи |
| **Система** | Аугментирует, учит ONNX, отдаёт в ассистент |

Задержка до WAKE:

```
≈ trigger_frames × 80 ms + ~3 ms inference
```

Настройки в [config/default.yaml](config/default.yaml).

---

## Структура проекта

```
custom-wake-word/
├── config/default.yaml       # лимиты, train, inference
├── data/
│   ├── negatives/            # ваш корпус шума (в git — только README)
│   └── features/             # shared_negatives.npy (генерируется)
├── workspace/recordings/     # WAV пользователя
├── catalog/                  # готовые .onnx + manifest.json
├── output/user_models/       # обученные модели
├── scripts/
│   ├── admin_build_negatives.py
│   ├── train_user_phrase.py
│   ├── validate_recordings.py
│   ├── listen.py
│   └── benchmark_latency.py
└── src/wakeword/             # Python SDK
    ├── service.py            # API для ассистента
    ├── inference.py          # WakeWordEngine
    ├── train_pipeline.py
    └── ...
```

---

## CLI — все команды

| Команда | Назначение |
|---------|------------|
| `admin_build_negatives.py` | Индексировать `data/negatives/` → `.npy` |
| `admin_build_negatives.py --features-only` | Только embeddings (staging уже готов) |
| `validate_recordings.py "фраза"` | Проверить 3–10 WAV |
| `train_user_phrase.py "фраза"` | Аугментация + train + ONNX |
| `listen.py model.onnx` | Слушать микрофон |
| `benchmark_latency.py model.onnx` | Замер p50/p99 latency |

---

## Встраивание в ассистент (Python)

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path("path/to/custom-wake-word/src")))

from wakeword.service import WakeWordPlatform
from wakeword.inference import WakeWordEngine

# Обучение
platform = WakeWordPlatform()
result = platform.train_user_model("айзек", Path("workspace/recordings"))
if result.success:
    print(result.model_path)

# Инференс
engine = WakeWordEngine(
    "output/user_models/айзек/айзек.onnx",
    threshold=0.6,
    trigger_frames=1,
)
score, fired = engine.process_chunk(audio_int16_16khz)
```

Каталог готовых фраз: `catalog/manifest.json` + `.onnx` в `catalog/models/`.

---

## Каталог готовых фраз

1. Положите `.onnx` в `catalog/models/`
2. Добавьте запись в `catalog/manifest.json`
3. `WakeWordEngine.from_catalog("id")`

---

## Настройка скорости и качества

В `config/default.yaml`:

```yaml
inference:
  frame_samples: 1280   # 80 ms — минимум openWakeWord
  threshold: 0.6
  trigger_frames: 1     # 1=~80ms, 2=~160ms, 3=~240ms
  refractory_sec: 2.0   # пауза между повторными WAKE
```

| `trigger_frames` | Задержка | Риск ложных |
|------------------|----------|-------------|
| 1 | ~80 ms | Выше |
| 2 | ~160 ms | Средний |
| 3 | ~240 ms | Ниже |

---

## Что коммитить на GitHub

**Коммитить:**
- `src/`, `scripts/`, `config/`, `README.md`, `requirements.txt`, `LICENSE`
- `data/negatives/README.md`, пустые `.gitkeep` в папках
- `catalog/manifest.json` (модели `.onnx` — по желанию, если маленькие)

**Не коммитить** (уже в `.gitignore`):
- `data/negatives/**/*.wav` — гигабайты датасетов
- `data/features/*.npy`
- `workspace/recordings/`, `output/`
- `.venv/`

---

## Источники данных для `data/negatives/`

Примеры бесплатных речевых датасетов:

- [M-AILABS](https://github.com/i-celeste-aurora/m-ailabs-dataset) — RU, EN, DE, PL, UK…
- [Mozilla Common Voice](https://github.com/common-voice/cv-dataset)

Шум: UrbanSound, ESC-50, свои записи с микрофона.

---

## Ограничения

- Embeddings openWakeWord лучше работают на **английском**; для **русского** нужен большой корпус негативов и 7–10 записей
- Качество на уровне Porcupine требует тестов **8+ часов** фона (FAR)
- `admin_build_negatives` на ~100k файлов может занять **часы**

---

## Публикация на GitHub

Пошагово: **[GITHUB.md](GITHUB.md)** (переименование папки, push, разные имена локально и на GitHub).

Кратко:

```bash
git init
git add .
git commit -m "Initial commit: custom wake word platform"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/phrase-wake.git
git push -u origin main
```

---

## Лицензия

MIT — см. [LICENSE](LICENSE).

Использует [openWakeWord](https://github.com/dscripka/openWakeWord) (Apache 2.0) для embeddings и inference.
