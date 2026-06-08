# Публикация на GitHub: имена папок

## Главное

**Локальная папка** и **имя репозитория на GitHub** — это разные вещи. Они **не обязаны совпадать**.

| Где | Пример имени | Что это |
|-----|----------------|---------|
| У вас на диске | `custom-wake-word` | Просто папка с файлами |
| На GitHub | `phrase-wake` или `open-wakeword-trainer` | Имя репозитория в URL |
| После `git clone` | как назвали репо | У другого человека своя папка |

Пример:

```text
Диск:     C:\Projects\custom-wake-word\
GitHub:   github.com/ivan/phrase-wake
Clone:    git clone https://github.com/ivan/phrase-wake.git
          → папка phrase-wake/  (не custom-wake-word — и это нормально)
```

Код внутри один и тот же. Имя папки на GitHub **не влияет** на работу программы.

---

## Шаг 1. Переименовать локальную папку (рекомендуется)

Сейчас: `test_wake_up` → лучше: **`custom-wake-word`**

1. **Закройте Cursor** (папка занята, пока IDE открыт).
2. В проводнике или PowerShell:

```powershell
Rename-Item "C:\Users\IVNse\IVN\IVNsell\test_wake_up" "custom-wake-word"
```

3. Откройте проект снова из `custom-wake-word`.

Другие варианты имени локальной папки:

- `custom-wake-word` — универсально
- `phrase-wake` — короче
- `wake-word-sdk` — если акцент на SDK

---

## Шаг 2. Создать репозиторий на GitHub

1. [github.com/new](https://github.com/new)
2. **Repository name** — любое, например:
   - `custom-wake-word`
   - `phrase-wake`
   - `open-wakeword-trainer`
3. Public, **без** README (у вас уже есть).
4. Create repository.

---

## Шаг 3. Залить код

```powershell
cd C:\Users\IVNse\IVN\IVNsell\custom-wake-word

git init
git add .
git status
git commit -m "Initial commit: custom wake word platform"
git branch -M main
git remote add origin https://github.com/ВАШ_ЛОГИН/phrase-wake.git
git push -u origin main
```

`phrase-wake` в URL — имя **репозитория на GitHub**.  
Локальная папка может называться `custom-wake-word` — **это нормально**.

---

## Что не попадёт на GitHub

См. `.gitignore`:

- `data/negatives/*.wav` (гигабайты датасетов)
- `data/features/*.npy`
- `workspace/recordings/`, `output/`
- `.venv/`

На GitHub только **код и документация**.

---

## Описание репозитория (About)

**Description:**

> Open-source wake word на Python: своя фраза из 3–10 записей, ONNX inference ~2 ms, обучение локально.

**Topics:** `wake-word`, `openwakeword`, `onnx`, `speech`, `voice-assistant`, `python`
