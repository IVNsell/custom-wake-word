# Publishing to GitHub: folder names

## Key point

Your **local folder name** and the **GitHub repository name** are different things. They **do not have to match**.

| Location | Example name | What it is |
|----------|--------------|------------|
| On your disk | `custom-wake-word` | Just a folder with files |
| On GitHub | `phrase-wake` or `open-wakeword-trainer` | Repository name in the URL |
| After `git clone` | whatever you named the repo | Each user has their own folder name |

Example:

```text
Disk:     C:\Projects\custom-wake-word\
GitHub:   github.com/ivan/phrase-wake
Clone:    git clone https://github.com/ivan/phrase-wake.git
          → folder phrase-wake/  (not custom-wake-word — that's fine)
```

The code inside is the same. The GitHub folder name **does not affect** how the program runs.

---

## Step 1. Rename the local folder (recommended)

Current: `test_wake_up` → better: **`custom-wake-word`**

1. **Close Cursor** (the folder is locked while the IDE is open).
2. In File Explorer or PowerShell:

```powershell
Rename-Item "C:\Users\IVNse\IVN\IVNsell\test_wake_up" "custom-wake-word"
```

3. Reopen the project from `custom-wake-word`.

Other local folder name options:

- `custom-wake-word` — general purpose
- `phrase-wake` — shorter
- `wake-word-sdk` — if you emphasize the SDK

---

## Step 2. Create a GitHub repository

1. [github.com/new](https://github.com/new)
2. **Repository name** — anything you like, for example:
   - `custom-wake-word`
   - `phrase-wake`
   - `open-wakeword-trainer`
3. Public, **without** README (you already have one).
4. Create repository.

---

## Step 3. Push the code

```powershell
cd C:\Users\IVNse\IVN\IVNsell\custom-wake-word

git init
git add .
git status
git commit -m "Initial commit: custom wake word platform"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/custom-wake-word.git
git push -u origin main
```

`custom-wake-word` in the URL is the **GitHub repository name**.  
Your local folder can still be named `custom-wake-word` — **that's fine**.

---

## What does not go to GitHub

See `.gitignore`:

- `data/negatives/*.wav` (gigabytes of datasets)
- `data/features/*.npy`
- `workspace/recordings/`, `output/`
- `.venv/`

Only **code and documentation** go to GitHub.

---

## Repository description (About)

**Description:**

> Open-source wake word in Python: train your own phrase from 3–10 recordings, ONNX inference ~2 ms, train locally.

**Topics:** `wake-word`, `openwakeword`, `onnx`, `speech`, `voice-assistant`, `python`
