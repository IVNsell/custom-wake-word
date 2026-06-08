# Shared negative audio corpus

Place **hours** of background audio here — anything that is **not** a user's wake phrase.

## Suggested folders

```
data/negatives/
  rain/           # rain, thunder
  road/           # traffic, street
  music/          # various genres
  home/           # household ambient
  speech_ru/      # Russian speech (no wake words)
  speech_en/      # English speech
  speech_other/   # other languages (de/, es/, …)
  tv/             # TV, podcasts, streams
  noise/          # misc ambient clips
```

**Formats:** `.wav`, `.flac`, `.mp3`, `.ogg`  
**File names:** any  
**Subfolders:** allowed (scanned recursively)

## After adding files

```bash
python scripts/admin_build_negatives.py
```

Creates `data/features/shared_negatives.npy` used by all user models.

## Free speech datasets

- [M-AILABS](https://github.com/i-celeste-aurora/m-ailabs-dataset)
- [Mozilla Common Voice](https://github.com/common-voice/cv-dataset)
