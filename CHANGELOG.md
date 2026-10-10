# Changelog

## 1.0.0 — 10 October 2026

First packaged release of the pipeline used for the ORYZN film voiceover.

- `voice_match.py`: pitch measurement, automatic Praat pre-shift, OpenVoice v2 conversion phrase by phrase, and polish (high-pass, presence, de-ess, 3:1 compression, −16 LUFS with a peak limiter).
- `--best-of`: six takes per phrase, judged on voice similarity (Resemblyzer) and on whether the words survive (a Bangla recogniser compares each take with the original phrase; no script needed).
- `--keep-takes`: saves every take, each original phrase and `takes.csv` so a take can be swapped by ear.
- Reads WAV, MP3 and M4A (M4A through FFmpeg).
- Loads the Bangla recogniser as safetensors, so the first `--best-of` run downloads 1.2 GB instead of 2.5 GB.
- `tools/evaluate.py`: scores files for voice similarity, char errors against a script, and loudness.
- Tested from a clean install on Linux, Python 3.13, CPU only. Versions pinned in `requirements.txt`.
