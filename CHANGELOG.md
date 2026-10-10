# Changelog

## 1.1.0 — 10 October 2026

The engine is now shared by the command-line tool and the Voice Match desktop app, and works fully offline.

- **One-time setup, then offline.** `voice_match.py setup` downloads the voice converter and the Bangla recogniser into one models folder (`~/Library/Application Support/Voice Match/models` on macOS). Every run loads them from there with `local_files_only=True` and sets `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE`; a missing model stops with a clear message instead of trying the network. Before, `--best-of` failed offline. Setup reuses models from older caches. `setup --status` shows what is installed; `setup --json` reports progress for the app.
- **`vmengine` package.** The pipeline moved out of `voice_match.py` into `vmengine/` (`audio`, `models`, `pipeline`, `server`). `voice_match.py` keeps the same options and gives the same results (checked with `tools/evaluate.py`).
- **`voice_match.py serve`.** A JSON-lines engine server that loads the models once: analyze, render one phrase with given settings, render all (fast or best-of) with progress and cancel, choose takes, export.
- **Manual settings** in the engine: pitch median, formant ratio (voice size), pitch range (intonation), match strength (blends the read's and the reference's tone-colour embeddings; 1.0 = the old behaviour), and finishing (presence, de-essing, loudness). Defaults are unchanged.
- **Faster.** The converter is driven directly: the reference's embedding is computed once instead of per take, and audio stays in memory instead of going through temporary files. Fast mode 19 s → 13 s and best-of 1 min 57 s → 1 min 26 s on a Mac mini M4. Each converted phrase is still peak-normalised, as coqui-tts's file writer did.
- **`--device mps`** runs the converter and the recogniser on the Apple GPU (best-of in 40 s, same quality). CPU stays the default.
- **macOS.** Tested on a Mac mini M4 with Python 3.13. Added `torchaudio` (imported by coqui-tts) and `setuptools<81` (webrtcvad needs `pkg_resources`) to the requirements; `pyproject.toml` and `uv.lock` pin every version for `uv sync`. README covers the macOS model folders.
- **`tools/offline_check.sh`** runs any command with the network blocked and every connection attempt logged. **`tools/server_check.py`** exercises every server command.
- `tools/evaluate.py` loads the recogniser from the models folder (offline).

## 1.0.0 — 10 October 2026

First packaged release of the pipeline used for the ORYZN film voiceover.

- `voice_match.py`: pitch measurement, automatic Praat pre-shift, OpenVoice v2 conversion phrase by phrase, and polish (high-pass, presence, de-ess, 3:1 compression, −16 LUFS with a peak limiter).
- `--best-of`: six takes per phrase, judged on voice similarity (Resemblyzer) and on whether the words survive (a Bangla recogniser compares each take with the original phrase; no script needed).
- `--keep-takes`: saves every take, each original phrase and `takes.csv` so a take can be swapped by ear.
- Reads WAV, MP3 and M4A (M4A through FFmpeg).
- Loads the Bangla recogniser as safetensors, so the first `--best-of` run downloads 1.2 GB instead of 2.5 GB.
- `tools/evaluate.py`: scores files for voice similarity, char errors against a script, and loudness.
- Tested from a clean install on Linux, Python 3.13, CPU only. Versions pinned in `requirements.txt`.
