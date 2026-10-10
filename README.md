# Voice Match

Make a recording sound like another person's voice. The words, timing and emotion come from the recording. The voice comes from a short sample of the target speaker.

It was built for the ORYZN brand film: a team member read the Bangla script, and the tool turned that read into the voice of a reference sample from the ORYZN team, recorded with its owner's consent.

Research notes and every measurement are in [docs/RESEARCH.md](docs/RESEARCH.md), the story of how it was built is in [docs/HISTORY.md](docs/HISTORY.md), and changes to the tool are in [CHANGELOG.md](CHANGELOG.md).

> **Consent first.** Only convert into a voice whose owner has agreed to it, keep a note of that agreement for each voice, and never use it to make someone appear to say something they haven't approved. YouTube, Meta and TikTok have rules about labelling realistic AI-altered content; check them before posting.

## How it works

This is **voice conversion** (speech in, speech out), not text-to-speech. Nothing is generated from text, so the Bangla pronunciation and the acting come from a real person. That is why it sounds natural where Bangla text-to-speech still sounds robotic.

1. **Measure.** Praat measures the pitch of both voices. In the ORYZN test the team read sat around 233 Hz and the reference voice around 114 Hz.
2. **Pre-shift.** Praat's *Change gender* effect moves the read into the reference's register. It sets the pitch median to the reference's, scales the formants (how big the vocal tract sounds), and narrows or widens the intonation range. The settings are calculated automatically: formant ratio = (reference pitch ÷ read pitch)^0.2, and pitch range = 1.1 × reference range ÷ read range, with ranges measured in semitones. For the ORYZN read that gives 0.87 and 0.74.
3. **Convert.** OpenVoice v2's tone-colour converter copies the timbre of the reference sample onto the pre-shifted speech. The read is split at pauses and converted phrase by phrase, and each phrase goes back at its original time, so the timing never drifts.
4. **Pick the best take** (optional, `--best-of`). Every phrase is converted six times with slightly different pre-shift settings. Two judges score each take: a speaker-similarity model (Resemblyzer) for how close it sounds to the reference, and a Bangla speech recogniser that checks the words survived by comparing its transcript of the take with its transcript of the original phrase. The tool keeps the clearest take that still sounds like the reference.
5. **Polish.** High-pass at 75 Hz, a little presence at 3.2 kHz, de-essing, gentle 3:1 compression, then loudness to −16 LUFS with a peak limiter at −1 dB.

**Why two stages?** Measured on three test lines. Voice similarity runs from 0 to 1, and two clips of the same person usually score 0.85 or more.

| What was applied | Similarity to the reference |
|---|---|
| Nothing (the team member's read) | 0.54 |
| Pitch/formant pre-shift only | 0.61 |
| Neural conversion only | 0.82 |
| **Pre-shift, then conversion** | **0.86** |

I also tried FreeVC (0.76) and kNN-VC (0.87, but the words came out noticeably blurrier). OpenVoice v2 after the pre-shift gave the best balance.

**Results on the full 76-second ORYZN read**

| | Similarity | Script characters a Bangla recogniser gets wrong | Time, 2-core Linux CPU | Time, Mac mini M4 (CPU) |
|---|---|---|---|---|
| The team member's original read | 0.58 | 8.4% | | |
| `voice_match.py` | 0.90 | 18.8–20% | about 1 minute | 13 s |
| `voice_match.py --best-of` | 0.91–0.92 | 14–16% | about 6 minutes | 1 min 25 s |

Measured with `tools/evaluate.py`; the details are in [docs/RESEARCH.md](docs/RESEARCH.md). Results vary by about ±0.01 similarity and ±1 point of char errors between runs, because the converter samples a little noise.

The words come out a little softer than in the original read; `--best-of` wins back part of that.

## Install

You need Python 3.10–3.13 (3.13 is tested; not 3.14) and FFmpeg. Tested on Linux (CPU only) and on macOS on Apple Silicon.

**With uv (recommended; installs the exact versions that were tested)**

```bash
# macOS: brew install uv ffmpeg     Ubuntu/Debian: sudo apt install ffmpeg, then install uv from astral.sh
uv sync                              # creates .venv with Python 3.13 and every pinned package
source .venv/bin/activate
```

**With pip**

```bash
# 1. FFmpeg.  macOS: brew install ffmpeg    Ubuntu/Debian: sudo apt install ffmpeg

# 2. A virtual environment with Python 3.13
python3.13 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# 3. Linux without an NVIDIA GPU only: CPU PyTorch first (skips several GB of CUDA downloads)
pip install torch torchaudio torchcodec --index-url https://download.pytorch.org/whl/cpu

# 4. Everything else
pip install -r requirements.txt
```

**Then download the models, once (1.4 GB):**

```bash
python voice_match.py setup                  # converter (126 MB) and Bangla recogniser (1.2 GB)
python voice_match.py setup --no-recogniser  # converter only, if you only need fast mode
python voice_match.py setup --status         # what is installed
```

After setup the tool never uses the network: it loads every model from its own folder and sets
`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. If a model is missing it stops with a message telling you to run
setup, instead of trying to download. The models folder is shared with the Voice Match desktop app:

| System | Models folder |
|---|---|
| macOS | `~/Library/Application Support/Voice Match/models` |
| Linux | `~/.local/share/voice-match/models` (or `$XDG_DATA_HOME/voice-match/models`) |

Set `VOICE_MATCH_MODELS` or pass `--models-dir` to use another folder. Setup reuses models that older versions
of the tool downloaded into coqui-tts's cache (`~/Library/Application Support/tts` on macOS,
`~/.local/share/tts` on Linux) or the Hugging Face cache (`~/.cache/huggingface`), so it doesn't fetch them again;
after that you can delete those old caches. The similarity model comes inside the Resemblyzer package.

If something fails on macOS:

- **`webrtcvad` fails to build:** install Apple's command-line tools with `xcode-select --install`, then rerun the install.
- **"No module named 'pkg_resources'":** `pip install "setuptools<81"` (already pinned in `requirements.txt` and `uv.lock`).
- **"Could not load libtorchcodec":** torchcodec needs FFmpeg's shared libraries. `brew install ffmpeg` provides them; if it still fails, install FFmpeg through conda (`conda install -c conda-forge ffmpeg`), which is the route torchcodec documents.

## Use

```bash
# Fast: 13 s on a Mac mini M4, about 1 minute on a 2-core Linux CPU, for a 76-second read
python voice_match.py team_read.m4a my_voice.mp3 -o voiceover.wav

# Best quality: six takes per phrase, keeps the best one
python voice_match.py team_read.m4a my_voice.mp3 -o voiceover.wav --best-of

# Also save every take, so you can swap one by ear
python voice_match.py team_read.m4a my_voice.mp3 -o voiceover.wav --best-of --keep-takes takes/
```

Inputs can be WAV, MP3 or M4A (phone voice memos). The output is a 48 kHz, 24-bit WAV at −16 LUFS, exactly as long as the read, ready to drop into a video editor. At the end, the tool prints the voice similarity before and after.

| Option | What it does |
|---|---|
| `--best-of` | Six takes per phrase, keeps the best. About 6× slower. |
| `--keep-takes DIR` | Saves every take (`phraseNN_takeK.wav`), each original phrase, and `takes.csv` with every take's scores and start time. |
| `--lang bn` | Language for the word check in `--best-of` (default `bn`). `--lang none` picks takes by voice similarity only. |
| `--asr-model ID` | Uses another Hugging Face wav2vec2 model for the word check, for example for English. Install it first with `setup --asr-model ID`. |
| `--no-preshift` | Skips step 2. Useful when both voices are already very similar. |
| `--no-polish` | Skips step 5 and gives you the raw converted voice. |
| `--lufs -16` | Loudness of the finished file. |
| `--device mps` | Runs on the Apple GPU: same quality, best-of about 2× faster on an M4 (see RESEARCH.md). `cuda` for an NVIDIA GPU. Default: CPU. |
| `--models-dir DIR` | Where the models are, if not in the default folder. |

**Engine server.** `python voice_match.py serve` starts a long-running engine that loads the models once and takes
JSON-lines commands on stdin (analyze, render one phrase with given settings, render all, cancel, export). The
Voice Match desktop app drives it; the protocol is described at the top of `vmengine/server.py`, and
`tools/server_check.py` exercises every command.

**Testing offline.** `tools/offline_check.sh <command>` runs any command with the network blocked and logs every
connection attempt, for example
`tools/offline_check.sh python voice_match.py read.m4a ref.mp3 -o out.wav --best-of`. It prints
`netguard: 0 network attempt(s)` when the run stayed offline.

## Getting good results

The recordings matter more than any setting.

**The reference voice (20–60 seconds)**
- One person, no music, a quiet room with soft furnishings.
- Speak in the tone you want in the ad. A calm reference gives a calm result.
- Hold the phone 20–30 cm away. Don't use noise-reduction apps; the converter exaggerates their artefacts.

**The read**
- Same quiet room, same distance throughout.
- Act it. The result keeps the read's pace, emphasis and emotion; only the voice changes.
- Leave a short pause between sentences, because the tool splits at pauses.
- Read in the same language as the reference.

**Checking the result**
- Listen on headphones. If one phrase sounds mushy, rerun with `--keep-takes`, listen to that phrase's other takes, and drop the best one into your editor at the start time listed in `takes.csv`. Or re-record just that sentence.
- In a video, keep the music about 10 dB below the voice while it speaks.

## Limits

- The words come out slightly softer than the original read (see the table above).
- The converter works at 22 kHz, so the voice has a little less "air" than a studio recording. That's fine for social video.
- Accent and speaking style come from the read, not the reference.
- Very different voices, such as a child's read into an adult's voice, are harder.
- Fast mode runs a little faster than real time on a 2-core CPU; `--best-of` takes about five times the length of the read. On a Mac mini M4 they take about 0.2× and 1.1× the length of the read.

## Licences

Check these before you sell or hand out a product built on this tool.

| Part | Licence |
|---|---|
| OpenVoice v2 model (MyShell) | MIT |
| coqui-tts (loads the model) | MPL-2.0 |
| Praat / Parselmouth (measuring and pre-shift) | **GPL-3.0** |
| Resemblyzer (similarity judge) | Apache-2.0 |
| Bangla recogniser `arijitx/wav2vec2-xls-r-300m-bengali` | Apache-2.0 |
| transformers / librosa / pyloudnorm | Apache-2.0 / ISC / MIT |

GPL-3.0 conditions apply when you distribute software that contains it, for example an app you sell that bundles Parselmouth. Using the tool in-house, or running it on your own server, isn't distribution. If you plan to distribute it, replace the pre-shift with a permissively licensed pitch and formant shifter first.

## Turning it into a team tool

- To call it from your own code, use `vmengine.pipeline.Session` (analyze, render, export); `voice_match.py` shows how. The building blocks (`pitch_profile`, `preshift_settings`, `speech_chunks`, `Judge`, `polish`) live in `vmengine/`.
- The quickest team interface is a small Gradio app on one office machine: two upload boxes, a "best quality" checkbox, and a download link.
- As a service, wrap it in FastAPI with a job queue, and load the models once at start-up instead of on every request.
- Keep a folder of approved reference voices, each with its consent note, and let people pick a voice instead of uploading one.
