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

**Results on the full 76-second ORYZN read** (2-core CPU, no GPU):

| | Similarity | Script characters a Bangla recogniser gets wrong | Time |
|---|---|---|---|
| The team member's original read | 0.58 | 8.4% | |
| `voice_match.py` | 0.90 | 18.8% | about 1 minute |
| `voice_match.py --best-of` | 0.91 | 15.3% | about 6 minutes |

Measured from a clean install with `tools/evaluate.py`; the details are in [docs/RESEARCH.md](docs/RESEARCH.md).

The words come out a little softer than in the original read; `--best-of` wins back part of that.

## Install

You need Python 3.10–3.13 and FFmpeg. Tested on Linux with Python 3.13 and no GPU. It should also work on macOS, but that hasn't been tested.

```bash
# 1. FFmpeg.  macOS: brew install ffmpeg    Ubuntu/Debian: sudo apt install ffmpeg

# 2. A virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# 3. Linux without an NVIDIA GPU only: CPU PyTorch first (skips several GB of CUDA downloads)
pip install torch torchaudio torchcodec --index-url https://download.pytorch.org/whl/cpu

# 4. Everything else
pip install -r requirements.txt
```

If something fails on macOS:

- **`webrtcvad` fails to build:** install Apple's command-line tools with `xcode-select --install`, then rerun step 4.
- **"Could not load libtorchcodec":** install FFmpeg through conda instead (`conda install -c conda-forge ffmpeg`), which is the route torchcodec documents.

Models download the first time they're needed: the voice converter (126 MB) on the first run, and the Bangla speech recogniser (1.2 GB) on the first `--best-of` run. They're cached in `~/.local/share/tts` and `~/.cache/huggingface`. The similarity model comes inside the Resemblyzer package.

## Use

```bash
# Fast: about 1 minute for a 76-second read on a 2-core CPU
python voice_match.py team_read.m4a my_voice.mp3 -o voiceover.wav

# Best quality: six takes per phrase, keeps the best one (about 6 minutes)
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
| `--asr-model ID` | Uses another Hugging Face wav2vec2 model for the word check, for example for English. |
| `--no-preshift` | Skips step 2. Useful when both voices are already very similar. |
| `--no-polish` | Skips step 5 and gives you the raw converted voice. |
| `--lufs -16` | Loudness of the finished file. |
| `--device cuda` | Runs on an NVIDIA GPU. |

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
- Fast mode runs a little faster than real time on a 2-core CPU; `--best-of` takes about five times the length of the read.

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

- To call it from your own code, move the body of `main()` into a function that takes the two file paths and options. The building blocks (`pitch_profile`, `preshift_settings`, `speech_chunks`, `Judge`, `polish`) are already separate functions.
- The quickest team interface is a small Gradio app on one office machine: two upload boxes, a "best quality" checkbox, and a download link.
- As a service, wrap it in FastAPI with a job queue, and load the models once at start-up instead of on every request.
- Keep a folder of approved reference voices, each with its consent note, and let people pick a voice instead of uploading one.
