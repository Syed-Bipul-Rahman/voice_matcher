# Research log

Everything that was tried while building the voice-matching pipeline, with the measurements behind each decision. Measured on a cloud machine with 2 CPU cores, 7 GB RAM and no GPU.

## The problem

The ORYZN brand film (Bangla, 9:16, about 1:24) needed a voiceover in a specific person's voice. A team member recorded the script; the goal was to make that read sound like a reference voice from the ORYZN team, used with its owner's consent.

## Options considered

| Option | Verdict |
|---|---|
| Open-source Bangla text-to-speech plus voice cloning | Rejected without testing. Free Bangla TTS tends to sound flat and robotic, which doesn't suit a premium brand film. |
| ElevenLabs voice cloning (supports Bengali; an Instant Voice Clone needs about 30 s of recording) | A good paid route, offered as an alternative. Not used. |
| **A real read converted into the reference voice (speech-to-speech)** | **Chosen.** The Bangla pronunciation, timing and acting come from a person; only the voice changes. |
| The narrator of the reference ad (আমার লিপি) | Declined: there was no consent from that person. A speaker check confirmed the reference sample is a different person (below). |

## The two recordings

| | Team member's read | Reference voice |
|---|---|---|
| Length, format | 75.8 s, MP3, 44.1 kHz mono | 37.6 s, MP3, 48 kHz stereo |
| Loudness, peak | −21.4 LUFS, −1.7 dBFS | −17.5 LUFS, −1.0 dBFS |
| Pitch median (10th–90th percentile) | 233 Hz (148–339) | 114 Hz (92–161) |
| Formant medians F1 / F2 | 581 / 1840 Hz | 492 / 1946 Hz |
| Noise | Near digital silence between phrases | Clean, short pauses between phrases |

**Speaker check** (Resemblyzer cosine similarity on 6 s windows; one person usually scores 0.85+):

| Pair | Similarity |
|---|---|
| Reference with itself (different windows) | 0.95 |
| Reference vs the team member's read | 0.55 |
| Reference vs the আমার লিপি narrator | 0.64 mean, 0.70 max |

So the reference is not the narrator of the reference ad.

## Measuring quality

- **Voice similarity:** Resemblyzer speaker embeddings, cosine similarity to the reference.
- **Are the words still there:** a Bangla wav2vec2 recogniser (`arijitx/wav2vec2-xls-r-300m-bengali`) transcribes each phrase, then the character error rate against the script. On the original read it gets 8.6% of characters wrong, so that is the floor.
- **Audio quality estimates:** torchaudio SQUIM (objective STOI, PESQ and SI-SDR; subjective MOS with the reference as the non-matching reference).
- Whisper small could not transcribe this Bangla audio; it returned text in the wrong scripts. faster-whisper also fails to open files with the current PyAV (`unexpected keyword argument 'metadata_errors'`), though passing a NumPy array works.

## Experiment 1: which converter, and does a pre-shift help?

Three test lines (lines 1, 6 and 17 of the script). Pre-shift = Praat *Change gender* with formant ratio 0.86, pitch median 114 Hz, pitch range 0.8.

| Method | Similarity | Pitch medians per line (Hz) | Char errors | PESQ | MOS |
|---|---|---|---|---|---|
| Original read | 0.54 | 272 / 228 / 234 | 15% | 3.70 | 4.41 |
| Pre-shift only (Praat) | 0.61 | 115 / 114 / 114 | 17% | 1.99 | 3.97 |
| FreeVC 24 kHz | 0.76 | 138 / 130 / 121 | 20% | 3.77 | 4.25 |
| FreeVC after pre-shift | 0.77 | 112 / 115 / 115 | 17% | 4.05 | 4.22 |
| kNN-VC | 0.87 | 124 / 113 / 112 | 24% | 3.67 | 4.40 |
| kNN-VC after pre-shift | 0.87 | 116 / 109 / 106 | 24% | 3.46 | 4.37 |
| OpenVoice v2 | 0.82 | 132 / 116 / 117 | 24% | 3.31 | 4.27 |
| **OpenVoice v2 after pre-shift** | **0.86** | 130 / 119 / 126 | **13%** | 3.30 | **4.41** |

Reference pitch median: 114 Hz. kNN-VC matched the voice as well but blurred the words; OpenVoice v2 after the pre-shift gave the best balance. Speed on 2 CPUs for the three test lines (about 12 s of audio): OpenVoice v2 about 6 s (126 MB model), FreeVC about 8 s (896 MB model), kNN-VC about 18 s.

## Experiment 2: pre-shift settings (OpenVoice v2, same three lines)

| Formant ratio | Pitch range | Similarity | Char errors |
|---|---|---|---|
| 0.80 | 0.7 | 0.854 | 20.1% |
| 0.80 | 0.9 | 0.845 | 16.7% |
| **0.86** | **0.7** | **0.868** | **16.0%** |
| 0.86 | 0.9 | 0.854 | 19.4% |
| 0.92 | 0.7 | 0.875 | 18.1% |
| 0.92 | 0.9 | 0.855 | 19.4% |

Chosen: formant 0.86, range 0.75. The packaged tool calculates these automatically, and its formulas reproduce them for this read:

- formant ratio = (reference pitch ÷ read pitch)^0.2 = (114 ÷ 233)^0.2 = 0.87
- pitch range = 1.1 × reference range ÷ read range, with ranges in semitones between the 10th and 90th percentiles = 1.1 × 9.7 ÷ 14.3 = 0.74

## Experiment 3: the full read (23 script lines)

| Version | Similarity | Char errors | MOS |
|---|---|---|---|
| Original read | 0.58 | 8.6% | 4.43 |
| One setting for every line (0.86 / 0.75) | 0.89 | 22.9% | 4.43 |
| Same, plus EQ matched to the reference's spectrum | 0.89 | 24.6% | 4.43 |
| **Best of 6 settings per line** | **0.90** | **16.7%** | **4.44** |

- EQ matching made the words worse, so it was dropped.
- Best-of scoring: lowest of (char errors − 0.6 × (similarity − 0.85) + 0.5 if similarity < 0.82). Settings tried: (0.86, 0.75), (0.80, 0.9), (0.86, 0.9), (0.92, 0.7), (0.92, 0.9), (0.80, 0.7). Each was the winner on 6, 2, 2, 6, 3 and 4 lines respectively, so no single setting is best everywhere.
- The weakest lines contain the Latin-script brand name "ORYZN", which the Bangla recogniser has no spelling for, and dense phrases such as "তিনশো ফুট মসলিন গলে যেত".

## Fitting the voice into the film

- **Line alignment.** Pauses in the read were found with 10 ms frames of 120–4000 Hz band energy below −45 dB lasting at least 90 ms. A dynamic-programming fit then matched the 23 lines to those pauses, using expected durations from Bangla letter counts (a year counts as 11 letters, other numbers as 6, ORYZN as ওরাইজন). Word-by-word caption timing came from voiced time inside each line.
- **Re-timing.** The film followed the voice: a 0.6 s lead-in, plus extra pauses after line 16 (+0.45 s), line 17 (+1.15 s, for the logo-stitch moment) and line 21 (+0.3 s). The film grew from 1:22 to 1:24. Voice onset vs caption cue: mean −0.006 s, worst 0.04 s, across all 23 lines.
- **Voice chain.** High-pass at 75 Hz, +0.18 parallel peak at 3.2 kHz, de-esser above 4.8 kHz, 3:1 compression.
- **Mix.** Music ducked 10 dB under the voice (80 ms attack, 450 ms release), with the voice 7–15 dB above the music while speaking. The final mix was normalised to −14 LUFS with a −1.4 dBFS peak.

## The packaged tool (v1.0)

Differences from the hand-run pipeline:

- Pre-shift settings come from the formulas above instead of a sweep.
- The read is split at pauses of 0.3 s or more (no script needed); the ORYZN read gives 27 phrases.
- The best-of word check needs no script: it compares the recogniser's transcript of each take with its transcript of the original phrase.
- A peak limiter replaced whole-file gain, which had left the output at −19 LUFS instead of −16.

**Clean-install test** (fresh virtual environment, empty model caches, commands copied from the README):

| | Time | Similarity | Char errors (script, 27 phrase windows) | Loudness |
|---|---|---|---|---|
| Install (step 3 + step 4) | 60 s | | | |
| Original read (as M4A) | | 0.58 | 8.4% | −21.4 LUFS |
| `voice_match.py` | 56 s | 0.90 | 18.8% | −16.0 LUFS |
| `voice_match.py --best-of` | 6 min 27 s, including the first-time recogniser download | 0.91 | 15.3% | −16.0 LUFS |

Measured with `tools/evaluate.py --reference my_voice.mp3 --script ORYZN_story_bn.srt team_read.m4a voiceover_fast.wav voiceover.wav`. Peaks: −0.9 dBFS. Char errors here are measured over the tool's 27 phrase windows against the whole script, so they differ slightly from the 23-line numbers in experiment 3. Similarity varies by about ±0.01 between runs of the same command.

The first packaged version, which still used whole-file gain instead of the limiter, scored 0.89 / 18.0% (fast) and 0.90 / 15.3% (best-of) at about −19 LUFS.

## Installation problems found (and fixed in requirements.txt)

- transformers 5.x breaks coqui-tts (`cannot import name 'isin_mps_friendly'`), and versions below 4.57 fail too (`is_torchcodec_available`). Pinned to 4.57.6.
- With torch 2.9 or later, coqui-tts needs torchcodec for audio input/output, so the requirement is `coqui-tts[codec]`.
- librosa 1.0 can't read M4A, so the tool falls back to FFmpeg for phone voice memos.
- The Bangla recogniser's repository has only `.bin` weights on its main branch. Loading them makes transformers start a background thread that also downloads an auto-converted safetensors copy, 2.5 GB in total. `use_safetensors=False` doesn't stop that thread (tested). Loading with `use_safetensors=True` fetches only the converted copy: 1.2 GB on disk, and the recogniser gives identical transcripts.
- On Linux, installing CPU PyTorch first avoids several GB of CUDA downloads.

## macOS, offline engine and server (v1.1, 10 October 2026)

Measured on a Mac mini M4 (10 cores, 16 GB) with Python 3.13 and the same package versions as on Linux. The test
read here is the MP3 version of the read (`team_read.m4a.mp3`, 75.8 s); its char errors are 8.6% instead of 8.4%
for the M4A. Every row is scored by `tools/evaluate.py` with the same reference and script.

| Version | Device | Time | Similarity | Char errors | Loudness | Length |
|---|---|---|---|---|---|---|
| Original read | | | 0.576 | 8.6% | −21.4 LUFS | 75.8 s |
| v1.0, fast | CPU | 19 s | 0.904 | 20.5% | −16.0 | 75.8 s |
| v1.0, best-of | CPU | 1 min 57 s | 0.919 | 14.2% | −16.0 | 75.8 s |
| **v1.1, fast** | CPU | **13 s** | 0.904 | 19.5% | −16.0 | 75.8 s |
| **v1.1, best-of** | CPU | **1 min 26 s** | 0.915 | 14.8% | −16.0 | 75.8 s |
| v1.1, fast, network blocked | CPU | 13 s | 0.899 | 20.1% | −16.0 | 75.8 s |
| v1.1, best-of, network blocked | CPU | 1 min 23 s | 0.921 | 15.6% | −16.0 | 75.8 s |
| v1.1 server, fast | CPU | 16 s | 0.903 | 19.7% | −16.0 | 75.8 s |
| v1.1 server, best-of, network blocked | CPU | 75 s | 0.919 | 15.5% | −16.0 | 75.8 s |
| v1.1, fast | MPS (converter) | 11 s | 0.901 | 18.0% | −16.0 | 75.8 s |
| v1.1, best-of | MPS (converter) | 58 s | 0.916 | 15.8% | −16.0 | 75.8 s |
| v1.1, best-of | MPS (converter + recogniser) | 40 s | 0.920 | 15.1% | −16.0 | 75.8 s |

Baseline from the clean Linux install: fast 0.90 / 18.8%, best-of 0.91 / 15.3%. Every v1.1 row is within ±0.02
similarity and ±2 points of char errors of it. Peaks: −1.0 dBFS. The server's best-of row includes one take picked
by hand (phrase 3, take 4), which the server check does on purpose.

**Run-to-run spread.** The same command gives 19.5–20.5% char errors in fast mode and 14.8–15.6% in best-of, and
similarity within ±0.005. OpenVoice's posterior encoder samples noise (`tau`), so no two runs are identical.

**Peak normalisation was hidden in coqui-tts.** v1.0 wrote each converted phrase with coqui-tts's
`voice_conversion_to_file`, whose `save_wav` scales every file to full scale (`wav * 32767 / max(0.01, max|wav|)`).
So every phrase was peak-normalised before it was placed back into the timeline, and the polish stage was tuned on
that. v1.1 calls the model directly (`extract_se`, then `inference` with `g_src` and `g_tgt`) and copies that
normalisation explicitly.

**Speed-ups in v1.1** (same quality): the reference's tone-colour embedding is computed once instead of once per
take, and audio stays in memory (no WAV round trips for the pre-shift and the converter).

**Offline.** `tools/offline_check.sh` blocks every internet socket and DNS lookup in all Python processes and sets
`HTTP(S)_PROXY` to a closed port. v1.1 fast, best-of and the full server session: **0 attempts**. v1.0 best-of:
24 blocked attempts (HEAD requests for `preprocessor_config.json`), then failure. The recogniser's main branch only
has `.bin` weights and v1.0 loaded an auto-converted safetensors copy from a Hub pull-request revision, which can't
be resolved offline. Setup now saves the loaded model with `save_pretrained(safe_serialization=True)` into a plain
folder (1.26 GB), loaded with `local_files_only=True`; transcripts are the same.

**Match strength.** The converter maps the read's speech through the source speaker's tone-colour embedding
(`g_src`) into the target's (`g_tgt`). Blending the target embedding,
`g_tgt = α · reference + (1 − α) · read`, gives a smooth control. Phrase 1, automatic pre-shift, two runs:

| α | Similarity | Words kept |
|---|---|---|
| 1.0 (default, = v1.0) | 0.88–0.90 | 0.87–0.89 |
| 0.5 | 0.75–0.78 | 0.89 |
| 0.0 | 0.62–0.63 | 0.89–0.91 |

At 0.0 the pre-shifted read keeps its own timbre, so similarity drops to near the pre-shift-only level (0.61 in
experiment 1). Values between 0.8 and 1.0 are the useful range for fine-tuning.

**Single-phrase preview** (server, models loaded, CPU): about 0.9 s per phrase including the similarity and
words-kept scores. The first call loads the converter and recogniser (about 4 s).

**Apple GPU (MPS).** Quality matches the CPU within the limits, and best-of runs about 2× faster with both the
converter and the recogniser on the GPU. CPU stays the default until MPS has been tested on more recordings.

## Licences (checked 10 October 2026)

| Part | Licence | Where it was checked |
|---|---|---|
| OpenVoice v2 model (MyShell) | MIT | Hugging Face model card `myshell-ai/OpenVoiceV2`; coqui-tts model registry |
| coqui-tts | MPL-2.0 | Package metadata |
| praat-parselmouth | GPL-3.0 | Package metadata |
| Resemblyzer | Apache-2.0 | Licence file shipped in the package |
| `arijitx/wav2vec2-xls-r-300m-bengali` | Apache-2.0 | Hugging Face model card |
| transformers, librosa, pyloudnorm | Apache-2.0, ISC, MIT | Package metadata |

## Sources

- [ElevenLabs: voice cloning in 12 Indian languages, including Bengali](https://elevenlabs.io/blog/voice-cloning-in-12-indian-languages-demonstrated-live-at-iit-delhi)
- [OpenVoice V2 model card](https://huggingface.co/myshell-ai/OpenVoiceV2)
- [Bangla wav2vec2 recogniser model card](https://huggingface.co/arijitx/wav2vec2-xls-r-300m-bengali)
- [torchcodec: platforms and FFmpeg requirements](https://github.com/meta-pytorch/torchcodec)
- [coqui-tts (Idiap fork)](https://pypi.org/project/coqui-tts/)
