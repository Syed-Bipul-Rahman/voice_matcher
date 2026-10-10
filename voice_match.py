#!/usr/bin/env python3
"""
voice_match.py - make a recording sound like another person's voice.

  SOURCE     the recording whose words, timing and emotion you want to keep
             (for example a team member reading the script)
  REFERENCE  20-60 s of clean speech in the voice you want (for example yours)

How it works (the pipeline used for the ORYZN film voiceover):
  1. Measure the pitch of both voices (Praat).
  2. Pre-shift the source's pitch and formants into the reference's register
     (Praat "Change gender"). Large jumps such as female -> male need this.
  3. Split the source at pauses and convert each chunk with OpenVoice v2's
     tone-colour converter, which copies the reference's timbre while keeping
     the words and timing.
  4. --best-of: make 6 takes of every chunk with different pre-shift settings and
     keep the one that sounds most like the reference while its words still come
     through (checked with a speaker-similarity model and a speech recogniser).
  5. Polish: high-pass, a little presence, de-ess, 3:1 compression, loudness.

Only convert into a voice whose owner has agreed to it.

Examples:
  python voice_match.py read.m4a my_voice.mp3 -o voiceover.wav
  python voice_match.py read.m4a my_voice.mp3 -o voiceover.wav --best-of
"""
import argparse
import csv
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

OUT_SR = 48000        # sample rate of the finished file
WORK_SR = 24000       # sample rate of the chunks handed to the converter
VC_MODEL = "voice_conversion_models/multilingual/multi-dataset/openvoice_v2"
ASR_MODELS = {"bn": "arijitx/wav2vec2-xls-r-300m-bengali"}   # word check for --best-of


def log(msg=""):
    print(msg, flush=True)


# ---------------------------------------------------------------- audio in/out
def load_audio(path, sr):
    """Load any audio file as mono float32 at `sr`; falls back to FFmpeg for m4a/aac and friends."""
    import librosa
    try:
        y, _ = librosa.load(path, sr=sr, mono=True)
        if len(y):
            return y.astype(np.float32)
    except Exception:
        pass
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        sys.exit(f"Could not read {path}. Install FFmpeg (macOS: brew install ffmpeg) or convert it to WAV first.")
    res = subprocess.run([ffmpeg, "-v", "error", "-i", path, "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
                         capture_output=True)
    if res.returncode != 0 or not res.stdout:
        sys.exit(f"Could not decode {path}: {res.stderr.decode(errors='ignore')[:300]}")
    return np.frombuffer(res.stdout, dtype=np.float32).copy()


def save_audio(path, y, sr):
    import soundfile as sf
    subtype = "PCM_24" if path.lower().endswith((".wav", ".flac")) else None
    sf.write(path, np.asarray(y, dtype=np.float32), sr, subtype=subtype)


def resample(y, sr_in, sr_out):
    import librosa
    return librosa.resample(y, orig_sr=sr_in, target_sr=sr_out) if sr_in != sr_out else y


# ---------------------------------------------------------------- 1. measure
def pitch_profile(y, sr):
    """(median, 10th percentile, 90th percentile) of the voice's pitch in Hz, or None."""
    import parselmouth
    snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=sr)
    f0 = snd.to_pitch(time_step=0.01, pitch_floor=60, pitch_ceiling=500).selected_array["frequency"]
    f0 = f0[f0 > 0]
    if len(f0) < 50:
        return None
    return float(np.median(f0)), float(np.percentile(f0, 10)), float(np.percentile(f0, 90))


def preshift_settings(src, ref):
    """Base pre-shift that moves the source into the reference's register.
    Formant ratio = (pitch ratio) ** 0.2 and pitch range scaled to the reference's range.
    For the ORYZN read (233 Hz female -> 114 Hz male) this gives 0.86 and 0.75,
    the values that tested best by ear and by measurement."""
    ratio = ref[0] / src[0]
    formant = float(np.clip(ratio ** 0.2, 0.80, 1.20))
    src_range = 12 * math.log2(src[2] / src[1])          # semitones between 10th and 90th percentile
    ref_range = 12 * math.log2(ref[2] / ref[1])
    rng = float(np.clip(1.1 * ref_range / max(src_range, 1.0), 0.6, 1.2))
    return ref[0], formant, rng, ratio


# ---------------------------------------------------------------- 2. pre-shift
def preshift(in_path, out_path, median, formant, rng, floor, ceiling):
    import parselmouth
    from parselmouth.praat import call
    snd = parselmouth.Sound(in_path)
    # Change gender: pitch floor, pitch ceiling, formant shift ratio, new pitch median, pitch range factor, duration factor
    call(snd, "Change gender", floor, ceiling, float(formant), float(median), float(rng), 1.0).save(out_path, "WAV")


# ---------------------------------------------------------------- 3. split at pauses
def speech_chunks(y, sr, min_pause=0.30, min_len=1.0, max_len=15.0, lead=0.25, tail=0.30):
    """Split the speech at pauses. Returns non-overlapping (start, end) times in seconds."""
    from scipy import signal
    hop = int(0.01 * sr)
    yb = signal.sosfiltfilt(signal.butter(4, [120, 4000], btype="band", fs=sr, output="sos"), y)
    n = len(yb) // hop
    db = 20 * np.log10(np.sqrt(np.mean(yb[:n * hop].reshape(n, hop) ** 2, axis=1)) + 1e-9)
    db = np.convolve(db, np.ones(3) / 3, mode="same")
    thr = max(np.percentile(db, 95) - 30, np.percentile(db, 10) + 6)
    act = db > thr
    frames = np.flatnonzero(act)
    if len(frames) == 0:
        sys.exit("No speech found in the source recording.")
    first, last = frames[0], frames[-1] + 1

    pauses, i = [], first
    while i < last:
        if not act[i]:
            j = i
            while j < last and not act[j]:
                j += 1
            if (j - i) * 0.01 >= min_pause:
                pauses.append((i * 0.01, j * 0.01))
            i = j
        else:
            i += 1

    spans, s = [], first * 0.01
    for ps, pe in pauses:
        spans.append([s, ps])
        s = pe
    spans.append([s, last * 0.01])

    merged = []                                   # fold very short phrases into a neighbour
    for sp in spans:
        if merged and (merged[-1][1] - merged[-1][0] < min_len or sp[1] - sp[0] < min_len) \
                and sp[1] - merged[-1][0] <= max_len:
            merged[-1][1] = sp[1]
        else:
            merged.append(sp)

    final = []                                    # long stretches with no pause: cut at the quietest moment
    for s, e in merged:
        while e - s > max_len:
            a, b = int((s + max_len * 0.5) * 100), int((s + max_len) * 100)
            cut = (a + int(np.argmin(db[a:b]))) * 0.01
            final.append([s, cut])
            s = cut
        final.append([s, e])

    dur = len(y) / sr
    out = []
    for k, (s, e) in enumerate(final):
        lo = 0.0 if k == 0 else (final[k - 1][1] + s) / 2
        hi = dur if k == len(final) - 1 else (e + final[k + 1][0]) / 2
        out.append((max(lo, s - lead), min(hi, e + tail)))
    return out


# ---------------------------------------------------------------- 4. judge takes (--best-of)
_PUNCT = re.compile(r"[\s.,!?;:'\"“”‘’()\[\]\-–—।॥…]+")


def char_error_rate(ref, hyp):
    a, b = _PUNCT.sub("", ref.lower()), _PUNCT.sub("", hyp.lower())
    if not a:
        return 0.0 if not b else 1.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1] / len(a)


class Judge:
    """Voice similarity to the reference (Resemblyzer speaker embedding) and, when a speech
    recogniser exists for the language, how many of the source's words survive in a take."""

    def __init__(self, ref_y16, asr_model_id=None):
        from resemblyzer import VoiceEncoder, preprocess_wav
        self._pre = preprocess_wav
        self.enc = VoiceEncoder(device="cpu", verbose=False)
        self.ref = self._embed(ref_y16)
        self.asr = None
        if asr_model_id:
            import torch
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
            self.torch = torch
            self.proc = Wav2Vec2Processor.from_pretrained(asr_model_id)
            try:
                # Ask for safetensors directly. The Bangla model's main branch only has a .bin file; loading
                # that makes transformers fetch a converted safetensors copy in the background as well,
                # doubling the download (2.5 GB). Requesting safetensors fetches just the converted copy.
                self.asr = Wav2Vec2ForCTC.from_pretrained(asr_model_id, use_safetensors=True).eval()
            except (OSError, ValueError):
                self.asr = Wav2Vec2ForCTC.from_pretrained(asr_model_id).eval()

    def _embed(self, y16):
        wav = self._pre(np.asarray(y16, dtype=np.float32), source_sr=16000)
        if len(wav) < 1600:
            wav = np.asarray(y16, dtype=np.float32)
        return self.enc.embed_utterance(wav)

    def similarity(self, y16):
        try:
            e = self._embed(y16)
        except Exception:
            return 0.0
        return float(np.dot(e, self.ref) / (np.linalg.norm(e) * np.linalg.norm(self.ref) + 1e-9))

    def transcribe(self, y16):
        if self.asr is None:
            return None
        with self.torch.inference_mode():
            x = self.proc(y16, sampling_rate=16000, return_tensors="pt").input_values
            ids = self.asr(x).logits.argmax(-1)
        return self.proc.batch_decode(ids)[0]


# ---------------------------------------------------------------- 5. polish
def polish(y, sr, target_lufs=-16.0):
    from scipy import signal

    def env(x, rel):
        c = math.exp(-1 / (rel * sr))
        return signal.lfilter([1 - c], [1, -c], np.abs(x))

    y = signal.sosfiltfilt(signal.butter(2, 75, "high", fs=sr, output="sos"), y)        # rumble
    b, a = signal.iirpeak(3200 / (sr / 2), 1.2)
    y = y + 0.18 * signal.filtfilt(b, a, y)                                               # presence
    lo = signal.sosfiltfilt(signal.butter(4, 4800, "low", fs=sr, output="sos"), y)
    hi = y - lo
    ratio = 20 * np.log10((env(hi, 0.03) + 1e-6) / (env(lo, 0.03) + 1e-6))
    y = lo + hi * 10 ** (-np.clip((ratio + 6) * 0.6, 0, 6) / 20)                          # de-ess
    lvl = 20 * np.log10(env(y, 0.12) + 1e-6)
    active = lvl > np.percentile(lvl, 95) - 35
    if active.any():
        thr = np.percentile(lvl[active], 60)
        y = y * 10 ** (-np.where(lvl > thr, (lvl - thr) * (1 - 1 / 3.0), 0) / 20)       # 3:1 compression
    try:
        import pyloudnorm as pyln
        meter = pyln.Meter(sr)
        for _ in range(3):                    # loudness target, then limit peaks, then re-check
            y = y * 10 ** ((target_lufs - meter.integrated_loudness(y)) / 20)
            y = limit(y, sr)
    except Exception:
        y = limit(y / (np.max(np.abs(y)) + 1e-9) * 0.89, sr)
    return y


def limit(y, sr, ceiling=0.89):
    """Peak limiter (about -1 dBFS): turns down only the brief peaks, not the whole file."""
    from scipy.ndimage import minimum_filter1d, uniform_filter1d
    need = np.minimum(1.0, ceiling / (np.abs(y) + 1e-9))
    w = int(0.02 * sr)
    gain = uniform_filter1d(minimum_filter1d(need, size=w), size=w // 2)
    return np.clip(y * gain, -ceiling, ceiling)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(
        description="Make a recording sound like another person's voice (speech-to-speech voice conversion).")
    ap.add_argument("source", help="recording whose words and timing you want to keep (wav, mp3, m4a...)")
    ap.add_argument("reference", help="20-60 s of clean speech in the target voice")
    ap.add_argument("-o", "--out", default="converted.wav", help="output file (default: converted.wav)")
    ap.add_argument("--best-of", action="store_true",
                    help="make 6 takes per phrase and keep the best (about 6x slower, clearer words)")
    ap.add_argument("--lang", default="bn",
                    help="language of the recording, for the word check in --best-of (default: bn; 'none' to skip)")
    ap.add_argument("--asr-model", default=None, help="Hugging Face wav2vec2 model id for the word check")
    ap.add_argument("--keep-takes", metavar="DIR",
                    help="save every take plus takes.csv here, so you can swap a take by ear")
    ap.add_argument("--no-preshift", action="store_true", help="skip the pitch/formant pre-shift")
    ap.add_argument("--no-polish", action="store_true", help="skip EQ, de-ess, compression and loudness")
    ap.add_argument("--lufs", type=float, default=-16.0, help="loudness of the finished voice (default -16)")
    ap.add_argument("--device", default="auto", help="auto, cpu or cuda")
    args = ap.parse_args()

    t_start = time.time()
    import librosa
    import torch
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

    log("Reading audio...")
    src = load_audio(args.source, OUT_SR)
    ref = load_audio(args.reference, OUT_SR)
    ref, _ = librosa.effects.trim(ref, top_db=35)
    if len(ref) > 60 * OUT_SR:
        ref = ref[:60 * OUT_SR]
    if len(ref) < 8 * OUT_SR:
        log("  Warning: the reference is under 8 s. 20-60 s of speech gives a closer match.")

    sp, rp = pitch_profile(src, OUT_SR), pitch_profile(ref, OUT_SR)
    if sp is None:
        sys.exit("Could not find clear speech in the source recording.")
    if rp is None:
        sys.exit("Could not find clear speech in the reference.")
    median, formant, rng, ratio = preshift_settings(sp, rp)
    floor = float(np.clip(sp[1] * 0.6, 50, 100))
    ceiling = float(np.clip(sp[2] * 1.8, 300, 700))
    log(f"  Source voice pitch {sp[0]:.0f} Hz, reference {rp[0]:.0f} Hz")

    if args.no_preshift:
        takes = [None]
    elif args.best_of:
        f, r = formant, rng
        takes = [(f, r), (f * 0.93, r + 0.15), (f, r + 0.15), (f * 1.07, r - 0.05), (f * 1.07, r + 0.15), (f * 0.93, r - 0.05)]
        takes = [(float(np.clip(a, 0.75, 1.25)), float(np.clip(b, 0.5, 1.3))) for a, b in takes]
        if 0.8 < ratio < 1.25:
            takes.append(None)              # voices already close: also try no pre-shift
    else:
        takes = [(formant, rng)] if abs(math.log(ratio)) > math.log(1.03) else [None]
    if takes[0] is not None:
        log(f"  Pre-shift: pitch median -> {median:.0f} Hz, formant ratio {takes[0][0]:.2f}, pitch range {takes[0][1]:.2f}")

    chunks = speech_chunks(src, OUT_SR)
    log(f"  {len(chunks)} phrases to convert, {len(takes)} take(s) each")

    tmp = tempfile.mkdtemp(prefix="voice_match_")
    try:
        ref_path = os.path.join(tmp, "reference.wav")
        save_audio(ref_path, resample(ref, OUT_SR, WORK_SR), WORK_SR)

        log("Loading the voice converter (the first run downloads it)...")
        from TTS.api import TTS
        vc = TTS(VC_MODEL, progress_bar=False).to(device)

        judge = None
        if args.best_of:
            asr_id = args.asr_model or (None if args.lang.lower() == "none" else ASR_MODELS.get(args.lang.lower()))
            if not asr_id:
                log(f"  No word checker for language '{args.lang}': picking takes by voice similarity only.")
            log("Loading the take judges (the first run downloads the speech recogniser)...")
            judge = Judge(resample(ref, OUT_SR, 16000), asr_id)

        if args.keep_takes:
            os.makedirs(args.keep_takes, exist_ok=True)
        rows = []
        out = np.zeros(len(src), dtype=np.float32)
        fade = int(0.01 * OUT_SR)

        for ci, (s0, s1) in enumerate(chunks):
            seg = src[int(s0 * OUT_SR):int(s1 * OUT_SR)]
            src_path = os.path.join(tmp, f"src_{ci}.wav")
            save_audio(src_path, resample(seg, OUT_SR, WORK_SR), WORK_SR)
            src_text = judge.transcribe(resample(seg, OUT_SR, 16000)) if judge else None

            results = []
            for k, t in enumerate(takes):
                inp = src_path
                if t is not None:
                    inp = os.path.join(tmp, f"pre_{ci}_{k}.wav")
                    preshift(src_path, inp, median, t[0], t[1], floor, ceiling)
                outp = os.path.join(tmp, f"vc_{ci}_{k}.wav")
                vc.voice_conversion_to_file(source_wav=inp, target_wav=ref_path, file_path=outp)
                y = load_audio(outp, OUT_SR)
                sim = kept = None
                if judge:
                    y16 = resample(y, OUT_SR, 16000)
                    sim = judge.similarity(y16)
                    if src_text:
                        kept = 1.0 - min(1.0, char_error_rate(src_text, judge.transcribe(y16)))
                results.append({"k": k, "y": y, "sim": sim, "kept": kept, "path": outp})

            if judge and len(results) > 1:
                best_sim = max(r["sim"] for r in results)
                for r in results:
                    r["score"] = (0.6 * r["sim"] + (r["kept"] if r["kept"] is not None else 0.0)
                                  - (0.5 if r["sim"] < best_sim - 0.05 else 0.0)   # don't trade away the voice
                                  - (0.02 if r["k"] else 0.0))                      # slight preference for the base take
                chosen = max(results, key=lambda r: r["score"])
            else:
                chosen = results[0]

            y = chosen["y"][:len(seg)]
            if len(y) < len(seg):
                y = np.pad(y, (0, len(seg) - len(y)))
            y = y.copy()
            if len(y) > 2 * fade:
                y[:fade] *= np.linspace(0, 1, fade)
                y[-fade:] *= np.linspace(1, 0, fade)
            i0 = int(s0 * OUT_SR)
            out[i0:i0 + len(y)] = y[:len(out) - i0]

            msg = f"  phrase {ci + 1}/{len(chunks)} ({s1 - s0:.1f} s)"
            if judge and len(results) > 1:
                msg += f": take {chosen['k'] + 1} of {len(results)}, voice similarity {chosen['sim']:.2f}"
                if chosen["kept"] is not None:
                    msg += f", words kept {chosen['kept']:.0%}"
            log(msg)

            if args.keep_takes:
                save_audio(os.path.join(args.keep_takes, f"phrase{ci + 1:02d}_source.wav"), seg, OUT_SR)
                for r in results:
                    name = f"phrase{ci + 1:02d}_take{r['k'] + 1}.wav"
                    save_audio(os.path.join(args.keep_takes, name), r["y"], OUT_SR)
                    t = takes[r["k"]]
                    rows.append([ci + 1, f"{s0:.2f}", f"{s1:.2f}", r["k"] + 1,
                                 "-" if t is None else f"{t[0]:.2f}", "-" if t is None else f"{t[1]:.2f}",
                                 "" if r["sim"] is None else f"{r['sim']:.3f}",
                                 "" if r["kept"] is None else f"{r['kept']:.2f}",
                                 "yes" if r is chosen else ""])

        if args.keep_takes and rows:
            with open(os.path.join(args.keep_takes, "takes.csv"), "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["phrase", "start_s", "end_s", "take", "formant_ratio", "pitch_range",
                            "voice_similarity", "words_kept", "chosen"])
                w.writerows(rows)

        if not args.no_polish:
            out = polish(out, OUT_SR, args.lufs)
        d = os.path.dirname(os.path.abspath(args.out))
        os.makedirs(d, exist_ok=True)
        save_audio(args.out, out, OUT_SR)
        mins, secs = divmod(int(time.time() - t_start), 60)
        log(f"Saved {args.out} ({len(out) / OUT_SR:.1f} s of audio, took {mins}m{secs:02d}s)")

        try:
            j = judge or Judge(resample(ref, OUT_SR, 16000))
            before = j.similarity(resample(src, OUT_SR, 16000))
            after = j.similarity(resample(out, OUT_SR, 16000))
            log(f"Voice similarity to the reference: {before:.2f} before, {after:.2f} after "
                f"(two clips of the same person usually score 0.85 or more)")
        except Exception:
            pass
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
