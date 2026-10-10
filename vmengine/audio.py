"""Audio building blocks: read and write, measure pitch, pre-shift, split at pauses, polish."""
import math
import re
import shutil
import subprocess

import numpy as np

from . import EngineError

OUT_SR = 48000        # sample rate of the finished file
WORK_SR = 24000       # sample rate of the chunks handed to the pre-shift and converter


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
        raise EngineError(f"Could not read {path}. Install FFmpeg (macOS: brew install ffmpeg) "
                          "or convert it to WAV first.", "unreadable_audio")
    res = subprocess.run([ffmpeg, "-v", "error", "-i", path, "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
                         capture_output=True)
    if res.returncode != 0 or not res.stdout:
        raise EngineError(f"Could not decode {path}: {res.stderr.decode(errors='ignore')[:300]}",
                          "unreadable_audio")
    return np.frombuffer(res.stdout, dtype=np.float32).copy()


def save_audio(path, y, sr):
    import soundfile as sf
    subtype = "PCM_24" if path.lower().endswith((".wav", ".flac")) else None
    sf.write(path, np.asarray(y, dtype=np.float32), sr, subtype=subtype)


def resample(y, sr_in, sr_out):
    import librosa
    return librosa.resample(y, orig_sr=sr_in, target_sr=sr_out) if sr_in != sr_out else y


def peaks(y, bins=1200):
    """Min/max pairs for drawing a waveform: [[min, max], ...] with `bins` entries."""
    n = len(y) // bins
    if n < 1:
        return [[float(v), float(v)] for v in y]
    blocks = np.asarray(y[:n * bins]).reshape(bins, n)
    return np.round(np.stack([blocks.min(axis=1), blocks.max(axis=1)], axis=1), 4).tolist()


# ---------------------------------------------------------------- 1. measure
def _f0(y, sr):
    import parselmouth
    snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=sr)
    return snd.to_pitch(time_step=0.01, pitch_floor=60, pitch_ceiling=500).selected_array["frequency"]


def pitch_profile(y, sr):
    """(median, 10th percentile, 90th percentile) of the voice's pitch in Hz, or None."""
    f0 = _f0(y, sr)
    f0 = f0[f0 > 0]
    if len(f0) < 50:
        return None
    return float(np.median(f0)), float(np.percentile(f0, 10)), float(np.percentile(f0, 90))


def pitch_contour(y, sr, step=0.05):
    """Pitch every `step` seconds (Hz, 0 where unvoiced), for drawing."""
    f0 = _f0(y, sr)
    k = max(1, int(round(step / 0.01)))
    n = len(f0) // k
    out = []
    for i in range(n):
        v = f0[i * k:(i + 1) * k]
        v = v[v > 0]
        out.append(round(float(np.median(v)), 1) if len(v) else 0.0)
    return out


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
def preshift(y, sr, median, formant, rng, floor, ceiling):
    """Praat "Change gender" on an array: returns the shifted array at the same rate."""
    import parselmouth
    from parselmouth.praat import call
    snd = parselmouth.Sound(np.asarray(y, dtype=np.float64), sampling_frequency=sr)
    # Change gender: pitch floor, pitch ceiling, formant shift ratio, new pitch median, pitch range factor, duration factor
    out = call(snd, "Change gender", floor, ceiling, float(formant), float(median), float(rng), 1.0)
    v = out.values[0]
    if int(round(out.sampling_frequency)) != sr:
        v = resample(v, int(round(out.sampling_frequency)), sr)
    return np.asarray(v, dtype=np.float32)


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
        raise EngineError("No speech found in the source recording.", "no_speech")
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


# ---------------------------------------------------------------- word check
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


# ---------------------------------------------------------------- 5. polish
DEFAULT_PRESENCE = 0.18     # parallel peak at 3.2 kHz
DEFAULT_DEESS = 0.6         # de-esser slope above 4.8 kHz (0 = off)


def polish(y, sr, target_lufs=-16.0, presence=DEFAULT_PRESENCE, deess=DEFAULT_DEESS):
    from scipy import signal

    def env(x, rel):
        c = math.exp(-1 / (rel * sr))
        return signal.lfilter([1 - c], [1, -c], np.abs(x))

    y = signal.sosfiltfilt(signal.butter(2, 75, "high", fs=sr, output="sos"), y)        # rumble
    if presence:
        b, a = signal.iirpeak(3200 / (sr / 2), 1.2)
        y = y + presence * signal.filtfilt(b, a, y)                                       # presence
    if deess:
        lo = signal.sosfiltfilt(signal.butter(4, 4800, "low", fs=sr, output="sos"), y)
        hi = y - lo
        ratio = 20 * np.log10((env(hi, 0.03) + 1e-6) / (env(lo, 0.03) + 1e-6))
        y = lo + hi * 10 ** (-np.clip((ratio + 6) * deess, 0, 6) / 20)                    # de-ess
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


def loudness(y, sr):
    import pyloudnorm as pyln
    return float(pyln.Meter(sr).integrated_loudness(y))
