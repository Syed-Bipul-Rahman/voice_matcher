"""The conversion pipeline as a reusable session: analyze once, render phrases with any settings, export.

  s = Session(models_dir, device="cpu")
  info = s.analyze("read.m4a", "reference.mp3")          # pitch, automatic settings, phrases
  s.render_all(info["auto"], mode="best")                  # or render_phrase(i, settings) for previews
  s.export("voiceover.wav")

Settings (one dict for the whole read, optionally overridden per phrase):
  preshift      "auto" | "on" | "off"   auto: on when the voices' pitch differs by more than 3 %
  pitch_median  Hz the pre-shift moves the read's pitch median to (auto: the reference's median)
  formant       formant ratio, "voice size" (auto: (reference pitch / read pitch) ** 0.2)
  pitch_range   intonation range factor (auto: 1.1 x reference range / read range, in semitones)
  match         match strength, 0..1: blends the read's tone-colour embedding (0) with the reference's (1)
Finish settings: presence, deess, lufs, enabled (see audio.polish).
"""
import csv
import math
import os
import shutil
import tempfile
import threading

import numpy as np

from . import Cancelled, EngineError
from . import audio as A
from . import models as M

FINISH_DEFAULT = {"enabled": True, "presence": A.DEFAULT_PRESENCE, "deess": A.DEFAULT_DEESS, "lufs": -16.0}
SETTING_KEYS = ("preshift", "pitch_median", "formant", "pitch_range", "match")


def _r(x):
    return None if x is None else round(float(x), 4)


class Judge:
    """Voice similarity to the reference (Resemblyzer speaker embedding) and, when a recogniser is given,
    a transcript to check how many of the source's words survive in a take."""

    def __init__(self, ref_y16, recogniser=None, device="cpu"):
        from resemblyzer import VoiceEncoder, preprocess_wav
        self.device = device
        self._pre = preprocess_wav
        self.enc = VoiceEncoder(device="cpu", verbose=False)
        self.ref = self._embed(ref_y16)
        self.proc, self.asr = recogniser if recogniser else (None, None)
        if self.asr is not None and device != "cpu":
            self.asr = self.asr.to(device)

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
        import torch
        with torch.inference_mode():
            x = self.proc(y16, sampling_rate=16000, return_tensors="pt").input_values.to(self.device)
            ids = self.asr(x).logits.argmax(-1).cpu()
        return self.proc.batch_decode(ids)[0]


class Converter:
    """OpenVoice v2 tone-colour converter, driven directly so the reference embedding is computed once
    and can be blended with the read's (match strength)."""

    def __init__(self, models_dir, device="cpu"):
        self.device = device
        self.model = M.load_converter(models_dir, device)
        self.sr = int(self.model.config.audio.input_sample_rate)       # 22050
        self._ref_key = None
        self._ref_se = None

    def set_reference(self, key, ref24):
        if key != self._ref_key:
            se, _ = self.model.extract_se(A.resample(ref24, A.WORK_SR, self.sr).astype(np.float32))
            self._ref_key, self._ref_se = key, se

    def convert(self, x24, match=1.0):
        """Convert a 24 kHz chunk; returns audio at 48 kHz, peak-normalised like coqui-tts's own file writer
        (which the original tool went through, and whose levels the polish stage was tuned on)."""
        import torch
        src_se, spec = self.model.extract_se(A.resample(x24, A.WORK_SR, self.sr).astype(np.float32))
        g_tgt = self._ref_se if match >= 1.0 else match * self._ref_se + (1.0 - match) * src_se
        with torch.inference_mode():
            out = self.model.inference(spec, {"g_src": src_se, "g_tgt": g_tgt})["model_outputs"]
        y = out[0, 0].detach().float().cpu().numpy()
        y = y / max(0.01, float(np.max(np.abs(y))))
        return A.resample(y, self.sr, A.OUT_SR).astype(np.float32)


class Session:
    def __init__(self, models_dir=None, device="cpu", workdir=None, log=None):
        import torch
        torch.set_num_threads(max(1, os.cpu_count() or 1))
        self.models_dir = str(models_dir or M.default_models_dir())
        self.device = self._pick_device(device)
        self.log = log or (lambda msg: None)
        self._own_workdir = workdir is None
        self.workdir = workdir or tempfile.mkdtemp(prefix="voice_match_")
        os.makedirs(self.workdir, exist_ok=True)
        self.converter = None
        self.judge = None
        self._judge_asr = None
        self.asr_id = None
        self.cancel_event = threading.Event()
        self.src = self.ref = None
        self._cache = {}          # (phrase, key) -> 48 kHz take
        self.results = {}         # phrase -> {"takes": [...], "chosen": k}

    @staticmethod
    def _pick_device(device):
        import torch
        if device in (None, "auto"):
            return "cuda" if torch.cuda.is_available() else "cpu"
        if device == "mps" and not torch.backends.mps.is_available():
            raise EngineError("Apple GPU (MPS) isn't available on this machine.", "bad_device")
        return device

    def close(self):
        if self._own_workdir:
            shutil.rmtree(self.workdir, ignore_errors=True)

    def _check_cancel(self):
        if self.cancel_event.is_set():
            raise Cancelled()

    # ------------------------------------------------------------ models
    def ensure_converter(self):
        if self.converter is None:
            self.log("Loading the voice converter...")
            self.converter = Converter(self.models_dir, self.device)
        if self.ref is not None:
            self.converter.set_reference(self.ref_path, self.ref24)
        return self.converter

    def ensure_judge(self, asr_id=None):
        """Load the similarity judge, plus the recogniser `asr_id` if given (None = similarity only)."""
        if self.judge is None or (asr_id and self._judge_asr != asr_id):
            rec = None
            if asr_id:
                self.log("Loading the speech recogniser...")
                rec = M.load_recogniser(self.models_dir, asr_id)
            if self.judge is not None and rec:
                self.judge.proc, self.judge.asr = rec[0], rec[1].to(self.judge.device)
            else:
                self.judge = Judge(A.resample(self.ref, A.OUT_SR, 16000), rec,
                                   device="mps" if self.device == "mps" else "cpu")
            self._judge_asr = asr_id if rec else self._judge_asr
        return self.judge

    # ------------------------------------------------------------ 1. analyze
    def analyze(self, source, reference, with_display=False):
        import librosa
        for p, what in ((source, "read"), (reference, "reference")):
            if not os.path.isfile(p):
                raise EngineError(f"The {what} file doesn't exist: {p}", "file_not_found")
        src = A.load_audio(source, A.OUT_SR)
        ref = A.load_audio(reference, A.OUT_SR)
        ref, _ = librosa.effects.trim(ref, top_db=35)
        if len(ref) > 60 * A.OUT_SR:
            ref = ref[:60 * A.OUT_SR]
        warnings = []
        if len(ref) < 8 * A.OUT_SR:
            warnings.append("The reference is under 8 s. 20-60 s of speech gives a closer match.")
        sp, rp = A.pitch_profile(src, A.OUT_SR), A.pitch_profile(ref, A.OUT_SR)
        if sp is None:
            raise EngineError("Could not find clear speech in the source recording.", "no_speech")
        if rp is None:
            raise EngineError("Could not find clear speech in the reference.", "no_speech")
        median, formant, rng, ratio = A.preshift_settings(sp, rp)

        if (source, reference) != (getattr(self, "src_path", None), getattr(self, "ref_path", None)):
            self._cache.clear()
            self.results.clear()
            self.judge = None
            self._judge_asr = None
        self.src_path, self.ref_path = source, reference
        self.src, self.ref, self.sp, self.rp, self.ratio = src, ref, sp, rp, ratio
        self.ref24 = A.resample(ref, A.OUT_SR, A.WORK_SR)
        self.floor = float(np.clip(sp[1] * 0.6, 50, 100))
        self.ceiling = float(np.clip(sp[2] * 1.8, 300, 700))
        self.phrases = A.speech_chunks(src, A.OUT_SR)
        self.auto = {"preshift": "auto", "pitch_median": _r(median), "formant": _r(formant),
                     "pitch_range": _r(rng), "match": 1.0}
        info = {
            "read": {"path": source, "duration": len(src) / A.OUT_SR,
                     "pitch": {"median": sp[0], "p10": sp[1], "p90": sp[2]}},
            "reference": {"path": reference, "duration": len(ref) / A.OUT_SR,
                          "pitch": {"median": rp[0], "p10": rp[1], "p90": rp[2]}},
            "pitch_ratio": ratio,
            "auto": dict(self.auto),
            "preshift_auto_on": self._auto_preshift_on(),
            "phrases": [{"index": i, "start": round(s, 3), "end": round(e, 3)} for i, (s, e) in enumerate(self.phrases)],
            "warnings": warnings,
        }
        if with_display:
            info["read"]["peaks"] = A.peaks(src)
            info["reference"]["peaks"] = A.peaks(ref)
            info["read"]["contour"] = A.pitch_contour(src, A.OUT_SR)
            info["reference"]["contour"] = A.pitch_contour(ref, A.OUT_SR)
        return info

    def _need_analysis(self):
        if self.src is None:
            raise EngineError("Analyze a read and a reference first.", "not_analyzed")

    # ------------------------------------------------------------ settings -> takes
    def _auto_preshift_on(self):
        return abs(math.log(self.ratio)) > math.log(1.03)

    def effective(self, settings=None, override=None):
        s = dict(self.auto)
        s.update({k: v for k, v in (settings or {}).items() if k in SETTING_KEYS and v is not None})
        s.update({k: v for k, v in (override or {}).items() if k in SETTING_KEYS and v is not None})
        return s

    def take_plan(self, s, mode):
        """List of (formant, range) pre-shifts, or None for no pre-shift, one per take."""
        f, r = float(s["formant"]), float(s["pitch_range"])
        if s["preshift"] == "off":
            return [None]
        if mode == "best":
            takes = [(f, r), (f * 0.93, r + 0.15), (f, r + 0.15), (f * 1.07, r - 0.05), (f * 1.07, r + 0.15),
                     (f * 0.93, r - 0.05)]
            takes = [(float(np.clip(a, 0.75, 1.25)), float(np.clip(b, 0.5, 1.3))) for a, b in takes]
            if 0.8 < self.ratio < 1.25:
                takes.append(None)              # voices already close: also try no pre-shift
            return takes
        if s["preshift"] == "on" or self._auto_preshift_on():
            return [(f, r)]
        return [None]

    # ------------------------------------------------------------ 3. convert
    def phrase_audio(self, i):
        s0, s1 = self.phrases[i]
        return self.src[int(s0 * A.OUT_SR):int(s1 * A.OUT_SR)]

    def _render_take(self, i, seg24, shift, s):
        key = (i, None if shift is None else (_r(shift[0]), _r(shift[1]), _r(s["pitch_median"])), _r(s["match"]))
        if key in self._cache:
            return self._cache[key]
        x = seg24
        if shift is not None:
            x = A.preshift(seg24, A.WORK_SR, s["pitch_median"], shift[0], shift[1], self.floor, self.ceiling)
        y = self.converter.convert(x, float(s["match"]))
        self._cache[key] = y
        return y

    def render_phrase(self, i, settings=None, override=None, mode="fast", score=False, asr_id=None):
        """Render phrase i. In best mode every take is scored and the best one chosen.
        `score` also scores a fast take (similarity, and words kept if `asr_id` is given)."""
        self._need_analysis()
        if not 0 <= i < len(self.phrases):
            raise EngineError(f"There is no phrase {i + 1}.", "bad_phrase")
        self.ensure_converter()
        s = self.effective(settings, override)
        plan = self.take_plan(s, mode)
        judge = self.ensure_judge(asr_id) if (mode == "best" or score) else None
        seg = self.phrase_audio(i)
        seg24 = A.resample(seg, A.OUT_SR, A.WORK_SR)
        src_text = judge.transcribe(A.resample(seg, A.OUT_SR, 16000)) if judge and judge.asr is not None else None

        takes = []
        for k, shift in enumerate(plan):
            self._check_cancel()
            y = self._render_take(i, seg24, shift, s)
            sim = kept = None
            if judge:
                y16 = A.resample(y, A.OUT_SR, 16000)
                sim = judge.similarity(y16)
                if src_text:
                    kept = 1.0 - min(1.0, A.char_error_rate(src_text, judge.transcribe(y16)))
            takes.append({"k": k, "y": y, "sim": sim, "kept": kept,
                          "formant": None if shift is None else shift[0],
                          "pitch_range": None if shift is None else shift[1]})

        if mode == "best" and len(takes) > 1:
            best_sim = max(t["sim"] for t in takes)
            for t in takes:
                t["score"] = (0.6 * t["sim"] + (t["kept"] if t["kept"] is not None else 0.0)
                              - (0.5 if t["sim"] < best_sim - 0.05 else 0.0)   # don't trade away the voice
                              - (0.02 if t["k"] else 0.0))                      # slight preference for the base take
            chosen = max(takes, key=lambda t: t["score"])["k"]
        else:
            chosen = 0
        self.results[i] = {"takes": takes, "chosen": chosen, "mode": mode, "settings": s}
        return self.results[i]

    def render_all(self, settings=None, overrides=None, mode="fast", asr_id=None, score=False, progress=None):
        """Render every phrase. `overrides` maps phrase index -> settings dict. Raises Cancelled."""
        self._need_analysis()
        self.cancel_event.clear()
        overrides = overrides or {}
        n = len(self.phrases)
        for i in range(n):
            self._check_cancel()
            res = self.render_phrase(i, settings, overrides.get(i) or overrides.get(str(i)), mode=mode,
                                     score=score, asr_id=asr_id)
            if progress:
                progress(i, n, res)
        return self.results

    def choose(self, i, k):
        if i not in self.results or not 0 <= k < len(self.results[i]["takes"]):
            raise EngineError(f"Phrase {i + 1} has no take {k + 1}.", "bad_take")
        self.results[i]["chosen"] = k

    # ------------------------------------------------------------ assemble + export
    def assemble(self):
        """The chosen take of every phrase, each placed at its original time. Missing phrases are silent."""
        out = np.zeros(len(self.src), dtype=np.float32)
        fade = int(0.01 * A.OUT_SR)
        for i, (s0, s1) in enumerate(self.phrases):
            if i not in self.results:
                continue
            seg_len = int(s1 * A.OUT_SR) - int(s0 * A.OUT_SR)
            r = self.results[i]
            y = r["takes"][r["chosen"]]["y"][:seg_len]
            if len(y) < seg_len:
                y = np.pad(y, (0, seg_len - len(y)))
            y = y.copy()
            if len(y) > 2 * fade:
                y[:fade] *= np.linspace(0, 1, fade)
                y[-fade:] *= np.linspace(1, 0, fade)
            i0 = int(s0 * A.OUT_SR)
            out[i0:i0 + len(y)] = y[:len(out) - i0]
        return out

    def finish(self, y, finish=None):
        f = dict(FINISH_DEFAULT)
        f.update({k: v for k, v in (finish or {}).items() if v is not None})
        if not f["enabled"]:
            return y
        return A.polish(y, A.OUT_SR, float(f["lufs"]), presence=float(f["presence"]), deess=float(f["deess"]))

    def export(self, out_path, finish=None, takes_dir=None, report_csv=None, measure=True):
        self._need_analysis()
        missing = [i + 1 for i in range(len(self.phrases)) if i not in self.results]
        if missing:
            raise EngineError(f"Phrases {missing} haven't been rendered yet.", "not_rendered")
        out = self.finish(self.assemble(), finish)
        d = os.path.dirname(os.path.abspath(out_path))
        os.makedirs(d, exist_ok=True)
        A.save_audio(out_path, out, A.OUT_SR)
        rows = self._rows()
        if takes_dir:
            self.save_takes(takes_dir)
        if report_csv:
            self._write_csv(report_csv, rows)
        info = {"path": out_path, "duration": len(out) / A.OUT_SR, "read_duration": len(self.src) / A.OUT_SR,
                "peak_dbfs": 20 * math.log10(float(np.max(np.abs(out))) + 1e-9)}
        try:
            info["lufs"] = A.loudness(out, A.OUT_SR)
        except Exception:
            info["lufs"] = None
        if measure:
            j = self.ensure_judge(None)
            info["similarity_before"] = j.similarity(A.resample(self.src, A.OUT_SR, 16000))
            info["similarity_after"] = j.similarity(A.resample(out, A.OUT_SR, 16000))
        return info

    def _rows(self):
        rows = []
        for i, (s0, s1) in enumerate(self.phrases):
            r = self.results.get(i)
            if not r:
                continue
            for t in r["takes"]:
                rows.append([i + 1, f"{s0:.2f}", f"{s1:.2f}", t["k"] + 1,
                             "-" if t["formant"] is None else f"{t['formant']:.2f}",
                             "-" if t["pitch_range"] is None else f"{t['pitch_range']:.2f}",
                             "" if t["sim"] is None else f"{t['sim']:.3f}",
                             "" if t["kept"] is None else f"{t['kept']:.2f}",
                             "yes" if t["k"] == r["chosen"] else ""])
        return rows

    @staticmethod
    def _write_csv(path, rows):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["phrase", "start_s", "end_s", "take", "formant_ratio", "pitch_range",
                        "voice_similarity", "words_kept", "chosen"])
            w.writerows(rows)

    def save_takes(self, takes_dir):
        """Every take, each original phrase and takes.csv, so a take can be swapped by ear."""
        os.makedirs(takes_dir, exist_ok=True)
        for i in sorted(self.results):
            A.save_audio(os.path.join(takes_dir, f"phrase{i + 1:02d}_source.wav"), self.phrase_audio(i), A.OUT_SR)
            for t in self.results[i]["takes"]:
                A.save_audio(os.path.join(takes_dir, f"phrase{i + 1:02d}_take{t['k'] + 1}.wav"), t["y"], A.OUT_SR)
        self._write_csv(os.path.join(takes_dir, "takes.csv"), self._rows())
