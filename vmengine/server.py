"""Long-running engine server for the desktop app: JSON lines over stdin/stdout, models loaded once.

Requests (one JSON object per line on stdin):
  {"id": 7, "cmd": "analyze", "args": {...}}

Replies and events (one JSON object per line on stdout):
  {"type": "ready", "version": "1.1.0", "protocol": 1, "device": "cpu", "models": {...}}
  {"type": "result", "id": 7, "ok": true, "data": {...}}
  {"type": "result", "id": 7, "ok": false, "error": {"code": "models_missing", "message": "..."}}
  {"type": "event", "id": 7, "event": "progress", "data": {...}}
  {"type": "log", "message": "..."}

Commands (all args optional unless marked *):
  ping, status
  load_models   {words}                                   preload converter (+ recogniser when words)
  profile       {path*}                                   pitch profile, contour and waveform of one file
  analyze       {read*, reference*}                       pitch of both voices, automatic settings, phrases
  render_phrase {index*, settings, override, mode, words, finish}
                                                          one phrase -> takes with audio paths and scores
  render_all    {settings, overrides, mode, words}        every phrase, with "progress" events; cancellable
  take_audio    {index*, take*, finish}                   path of one take (finished like the preview)
  choose        {index*, take*}                           pick the take used for export
  mix           {finish}                                  the whole read with the chosen takes, for playback
  export        {out*, finish, takes_dir, report_csv}     the finished WAV (48 kHz, 24-bit)
  cancel        {}                                        stop the running render (handled immediately)
  shutdown      {}

Jobs run one at a time on a worker thread, so `cancel` and `ping` always answer at once. A queued
render_phrase that hasn't started is dropped (error "superseded") when a newer one arrives.
"""
import argparse
import hashlib
import json
import os
import queue
import sys
import threading
import traceback

PROTOCOL = 1


class Server:
    def __init__(self, models_dir, device, workdir=None):
        self._out = sys.stdout
        sys.stdout = sys.stderr                     # library prints must never corrupt the protocol
        self._lock = threading.Lock()
        self._jobs = queue.Queue()
        self._pending = []                          # queued (not started) jobs, for superseding
        self._pending_lock = threading.Lock()
        self.models_dir = models_dir
        self.device = device
        self.workdir = workdir
        self.session = None

    # ------------------------------------------------------------ output
    def send(self, obj):
        line = json.dumps(obj, ensure_ascii=False, default=_jsonable)
        with self._lock:
            self._out.write(line + "\n")
            self._out.flush()

    def log(self, msg):
        self.send({"type": "log", "message": msg})

    def reply(self, rid, data=None, error=None):
        if error is None:
            self.send({"type": "result", "id": rid, "ok": True, "data": data})
        else:
            self.send({"type": "result", "id": rid, "ok": False, "error": error})

    # ------------------------------------------------------------ main loops
    def run(self):
        from . import __version__
        from . import models as M
        M.go_offline()
        from .pipeline import Session
        self.session = Session(self.models_dir, device=self.device, workdir=self.workdir, log=self.log)
        os.makedirs(os.path.join(self.session.workdir, "audio"), exist_ok=True)
        self.send({"type": "ready", "version": __version__, "protocol": PROTOCOL, "device": self.session.device,
                   "models": M.status(self.session.models_dir), "workdir": self.session.workdir})
        worker = threading.Thread(target=self._work, daemon=True)
        worker.start()
        try:
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue
                try:
                    req = json.loads(line)
                except json.JSONDecodeError as e:
                    self.reply(None, error={"code": "bad_request", "message": f"Not JSON: {e}"})
                    continue
                rid, cmd, args = req.get("id"), req.get("cmd"), req.get("args") or {}
                if cmd == "cancel":
                    self.session.cancel_event.set()
                    self.reply(rid, {"cancelled": True})
                elif cmd == "ping":
                    self.reply(rid, {"pong": True})
                elif cmd == "shutdown":
                    self.reply(rid, {"bye": True})
                    break
                else:
                    job = {"id": rid, "cmd": cmd, "args": args, "dropped": False}
                    with self._pending_lock:
                        if cmd == "render_phrase":
                            for old in self._pending:
                                if old["cmd"] == "render_phrase":
                                    old["dropped"] = True
                        self._pending.append(job)
                    self._jobs.put(job)
        finally:
            self._jobs.put(None)
            self.session.cancel_event.set()
            worker.join(timeout=5)
            self.session.close()

    def _work(self):
        from . import Cancelled, EngineError
        while True:
            job = self._jobs.get()
            if job is None:
                return
            with self._pending_lock:
                if job in self._pending:
                    self._pending.remove(job)
            if job["dropped"]:
                self.reply(job["id"], error={"code": "superseded", "message": "A newer preview replaced this one."})
                continue
            handler = getattr(self, "cmd_" + str(job["cmd"]), None)
            if handler is None:
                self.reply(job["id"], error={"code": "bad_request", "message": f"Unknown command {job['cmd']!r}"})
                continue
            self.session.cancel_event.clear()
            try:
                self.reply(job["id"], handler(job["id"], **job["args"]))
            except Cancelled:
                self.reply(job["id"], error={"code": "cancelled", "message": "Cancelled."})
            except EngineError as e:
                self.reply(job["id"], error={"code": e.code, "message": str(e)})
            except TypeError as e:
                self.reply(job["id"], error={"code": "bad_request", "message": str(e)})
            except Exception as e:
                traceback.print_exc()
                self.reply(job["id"], error={"code": "internal", "message": f"{type(e).__name__}: {e}"})

    # ------------------------------------------------------------ helpers
    def _path(self, *parts):
        return os.path.join(self.session.workdir, "audio", *parts)

    def _write(self, name, y):
        from . import audio as A
        p = self._path(name)
        if not os.path.exists(p):
            A.save_audio(p, y, A.OUT_SR)
        return p

    @staticmethod
    def _tag(*items):
        return hashlib.sha1(json.dumps(items, sort_keys=True, default=_jsonable).encode()).hexdigest()[:10]

    def _take_file(self, i, t, finish):
        """A take as a finished file, level-matched with the export, for preview playback."""
        s = self.session
        tag = self._tag(t["formant"], t["pitch_range"], s.results[i]["settings"], finish)
        name = f"p{i + 1:02d}_t{t['k'] + 1}_{tag}.wav"
        if os.path.exists(self._path(name)):
            return self._path(name)
        return self._write(name, s.finish(t["y"], finish))

    def _phrase_summary(self, i, finish=None, with_audio=True):
        s = self.session
        r = s.results[i]
        out = {"index": i, "start": s.phrases[i][0], "end": s.phrases[i][1], "chosen": r["chosen"],
               "mode": r["mode"], "settings": r["settings"], "takes": []}
        for t in r["takes"]:
            item = {"take": t["k"], "similarity": t["sim"], "words_kept": t["kept"],
                    "formant": t["formant"], "pitch_range": t["pitch_range"]}
            if with_audio:
                item["path"] = self._take_file(i, t, finish)
            out["takes"].append(item)
        if with_audio:
            out["source_path"] = self._write(f"p{i + 1:02d}_source.wav", s.phrase_audio(i))
        return out

    def _asr(self, words):
        from . import models as M
        if not words:
            return None
        return M.resolve_asr_id(words if isinstance(words, str) and words not in ("1", "true") else "bn")

    # ------------------------------------------------------------ commands
    def cmd_status(self, rid):
        from . import models as M
        s = self.session
        return {"device": s.device, "models": M.status(s.models_dir), "analyzed": s.src is not None,
                "converter_loaded": s.converter is not None, "recogniser_loaded": s._judge_asr is not None}

    def cmd_load_models(self, rid, words=False):
        self.session.ensure_converter()
        self.session.ensure_judge(self._asr(words))
        return self.cmd_status(rid)

    def cmd_profile(self, rid, path):
        from . import EngineError
        from . import audio as A
        if not os.path.isfile(path):
            raise EngineError(f"The file doesn't exist: {path}", "file_not_found")
        import librosa
        y = A.load_audio(path, A.OUT_SR)
        trimmed, _ = librosa.effects.trim(y, top_db=35)
        p = A.pitch_profile(trimmed, A.OUT_SR)
        return {"path": path, "duration": len(y) / A.OUT_SR, "speech_duration": len(trimmed) / A.OUT_SR,
                "pitch": None if p is None else {"median": p[0], "p10": p[1], "p90": p[2]},
                "contour": A.pitch_contour(trimmed, A.OUT_SR), "peaks": A.peaks(y)}

    def cmd_analyze(self, rid, read, reference, display=True):
        info = self.session.analyze(read, reference, with_display=display)
        return info

    def cmd_render_phrase(self, rid, index, settings=None, override=None, mode="fast", words=False, finish=None):
        s = self.session
        s.render_phrase(int(index), settings, override, mode=mode, score=True, asr_id=self._asr(words))
        return self._phrase_summary(int(index), finish)

    def cmd_render_all(self, rid, settings=None, overrides=None, mode="fast", words=None):
        s = self.session
        if words is None:
            words = mode == "best"
        asr_id = self._asr(words)
        if mode == "best" or asr_id:
            s.ensure_judge(asr_id)

        def progress(i, n, res):
            self.send({"type": "event", "id": rid, "event": "progress",
                       "data": {"done": i + 1, "total": n, "phrase": self._phrase_summary(i, with_audio=False)}})

        s.render_all(settings, {int(k): v for k, v in (overrides or {}).items()}, mode=mode, asr_id=asr_id,
                     score=True, progress=progress)
        return {"phrases": [self._phrase_summary(i, with_audio=False) for i in sorted(s.results)]}

    def cmd_take_audio(self, rid, index, take, finish=None):
        s = self.session
        i, k = int(index), int(take)
        if i not in s.results or not 0 <= k < len(s.results[i]["takes"]):
            from . import EngineError
            raise EngineError(f"Phrase {i + 1} has no take {k + 1}.", "bad_take")
        return {"path": self._take_file(i, s.results[i]["takes"][k], finish),
                "source_path": self._write(f"p{i + 1:02d}_source.wav", s.phrase_audio(i))}

    def cmd_choose(self, rid, index, take):
        self.session.choose(int(index), int(take))
        return {"index": int(index), "chosen": int(take)}

    def cmd_mix(self, rid, finish=None):
        s = self.session
        y = s.finish(s.assemble(), finish)
        tag = self._tag(sorted((i, r["chosen"], r["settings"]) for i, r in s.results.items()), finish)
        p = self._path(f"mix_{tag}.wav")
        from . import audio as A
        A.save_audio(p, y, A.OUT_SR)
        return {"path": p, "rendered": sorted(s.results)}

    def cmd_export(self, rid, out, finish=None, takes_dir=None, report_csv=None):
        return self.session.export(out, finish, takes_dir=takes_dir, report_csv=report_csv)


def _jsonable(o):
    import numpy as np
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    return str(o)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="voice_match.py serve", description="JSON-lines engine server for the app.")
    ap.add_argument("--models-dir", help="where the models are")
    ap.add_argument("--device", default="cpu", help="cpu (default), mps or cuda")
    ap.add_argument("--workdir", help="folder for preview audio (default: a temporary folder)")
    args = ap.parse_args(argv)
    try:
        Server(args.models_dir, args.device, args.workdir).run()
    except Exception as e:  # start-up failure: report it on the protocol so the app can show it
        print(json.dumps({"type": "fatal", "error": {"code": getattr(e, "code", "internal"), "message": str(e)}}),
              file=sys.__stdout__, flush=True)
        return 1
    return 0
