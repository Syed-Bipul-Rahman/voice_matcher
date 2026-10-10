#!/usr/bin/env python3
"""
server_check.py - drive `voice_match.py serve` the way the desktop app does, and check the replies.

  python tools/server_check.py READ REFERENCE OUT_DIR [--mode fast|best] [--device cpu]

Runs: analyze -> render_phrase (with two settings and match strength) -> cancel test -> render_all -> export,
and prints timings. Exits non-zero if any reply is wrong.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Client:
    def __init__(self, cmd, env=None):
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  text=True, bufsize=1, env=env)
        self.n = 0
        self.ready = self._next()
        threading.Thread(target=self._drain_err, daemon=True).start()

    def _drain_err(self):
        for _ in self.p.stderr:
            pass

    def _next(self):
        line = self.p.stdout.readline()
        if not line:
            raise SystemExit("server closed its output")
        return json.loads(line)

    def send(self, cmd, **args):
        self.n += 1
        self.p.stdin.write(json.dumps({"id": self.n, "cmd": cmd, "args": args}) + "\n")
        self.p.stdin.flush()
        return self.n

    def wait(self, rid, on_event=None):
        while True:
            m = self._next()
            if m.get("type") == "result" and m.get("id") == rid:
                return m
            if m.get("type") == "event" and on_event:
                on_event(m)

    def call(self, cmd, on_event=None, **args):
        t = time.time()
        m = self.wait(self.send(cmd, **args), on_event)
        return m, time.time() - t


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg, flush=True)
    if not cond:
        check.failed = True


check.failed = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("read")
    ap.add_argument("reference")
    ap.add_argument("out_dir")
    ap.add_argument("--mode", default="fast")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    t0 = time.time()
    c = Client([args.python, os.path.join(ROOT, "voice_match.py"), "serve", "--device", args.device], os.environ.copy())
    check(c.ready.get("type") == "ready", f"ready ({time.time() - t0:.1f} s): {c.ready.get('version')}, "
                                          f"device {c.ready.get('device')}, models ready {c.ready['models']['ready']}")

    m, dt = c.call("analyze", read=args.read, reference=args.reference)
    check(m["ok"], f"analyze ({dt:.1f} s)")
    info = m["data"]
    print(f"       read {info['read']['pitch']['median']:.0f} Hz, reference {info['reference']['pitch']['median']:.0f} Hz,"
          f" auto {info['auto']}, {len(info['phrases'])} phrases")

    m, dt = c.call("render_phrase", index=0, words=True)
    check(m["ok"] and os.path.isfile(m["data"]["takes"][0]["path"]),
          f"render_phrase 1, auto settings, first call loads models ({dt:.1f} s): "
          f"similarity {m['data']['takes'][0]['similarity']:.3f}, words kept {m['data']['takes'][0]['words_kept']}")
    base = m["data"]["takes"][0]
    for label, settings in (("formant 1.0", {"formant": 1.0}), ("pitch 160 Hz", {"pitch_median": 160}),
                            ("match 0.5", {"match": 0.5}), ("match 0.0", {"match": 0.0})):
        m, dt = c.call("render_phrase", index=0, settings=settings, words=True)
        t = m["data"]["takes"][0]
        check(m["ok"] and t["path"] != base["path"],
              f"render_phrase 1, {label} ({dt:.1f} s): similarity {t['similarity']:.3f}, words kept {t['words_kept']}")

    # superseding: three previews queued back to back, only the last may run
    ids = [c.send("render_phrase", index=1, settings={"formant": f}) for f in (0.9, 0.92, 0.94)]
    replies = [c.wait(i) for i in ids]
    codes = [r["ok"] or r["error"]["code"] for r in replies]
    check(replies[-1]["ok"] and "superseded" in codes, f"queued previews superseded: {codes}")

    # cancel during render_all
    rid = c.send("render_all", mode="best")
    time.sleep(4)
    c.call("cancel")
    m = c.wait(rid)
    check(not m["ok"] and m["error"]["code"] == "cancelled", "render_all cancelled")

    events = []
    m, dt = c.call("render_all", mode=args.mode, on_event=events.append)
    check(m["ok"] and len(events) == len(info["phrases"]),
          f"render_all {args.mode} ({dt:.1f} s), {len(events)} progress events")

    if args.mode == "best":
        m, _ = c.call("choose", index=2, take=3)
        check(m["ok"], "choose take 4 for phrase 3")
        m, _ = c.call("take_audio", index=2, take=3)
        check(m["ok"] and os.path.isfile(m["data"]["path"]), "take_audio")

    out = os.path.join(args.out_dir, f"server_{args.mode}.wav")
    m, dt = c.call("export", out=out, takes_dir=os.path.join(args.out_dir, f"takes_{args.mode}"),
                   report_csv=os.path.join(args.out_dir, f"report_{args.mode}.csv"))
    d = m.get("data") or {}
    check(m["ok"] and abs(d["duration"] - info["read"]["duration"]) < 1e-3 and abs(d["lufs"] + 16) < 0.15,
          f"export ({dt:.1f} s): {d.get('duration', 0):.3f} s (read {info['read']['duration']:.3f} s), "
          f"{d.get('lufs', 0):.1f} LUFS, peak {d.get('peak_dbfs', 0):.1f} dBFS, similarity {d.get('similarity_after', 0):.3f}")

    m, _ = c.call("shutdown")
    c.p.wait(timeout=10)
    check(c.p.returncode == 0, f"clean shutdown, total {time.time() - t0:.0f} s")
    sys.exit(1 if check.failed else 0)


if __name__ == "__main__":
    main()
