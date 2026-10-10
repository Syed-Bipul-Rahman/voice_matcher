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

The pipeline lives in the vmengine package, shared with the Voice Match desktop app.
Models are downloaded once with `setup`; after that the tool never uses the network.

Only convert into a voice whose owner has agreed to it.

Examples:
  python voice_match.py setup                         # one-time model download (1.3 GB)
  python voice_match.py read.m4a my_voice.mp3 -o voiceover.wav
  python voice_match.py read.m4a my_voice.mp3 -o voiceover.wav --best-of
  python voice_match.py serve                         # JSON-lines server for the desktop app
"""
import argparse
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from vmengine import Cancelled, EngineError  # noqa: E402
from vmengine import models as M  # noqa: E402
from vmengine.audio import (OUT_SR, WORK_SR, char_error_rate, load_audio, pitch_profile, polish,  # noqa: E402,F401
                            preshift_settings, resample, save_audio, speech_chunks)
from vmengine.models import ASR_MODELS, VC_MODEL  # noqa: E402,F401


def log(msg=""):
    print(msg, flush=True)


# ---------------------------------------------------------------- setup
def cmd_setup(argv):
    ap = argparse.ArgumentParser(prog="voice_match.py setup",
                                 description="Download the models once, so every later run works offline.")
    ap.add_argument("--models-dir", help=f"where to keep the models (default: {M.default_models_dir()})")
    ap.add_argument("--asr-model", action="append",
                    help="also install this Hugging Face wav2vec2 recogniser (repeatable)")
    ap.add_argument("--no-recogniser", action="store_true",
                    help="skip the 1.2 GB Bangla recogniser (fast mode doesn't need it)")
    ap.add_argument("--status", action="store_true", help="only show what is installed")
    ap.add_argument("--json", action="store_true", help="print progress as JSON lines (used by the desktop app)")
    args = ap.parse_args(argv)
    md = args.models_dir or M.default_models_dir()
    ids = [] if args.no_recogniser else list(ASR_MODELS.values())
    ids += [a for a in (args.asr_model or []) if a not in ids]

    def emit(obj):
        print(json.dumps(obj), flush=True)

    if args.status:
        st = M.status(md, ids or None)
        if args.json:
            emit({"event": "status", "data": st})
        else:
            log(f"Models folder: {st['models_dir']}")
            for item in [st["converter"]] + st["recognisers"]:
                log(f"  {'ok     ' if item['present'] else 'missing'} {item['id']} ({item['bytes'] / 1e6:.0f} MB)")
        return 0 if st["ready"] else 1

    last = {}

    def report(name, stage, done, total):
        if args.json:
            emit({"event": "progress", "data": {"model": name, "stage": stage, "done": done, "total": total}})
        elif stage != "downloading" or time.time() - last.get(name, 0) > 5:
            last[name] = time.time()
            log(f"  {name}: {stage} {done / 1e6:.0f}/{total / 1e6:.0f} MB")

    if not args.json:
        log(f"Installing models into {md}")
    try:
        st = M.setup(md, ids, report)
    except EngineError as e:
        if args.json:
            emit({"event": "error", "error": {"code": e.code, "message": str(e)}})
        else:
            log(f"Error: {e}")
        return 1
    except Exception as e:  # network errors and the like
        msg = f"Setup failed: {type(e).__name__}: {e}. Check the internet connection and try again."
        if args.json:
            emit({"event": "error", "error": {"code": "download_failed", "message": msg}})
        else:
            log(msg)
        return 1
    if args.json:
        emit({"event": "done", "data": st})
    else:
        log("Done. Everything now works offline.")
    return 0


# ---------------------------------------------------------------- convert
def cmd_convert(argv):
    ap = argparse.ArgumentParser(
        description="Make a recording sound like another person's voice (speech-to-speech voice conversion). "
                    "First time: python voice_match.py setup")
    ap.add_argument("source", help="recording whose words and timing you want to keep (wav, mp3, m4a...)")
    ap.add_argument("reference", help="20-60 s of clean speech in the target voice")
    ap.add_argument("-o", "--out", default="converted.wav", help="output file (default: converted.wav)")
    ap.add_argument("--best-of", action="store_true",
                    help="make 6 takes per phrase and keep the best (about 6x slower, clearer words)")
    ap.add_argument("--lang", default="bn",
                    help="language of the recording, for the word check in --best-of (default: bn; 'none' to skip)")
    ap.add_argument("--asr-model", default=None, help="Hugging Face wav2vec2 model id for the word check "
                                                      "(install it first with: setup --asr-model ID)")
    ap.add_argument("--keep-takes", metavar="DIR",
                    help="save every take plus takes.csv here, so you can swap a take by ear")
    ap.add_argument("--no-preshift", action="store_true", help="skip the pitch/formant pre-shift")
    ap.add_argument("--no-polish", action="store_true", help="skip EQ, de-ess, compression and loudness")
    ap.add_argument("--lufs", type=float, default=-16.0, help="loudness of the finished voice (default -16)")
    ap.add_argument("--device", default="auto", help="auto, cpu, mps (Apple GPU) or cuda")
    ap.add_argument("--models-dir", help=f"where the models are (default: {M.default_models_dir()})")
    args = ap.parse_args(argv)

    M.go_offline()
    from vmengine.pipeline import Session

    t_start = time.time()
    s = Session(args.models_dir, device=args.device, log=log)
    try:
        log("Reading audio...")
        info = s.analyze(args.source, args.reference)
        for w in info["warnings"]:
            log(f"  Warning: {w}")
        log(f"  Source voice pitch {info['read']['pitch']['median']:.0f} Hz, "
            f"reference {info['reference']['pitch']['median']:.0f} Hz")
        settings = {"preshift": "off" if args.no_preshift else "auto"}
        mode = "best" if args.best_of else "fast"
        eff = s.effective(settings)
        plan = s.take_plan(eff, mode)
        if plan[0] is not None:
            log(f"  Pre-shift: pitch median -> {eff['pitch_median']:.0f} Hz, formant ratio {plan[0][0]:.2f}, "
                f"pitch range {plan[0][1]:.2f}")
        log(f"  {len(s.phrases)} phrases to convert, {len(plan)} take(s) each")

        asr_id = None
        s.ensure_converter()
        if args.best_of:
            asr_id = M.resolve_asr_id(args.lang, args.asr_model)
            if not asr_id:
                log(f"  No word checker for language '{args.lang}': picking takes by voice similarity only.")
            log("Loading the take judges...")
            s.ensure_judge(asr_id)

        def progress(i, n, res):
            s0, s1 = s.phrases[i]
            msg = f"  phrase {i + 1}/{n} ({s1 - s0:.1f} s)"
            if len(res["takes"]) > 1:
                t = res["takes"][res["chosen"]]
                msg += f": take {t['k'] + 1} of {len(res['takes'])}, voice similarity {t['sim']:.2f}"
                if t["kept"] is not None:
                    msg += f", words kept {t['kept']:.0%}"
            log(msg)

        s.render_all(settings, mode=mode, asr_id=asr_id, progress=progress)
        finish = {"enabled": not args.no_polish, "lufs": args.lufs}
        res = s.export(args.out, finish, takes_dir=args.keep_takes)
        mins, secs = divmod(int(time.time() - t_start), 60)
        log(f"Saved {args.out} ({res['duration']:.1f} s of audio, took {mins}m{secs:02d}s)")
        log(f"Voice similarity to the reference: {res['similarity_before']:.2f} before, "
            f"{res['similarity_after']:.2f} after (two clips of the same person usually score 0.85 or more)")
    except EngineError as e:
        log(f"Error: {e}")
        return 1
    except Cancelled:
        return 130
    finally:
        s.close()
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "setup" and not os.path.exists("setup"):
        return cmd_setup(argv[1:])
    if argv and argv[0] == "serve" and not os.path.exists("serve"):
        from vmengine.server import main as serve
        return serve(argv[1:])
    return cmd_convert(argv)


if __name__ == "__main__":
    sys.exit(main())
