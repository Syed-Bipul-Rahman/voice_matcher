#!/usr/bin/env python3
"""
evaluate.py - measure how well a conversion worked, so changes to the tool can be compared.

For each audio file it reports:
  similarity    voice similarity to the reference (Resemblyzer; two clips of one person usually score 0.85+)
  char errors   share of the script's characters a speech recogniser gets wrong (needs --script)
  loudness      integrated loudness in LUFS
Every file is transcribed over the same phrase windows, found in --split-by (default: the first file).

  python tools/evaluate.py --reference my_voice.mp3 --script script.srt team_read.m4a voiceover.wav

The numbers recorded in docs/RESEARCH.md were measured this way.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vmengine import audio as vm  # noqa: E402
from vmengine import models as M  # noqa: E402
from vmengine.pipeline import Judge  # noqa: E402


def read_script(path):
    text = open(path, encoding="utf-8").read()
    if path.lower().endswith(".srt"):
        lines = []
        for block in text.strip().split("\n\n"):
            lines += block.strip().split("\n")[2:]          # skip the index and timing lines
        return " ".join(lines)
    return " ".join(text.split())


def main():
    ap = argparse.ArgumentParser(description="Score recordings against a reference voice and a script.")
    ap.add_argument("files", nargs="+", help="audio files to score (for example the original read and the outputs)")
    ap.add_argument("--reference", required=True, help="the reference voice sample")
    ap.add_argument("--script", help="the script as .txt or .srt (enables the char-error check)")
    ap.add_argument("--split-by", help="file whose pauses define the phrase windows (default: first file)")
    ap.add_argument("--lang", default="bn", help="recogniser language (default bn)")
    ap.add_argument("--asr-model", help="Hugging Face wav2vec2 model id for the recogniser")
    ap.add_argument("--models-dir", help="where the models are (default: the one `voice_match.py setup` uses)")
    args = ap.parse_args()
    M.go_offline()
    models_dir = args.models_dir or M.default_models_dir()

    import torch
    import pyloudnorm as pyln
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    script = read_script(args.script) if args.script else None
    asr_id = M.resolve_asr_id(args.lang, args.asr_model) if script else None
    if script and not asr_id:
        sys.exit(f"No recogniser known for language '{args.lang}'. Pass --asr-model.")

    ref = vm.load_audio(args.reference, 16000)
    judge = Judge(ref, M.load_recogniser(models_dir, asr_id) if asr_id else None)
    windows = vm.speech_chunks(vm.load_audio(args.split_by or args.files[0], vm.OUT_SR), vm.OUT_SR)
    meter = pyln.Meter(vm.OUT_SR)

    print(f"{'file':32s} {'similarity':>10s} {'char errors':>12s} {'loudness':>10s} {'length':>8s}")
    for path in args.files:
        y48 = vm.load_audio(path, vm.OUT_SR)
        y16 = vm.resample(y48, vm.OUT_SR, 16000)
        sim = judge.similarity(y16)
        cer = ""
        if script:
            hyp = " ".join(judge.transcribe(y16[int(a * 16000):int(b * 16000)]) for a, b in windows)
            cer = f"{vm.char_error_rate(script, hyp):.1%}"
        lufs = meter.integrated_loudness(y48)
        print(f"{os.path.basename(path)[:32]:32s} {sim:10.3f} {cer:>12s} {lufs:7.1f} LUFS {len(y48) / vm.OUT_SR:6.1f} s",
              flush=True)


if __name__ == "__main__":
    main()
