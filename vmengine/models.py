"""Where the models live, the one-time download, and loading them without touching the network.

All models sit in one folder we control:

  <models>/tts/voice_conversion_models--multilingual--multi-dataset--openvoice_v2/   OpenVoice v2 (126 MB)
  <models>/asr/arijitx--wav2vec2-xls-r-300m-bengali/                                Bangla recogniser (1.2 GB)

Default folder: ~/Library/Application Support/Voice Match/models on macOS (shared with the desktop app),
$XDG_DATA_HOME/voice-match/models (or ~/.local/share/voice-match/models) elsewhere.
Override with VOICE_MATCH_MODELS or --models-dir.

`setup()` is the only code that may use the network. Everything else calls `go_offline()` first and loads
from local paths with local_files_only=True; a missing model raises EngineError("models_missing").
"""
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path

from . import EngineError

VC_MODEL = "voice_conversion_models/multilingual/multi-dataset/openvoice_v2"
VC_DIRNAME = VC_MODEL.replace("/", "--")
ASR_MODELS = {"bn": "arijitx/wav2vec2-xls-r-300m-bengali"}   # word check for --best-of
VC_SIZE = 131_327_000          # approximate download sizes, for progress bars
ASR_SIZE = 1_262_300_000


def default_models_dir():
    env = os.environ.get("VOICE_MATCH_MODELS")
    if env:
        return Path(env).expanduser()
    if sys.platform == "darwin":
        return Path("~/Library/Application Support/Voice Match/models").expanduser()
    base = os.environ.get("XDG_DATA_HOME") or "~/.local/share"
    return Path(base).expanduser() / "voice-match" / "models"


def go_offline():
    """Make sure the Hugging Face libraries never try the network. Call before importing transformers."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def vc_dir(models_dir):
    return Path(models_dir) / "tts" / VC_DIRNAME


def asr_dir(models_dir, model_id):
    return Path(models_dir) / "asr" / model_id.replace("/", "--")


def resolve_asr_id(lang=None, asr_model=None):
    """The recogniser for the word check, or None when there isn't one for this language."""
    if asr_model:
        return asr_model
    if not lang or lang.lower() == "none":
        return None
    return ASR_MODELS.get(lang.lower())


def _dir_size(p):
    p = Path(p)
    if not p.exists():
        return 0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file() and not f.is_symlink())


def _vc_present(d):
    return (d / "model.pth").is_file() and (d / "config.json").is_file()


def _asr_present(d):
    return (d / "config.json").is_file() and (d / "model.safetensors").is_file() \
        and (d / "preprocessor_config.json").is_file() and (d / "vocab.json").is_file()


def status(models_dir=None, asr_ids=None):
    """What is installed. Used by `setup --status` and the app's settings screen."""
    md = Path(models_dir or default_models_dir())
    asr_ids = asr_ids or list(ASR_MODELS.values())
    v = vc_dir(md)
    out = {"models_dir": str(md),
           "converter": {"id": VC_MODEL, "path": str(v), "present": _vc_present(v), "bytes": _dir_size(v)},
           "recognisers": []}
    for mid in asr_ids:
        a = asr_dir(md, mid)
        out["recognisers"].append({"id": mid, "path": str(a), "present": _asr_present(a), "bytes": _dir_size(a)})
    out["ready"] = out["converter"]["present"] and all(r["present"] for r in out["recognisers"])
    return out


def _missing(what, models_dir, extra=""):
    return EngineError(
        f"The {what} isn't installed in {models_dir}. Run the one-time setup "
        f"(`python voice_match.py setup{extra}`, or Settings > Models in the app) while online.",
        "models_missing")


# ---------------------------------------------------------------- loading (offline)
def load_converter(models_dir, device="cpu"):
    """OpenVoice v2, loaded straight from its folder (no model registry, no network)."""
    go_offline()
    d = vc_dir(models_dir)
    if not _vc_present(d):
        raise _missing("voice converter", models_dir)
    from TTS.vc.configs.openvoice_config import OpenVoiceConfig
    from TTS.vc.models.openvoice import OpenVoice
    cfg = OpenVoiceConfig()
    model = OpenVoice.init_from_config(cfg)
    model.load_checkpoint(cfg, d / "model.pth", eval=True)
    return model.to(device)


def load_recogniser(models_dir, model_id):
    """(processor, model) for the word check, from the local folder only."""
    go_offline()
    d = asr_dir(models_dir, model_id)
    if not _asr_present(d):
        extra = "" if model_id in ASR_MODELS.values() else f" --asr-model {model_id}"
        raise _missing(f"speech recogniser ({model_id})", models_dir, extra)
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    proc = Wav2Vec2Processor.from_pretrained(str(d), local_files_only=True)
    model = Wav2Vec2ForCTC.from_pretrained(str(d), local_files_only=True, use_safetensors=True).eval()
    return proc, model


# ---------------------------------------------------------------- one-time setup (online)
class _Watcher:
    """Reports the size of a folder while something downloads into it."""

    def __init__(self, path, name, total, report):
        self.path, self.name, self.total, self.report = Path(path), name, total, report
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.wait(0.5):
            self.report(self.name, "downloading", min(_dir_size(self.path), self.total), self.total)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()


def _old_vc_dirs():
    """Converter downloads left by older versions of the tool (coqui-tts's own cache)."""
    homes = [os.environ.get("TTS_HOME"), os.environ.get("XDG_DATA_HOME"),
             "~/Library/Application Support", "~/.local/share"]
    return [Path(h).expanduser() / "tts" / VC_DIRNAME for h in homes if h]


def setup(models_dir=None, asr_ids=None, report=None):
    """Download every model into `models_dir`. Reuses copies from older caches when present.
    `report(name, stage, done_bytes, total_bytes)` is called as it goes."""
    md = Path(models_dir or default_models_dir())
    md.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_XET_CACHE", str(md / ".download" / "xet"))   # keep download leftovers in one place
    report = report or (lambda *a: None)
    asr_ids = asr_ids if asr_ids is not None else list(ASR_MODELS.values())

    # The voice converter
    d = vc_dir(md)
    if _vc_present(d):
        report("converter", "done", _dir_size(d), _dir_size(d))
    else:
        old = next((o for o in _old_vc_dirs() if _vc_present(o) and o.resolve() != d.resolve()), None)
        if old:
            report("converter", "copying", 0, VC_SIZE)
            shutil.copytree(old, d, dirs_exist_ok=True)
        else:
            from TTS.api import TTS
            from TTS.utils.manage import ModelManager
            mm = ModelManager(models_file=TTS.get_models_file_path(), output_prefix=str(md), progress_bar=False)
            with _Watcher(d, "converter", VC_SIZE, report):
                mm.download_model(VC_MODEL)
        if not _vc_present(d):
            raise EngineError(f"The voice converter download did not finish ({d}).", "download_failed")
        report("converter", "done", _dir_size(d), _dir_size(d))

    # The speech recognisers, saved as safetensors so they load offline from a plain folder
    for mid in asr_ids:
        a = asr_dir(md, mid)
        if _asr_present(a):
            report(mid, "done", _dir_size(a), _dir_size(a))
            continue
        tmp_cache = md / ".download"
        proc = model = None
        try:
            # Reuse a copy in the default Hugging Face cache (left by older versions) if there is one.
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
            proc = Wav2Vec2Processor.from_pretrained(mid, local_files_only=True)
            model = Wav2Vec2ForCTC.from_pretrained(mid, local_files_only=True, use_safetensors=True)
            report(mid, "copying", 0, ASR_SIZE)
        except Exception:
            proc = model = None
        if model is None:
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
            with _Watcher(tmp_cache, mid, ASR_SIZE, report):
                proc = Wav2Vec2Processor.from_pretrained(mid, cache_dir=str(tmp_cache))
                # The Bangla model's main branch only has .bin weights; asking for safetensors fetches just the
                # auto-converted copy (1.2 GB) instead of both (2.5 GB).
                model = Wav2Vec2ForCTC.from_pretrained(mid, cache_dir=str(tmp_cache), use_safetensors=True)
        report(mid, "saving", ASR_SIZE, ASR_SIZE)
        part = a.with_name(a.name + ".partial")
        shutil.rmtree(part, ignore_errors=True)
        model.save_pretrained(str(part), safe_serialization=True)
        proc.save_pretrained(str(part))
        shutil.rmtree(a, ignore_errors=True)
        part.rename(a)
        shutil.rmtree(tmp_cache, ignore_errors=True)
        if not _asr_present(a):
            raise EngineError(f"The recogniser download did not finish ({a}).", "download_failed")
        report(mid, "done", _dir_size(a), _dir_size(a))

    with open(md / "manifest.json", "w") as fh:
        json.dump({"converter": VC_MODEL, "recognisers": asr_ids, "installed": time.strftime("%Y-%m-%d %H:%M:%S")},
                  fh, indent=2)
    return status(md, asr_ids)
