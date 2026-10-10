# Notes for Claude

This repository holds the voice-matching tool built for the ORYZN film, plus its research and history. Work here survives machine resets only once it's committed and pushed, so commit and push as you go.

- Read `docs/HISTORY.md` and `docs/RESEARCH.md` first. They explain what was tried, what won and why.
- Set up with the README's Install steps. On Linux without a GPU, install CPU PyTorch first.
- Test recordings are not in the repository: it's public, and real voice samples must stay out of it. Ask the user to attach the read and the reference again if you need to re-run tests.
- Judge every change with `tools/evaluate.py` on the same recordings, and compare with the numbers in `docs/RESEARCH.md`.
- Record changes in `CHANGELOG.md` and new findings in `docs/RESEARCH.md`.
- Only convert into voices whose owners have agreed.

## Engine layout (v1.1)

- The pipeline lives in `vmengine/` (`audio`, `models`, `pipeline.Session`, `server`). `voice_match.py` is a thin CLI on top, with `setup` and `serve` subcommands. Keep the CLI's options and results unchanged.
- The Voice Match desktop app (Tauri) is a separate project in the sibling folder `../voice_match_app`. It runs `voice_match.py serve` and `voice_match.py setup --json`, and bundles `voice_match.py`, `vmengine/`, `pyproject.toml` and `uv.lock` from here. Protocol changes must stay compatible with it, or bump `PROTOCOL` in `vmengine/server.py`.
- Models load only from the models folder (`voice_match.py setup`); nothing else may touch the network. Check with `tools/offline_check.sh` (0 attempts expected), and check the server with `tools/server_check.py`.
- Set up with `uv sync` (Python 3.13). On this project's Mac the uv cache and Rust toolchain live on an external disk (`UV_CACHE_DIR`, `RUSTUP_HOME`, `CARGO_HOME`); see the user's shell profile.
- Ask before changing default quality behaviour (default device stays CPU; match strength 1.0; finishing defaults).
