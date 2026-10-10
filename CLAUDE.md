# Notes for Claude

This repository holds the voice-matching tool built for the ORYZN film, plus its research and history. Work here survives machine resets only once it's committed and pushed, so commit and push as you go.

- Read `docs/HISTORY.md` and `docs/RESEARCH.md` first. They explain what was tried, what won and why.
- Set up with the README's Install steps. On Linux without a GPU, install CPU PyTorch first.
- Test recordings are not in the repository: it's public, and real voice samples must stay out of it. Ask the user to attach the read and the reference again if you need to re-run tests.
- Judge every change with `tools/evaluate.py` on the same recordings, and compare with the numbers in `docs/RESEARCH.md`.
- Record changes in `CHANGELOG.md` and new findings in `docs/RESEARCH.md`.
- Only convert into voices whose owners have agreed.
