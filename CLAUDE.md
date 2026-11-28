# Vektr64

A from-scratch Rust chess engine playing on Lichess as the bot Vektr64, with Python tooling that
measures and promotes new versions. Design: `docs/superpowers/specs/2026-11-27-vektr64-design.md`.
Plans: `docs/superpowers/plans/`.

- `engine/` – the engine (Rust). Test with `(cd engine && cargo test --release)`.
- `ops/` – gate, anchors, sync, page (Python, uv). Test with `uv run --project ops pytest ops/tests`.
- `deploy/` – scripts and units for the play machine.

## Executing plans

Execute plans with subagents (superpowers:subagent-driven-development), and set the model
explicitly on every dispatch:

- Implementer subagents: `model: sonnet`.
- Per-task reviewer subagents: `model: sonnet`.
- The final whole-branch review: `model: fable`.

Never leave the model out of a dispatch; an omitted model inherits the session's.

## Rules

- Secrets (`LICHESS_TOKEN`, `HF_TOKEN`, `RUNPOD_API_KEY`) live only in the git-ignored `.env`.
  Never print, log, commit or upload them.
- Work on a feature branch; never commit to `main`. Commit and push only when asked. Never merge
  pull requests.
- Spending money (RunPod) and changing what the bot plays need the owner's explicit go-ahead
  each time.
