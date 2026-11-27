# Runbook

The bot runs on `irom-sei` (Ubuntu 24.04, 4 cores / 8 threads, reached over Tailscale at
`100.114.82.19`, user `irom`) from `/home/irom/Vektr64`, as the systemd service `vektr64`.

## What was set up (2026-11-27)

- `deploy/install-box.sh` installed Rust (rustup, stable) and `uv`, created the bridge's
  Python environment in `bot/lichess-bot/.venv`, built and tested the engine, and installed
  `/etc/systemd/system/vektr64.service` without enabling it.
- `/home/irom/Vektr64/.env` holds `LICHESS_TOKEN` (mode 600, never in git).
- The Lichess account already carried the BOT title, so no upgrade was run.
- Smoke test: one casual 3+2 game from the box (lichess.org/g3KDGGwq, lost by checkmate to a
  2939-rated bot at move 44). No illegal moves, no engine errors, lowest clock 37 s,
  search depth 8–9 at about 2.7M positions per second.

## Everyday commands (on the box, in `/home/irom/Vektr64`)

```bash
uv run --project ops vektr-ops status      # service state, blitz rating, deviation, game count
sudo systemctl enable --now vektr64        # start rated play, and on every boot
sudo systemctl stop vektr64                # stop now (a game in progress is abandoned)
sudo systemctl disable --now vektr64       # stop and do not start on boot
sudo systemctl restart vektr64             # restart; rebuilds the engine first
journalctl -u vektr64 -f                   # live log
journalctl -u vektr64 --since "1 hour ago" | grep -iE "error|exception|illegal"
ls bot/game_records/                       # one PGN per finished game
```

## Updating the engine

```bash
git pull && git submodule update --init
(cd engine && cargo test --release)
sudo systemctl restart vektr64
```

The bridge starts a fresh engine process for every game, and `vektr-ops play` rebuilds the
engine at start, so a restart between games is all an update needs.

## Changing how it plays

Edit `bot/config.template.yml` (time controls, opponents, matchmaking pace) and restart.
`bot/config.yml` is generated from it on every start; do not edit that file.

- `matchmaking.challenge_timeout`: minutes idle before the bot challenges someone.
- `matchmaking.opponent_rating_difference`: how far from its own rating it looks for opponents.
- `move_overhead` and `engine.uci_options.Move Overhead`: raise these if games are lost on time.

## If something is wrong

| Symptom | Check |
|---|---|
| Service keeps restarting | `journalctl -u vektr64 -n 80`. A bad or revoked token shows as an authorization error: fix `.env`, restart |
| No games for hours | `vektr-ops status`; the bot only challenges after 35 idle minutes and only bots within 300 points |
| Losses on time | `grep -l "Time forfeit" bot/game_records/*.pgn`; raise the overheads above |
| Box unreachable from the laptop | Tailscale key expiry or the SSH browser check; the bot itself does not depend on the laptop |

## Not verified

- Recovery from a network drop was not tested: the box is on Wi-Fi only and is administered
  over the same link, so there was no safe way to cut it remotely. The bridge's own
  reconnect logic is relied on.
