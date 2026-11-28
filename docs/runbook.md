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

## Changing the engine: every change goes through the gate

The engine that plays is the **champion** named in the manifest in the external store, not
whatever is checked out. To change it:

```bash
git push origin my-branch                      # the commit must be on GitHub
uv run --project ops vektr-ops gate my-branch  # on the laptop; plays until decided
```

- The candidate plays the champion in batches of 100 games. Each batch prints
  `games=… llr=… elo=… ±…`. The test ends at `accept` (better by about 5 Elo) or `reject`
  (not better, or 20,000 games without a decision).
- The box's worker joins the same test on its own; results from every machine are pooled.
  You can stop the laptop at any time and the test carries on, or resumes when you rerun it.
- On `accept` the candidate becomes the champion. Within ten minutes the box builds it,
  checks that it starts and is not more than 10% slower, and switches at the next game.
- `uv run --project ops vektr-ops rollback` puts the previous champion back.
- `uv run --project ops vektr-ops reject <id>` withdraws an open candidate; its test stops on every machine.
- `uv run --project ops vektr-ops anchors` measures the champion against fixed-strength
  opponents (the box does this by itself after a promotion).

Bridge or ops changes (anything outside `engine/`) do not need the gate:
`git pull && git submodule update --init && sudo systemctl restart vektr64`.

## The status page

`http://100.114.82.19:8064/` (or the box's LAN address, port 8064), regenerated every ten
minutes. It shows the Lichess blitz rating over time, leaderboard eligibility, the strength
of each version on the anchor scale, and every gate decision.

## Background units on the box

| Unit | What it does |
|---|---|
| `vektr64-sync.timer` | Every 10 minutes: follow the champion, upload finished game records, rebuild the page |
| `vektr64-worker.service` | Gate and anchor games, 2 at a time at the lowest priority |
| `vektr64-page.service` | Serves the page |

To stop background games (for example if rated games start losing on time):
`sudo systemctl stop vektr64-worker`.

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
| `sync error:` in `vektr-ops status`, or a notice on the page | The box could not put the champion into play: the store was unreachable, the commit did not build, the build did not start, or it was too slow. The bot keeps playing the previous engine. The message says which; `journalctl -u vektr64-sync -n 40` has the detail. It clears itself on the next good sync |
| Page says the engine in play is not the champion | Normal for up to ten minutes after a promotion; otherwise see `sync error` |
| A candidate is never played on one machine | It failed to build there: the reason is in `builds/<id>.failed`, and the build is tried again an hour later (delete the file to retry now) |
| A test never finishes | `vektr-ops status` shows open tests; batches with time forfeits or crashes are discarded (`gate/<test>/discarded/` in the data repo) |
| Box unreachable from the laptop | Tailscale key expiry or the SSH browser check; the bot itself does not depend on the laptop |

## Not verified

- Recovery from a network drop was not tested: the box is on Wi-Fi only and is administered
  over the same link, so there was no safe way to cut it remotely. The bridge's own
  reconnect logic is relied on.
