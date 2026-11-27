# Vektr64

A chess engine written from scratch that plays rated blitz on Lichess as the BOT account
[Vektr64](https://lichess.org/@/Vektr64), and is meant to get measurably stronger over time
on a minimal budget.

- **Design:** [docs/superpowers/specs/2026-11-27-vektr64-design.md](docs/superpowers/specs/2026-11-27-vektr64-design.md)
- **Stage 1 plan (the engine plays):** [docs/superpowers/plans/2026-11-27-stage1-engine-plays.md](docs/superpowers/plans/2026-11-27-stage1-engine-plays.md)
- **Stage 2 plan (measuring tools):** [docs/superpowers/plans/2026-11-27-stage2-measuring-tools.md](docs/superpowers/plans/2026-11-27-stage2-measuring-tools.md)
- **Runbook:** [docs/runbook.md](docs/runbook.md)
- **Bench baselines:** [docs/bench.md](docs/bench.md)

## Layout

| Path | What |
|---|---|
| `engine/` | The engine (Rust). One binary, `vektr`: a UCI engine, plus `vektr bench` |
| `ops/` | `vektr-ops`, the Python CLI that runs and inspects the bot |
| `bot/` | The pinned [lichess-bot](https://github.com/lichess-bot-devs/lichess-bot) bridge and its settings |
| `deploy/` | systemd unit and machine setup script |

## Running it

Set up a machine once (installs Rust and `uv` if missing, builds and tests the engine,
installs the service without starting it):

```bash
git clone --recurse-submodules https://github.com/irom77/Vektr64.git && cd Vektr64
./deploy/install-box.sh
```

Put the Lichess token (scope `bot:play`) in `.env`, which git ignores:

```bash
cp .env.example .env && chmod 600 .env   # then edit LICHESS_TOKEN=
```

Then:

```bash
uv run --project ops vektr-ops play --dry-run   # show the bridge settings for this machine
uv run --project ops vektr-ops play             # play in the foreground
sudo systemctl enable --now vektr64             # play as a service
uv run --project ops vektr-ops status           # service state, rating, deviation, game count
```

Measuring and changing the engine (needs `HF_TOKEN` in `.env`; see [docs/runbook.md](docs/runbook.md)):

```bash
uv run --project ops vektr-ops gate <git-ref>   # test a candidate against the champion
uv run --project ops vektr-ops anchors          # place the champion on the absolute scale
uv run --project ops vektr-ops rollback         # put the previous champion back
uv run --project ops vektr-ops page             # write the status page to site/
uv run --project ops vektr-ops sync             # play machine only: follow the champion
```

Nothing is edited per machine. `vektr-ops play` builds the engine, sizes its hash table
from the memory available, and writes `bot/config.yml` from `bot/config.template.yml`.
The bridge runs from `bot/lichess-bot/.venv`, which the install script creates from the
bridge's `requirements.txt`.

## Tests

```bash
(cd engine && cargo test --release)
uv run --project ops pytest ops/tests
```
