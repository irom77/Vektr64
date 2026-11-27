# Vektr64 — design

Date: 2026-11-27. Status: awaiting approval.

Lines marked **(call)** are choices made by the designer, not by the owner. They can be overturned.

## Intent

Vektr64 is a Lichess BOT account (brand new, zero games) run by one person as a hands-on learning project. The owner wants to build and fine-tune their **own** chess engine, see how high it can climb the bot ladder on a strict budget, and watch its playing strength visibly improve over time.

The learning and the visible improvement are the point. Rating is the scoreboard, not the goal at any price.

## Goals

1. An original chess engine, written from scratch in this repo, that plays rated games on Lichess as Vektr64, unattended, 24/7.
2. A self-improvement loop: the engine's evaluation network is retrained on its own data in short bursts, and a new network replaces the current one only if it wins a statistical test match.
3. Improvement that can be seen: an Elo curve per network version, the Lichess rating history, and a record of every gating decision.
4. The same repo and the same commands work on every machine with no config edits: the always-on Ubuntu box, the dev laptop, and a rented instance.
5. Nothing learned is lost when a rented instance is torn down. Any machine can resume from the latest checkpoint on boot.
6. Stay leaderboard eligible: 50+ rated games, low rating deviation, no time losses from infrastructure faults.
7. Minimal spend: target a few dollars a month in rented compute, zero in fixed costs.

## Non-goals

- Beating the Stockfish-on-big-hardware bots at the top of the lists.
- Chess variants, Chess960, endgame tablebases, pondering, multi-threaded search. All can come later.
- A GPU at play time. Play is CPU only by design.
- A web app. Monitoring is a generated static page plus a CLI.

## Anti-goals

- Using another engine's code or network as the playing engine. It would defeat the purpose.
- Promoting a network or parameter set that has not passed the gate.
- A rented instance left running after its job ends or fails.
- Secrets (Lichess, storage, RunPod tokens) in git or in the external store.
- Background work (data generation, test matches) causing the bot to lose rated games on time.

## Constraints

- **Play machine:** `irom-sei`, reached by SSH over Tailscale (100.114.82.19, user `irom`). Read on 2026-11-27:
  - Ubuntu 24.04, x86-64, Intel i5-8259U: 4 cores / 8 threads, AVX2. No ARM code path is needed.
  - 7 GB RAM, of which only about 2 GB was available: other things run on this machine. Memory is the tight resource. The engine's hash table and the number of parallel background games must be sized from available memory, not just core count **(call)**.
  - 172 GB free disk. Python 3, git and Docker are installed; Rust and `uv` are not and must be installed in stage 1.
  - Tailscale SSH asks the owner for a browser check from time to time, and the node key can expire. Unattended jobs must not depend on the laptop reaching the box; the box pulls and pushes to the external store on its own.
  - Jobs still detect cores, memory and CPU features at run time rather than hard-coding these values.
- **Dev laptop:** WSL2, 22 threads, 15 GB RAM, AVX2, no GPU. Rust and `uv` are installed. It is a useful free source of CPU for data generation and test matches when it is on.
- **GPU:** rented only, in bursts of 1–3 hours. No multi-day training.
- **Account:** upgrading Vektr64 to a BOT account is irreversible. It must happen only on the owner's explicit go-ahead, at the moment the first engine is ready to play.

## Approaches considered

1. **Classic alpha-beta search with a small NNUE-style evaluation network trained in-house — chosen.** The network is tiny and runs on CPU, so play needs no GPU. Training is the only GPU job and fits a 30–90 minute burst. The first network can be trained on Lichess's free database of pre-evaluated positions, so there is no from-scratch phase. It is the most engineering work, but every gain is measurable Elo, which is what the owner wants to watch.
2. **AlphaZero-style tree search with a policy/value network.** Rejected: it needs a GPU at play time, and on a low-CPU box it would search a handful of positions per second.
3. **Searchless network that predicts the move directly.** Rejected: simplest code, but tactically fragile, lower ceiling, and it learns little from its own games.

Within approach 1, the owner chose to write the engine **from scratch** rather than fork an existing one.

## Architecture

```
            ┌────────────── always-on Ubuntu box ──────────────┐
 Lichess ◄──┤ lichess-bot bridge ──UCI──► vektr engine (Rust)   │
            │ idle worker: self-play datagen, gate matches     │
            │ status page generator                            │
            └───────┬──────────────────────────────▲───────────┘
                    │ data, logs, gate results     │ champion net
                    ▼                              │
            ┌──────────── external store (Hugging Face Hub) ───┐
            │ nets + manifest │ data shards │ game logs        │
            └───────▲──────────────────────────────┬───────────┘
                    │ candidate net, checkpoints   │ data, checkpoint
            ┌───────┴──────────────────────────────▼───────────┐
            │ burst instance (RunPod GPU): trainer (PyTorch)   │
            └──────────────────────────────────────────────────┘
```

### Repo layout **(call)**

| Path | What |
|---|---|
| `engine/` | Rust crate. One binary, `vektr`: UCI engine, plus `datagen` and `bench` subcommands |
| `trainer/` | Python (PyTorch) network trainer and data loaders |
| `ops/` | Python CLI `vektr-ops`: `play`, `datagen`, `gate`, `train`, `burst`, `sync`, `promote`, `rollback`, `status` |
| `bot/` | Pinned `lichess-bot` (git submodule) and its config |
| `params/` | Search and time-management parameters, versioned in git |
| `deploy/` | systemd units for the box, burst bootstrap script |
| `docs/` | This design, plans, run notes |

Python is managed with `uv`. **(call)**

### The engine

- **Move generation:** the `cozy-chess` crate **(call)**. Everything above it is original: search, evaluation, time management, UCI.
- **Search:** iterative deepening, principal-variation alpha-beta, transposition table, quiescence search, and the standard pruning and ordering techniques added one at a time, each as its own gated change (null move, late-move reductions, killer and history ordering, aspiration windows, static exchange evaluation). Single-threaded.
- **Evaluation, stage 1:** a hand-written material and piece-square evaluation, so the bot can play and the measuring tools can be proven before any network exists.
- **Evaluation, stage 2:** an NNUE-style network. Starting shape **(call)**: 768 inputs (piece × colour × square) → one hidden layer of 128, two perspectives → one output. Updated incrementally as moves are made, quantized to integers, AVX2 when the CPU has it and a plain fallback when it does not, chosen at run time. Size grows only when a bigger network passes the gate at equal time.
- **Network file:** loaded at start-up from a path (UCI option `EvalFile`). A default network is compiled in so the binary always works alone.
- **Time management:** in the engine. A soft and a hard limit per move from remaining time and increment, with a `Move Overhead` option for network lag.
- **Parameters:** search and time constants are read from `params/` at build time **(call)**, so a tuned set is a git commit.

### The Lichess connection

- The official `lichess-bot` bridge handles the event stream, challenges, matchmaking, reconnects and back-off. It is not rewritten.
- Run as a systemd service with restart-on-failure.
- **Games:** standard chess only, rated, against other bots. The owner chose the **blitz** leaderboard as the one to climb, so all rated play is blitz. Time controls **(call)**: 3+2 and 5+3, with increment to protect against lag. One game at a time. Gate matches mirror blitz thinking time.
- **Matchmaking (call):** the bot issues its own challenges to bots within a rating window, and accepts incoming ones that match. A daily game budget (start at about 30) leaves idle windows for background work and is far more than eligibility needs.
- The bridge starts a fresh engine process per game, so a newly promoted network takes effect at the next game without a restart.

### The self-improvement loop

1. **Collect.** Three sources of training positions, all written as shards to the external store:
   - **Bootstrap:** Lichess's public evaluation database (positions with Stockfish scores). Used for the first network and mixed in afterwards.
   - **Self-play:** `vektr datagen` plays fast fixed-node games against itself from randomized openings and records position, search score and final result.
   - **Experience:** the bot's own rated games are downloaded from Lichess. Positions from games it lost are used as self-play starting points, so the engine generates more data exactly where it played badly. **(call)**
2. **Train.** A burst pulls the latest checkpoint and the data shards, trains, and pushes a candidate network. The target blends search score and game result.
3. **Gate.** The candidate plays the current champion in a sequential statistical test (SPRT) using `fastchess` **(call)** with a fixed opening set and short time control. Bounds **(call)**: accept if better by about 5 Elo, reject if not better than 0, 5% error rates. The engine and trainer must also agree bit-for-bit on evaluation for a set of test positions before any match starts.
4. **Promote or reject.** A pass updates the champion pointer in the manifest; the box picks it up at the next game. A fail is recorded and the candidate is kept for reference but never played. `vektr-ops rollback` restores the previous champion in one command.
5. **Measure.** Each promoted version also plays a fixed set of anchor opponents (Stockfish at fixed limited strengths, used only as a yardstick) to place it on one absolute Elo scale. That gives the improvement curve.

Search and time-management parameters follow the same rule: a change, whether hand-made or from automated tuning, is a candidate that must pass the same gate.

**What a version is (call).** A version is a pair: an engine git tag and a network file. The manifest records both, the gate tests the pair, and promotion and rollback move both together. The box builds the engine from the tagged commit when a version is promoted and keeps the previous build on disk, so rollback is instant. A network with a new shape is therefore always promoted with the engine build that can read it.

**Where the gate runs (call).** The match can run on any machine and is resumable: results are stored per batch of games in the external store and pooled, so the laptop, the box and a rented CPU instance can all contribute to one test. Because the box is slower than the laptop, gate time controls are scaled by the ratio of each machine's `vektr bench` speed to the box's recorded speed, so every machine tests at box-equivalent thinking time. Before a promotion takes effect, the box runs a short check of its own: the new build starts, loads the network, and its `bench` speed has not dropped by more than a set margin.

### Where the compute runs

| Job | Needs | Runs on |
|---|---|---|
| Rated play | 1 CPU core | the box, always |
| Self-play datagen | CPU, many cores help | the box when no game is in progress; the laptop when it is on; optionally a rented CPU instance |
| Gate and anchor matches | CPU | same as datagen |
| Training | GPU | rented burst; falls back to CPU for small smoke runs |

- The trainer picks its device automatically: CUDA, then Apple MPS, then CPU.
- Every job sizes itself from the detected core count.
- **Background work on the box (call):** the box has 4 physical cores, so background work keeps running during rated games but is limited to 2 parallel games at low priority, leaving the engine a core of its own. On a machine with fewer than four cores it would pause during rated games instead. If time losses or a speed drop show up in rated play, the fallback is to pause during games here too.
- The original brief asked for fallback to "a lightweight remote inference endpoint". With this design play never needs remote inference, so that requirement is met by not needing it.

### Persistence

- **Git (this repo):** code, parameters, configs, design and run notes.
- **Hugging Face Hub (call), private:** one model repo for networks, training checkpoints and `manifest.json`; one dataset repo for data shards, game logs (PGN) and gate results.
- **Manifest:** one entry per network version: id, parent, data shards used, hyperparameters, gate result, anchor Elo, timestamp, status (`candidate`, `champion`, `rejected`, `retired`).
- **Who writes what:** bursts write only checkpoints and candidates. Only the gate runner writes promotions. Data shards are append-only with unique names, so machines never overwrite each other.
- **Resume:** every job starts with a pull and ends with a push. Training also pushes a checkpoint every 10 minutes **(call)**, so a teardown loses at most that.
- **Secrets:** `LICHESS_TOKEN`, `HF_TOKEN`, `RUNPOD_API_KEY` come from the environment or a git-ignored `.env`. No machine-specific settings live anywhere else.

### Burst instances

- `vektr-ops burst` starts a RunPod instance from a stock PyTorch image, which runs `deploy/burst.sh`: clone the repo, install, pull, train to a time budget, push, terminate itself.
- Termination runs on success, on failure and on a hard time limit, so a broken run cannot keep billing.
- Stage one: the owner launches each burst by hand with that one command. Stage two: the box launches one automatically when enough new data has accumulated, within a monthly spend cap.

### Monitoring

- `vektr-ops status`: service health, current champion, last games, data waiting, last gate result.
- A generated static page: Lichess rating history per time control, the anchor Elo curve by version, the gate history, games played and rating deviation against the eligibility thresholds. Served on the local network from the box. **(call)**
- The service logs to journald; game PGNs go to the external store daily.

## Build order

Each stage ends with something that works and can be measured.

1. **Engine plays.** Search with the hand-written evaluation, UCI, time management, correctness tests. Bot account upgraded on the owner's go-ahead. Bot plays rated games from the box under systemd.
2. **Measuring tools.** Gate runner, anchor ladder, external store and manifest, status CLI and page. Every later change goes through these.
3. **First network.** Network inference in the engine, the trainer, bootstrap data from the Lichess evaluation database, the first GPU burst, the first gated promotion.
4. **The loop.** Self-play datagen, ingestion of the bot's own games, repeated burst → gate → promote cycles launched by hand.
5. **Automation and tuning.** Automatic burst trigger with spend cap, automated parameter tuning, larger networks.

## Testing

- **Engine:** move-count (perft) tests against known values, mate-finding test positions, incremental network update equals full recomputation, a fixed-depth `bench` node count that must not change unless search changes.
- **Engine ↔ trainer:** the same network file gives the same evaluation in both.
- **Data:** shard write/read round trip.
- **Ops:** gate decision logic against scripted results; promote and rollback against a local fake store; a full burst dry run on CPU with a tiny dataset.
- **Bot:** a casual (unrated) game against a known bot before rated play is switched on.

## Left to the builder

- Internal module structure of each component and the exact shard format.
- Training hyperparameters, the score/result blend, and how bootstrap data is filtered.
- The order in which search techniques are added, as long as each is gated.
- The opening set, time control and game cap for gate matches on a slow box.
- The automated tuning method in stage 5.

## Expectations

Engines built this way typically start around 1800–2000 with a hand-written evaluation and can pass 2800 on a rating list as search and network mature. Lichess bot ratings depend on the opponent pool and on this hardware, so these are rough guides, not targets.
