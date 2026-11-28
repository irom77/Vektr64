# Stage 4: The Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The bot produces its own training data: the engine plays itself and records positions, the bot's lost Lichess games become starting points for those games, and the trainer mixes this data with the Lichess evaluation data, so that burst → gate → promote can be repeated by hand.

**Architecture:** `vektr datagen` plays fixed-node games against itself and writes fixed-size binary records that the trainer reads directly, so the play machine needs no trainer install. `vektr-ops` wraps it: it runs the engine with the champion's network, compresses the output and uploads it to the data repo under a unique, time-ordered name. The box's background worker generates one shard whenever it has no gate or anchor work. Training selects all Lichess shards plus the newest self-play shards; everything after training (install, gate, promote) is unchanged from Stage 3.

**Tech Stack:** Rust (engine, no new crates); Python 3.12, `uv`, numpy, PyTorch (trainer); typer, `huggingface_hub`, `chess` (python-chess, new in ops), pytest (ops).

**Spec:** `docs/superpowers/specs/2026-11-27-vektr64-design.md` (sections "The self-improvement loop", "Where the compute runs", build order item 4)

## Global Constraints

- Work on branch `stage4-the-loop`, cut from `main`.
- No other engine's code or network is used in the playing engine.
- Search and evaluation do not change: `vektr bench` still prints `27251534 nodes`, and `vektr netcheck` still prints `netcheck ok 256 positions`.
- No new Rust crate. Randomness in the engine is an own small generator.
- Data shards are append-only with unique names; no machine overwrites another's file.
- Background work must not cost rated games: on the box it runs at most 2 engine threads at the lowest priority, and not at all on a play machine with fewer than 4 physical cores (`match.concurrency`).
- Nothing is promoted unless its gate decision is `accept`. Spending money (a burst) and changing what the bot plays need the owner's go-ahead each time.
- Secrets (`LICHESS_TOKEN`, `HF_TOKEN`, `RUNPOD_API_KEY`) come from the environment or the git-ignored `.env`. They are never written to git, the store, logs or the page.
- Tests never touch the network, Hugging Face, Lichess or RunPod: they use `LocalStore` in `tmp_path`, scripted inputs and fake runners.
- No field is added to the manifest's `Version` record.
- Bounded test commands on the laptop: `(cd engine && timeout -s KILL 550 cargo test --release)`, `(ulimit -v 6000000; timeout -s KILL 280 uv run --project ops pytest ops/tests -q -p no:cacheprovider)`, `(cd trainer && ulimit -v 6000000; timeout -s KILL 280 uv run pytest -q -p no:cacheprovider)`.
- Commit messages end with `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>`.

## Decisions this plan makes (owner can overturn)

- **Self-play games:** 5,000 nodes per move, a fresh 4 MiB table per game. A game starts from the initial position after 8 or 9 random moves; it is dropped if the first search scores it beyond ±600. One game in ten starts instead from a position of a lost Lichess game, followed by 2 random moves.
- **Game end:** mate, stalemate, the engine's own draw rules, 400 plies (draw), or a score of ±2000 or more on 5 searches in a row (win for the side ahead).
- **Positions kept:** side to move not in check, best move neither a capture nor a promotion, score not a mate score. Scores are clipped to ±3000. These match the filters on the Lichess data.
- **Record format:** 68 bytes per position, defined below. A shard is 500,000 positions (34 MB raw), gzip-compressed, stored as `data:selfplay/<UTC time>-<network>-<host>-<random>.vkd.gz`.
- **Which engine generates:** the engine built from the repository's checked-out commit, loading the champion's network from the store (`nets:nets/<hash>.vknn`) with `--net`. A commit that only adds data generation is never gated, because it never plays.
- **How much:** generation for a network stops at 120 shards (60 million positions, about 1.8 GB in the store). A new champion network starts a new count.
- **Lost games:** a game counts when the variant is standard, Vektr64 lost and the termination is `Normal` (no time forfeits). Every position from ply 12 on becomes a starting point. The box extracts them during its ten-minute sync from the game records it already keeps.
- **Training mix:** every Lichess shard (30 million positions) plus the newest 120 self-play shards, trained from scratch for the same 45 minutes. Self-play positions carry the game result, so the existing blend (0.75 score, 0.25 result) applies to them.
- **Validation loss** is measured on a seeded random 1% of all positions, not on the last 1%, so it covers both kinds of data.
- **Kill switch:** `VEKTR_DATAGEN=0` in `.env` stops self-play on that machine without stopping gate and anchor work.

Not known in advance: whether a network trained on this mix beats the current one. Its labels come from a 5,000-node search of a small network, the Lichess labels from deep Stockfish analysis. A `reject` at the gate is a valid result of the first round.

## Review Focus

1. `vektr datagen` is killed mid-run, or the checked-out engine has no `datagen` command: no half-written or wrong-sized shard reaches the store. (Tasks 2, 5)
2. The starts file has a malformed line, a mated or stalemated position, or is empty: those lines are skipped and generation carries on with random openings. (Tasks 1, 2)
3. A truncated or corrupt record file is among the training shards: training stops before the first step and names the file. (Task 3)
4. Game records as the bridge writes them (variations in parentheses, clock and evaluation comments), aborted games with no moves, games shorter than 12 plies, wins, draws, time forfeits: only real losses give starting points, and a malformed record never fails the sync. (Task 4)
5. The store is unreachable when a finished shard is uploaded: the shard stays on disk and is uploaded by the next run, not generated again. (Task 5)

## File Structure

| File | Responsibility |
|---|---|
| `engine/src/datagen.rs` | Random generator, record encoding, one self-play game, the multi-threaded run |
| `engine/src/main.rs` | `datagen` subcommand: arguments, network and starts files, output file |
| `engine/tests/datagen.rs` | Tests for all of the above |
| `trainer/src/vektr_trainer/records.py` | Read and write record files (`.vkd`, `.vkd.gz`) |
| `trainer/src/vektr_trainer/train.py` | Load both shard kinds, compact memory, random held-out sample |
| `ops/src/vektr_ops/experience.py` | Lost game record → starting positions; upload during sync |
| `ops/src/vektr_ops/selfplay.py` | Generate, compress and upload one shard; the worker's idle step |
| `ops/src/vektr_ops/training.py` | Shard selection includes self-play |
| `ops/src/vektr_ops/gate.py` | Worker loop calls the idle step |
| `ops/src/vektr_ops/cli.py` | `datagen` command, `train --selfplay`, status line, sync hook |
| `docs/runbook.md` | "The loop" section |

## The record, defined once

One position is 68 bytes, little-endian, no header:

| Bytes | Type | Meaning |
|---|---|---|
| 0–63 | 32 × `u16` | White-view feature indices `colour*384 + piece*64 + square` in ascending order, padded with 768 |
| 64–65 | `i16` | Search score in centipawns for the side to move, clipped to ±3000 |
| 66 | `u8` | Side to move: 0 white, 1 black |
| 67 | `u8` | Game result for the side to move: 0 loss, 1 draw, 2 win |

Reference record, asserted by both the engine tests and the trainer tests. Position `4k3/8/8/8/8/8/4P3/4K3 b - - 0 1`, score −37, White won the game (so the side to move lost):

```
REFERENCE = "0c00" "4401" "fc02" + "0003" * 29 + "dbff" "01" "00"
```

(features 12, 324, 764; score −37; side to move 1; result 0).

---

### Task 1: Engine — records and one self-play game

**Files:**
- Create: `engine/src/datagen.rs`, `engine/tests/datagen.rs`
- Modify: `engine/src/lib.rs` (add `pub mod datagen;`), `engine/src/nnue.rs` (make `features` `pub(crate)`)

**Interfaces:**
- Consumes: `Searcher::new(hash_mb)`, `Searcher::set_network`, `Searcher::new_game`, `Searcher::search(&mut Position, &Limits, on_iter)`, `Limits { nodes, .. }`, `Position::{startpos, from_fen, play, legal_moves, is_draw}`, `nnue::features(color, piece, square) -> (white index, black index)`.
- Produces:

```rust
pub const RECORD_BYTES: usize = 68;
pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Rng;          // any seed, including 0, gives a usable generator
    pub fn next(&mut self) -> u64;
    pub fn below(&mut self, n: usize) -> usize;  // 0..n, n > 0
}
pub struct Settings {
    pub nodes: u64,            // per move
    pub starts: Vec<String>,   // FENs accepted by load_starts
    pub start_share: u32,      // percent of games that begin from `starts`
}
pub struct Sample { pub board: Board, pub score: i32 }   // score for the side to move
pub struct Game { pub samples: Vec<Sample>, pub result: u8 } // White's view: 0 loss, 1 draw, 2 win
pub fn load_starts(text: &str) -> Vec<String>;  // lines that parse and have a legal move
pub fn encode(board: &Board, score: i32, white_result: u8) -> [u8; RECORD_BYTES];
pub fn play_game(searcher: &mut Searcher, rng: &mut Rng, settings: &Settings) -> Option<Game>;
```

- [ ] **Step 1: Write the failing tests** in `engine/tests/datagen.rs`:

```rust
const REFERENCE: &str = /* the hex from "The record, defined once" */;

#[test] fn encode_matches_the_defined_record()
// board from "4k3/8/8/8/8/8/4P3/4K3 b - - 0 1": hex(encode(&board, -37, 2)) == REFERENCE

#[test] fn encode_clips_the_score_and_turns_the_result()
// white to move, score 9999, white_result 2 -> score bytes 3000, result byte 2
// black to move, score -9999, white_result 2 -> score bytes -3000, result byte 0; white_result 1 -> 1

#[test] fn rng_repeats_and_stays_in_range()
// two Rng::new(7) give the same 100 values; Rng::new(0) does not return only zeros;
// below(5) < 5 for 1000 draws and hits every value 0..5

#[test] fn load_starts_keeps_only_playable_positions()
// "garbage\n\n4k3/8/8/8/8/8/4P3/4K3 w - - 0 1\n7k/5Q2/6K1/8/8/8/8/8 b - - 0 1\n" -> 1 entry (stalemate dropped)

#[test] fn a_game_gives_quiet_positions_and_a_result()
// settings {nodes: 300, starts: [], start_share: 0}; the first Some(game) over seeds 1..=20:
// result <= 2, samples not empty, every sample: board.checkers() empty, |score| < MATE - MAX_PLY

#[test] fn the_same_seed_plays_the_same_game()
// two fresh searchers and Rng::new(3): equal results and equal encoded samples

#[test] fn a_crushing_start_is_adjudicated_as_a_win()
// starts ["4k3/8/8/8/8/8/8/QQQQK3 w - - 0 1"], start_share 100, nodes 300, seeds 1..=10:
// no game has result 0, at least 8 have result 2 (the two random moves may throw a win away), none reaches 400 plies

#[test] fn unplayable_starts_give_no_game()
// starts built by hand with only the stalemate FEN, start_share 100 -> None, no panic
```

- [ ] **Step 2: Run** `(cd engine && timeout -s KILL 550 cargo test --release --test datagen)`. Expected: does not compile, `datagen` not found.

- [ ] **Step 3: Implement `engine/src/datagen.rs`.** `Rng` is splitmix64. `play_game`:
  1. Start: with probability `start_share`% and non-empty `starts`, a random start then 2 random legal moves; otherwise the initial position then 8 random legal moves, plus one more when the next random bit is set. Return `None` if a position on the way has no legal move or `is_draw(0)`.
  2. `searcher.new_game()`, then loop: no legal move → mate (the side to move lost) or stalemate; `is_draw(0)` or 400 plies since the start → draw; search with `Limits { nodes: Some(settings.nodes), ..Default::default() }`.
  3. From a random opening only: if the first search's score is beyond ±600, return `None`.
  4. Keep a `Sample` when the side to move is not in check, the best move does not capture (including en passant) or promote, and `|score| < MATE - MAX_PLY`.
  5. A score of ±2000 or more on 5 searches in a row ends the game as a win for the side ahead. Otherwise play the best move.

- [ ] **Step 4: Run** the same command. Expected: 8 tests pass. Then the whole suite `(cd engine && timeout -s KILL 550 cargo test --release)`: all pass.

- [ ] **Step 5: Commit** `feat(engine): self-play games and position records`.

---

### Task 2: Engine — the `datagen` command

**Files:**
- Modify: `engine/src/datagen.rs`, `engine/src/main.rs`, `engine/tests/datagen.rs`, `docs/bench.md`

**Interfaces:**
- Consumes: Task 1's `Settings`, `Rng`, `play_game`, `encode`, `load_starts`; `Network::load(path)`, `Network::embedded()`.
- Produces:

```rust
pub struct Summary { pub games: u64, pub positions: u64, pub white_wins: u64, pub draws: u64, pub black_wins: u64 }
pub fn run(settings: &Settings, net: Arc<Network>, positions: u64, threads: usize, seed: u64,
           out: &mut (dyn Write + Send)) -> std::io::Result<Summary>;
```

  Command line: `vektr datagen --out FILE --positions N [--threads T=1] [--nodes K=5000] [--seed S=1] [--net FILE] [--starts FILE]`. Last stdout line: `datagen: <games> games, <positions> positions, +<white wins> =<draws> -<black wins>`. Exit status 0 on success, 1 when the network or starts file cannot be read or the output cannot be written, 2 for missing or unknown arguments (usage on stderr).

- [ ] **Step 1: Write the failing tests** (added to `engine/tests/datagen.rs`):

```rust
#[test] fn run_writes_exactly_the_requested_positions()
// positions 200, threads 1, nodes 200 into a Vec<u8>: len == 200 * 68, summary.positions == 200,
// white_wins + draws + black_wins == games

#[test] fn single_thread_runs_repeat()
// seed 5 twice -> identical bytes; seed 6 -> different bytes

#[test] fn threads_share_one_target()
// threads 3, positions 300: len == 300 * 68; every record: indices ascending then only 768,
// byte 66 <= 1, byte 67 <= 2

#[test] fn the_command_writes_the_file_and_leaves_no_part()
// Command::new(env!("CARGO_BIN_EXE_vektr")) datagen --out <tmp>/a.vkd --positions 100 --nodes 200 --seed 7:
// status 0, file is 6800 bytes, <tmp>/a.vkd.part absent, last stdout line starts with "datagen: "

#[test] fn the_command_skips_bad_start_lines()
// --starts file "garbage\n4k3/8/8/8/8/8/8/QQQQK3 w - - 0 1\n": status 0, 6800 bytes

#[test] fn a_missing_network_fails_without_an_output_file()
// --net <tmp>/none.vknn: status 1, <tmp>/a.vkd and its .part absent

#[test] fn missing_arguments_print_usage()
// `datagen` alone: status 2
```

- [ ] **Step 2: Run** `(cd engine && timeout -s KILL 550 cargo test --release --test datagen)`. Expected: the new tests fail to compile (`run` not found).

- [ ] **Step 3: Implement `run` and the subcommand.** Each thread owns a `Searcher::new(4)` with the given network and an `Rng::new(seed + thread index)`; finished games go through one mutex that writes whole games, counts positions and cuts the last game so the total is exact. `start_share` is 10 when a starts file gave at least one position, else 0. `main.rs` writes to `<out>.part` and renames it to `<out>` only after `run` succeeded; on any failure it removes the `.part` file.

- [ ] **Step 4: Run** the engine suite. Expected: all pass. Then `cargo build --release && ./target/release/vektr bench` prints `27251534 nodes`, and `./target/release/vektr netcheck` prints `netcheck ok 256 positions`.

- [ ] **Step 5: Measure.** Run `./target/release/vektr datagen --out /tmp/…/speed.vkd --positions 100000 --threads <physical cores − 1>` in the scratchpad, time it, and add a "Self-play speed" section to `docs/bench.md` with the laptop's positions per second and the share of decisive games from the summary line.

- [ ] **Step 6: Commit** `feat(engine): datagen command`.

---

### Task 3: Trainer — read records, mix both kinds of shard

**Files:**
- Create: `trainer/src/vektr_trainer/records.py`, `trainer/tests/test_records.py`
- Modify: `trainer/src/vektr_trainer/train.py`, `trainer/tests/test_train.py`

**Interfaces:**
- Produces:

```python
RECORD_BYTES = 68
def read_records(path: Path) -> dict[str, np.ndarray]   # keys features (N,32) uint16, stm, score, result; like read_shard
def write_records(path: Path, features, stm, score, result) -> None  # gzip when the name ends in .gz
# train.py
def data_files(shards: Path) -> list[Path]   # sorted *.npz, *.vkd, *.vkd.gz
def split(total: int, holdout: float, seed: int) -> tuple[np.ndarray, np.ndarray]  # train indices, held-out indices
```

  `read_records` raises `ValueError` naming the file when the size is not a multiple of 68, the gzip stream is cut, a feature exceeds 768, side to move exceeds 1 or the result exceeds 2.

- [ ] **Step 1: Write the failing tests.**

```python
# test_records.py
def test_reference_record(tmp_path):
    # bytes.fromhex(REFERENCE) written to a.vkd: features[0][:3] == [12, 324, 764], features[0][3:] all 768,
    # score[0] == -37, stm[0] == 1, result[0] == 0
def test_round_trip_plain_and_gzip(tmp_path)        # write_records then read_records, .vkd and .vkd.gz, 1000 rows
def test_truncated_file_names_the_file(tmp_path)    # 100 bytes -> ValueError containing the file name
def test_cut_gzip_names_the_file(tmp_path)          # first half of a .vkd.gz -> ValueError containing the file name
def test_bad_values_are_refused(tmp_path)           # feature 769; stm 2; result 3 -> ValueError each

# test_train.py
def test_training_reads_both_kinds(tmp_path)
    # one .npz of 4096 rows and one .vkd.gz of 4096 rows with results: info["positions"] == 8192
def test_held_out_sample_is_spread_and_repeatable()
    # split(10_000, 0.01, 1): 100 held out, fewer than 50 of them >= 9_900, equal on a second call,
    # train and held-out indices together are exactly 0..9_999
def test_resume_refuses_an_added_record_file(tmp_path)
    # train, add a .vkd.gz, train again with the same out -> ValueError("different data")
def test_a_corrupt_shard_stops_before_training(tmp_path)
    # a 100-byte b.vkd beside a good .npz -> ValueError naming b.vkd, no checkpoint.pt written
```

- [ ] **Step 2: Run** `(cd trainer && ulimit -v 6000000; timeout -s KILL 280 uv run pytest -q -p no:cacheprovider)`. Expected: the new tests fail on the missing module and functions.

- [ ] **Step 3: Implement.** `records.py` uses one numpy structured dtype for the record. In `train.py`: `_load` reads every file from `data_files` into preallocated arrays (peak memory about one copy of the data) and puts them on the device as `int16` features, `int16` score, `uint8` side to move and result; `_loss` converts each batch to the types the model takes. The held-out set is `split(...)`'s second array, gathered once; training batches index through the first. `_fingerprint` lists every file from `data_files`.

- [ ] **Step 4: Run** the trainer suite. Expected: all pass, including every existing test.

- [ ] **Step 5: Commit** `feat(trainer): read self-play records; held-out sample across all data`.

---

### Task 4: Ops — starting positions from lost games

**Files:**
- Create: `ops/src/vektr_ops/experience.py`, `ops/tests/test_experience.py`
- Modify: `ops/pyproject.toml` (add `chess>=1.11`; run `uv lock --project ops`), `ops/src/vektr_ops/cli.py` (`sync` command), `ops/tests/test_cli.py`

**Interfaces:**
- Consumes: `Store`, `results.headers(text)`.
- Produces:

```python
ACCOUNT = "Vektr64"
FIRST_PLY = 12
def lost_starts(pgn: str, account: str = ACCOUNT) -> list[str]
    # FENs of the main-line positions from ply FIRST_PLY up to, not including, the final one; [] unless
    # Variant is Standard, there is no FEN header, Termination is Normal and `account` lost
def upload_experience(store: Store, root: Path, *, min_age_s: float = 1800) -> int
    # for each bot/game_records/*.pgn old enough and without data:experience/<stem>.fens:
    # write that file (one FEN per line) when lost_starts is not empty; returns files written
```

- [ ] **Step 1: Write the failing tests.** `LOSS` is a 30-ply game in the bridge's format, with `[White "x"] [Black "Vektr64"] [Result "1-0"] [Variant "Standard"] [Termination "Normal"]`, clock comments and a parenthesised variation after every black move.

```python
def test_a_loss_gives_every_position_from_ply_12()
    # len == 30 - 12; each FEN loads in chess.Board; the first equals the board after 12 main-line plies
def test_variations_are_not_followed()        # no FEN equals a position that occurs only inside a variation
def test_wins_draws_and_forfeits_give_nothing()
    # Result "0-1"; "1/2-1/2"; Termination "Time forfeit"; Vektr64 as White with "1-0" -> [] each
def test_other_variants_and_set_up_positions_give_nothing()   # Variant "Chess960"; a FEN header -> []
def test_short_empty_and_broken_records_give_nothing()        # 8 plies; headers only; "not a pgn"; "" -> []
def test_upload_writes_one_file_per_lost_game_once(tmp_path)
    # records: a loss, a win, a loss touched just now -> returns 1, data:experience/<stem>.fens holds 18 lines;
    # a second call returns 0
def test_a_broken_record_does_not_stop_the_others(tmp_path)   # garbage file beside a loss -> returns 1
# test_cli.py
def test_sync_reports_lost_games(…)           # output has "lost games added: 1"; a StoreUnavailable there does not fail the sync
```

- [ ] **Step 2: Run** `(ulimit -v 6000000; timeout -s KILL 280 uv run --project ops pytest ops/tests/test_experience.py -q -p no:cacheprovider)`. Expected: fails, no module `vektr_ops.experience`.

- [ ] **Step 3: Implement.** Headers are checked with `results.headers` before `chess.pgn.read_game` parses the moves; any exception while reading one record counts as "no starts". In `cli.sync`, call `upload_experience` after `upload_games` and print `lost games added: N`, handled like the game-record upload.

- [ ] **Step 4: Run** the whole ops suite. Expected: all pass.

- [ ] **Step 5: Commit** `feat(ops): starting positions from lost games`.

---

### Task 5: Ops — generate and upload a shard

**Files:**
- Create: `ops/src/vektr_ops/selfplay.py`, `ops/tests/test_selfplay.py`
- Modify: `ops/src/vektr_ops/cli.py` (`datagen` command), `ops/tests/test_cli.py`

**Interfaces:**
- Consumes: `builds.ensure_build(root, commit) -> Path`, `manifest.load`, `cli.resolve_commit`, `Store`, `match.concurrency`, `gate.machine`, `match.bot_here`.
- Produces:

```python
SHARD_POSITIONS = 500_000
NODES = 5000
PER_NET = 120
PREFIX = "selfplay/"
RECORD_BYTES = 68
class DatagenFailed(Exception): ...
def shard_path(net: str, host: str, now: datetime, rand: str) -> str
    # "selfplay/20261128T170102Z-4a10db9d9f3f084c-irom-SEi-9f3c.vkd.gz"; net is "none" for a champion without a network
def net_of(path: str) -> str                      # the network part of a shard path
def shards_for(store: Store, net: str) -> list[str]
def generate(store: Store, root: Path, *, commit: str, threads: int, positions: int = SHARD_POSITIONS,
             build=ensure_build, run=subprocess.run) -> str     # store path of the uploaded shard
def flush(store: Store, root: Path) -> int        # upload shards left in state/datagen/ by an earlier failure
```

  CLI: `vektr-ops datagen [--shards N=1] [--threads T]`; `T` defaults to `concurrency(*machine(), bot_here())` and the command exits with status 1 and a message when that is 0. Each shard prints its store path and positions per second.

- [ ] **Step 1: Write the failing tests** (fake `build` returns a path; fake `run` writes `positions * 68` zero bytes to the `--out` argument and returns status 0):

```python
def test_shard_path_sorts_by_time_and_names_the_network()
def test_generate_uploads_one_gzipped_shard(tmp_path)
    # the returned path starts with "selfplay/" and contains the champion's network; gunzipped length ==
    # positions * 68; state/datagen/ is empty afterwards
def test_the_champions_network_is_given_to_the_engine(tmp_path)
    # nets:nets/<hash>.vknn in the store -> the command has "--net" followed by a file with those bytes,
    # and "--nodes", "5000", "--positions", str(positions), "--threads", "2"
def test_a_champion_without_a_network_uses_the_built_in_one(tmp_path)   # no "--net"; path contains "-none-"
def test_a_network_missing_from_the_store_refuses(tmp_path)             # DatagenFailed, run never called
def test_lost_game_positions_become_the_starts_file(tmp_path)
    # two data:experience/*.fens -> "--starts" names a file with all their lines; none -> no "--starts"
def test_a_wrong_sized_output_is_not_uploaded(tmp_path)     # run writes 100 bytes -> DatagenFailed, store unchanged
def test_no_output_is_not_uploaded(tmp_path)                # run returns 0 and writes nothing -> DatagenFailed
def test_an_engine_failure_carries_its_output(tmp_path)     # status 1, stderr "boom" -> DatagenFailed containing "boom"
def test_an_upload_failure_keeps_the_shard_for_flush(tmp_path)
    # store raises StoreUnavailable on write -> generate raises it, one *.vkd.gz remains in state/datagen/;
    # flush on a working store returns 1, the shard is in the store, the directory is empty
# test_cli.py
def test_datagen_refuses_when_this_machine_must_not_play(…)   # concurrency 0 -> exit 1
```

- [ ] **Step 2: Run** `(ulimit -v 6000000; timeout -s KILL 280 uv run --project ops pytest ops/tests/test_selfplay.py -q -p no:cacheprovider)`. Expected: fails, no module `vektr_ops.selfplay`.

- [ ] **Step 3: Implement.** `generate`: `flush` first; build the engine for `commit`; read the champion from the manifest and write its network to `state/nets/<hash>.vknn` (once); join every `data:experience/*.fens` into `state/datagen/starts.txt` (files already fetched are kept in `state/experience/`); run `[binary, "datagen", "--out", …, "--positions", …, "--threads", …, "--nodes", "5000", "--seed", <secrets.randbits(63)>]` with `stdin=DEVNULL`, captured output and a 3-hour timeout; check the size; gzip into `state/datagen/` under the store path with `/` written as `__`; upload; delete the local files. The CLI passes `resolve_commit(root, "HEAD")`.

- [ ] **Step 4: Run** the whole ops suite. Expected: all pass.

- [ ] **Step 5: Try it for real on the laptop against a local store:** `VEKTR_STORE=<scratchpad>/store` with a bootstrapped manifest is enough to see one shard appear and read back with `vektr_trainer.records.read_records`. Report positions per second.

- [ ] **Step 6: Commit** `feat(ops): datagen command uploads self-play shards`.

---

### Task 6: Ops — training uses self-play shards

**Files:**
- Modify: `ops/src/vektr_ops/training.py`, `ops/src/vektr_ops/cli.py` (`train`), `ops/tests/test_training.py`, `ops/tests/test_cli.py`

**Interfaces:**
- Consumes: `selfplay.PREFIX`, `selfplay.PER_NET`.
- Produces:

```python
def pull_shards(store: Store, dest: Path, limit: int | None = None, selfplay: int = PER_NET) -> list[str]
    # the first `limit` bootstrap shards by path, then the newest `selfplay` self-play shards (0 = none)
def run_training(store, root, run, *, minutes, shards=None, selfplay=PER_NET, env, …) -> dict
```

  CLI: `vektr-ops train … [--selfplay N]`, default 120. `deploy/burst.sh` is unchanged, so a burst uses the default.

- [ ] **Step 1: Write the failing tests.**

```python
def test_pull_shards_adds_the_newest_selfplay(tmp_path)
    # 2 bootstrap .npz and 5 self-play .vkd.gz, selfplay=3 -> both bootstrap and the 3 with the latest times,
    # local names use "__" for "/", files are byte-identical
def test_selfplay_zero_takes_none(tmp_path)
def test_shards_no_longer_selected_are_removed(tmp_path)    # pull with selfplay=5, then 2 -> 3 local files gone
def test_the_run_record_lists_selfplay_shards(tmp_path)     # record["shards"] contains the self-play paths
# test_cli.py
def test_train_passes_selfplay(…)                           # --selfplay 7 reaches run_training
```

- [ ] **Step 2: Run** the ops suite. Expected: the new tests fail (`selfplay` is an unexpected argument).

- [ ] **Step 3: Implement** the selection and pass the option through.

- [ ] **Step 4: Run** the ops suite. Expected: all pass.

- [ ] **Step 5: Dry run of the whole path on the laptop:** with the local store from Task 5 holding a few small self-play shards and one small `.npz`, run `VEKTR_LOCAL=1 VEKTR_RUN=dry VEKTR_MINUTES=1 deploy/burst.sh`. Expected: a network in `nets:runs/dry/`, whose `run.json` lists both kinds of shard and has a `val_loss`.

- [ ] **Step 6: Commit** `feat(ops): training mixes in the newest self-play shards`.

---

### Task 7: Ops — the worker generates when idle; status; runbook

**Files:**
- Modify: `ops/src/vektr_ops/selfplay.py`, `ops/src/vektr_ops/gate.py` (`work`), `ops/src/vektr_ops/cli.py` (`gate --work`, `store_lines`), `deploy/vektr64-worker.service` (description only), `docs/runbook.md`, `ops/tests/test_selfplay.py`, `ops/tests/test_gate.py`, `ops/tests/test_cli.py`

**Interfaces:**
- Consumes: Task 5's `generate`, `shards_for`, `PER_NET`.
- Produces:

```python
# selfplay.py
def step(store: Store, root: Path, env: dict, *, commit: str, generate=generate) -> bool
    # False without generating when env["VEKTR_DATAGEN"] == "0", when concurrency(...) is 0, or when the
    # champion's network already has PER_NET shards; otherwise generates one shard and returns True
# gate.py
def work(store, root, *, once=False, idle_s=300, datagen=None) -> None
    # `datagen()` is attempted only in a pass where no test and no anchor work was done; a True result
    # counts as busy
```

  `vektr-ops status` gains the line `self-play: <n> of 120 shards for network <hash>, lost games: <m>`.

- [ ] **Step 1: Write the failing tests.**

```python
# test_selfplay.py
def test_step_generates_one_shard(tmp_path)                 # returns True, generate called once with threads == concurrency
def test_step_stops_at_the_target(tmp_path)                 # 120 shards for the champion's network -> False, generate not called
def test_shards_of_an_older_network_do_not_count(tmp_path)  # 120 shards for another network -> True
def test_the_switch_turns_it_off(tmp_path)                  # env {"VEKTR_DATAGEN": "0"} -> False
def test_step_does_nothing_where_play_would_suffer(tmp_path, monkeypatch)   # concurrency 0 -> False
# test_gate.py
def test_worker_generates_only_when_idle(tmp_path)
    # an open candidate whose step did work -> datagen not called; no candidates and no anchor work -> called once
def test_a_datagen_fault_does_not_stop_the_worker(tmp_path) # datagen raises RuntimeError -> work(once=True) returns
# test_cli.py
def test_status_shows_selfplay_progress(…)                  # "self-play: 2 of 120 shards for network …, lost games: 1"
```

- [ ] **Step 2: Run** the ops suite. Expected: the new tests fail.

- [ ] **Step 3: Implement.** `cli.gate --work` passes `datagen=lambda: selfplay.step(store, root, load_env(root), commit=resolve_commit(root, "HEAD"))`. Existing callers and tests of `work` keep `datagen=None`.

- [ ] **Step 4: Write the runbook section "The loop"** in `docs/runbook.md`: what the worker now does and at what priority; `vektr-ops status` shows progress; `VEKTR_DATAGEN=0` in `.env` plus `sudo systemctl restart vektr64-worker` turns self-play off; `vektr-ops datagen --shards N` adds shards from the laptop; one round by hand is `burst` → `install-net` → branch, commit, push → `gate`, exactly as in "Training a network"; `train --selfplay 0` trains on Lichess data alone. Update the "Background units" table row for the worker.

- [ ] **Step 5: Run** all three suites with the bounded commands. Expected: all pass.

- [ ] **Step 6: Commit** `feat(ops): the worker generates self-play data when idle`.

---

## After the code is merged: the first round by hand

Each numbered step that spends money or changes what plays waits for the owner's go-ahead.

1. **Update the box:** `git pull`, `uv sync --project ops`, `sudo systemctl restart vektr64-worker`. Confirm `vektr-ops status` still shows `playing: 16df60e265a9`.
2. **Watch the first day.** `vektr-ops status` shows the shard count rising. Check rated games for time forfeits (`grep -l "Time forfeit" bot/game_records/*.pgn`) and compare the bot's search depth in the journal with the day before. If either is worse, set `VEKTR_DATAGEN=0` and report.
3. **Optionally add laptop shards:** `vektr-ops datagen --shards 20`.
4. **Burst** when the count reaches 120, or earlier if the owner prefers: `vektr-ops burst --minutes 45 --run gen2-<date>` (about $0.40).
5. **Install, commit, gate:** `vektr-ops install-net gen2-<date>`, branch `net-gen2`, a `docs/bench.md` row, push, `vektr-ops gate net-gen2`. An `accept` promotes the network and the box switches within ten minutes; a `reject` leaves the champion in place and is recorded.
6. **Repeat** from step 2 with the new champion's network. A second round only makes sense after an `accept`; after a `reject`, the next experiment is one of: more self-play shards, `--selfplay` with a different count, or a different score/result blend, each as its own gated run.
