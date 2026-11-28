# Stage 3: First Network Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hand-written evaluation with a small neural network trained in-house: network inference in the engine, a trainer, training data from the Lichess evaluation database, a rented-GPU burst that cannot be left running, and the first network promoted through the gate.

**Architecture:** A network file is a few hundred kilobytes, so it is committed inside `engine/nets/` and compiled into the binary. A version is therefore still one engine commit, and the Stage 2 gate, builds, sync and rollback work unchanged. The trainer is a separate Python project (`trainer/`) that knows nothing about the store; `vektr-ops` wraps it with pulls and pushes. The engine and the trainer share one integer definition of the evaluation, and a check file of positions with expected scores travels with every network so that any build can prove it agrees.

**Tech Stack:** Rust (engine, `cozy-chess`, AVX2 intrinsics with a scalar fallback); Python 3.12, `uv`, PyTorch, numpy, zstandard (trainer); typer, `huggingface_hub`, pytest (ops); RunPod REST API v2; the Lichess evaluation database (CC0).

**Spec:** `docs/superpowers/specs/2026-11-27-vektr64-design.md`

## Global Constraints

- Work on branch `stage3-first-network`, cut from `main`.
- No other engine's code or network is used in the playing engine. Stockfish scores in the Lichess database are training labels only.
- Network shape: 768 inputs (piece × colour × square) → one hidden layer of 128, two perspectives → one output. Integer inference, updated incrementally, AVX2 when the CPU has it and a scalar fallback otherwise, chosen at run time. Both paths give identical scores.
- The engine and the trainer agree bit for bit on the check positions before any gate match starts.
- A default network is compiled into the binary; `EvalFile` loads another one at run time.
- Play is CPU only. The trainer picks its device automatically: CUDA, then MPS, then CPU.
- Training writes a checkpoint every 10 minutes and resumes from the latest one.
- A rented instance terminates itself on success, on failure and at a hard time limit.
- Secrets (`LICHESS_TOKEN`, `HF_TOKEN`, `RUNPOD_API_KEY`) come from the environment or the git-ignored `.env`. They are never written to git, the store, logs or the page.
- Nothing is promoted unless its gate decision is `accept` (bounds `elo0=0`, `elo1=5`, 5% error rates).
- Tests never touch the network, Hugging Face, Lichess or RunPod: they use `LocalStore` in `tmp_path`, scripted inputs and fake API senders.
- No field is added to the manifest's `Version` record: the box may still be running older code that rejects unknown fields. New facts go into the existing `net` string and the free-form `gate` dictionary.
- Commit messages end with `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>`.

## Decisions this plan makes (owner can overturn)

- **The network lives in git**, at `engine/nets/default.vknn` (197,392 bytes) with its check file `engine/nets/default.check`. The version id stays the engine commit's first 12 characters; the manifest's `net` field records the first 16 hex characters of the file's SHA-256. Training runs, checkpoints and every produced network are also kept in the store under `nets:runs/<run>/` and `nets:nets/<hash>.vknn`.
- **Activation and scaling:** squared clipped ReLU; weights quantised with `QA=255` (hidden layer) and `QB=64` (output), score scale 400.
- **Training data:** the first 30,000,000 usable positions of `lichess_db_eval.jsonl.zst` (22.5 GB compressed, 417,862,375 positions, lines in no particular order, scores from White's point of view; checked on 2026-11-28). About 4 GB is streamed, nothing is stored locally except the shards. For each position the deepest evaluation's first line is used. Dropped: forced mates (about 19% of lines) and positions whose best move is a capture. Scores are clipped to ±3000.
- **Shards:** 1,000,000 positions each, compressed numpy files in the data repo under `bootstrap/lichess-eval-<file date>/`.
- **Target:** `sigmoid(score / 400)`, blended with the game result at weight 0.75 score / 0.25 result when a shard has results. The Lichess data has none, so the first network learns from scores alone.
- **Optimiser:** Adam, batch 16,384, learning rate 0.001 falling on a cosine to 0.00001 over the time budget, 1% of positions held out for a validation loss.
- **Burst:** 45 minutes of training plus up to 20 minutes of setup; hard limit 65 minutes. GPU types tried in order: RTX A4000, RTX 3090, RTX A5000, RTX 4090. The launcher refuses any rate above $0.60 per hour and asks for confirmation after printing the rate and the worst-case cost.
- **The pod receives `HF_TOKEN`** in its environment, because it must pull shards and push checkpoints. The owner may prefer a separate fine-grained Hugging Face token limited to the two `vektr64-*` repos for this.
- **Box speed check:** a network engine searches fewer nodes per second than the hand-written one, so the Stage 2 rule (refuse a build more than 10% slower than the one it replaces) would refuse every network. The gate now records the candidate's speed relative to the champion on the machines that played, and the box refuses a build more than 10% slower than *that expectation*.
- **Two pull requests.** The code (this plan's Tasks 1–8, with a small network trained on the laptop) is reviewed and merged first, and the box is updated. The burst-trained network is then a second, small pull request that only changes the network files.

## Review Focus

1. `EvalFile` is given a missing, truncated or wrong-shape file: the engine reports the error, keeps the network it had, and still answers `go`. It never panics. (Tasks 2, 6)
2. Castling, en passant, promotion and capture-promotion: the incrementally updated accumulator equals a full recomputation. (Task 2)
3. A line of the Lichess file is malformed, has only a mate score, has no evaluations, or its best move is castling written as king-takes-own-rook: the preparation skips or keeps it correctly and never dies. (Task 3)
4. Training is killed mid-run: the next run with the same id continues from the stored checkpoint, and a half-written checkpoint is never uploaded. (Tasks 4, 5)
5. RunPod has no capacity, quotes a rate above the cap, or the container fails before training starts: no pod is left running. (Task 8)

Also checked in Task 7: rolling back to the Stage 1 champion, which has no network and no `netcheck` command, still works on the box.

## File Structure

| File | Responsibility |
|---|---|
| `trainer/pyproject.toml` | `vektr-trainer` project; PyTorch from the CPU wheel index under `uv` |
| `trainer/src/vektr_trainer/features.py` | Feature indices from a FEN |
| `trainer/src/vektr_trainer/vknn.py` | Network file format, quantised network, reference integer evaluation, check files |
| `trainer/src/vektr_trainer/check_fens.txt` | 256 fixed positions for the engine ↔ trainer check |
| `trainer/src/vektr_trainer/prep.py` | Lichess evaluation lines → shards |
| `trainer/src/vektr_trainer/model.py` | The PyTorch model and its quantisation |
| `trainer/src/vektr_trainer/train.py` | Time-budgeted training loop with checkpoints and resume |
| `engine/src/nnue.rs` | Network loading, accumulator, evaluation, AVX2 and scalar paths |
| `engine/nets/default.vknn`, `default.check` | The compiled-in network and its check positions |
| `ops/src/vektr_ops/data.py` | Run the preparation and upload shards |
| `ops/src/vektr_ops/training.py` | Pull shards, run the trainer, push checkpoints and results, install a network |
| `ops/src/vektr_ops/runpod.py` | Thin RunPod REST client |
| `ops/src/vektr_ops/burst.py` | Build the pod request, launch, watch, kill |
| `deploy/burst.sh` | What a burst instance runs |

## The evaluation, defined once

Both the engine and the trainer implement exactly this.

- **Feature index, White's view:** `colour * 384 + piece * 64 + square`, with colour 0 = white, 1 = black; piece 0–5 = pawn, knight, bishop, rook, queen, king; square a1 = 0 … h8 = 63.
- **Black's view of the same piece:** `(1 - colour) * 384 + piece * 64 + (square ^ 56)`.
- **Accumulator** for each view: `ft_b[j] + Σ ft_w[f][j]` over the features present, in 16-bit integers.
- **Output:** with `us` the side to move's accumulator and `them` the other, `sum = Σ_j clamp(us[j], 0, 255)² · out_w[j] + Σ_j clamp(them[j], 0, 255)² · out_w[128 + j]`, then `cp = (sum / 255 + out_b) * 400 / (255 * 64)`. Both divisions truncate toward zero. The engine clamps the result to ±20000.
- **File (`VKNN` version 1), little-endian:** `b"VKNN"`, `u32` version = 1, `u32` hidden = 128, `ft_w` as 768 × 128 `i16` (feature-major), `ft_b` as 128 `i16`, `out_w` as 256 `i16`, `out_b` as one `i32`. Total 197,392 bytes.
- **Valid ranges**, enforced by both readers: `|ft_w|, |ft_b| ≤ 505` and `|out_w| ≤ 127`. Within them no 16-bit accumulator and no 32-bit sum can overflow.
- **Check file:** one line per position, `<fen>;<cp>`.

---

### Task 1: Trainer core — features, network file, reference evaluation

**Files:**
- Create: `trainer/pyproject.toml`, `trainer/src/vektr_trainer/__init__.py`, `features.py`, `vknn.py`, `check_fens.txt`
- Create: `engine/nets/default.vknn`, `engine/nets/default.check` (a seeded random network, replaced in Task 6)
- Test: `trainer/tests/test_features.py`, `trainer/tests/test_vknn.py`

**Interfaces:**
- Produces:
  - `features.INPUTS = 768`, `HIDDEN = 128`, `EMPTY = 768`, `MAX_PIECES = 32`
  - `features.feature(colour: int, piece: int, square: int) -> int`
  - `features.flip(index: int) -> int` (Black's view; `flip(EMPTY) == EMPTY`)
  - `features.fen_features(fen: str) -> tuple[list[int], int]` — White-view indices in ascending order and side to move (0 white, 1 black); `ValueError` for a board field that is not 8 ranks of 8 files, has an unknown character, or more than 32 pieces. Only the first two FEN fields are read.
  - `vknn.QA = 255`, `QB = 64`, `SCALE = 400`
  - `vknn.QNet(ft_w: np.ndarray[int16, (768,128)], ft_b: np.ndarray[int16, (128,)], out_w: np.ndarray[int16, (256,)], out_b: int)`
  - `vknn.write(path: Path, net: QNet) -> None` (atomic), `vknn.read(path: Path) -> QNet` (`ValueError` on bad magic, version, hidden size, length or out-of-range values)
  - `vknn.random_net(seed: int) -> QNet`, `vknn.net_hash(path: Path) -> str` (first 16 hex of SHA-256)
  - `vknn.evaluate(net: QNet, fen: str) -> int`
  - `vknn.check_fens() -> list[str]`, `vknn.write_check(path: Path, net: QNet) -> None`, `vknn.read_check(path: Path) -> list[tuple[str, int]]`
  - `python -m vektr_trainer.vknn random --seed N --out DIR` writes `DIR/default.vknn` and `DIR/default.check`

- [ ] **Step 1: Create the project.** `trainer/pyproject.toml`: name `vektr-trainer`, `requires-python = ">=3.12"`, dependencies `numpy>=2`, `zstandard>=0.23`, `torch>=2.4`; dev group `pytest>=8`; hatchling, package `src/vektr_trainer`; `[tool.pytest.ini_options]` with `testpaths = ["tests"]` and `markers = ["slow"]`; and so that `uv` installs the small CPU build on Linux:

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cpu" }
```

Run: `uv sync --project trainer` — Expected: completes; `uv run --project trainer python -c "import torch; print(torch.__version__)"` prints a version ending `+cpu`.

- [ ] **Step 2: Write the failing feature tests**

```python
START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

def test_startpos_has_32_features_and_white_to_move():
    indices, stm = fen_features(START)
    assert len(indices) == 32 and stm == 0 and indices == sorted(indices)

def test_known_indices():
    indices, _ = fen_features(START)
    assert feature(0, 0, 8) == 8 and 8 in indices          # white pawn a2
    assert feature(1, 5, 60) == 764 and 764 in indices      # black king e8

def test_flip_is_the_black_view_and_its_own_inverse():
    assert flip(8) == 432                                   # white pawn a2 seen by Black
    assert all(flip(flip(i)) == i for i in range(768)) and flip(EMPTY) == EMPTY

def test_lichess_four_field_fen_and_black_to_move():
    assert fen_features("7r/1p3k2/p1bPR3/5p2/2B2P1p/8/PP4P1/3K4 b - -")[1] == 1

@pytest.mark.parametrize("fen", ["8/8/8 w - -", "9/8/8/8/8/8/8/8 w - -", "x7/8/8/8/8/8/8/8 w - -",
                                 "QQQQQQQQ/QQQQQQQQ/QQQQQQQQ/QQQQQQQQ/QQQQQQQQ/8/8/8 w - -", ""])
def test_bad_boards_are_rejected(fen):
    with pytest.raises(ValueError):
        fen_features(fen)
```

- [ ] **Step 3: Run** `uv run --project trainer pytest trainer/tests/test_features.py` — Expected: FAIL, `fen_features` not defined.
- [ ] **Step 4: Implement `features.py`.** Run again — Expected: PASS.

- [ ] **Step 5: Write the failing network-file tests**

```python
def zero_net(out_b=0):
    return QNet(np.zeros((768, 128), np.int16), np.zeros(128, np.int16), np.zeros(256, np.int16), out_b)

def mirror(fen):  # swap colours and flip ranks; the side to move swaps too
    board, stm = fen.split()[:2]
    return "/".join(r.swapcase() for r in reversed(board.split("/"))) + (" b" if stm == "w" else " w") + " - -"

def test_output_bias_alone_gives_the_scale():
    assert evaluate(zero_net(255 * 64), START) == 400

def test_division_truncates_toward_zero():
    assert evaluate(zero_net(-1), START) == 0 and evaluate(zero_net(-41), START) == -1

def test_one_feature_by_hand():
    net = zero_net()
    net.ft_w[feature(0, 5, 4), 0] = 100       # white king e1, first hidden unit
    net.out_w[0] = 64                         # "us" half
    net.out_w[128] = -64                      # "them" half
    # White to move: us sees the king (100² · 64), them does not.
    assert evaluate(net, "4k3/8/8/8/8/8/8/4K3 w - -") == (100 * 100 * 64 // 255) * 400 // (255 * 64)
    assert evaluate(net, "4k3/8/8/8/8/8/8/4K3 b - -") == -((100 * 100 * 64 // 255) * 400 // (255 * 64))

def test_mirrored_positions_score_the_same():
    net = random_net(7)
    for fen in check_fens():
        assert evaluate(net, fen) == evaluate(net, mirror(fen))

def test_write_read_round_trip(tmp_path):
    net = random_net(3); write(tmp_path / "n.vknn", net)
    back = read(tmp_path / "n.vknn")
    assert (tmp_path / "n.vknn").stat().st_size == 197_392
    assert (back.ft_w == net.ft_w).all() and (back.out_w == net.out_w).all() and back.out_b == net.out_b

@pytest.mark.parametrize("damage", ["truncate", "magic", "version", "hidden", "range"])
def test_read_rejects_bad_files(tmp_path, damage): ...   # each raises ValueError

def test_check_fens_are_256_valid_positions_with_both_sides_to_move():
    fens = check_fens()
    assert len(fens) == 256 and {fen_features(f)[1] for f in fens} == {0, 1}

def test_check_file_round_trip(tmp_path):
    net = random_net(5); write_check(tmp_path / "c", net)
    assert read_check(tmp_path / "c") == [(f, evaluate(net, f)) for f in check_fens()]
```

- [ ] **Step 6: Run** `uv run --project trainer pytest trainer/tests/test_vknn.py` — Expected: FAIL, module missing.
- [ ] **Step 7: Create `check_fens.txt`.** Line 1 is the start position, line 2 is `8/8/8/4k3/8/8/8/4K3 w - -`, lines 3–256 are the `fen` values of the first 254 lines of `https://database.lichess.org/lichess_db_eval.jsonl.zst` (a ranged request for the first 2 MB, decompressed with `zstandard`, is enough). One FEN per line.
- [ ] **Step 8: Implement `vknn.py`.** `random_net` draws `ft_w`, `ft_b` uniformly from −64…64, `out_w` from −127…127 and `out_b` from −20000…20000 with `numpy.random.default_rng(seed)`. `evaluate` uses Python integers and truncating division (`int(a / b)` is not exact for large values; use `abs(a) // b` with the sign restored). Run the tests — Expected: PASS.
- [ ] **Step 9: Produce the placeholder network.** Run: `uv run --project trainer python -m vektr_trainer.vknn random --seed 1 --out engine/nets` — Expected: `engine/nets/default.vknn` is 197,392 bytes and `default.check` has 256 lines.
- [ ] **Step 10: Commit** `feat(trainer): network file format and reference evaluation`.

### Task 2: Engine network module

**Files:**
- Create: `engine/src/nnue.rs`
- Modify: `engine/src/lib.rs` (add `pub mod nnue;`), `engine/src/main.rs` (the `netcheck` subcommand)
- Test: `engine/tests/nnue.rs`

**Interfaces:**
- Consumes: the file format and evaluation definition above; `engine/nets/default.vknn`, `default.check`.
- Produces:
  - `nnue::HIDDEN: usize = 128`, `nnue::DEFAULT_NET: &[u8]`, `nnue::DEFAULT_CHECK: &str`
  - `nnue::Network::from_bytes(bytes: &[u8]) -> Result<Network, String>`, `Network::load(path: &str) -> Result<Network, String>`, `Network::embedded() -> Arc<Network>` (parsed once)
  - `nnue::Accumulator { white: [i16; HIDDEN], black: [i16; HIDDEN] }`, 32-byte aligned, `Clone`
  - `Accumulator::refresh(net: &Network, board: &Board) -> Accumulator`
  - `Accumulator::update_from(&mut self, parent: &Accumulator, net: &Network, before: &Board, after: &Board)` — finds the changed squares by comparing each colour-and-piece bitboard of the two boards, so every move type is handled the same way
  - `Network::evaluate(&self, acc: &Accumulator, stm: Color) -> i32` (run-time dispatch), `Network::evaluate_scalar(...) -> i32`, `Network::evaluate_board(&self, board: &Board) -> i32`
  - `nnue::netcheck(net: &Network, check: &str) -> Result<usize, Vec<String>>` — the number of positions checked, or one line per disagreement
  - `vektr netcheck` prints `netcheck ok <n> positions` and exits 0, or prints `mismatch <fen>: engine <a> trainer <b>` lines and exits 1

- [ ] **Step 1: Write the failing tests** in `engine/tests/nnue.rs`

```rust
#[test] fn embedded_network_agrees_with_the_trainer() {
    assert_eq!(netcheck(&Network::embedded(), DEFAULT_CHECK), Ok(256));
}
#[test] fn incremental_update_equals_refresh_over_random_games() {
    // 200 games of up to 150 random legal moves (xorshift seed 1..=200) from the start position
    // and from each of SPECIALS; after every move update_from(parent, before, after) must equal
    // refresh(after) in both halves.
}
const SPECIALS: [&str; 4] = [
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",                 // castling, both sides and wings
    "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",                    // en passant
    "1n2k3/P7/8/8/8/8/7p/4K1N1 w - - 0 1",                  // promotion and capture-promotion
    "8/8/8/4k3/8/8/8/4K3 w - - 0 1",
];
#[test] fn each_special_move_is_actually_played() { /* O-O, O-O-O, e5xd6 e.p., a7xb8=Q occur in the walk */ }
#[test] fn simd_and_scalar_give_the_same_score() { /* every check position, and 10,000 random accumulators in −300..600 */ }
#[test] fn colour_mirrored_positions_score_the_same() { /* the pair from the old tests/eval.rs */ }
#[test] fn bad_network_bytes_are_errors_not_panics() {
    // empty, 100 bytes, wrong magic, version 2, hidden 256, one byte short, one byte long,
    // an ft weight of 506, an output weight of 128
}
#[test] fn netcheck_reports_a_disagreement() {
    assert!(netcheck(&Network::embedded(), "8/8/8/4k3/8/8/8/4K3 w - -;12345").is_err());
}
```

- [ ] **Step 2: Run** `(cd engine && cargo test --release --test nnue)` — Expected: FAIL to compile, `nnue` missing.
- [ ] **Step 3: Implement `nnue.rs`.** The AVX2 functions carry `#[target_feature(enable = "avx2")]` and are called only after `is_x86_feature_detected!("avx2")`; on other architectures only the scalar path is compiled. AVX2 output layer: clamp to 0…255, `mullo` by the output weight (fits 16 bits: 255 · 127), `madd` with the clamped value, sum 32-bit lanes.
- [ ] **Step 4: Add `netcheck` to `main.rs`.** Run: `(cd engine && cargo test --release --test nnue && cargo run --release -- netcheck)` — Expected: tests PASS; `netcheck ok 256 positions`.
- [ ] **Step 5: Run the whole engine suite and the bench.** `(cd engine && cargo test --release && cargo run --release -- bench 8)` — Expected: all PASS; `38791222 nodes` still (search does not use the network yet).
- [ ] **Step 6: Commit** `feat(engine): network inference with incremental updates`.

### Task 3: Training data from the Lichess evaluation database

**Files:**
- Create: `trainer/src/vektr_trainer/prep.py`, `ops/src/vektr_ops/data.py`
- Modify: `ops/src/vektr_ops/cli.py` (command `prep`)
- Test: `trainer/tests/test_prep.py`, `ops/tests/test_data.py`

**Interfaces:**
- Consumes: `features.fen_features`, `EMPTY`; `Store`.
- Produces:
  - `prep.SHARD_POSITIONS = 1_000_000`, `prep.SCORE_CLIP = 3000`, `prep.NO_RESULT = 255`
  - `prep.parse_line(line: bytes) -> tuple[list[int], int, int] | None` — White-view indices, side to move, score in centipawns for the side to move
  - `prep.write_shard(path: Path, features: np.ndarray, stm: np.ndarray, score: np.ndarray, result: np.ndarray) -> None` — `np.savez_compressed`, atomic; arrays `features` `uint16 [N, 32]` padded with `EMPTY`, `stm` `uint8`, `score` `int16`, `result` `uint8` (0 loss, 1 draw, 2 win for the side to move, `NO_RESULT` unknown)
  - `prep.read_shard(path: Path) -> dict[str, np.ndarray]`
  - `prep.prep(lines: Iterable[bytes], out: Path, *, positions: int, skip_shards: int = 0, shard_positions: int = SHARD_POSITIONS, on_shard=None) -> int` — writes `shard-00000.npz`, …; returns shards written; `ValueError` unless `positions` is a multiple of `shard_positions`; an input that ends early leaves the incomplete shard unwritten
  - `python -m vektr_trainer.prep --url URL --out DIR --positions N --skip K` streams and decompresses the URL and prints `SHARD <path>` after each shard
  - `data.SOURCE = "https://database.lichess.org/lichess_db_eval.jsonl.zst"`
  - `data.source_tag(url: str, fetch=urllib.request.urlopen) -> str` — `lichess-eval-YYYYMMDD` from the file's `Last-Modified` header
  - `data.run_prep(store: Store, root: Path, *, positions: int, url: str = SOURCE, runner=None) -> int` — counts the shards already under `data:bootstrap/<tag>/`, runs the preparation with `--skip` that many, uploads each shard as it is announced, deletes the local copy; returns shards uploaded
  - `training.trainer_cmd(env: dict, module: str, *args: str) -> list[str]` — `[env["VEKTR_TRAINER_PYTHON"], "-m", module, ...]` when that variable is set, otherwise `["uv", "run", "--project", "trainer", "python", "-m", module, ...]` (created here in `ops/src/vektr_ops/training.py`, extended in Task 5)
  - `vektr-ops prep --positions 30000000`

- [ ] **Step 1: Write the failing trainer tests**

```python
def line(fen, *evals):
    return json.dumps({"fen": fen, "evals": list(evals)}).encode()
def ev(depth, **pv):
    return {"depth": depth, "knodes": 1, "pvs": [pv]}

QUIET = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"

def test_deepest_evaluation_first_line_is_used():
    got = parse_line(line(QUIET, ev(20, cp=10, line="e2e4"), ev(36, cp=31, line="d2d4"), ev(30, cp=5, line="g1f3")))
    assert got[2] == 31 and got[1] == 0 and len(got[0]) == 32

def test_score_is_turned_to_the_side_to_move():
    black = "7r/1p3k2/p1bPR3/5p2/2B2P1p/8/PP4P1/3K4 b - -"
    assert parse_line(line(black, ev(30, cp=69, line="f7g7")))[2] == -69

def test_scores_are_clipped():
    assert parse_line(line(QUIET, ev(30, cp=9000, line="e2e4")))[2] == 3000

@pytest.mark.parametrize("raw", [
    b"not json", b"{}", line(QUIET), line(QUIET, ev(30, mate=3, line="e2e4")),
    line("8/8/8 w - -", ev(30, cp=1, line="e2e4")),
    line("4k3/8/8/3p4/4P3/8/8/4K3 w - -", ev(30, cp=50, line="e4d5")),          # best move captures
    line("4k3/8/8/3pP3/8/8/8/4K3 w - d6", ev(30, cp=50, line="e5d6")),          # en passant capture
])
def test_unusable_lines_are_skipped(raw):
    assert parse_line(raw) is None

def test_castling_as_king_takes_own_rook_is_not_a_capture():
    assert parse_line(line("r3k2r/8/8/8/8/8/8/R3K2R w KQkq -", ev(30, cp=0, line="e1h1"))) is not None

def test_a_line_without_a_move_list_is_kept():
    assert parse_line(line(QUIET, {"depth": 30, "knodes": 1, "pvs": [{"cp": 12}]}))[2] == 12

def test_shard_round_trip(tmp_path): ...            # dtypes, shapes and values survive; padding is EMPTY

def test_prep_writes_full_shards_and_can_skip(tmp_path):
    lines = [line(QUIET, ev(30, cp=i, line="e2e4")) for i in range(7)] + [b"junk"]
    assert prep(lines, tmp_path / "a", positions=6, shard_positions=2) == 3
    assert prep(lines, tmp_path / "b", positions=6, skip_shards=1, shard_positions=2) == 2
    assert not (tmp_path / "b" / "shard-00000.npz").exists()
    for name in ("shard-00001.npz", "shard-00002.npz"):
        assert (read_shard(tmp_path / "a" / name)["score"] == read_shard(tmp_path / "b" / name)["score"]).all()

def test_input_that_ends_early_leaves_no_partial_shard(tmp_path):
    assert prep([line(QUIET, ev(30, cp=1, line="e2e4"))] * 3, tmp_path, positions=4, shard_positions=2) == 1

def test_positions_must_be_whole_shards(tmp_path):
    with pytest.raises(ValueError):
        prep([], tmp_path, positions=3, shard_positions=2)
```

- [ ] **Step 2: Run** `uv run --project trainer pytest trainer/tests/test_prep.py` — Expected: FAIL, module missing.
- [ ] **Step 3: Implement `prep.py`.** A capture is a move whose destination holds a piece of the other colour, or a pawn move that changes file onto an empty square. Run — Expected: PASS.
- [ ] **Step 4: Write the failing ops tests** (`LocalStore`, a fake runner that writes shard files and yields `SHARD <path>` lines)

```python
def test_source_tag_comes_from_last_modified():
    assert source_tag("u", fetch=fake_head("Sun, 01 Nov 2026 17:23:27 GMT")) == "lichess-eval-20261101"

def test_shards_are_uploaded_under_the_tag_and_local_copies_removed(tmp_path): ...
def test_a_second_run_skips_what_the_store_already_has(tmp_path):
    # first run uploads 2 shards; the second, asked for 3, passes --skip 2 and uploads 1
def test_trainer_cmd_uses_the_named_python_when_set():
    assert trainer_cmd({"VEKTR_TRAINER_PYTHON": "python3"}, "m", "--x")[:3] == ["python3", "-m", "m"]
    assert trainer_cmd({}, "m")[:4] == ["uv", "run", "--project", "trainer"]
```

- [ ] **Step 5: Run** `uv run --project ops pytest ops/tests/test_data.py` — Expected: FAIL. **Implement** `data.py`, `training.trainer_cmd` and the `prep` command. Run — Expected: PASS.
- [ ] **Step 6: Try it for real, small.** Run: `VEKTR_STORE=/tmp/vektr-prep-try uv run --project ops vektr-ops prep --positions 1000000` — Expected: one shard announced within a few minutes; `ls /tmp/vektr-prep-try/data/bootstrap/*/` shows `shard-00000.npz`. Note its size and the lines-per-second rate in the ledger.
- [ ] **Step 7: Commit** `feat: training data from the Lichess evaluation database`.

### Task 4: The model and the training loop

**Files:**
- Create: `trainer/src/vektr_trainer/model.py`, `trainer/src/vektr_trainer/train.py`
- Test: `trainer/tests/test_model.py`, `trainer/tests/test_train.py`

**Interfaces:**
- Consumes: `features`, `vknn`, `prep.read_shard`.
- Produces:
  - `model.Net()` — `ft = nn.EmbeddingBag(769, 128, mode="sum", padding_idx=768)`, `ft_bias` (128), `out = nn.Linear(256, 1)`; `forward(features: LongTensor[N, 32], stm: Tensor[N]) -> Tensor[N]`, the score for the side to move in units of 400 centipawns; `clip_() -> None` clamps `ft`, `ft_bias` and `out.weight` to ±1.98
  - `model.quantise(net: Net) -> vknn.QNet`
  - `train.Config(minutes: float, batch: int = 16384, lr: float = 1e-3, lr_end: float = 1e-5, score_weight: float = 0.75, holdout: float = 0.01, checkpoint_s: float = 600, seed: int = 1)`
  - `train.pick_device() -> str`
  - `train.targets(score: Tensor, result: Tensor, score_weight: float) -> Tensor`
  - `train.train(shards: Path, out: Path, cfg: Config, *, device: str | None = None, clock=time.monotonic) -> dict` — loads every `*.npz` in `shards` onto the device; resumes from `out/checkpoint.pt` when present (model, optimiser, steps, elapsed seconds); writes `out/checkpoint.pt` by temp file and rename every `checkpoint_s` and at the end; then `out/net.vknn`, `out/net.check`, `out/run.json` (`net`, `positions`, `steps`, `epochs`, `elapsed_s`, `train_loss`, `val_loss`, `device`, `config`); returns the `run.json` contents
  - `python -m vektr_trainer.train --shards DIR --out DIR --minutes M [--checkpoint-s S]`

- [ ] **Step 1: Write the failing model tests**

```python
def test_float_model_and_integer_evaluation_are_the_same_function():
    torch.manual_seed(0); net = Net(); net.clip_(); q = quantise(net)
    for fen in check_fens()[:64]:
        idx, stm = fen_features(fen)
        x = torch.tensor([idx + [EMPTY] * (32 - len(idx))]); s = torch.tensor([stm])
        assert abs(net(x, s).item() * 400 - evaluate(q, fen)) <= 8

def test_quantised_values_are_inside_the_valid_ranges():
    net = Net(); [p.data.mul_(100) for p in net.parameters()]; net.clip_(); q = quantise(net)
    assert abs(q.ft_w).max() <= 505 and abs(q.ft_b).max() <= 505 and abs(q.out_w).max() <= 127

def test_targets_blend_score_and_result_and_ignore_unknown_results():
    t = targets(torch.tensor([0.0, 400.0, 400.0]), torch.tensor([255, 2, 0]), 0.75)
    assert t.tolist() == pytest.approx([0.5, 0.75 * 0.7311 + 0.25, 0.75 * 0.7311], abs=1e-3)
```

- [ ] **Step 2: Run** `uv run --project trainer pytest trainer/tests/test_model.py` — Expected: FAIL. **Implement `model.py` and `targets`.** Run — Expected: PASS.
- [ ] **Step 3: Write the failing training tests.** A helper writes one shard of 8,192 positions made from `check_fens()` repeated, scored by material count (pawn 100, knight 300, bishop 300, rook 500, queen 900, for the side to move). A fake clock advances one second per call so runs are short and deterministic.

```python
def test_training_lowers_the_validation_loss(tmp_path):
    first = train(shards, tmp_path / "a", Config(minutes=0.2, batch=512, checkpoint_s=5), device="cpu", clock=fake_clock())
    longer = train(shards, tmp_path / "b", Config(minutes=2.0, batch=512, checkpoint_s=5), device="cpu", clock=fake_clock())
    assert longer["val_loss"] < first["val_loss"]

def test_outputs_are_a_valid_network_with_a_matching_check_file(tmp_path):
    info = train(...); net = vknn.read(out / "net.vknn")
    assert info["net"] == vknn.net_hash(out / "net.vknn")
    assert all(vknn.evaluate(net, fen) == cp for fen, cp in vknn.read_check(out / "net.check"))

def test_a_second_run_resumes_from_the_checkpoint(tmp_path):
    a = train(shards, out, Config(minutes=0.5, batch=512), device="cpu", clock=fake_clock())
    b = train(shards, out, Config(minutes=1.0, batch=512), device="cpu", clock=fake_clock())
    assert b["steps"] > a["steps"] and b["elapsed_s"] >= 60
    c = train(shards, out, Config(minutes=1.0, batch=512), device="cpu", clock=fake_clock())
    assert c["steps"] == b["steps"]                      # budget already spent: no further steps

def test_a_damaged_checkpoint_is_an_error_not_a_silent_restart(tmp_path):
    (out / "checkpoint.pt").write_bytes(b"half"); 
    with pytest.raises(Exception): train(...)

def test_no_temporary_checkpoint_is_left_behind(tmp_path): ...   # only checkpoint.pt, net.vknn, net.check, run.json

def test_no_shards_is_an_error(tmp_path): ...                    # ValueError
```

- [ ] **Step 4: Run** `uv run --project trainer pytest trainer/tests/test_train.py` — Expected: FAIL. **Implement `train.py`.** Loss is the mean of `(sigmoid(output) − target)²`; the learning rate follows a cosine from `lr` to `lr_end` over `elapsed / (minutes · 60)`; each epoch draws a fresh permutation on the device; `clip_()` after every step; the last `holdout` share of positions is never trained on. Run — Expected: PASS.
- [ ] **Step 5: Run both suites.** `uv run --project trainer pytest trainer/tests` — Expected: all PASS.
- [ ] **Step 6: Commit** `feat(trainer): model, training loop, checkpoints and resume`.

### Task 5: Training and installing a network through the store

**Files:**
- Modify: `ops/src/vektr_ops/training.py`, `ops/src/vektr_ops/cli.py` (commands `train`, `install-net`)
- Test: `ops/tests/test_training.py`

**Interfaces:**
- Consumes: `Store`, `trainer_cmd`, the trainer's output files.
- Produces:
  - `training.pull_shards(store: Store, dest: Path, limit: int | None = None) -> list[str]` — store paths under `data:bootstrap/`, sorted, the first `limit`; files already in `dest` are not fetched again
  - `training.run_training(store: Store, root: Path, run: str, *, minutes: float, shards: int | None = None, env: dict, start=subprocess.Popen, poll_s: float = 15) -> dict` — work directory `state/train/<run>/`; restores `nets:runs/<run>/checkpoint.pt` if the store has one; starts the trainer; uploads `checkpoint.pt` whenever its modification time changes; after a zero exit uploads `checkpoint.pt`, `net.vknn`, `net.check` and `run.json` (with the shard paths added under `shards`) to `nets:runs/<run>/`, and `nets:nets/<hash>.vknn`; returns the run record; a non-zero exit uploads the latest checkpoint and raises `RuntimeError` with the tail of the trainer's output
  - `training.install_net(store: Store, root: Path, run: str) -> str` — writes `engine/nets/default.vknn` and `default.check` from the run; returns the network hash; `KeyError` if the run has no network
  - `vektr-ops train --run ID --minutes M [--shards N]`, `vektr-ops install-net RUN` (builds the engine and runs `vektr netcheck`; exits 1 with the mismatch lines if it fails)

- [ ] **Step 1: Write the failing tests** (`LocalStore`; `start` is a fake process whose `poll()` script writes files into the work directory)

```python
def test_shards_are_pulled_once_and_limited(tmp_path): ...
def test_results_are_uploaded_after_a_clean_run(tmp_path):
    # nets:runs/r1/{checkpoint.pt,net.vknn,net.check,run.json} and nets:nets/<hash>.vknn exist;
    # run.json lists the shard paths
def test_a_changed_checkpoint_is_uploaded_while_training_runs(tmp_path): ...
def test_an_existing_checkpoint_is_restored_before_training_starts(tmp_path): ...
def test_a_failed_run_keeps_its_checkpoint_and_raises(tmp_path): ...
def test_install_net_writes_both_files_and_returns_the_hash(tmp_path): ...
def test_install_net_for_an_unknown_run_is_an_error(tmp_path): ...
```

- [ ] **Step 2: Run** `uv run --project ops pytest ops/tests/test_training.py` — Expected: FAIL. **Implement.** Run — Expected: PASS.
- [ ] **Step 3: Run the whole ops suite.** `uv run --project ops pytest ops/tests` — Expected: all PASS.
- [ ] **Step 4: Commit** `feat(ops): train and install-net commands`.

### Task 6: The engine plays with the network

**Files:**
- Modify: `engine/src/search.rs`, `engine/src/uci.rs`, `engine/src/lib.rs`, `engine/nets/default.vknn`, `engine/nets/default.check`, `docs/bench.md`
- Delete: `engine/src/eval.rs`, `engine/tests/eval.rs`
- Test: `engine/tests/nnue.rs`, `engine/tests/uci.rs`, `engine/tests/search.rs`

**Interfaces:**
- Consumes: Task 2's `nnue`; Task 5's `train` and `install-net`.
- Produces:
  - `Searcher::set_network(&mut self, net: Arc<Network>)`; `Searcher::new(hash_mb)` starts with `Network::embedded()`
  - The searcher keeps one accumulator per ply. The root is refreshed at the start of `search`; after every `pos.play(mv)` in `negamax` and `quiescence` the child's accumulator is updated from the parent's; static evaluation is `net.evaluate(&accs[ply], side_to_move)` clamped to ±20000
  - UCI option `EvalFile` (`option name EvalFile type string default <embedded>`); a network that fails to load prints `info string error: <reason>` and leaves the current one in place; changing `Hash` keeps the loaded network

- [ ] **Step 1: Train the bootstrap network on the laptop.** Push the branch first (`git push -u origin stage3-first-network`). Then, with a scratch store so nothing reaches Hugging Face yet:

```bash
export VEKTR_STORE=/tmp/vektr-bootstrap
uv run --project ops vektr-ops prep --positions 3000000
uv run --project ops vektr-ops train --run bootstrap-cpu --minutes 15
uv run --project ops vektr-ops install-net bootstrap-cpu
```

Expected: `install-net` ends with `netcheck ok 256 positions`. Record the validation loss and the network hash in the ledger.

- [ ] **Step 2: Write the failing tests**

```rust
// tests/nnue.rs
#[test] fn the_embedded_network_knows_material() {
    let net = Network::embedded();
    assert!(net.evaluate_board(&b("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")) > 300);
    assert!(net.evaluate_board(&Board::default()).abs() < 150);
}
// tests/uci.rs
#[test] fn evalfile_is_advertised() { /* the uci reply has "option name EvalFile type string" */ }
#[test] fn a_missing_evalfile_is_reported_and_the_engine_still_moves() {
    // setoption name EvalFile value /no/such/file -> a line starting "info string error";
    // go depth 2 -> a bestmove that is not 0000
}
#[test] fn a_truncated_evalfile_is_reported_and_the_engine_still_moves() { /* first 1000 bytes of DEFAULT_NET */ }
#[test] fn a_loaded_evalfile_changes_the_score() {
    // DEFAULT_NET with out_b raised by 255*64 (the last four bytes), written to a temp file:
    // "go depth 1" reports a score about 400 higher than with the embedded network
}
#[test] fn the_loaded_network_survives_a_hash_change() { /* EvalFile, then Hash 8: the raised score remains */ }
// tests/search.rs
#[test] fn a_position_scores_the_same_however_it_was_reached() {
    // the position after 1.e4 e5 2.Nf3 Nc6, built by playing the moves and built from its FEN,
    // searched to depth 5 with a fresh Searcher each: same score and same node count
}
```

- [ ] **Step 3: Run** `(cd engine && cargo test --release)` — Expected: the new tests FAIL (no `EvalFile`; search still uses `eval::evaluate`).
- [ ] **Step 4: Switch the search to the network and add `EvalFile`.** Delete `eval.rs`, `tests/eval.rs` and the `pub mod eval;` line.
- [ ] **Step 5: Run** `(cd engine && cargo test --release)` — Expected: all PASS. The Stage 1 search tests stay as they are. If one that depends on evaluation quality fails with this small network (`finds_the_fine_70_key_move`, `reaches_depth_7_from_startpos_in_under_one_million_nodes`, `quiescence_does_not_grab_a_defended_pawn`, `wins_a_hanging_queen`), first rule out an inference bug with `vektr netcheck`; only then adjust that test's bound to the measured value with headroom, and ledger a ruling naming the test and both numbers.
- [ ] **Step 6: Record the bench.** Run `(cd engine && cargo run --release -- bench 8)` twice — Expected: the same node count both times, different from 38,791,222. Add a row to `docs/bench.md` with the node count and laptop speed, marked "bootstrap network, not gated".
- [ ] **Step 7: Sanity-check playing strength before any money is spent.** Commit and push, then with the scratch store: `vektr-ops promote be104b19b30e --bootstrap`, `vektr-ops gate HEAD --max-batches 3`. Expected: 300 games complete with no discarded batch. Record the Elo estimate in the ledger; it is information, not a pass mark. A result below −300 means something is wrong with inference or data, and is investigated before going on.
- [ ] **Step 8: Commit** `feat(engine): evaluate with the network; EvalFile option`.

### Task 7: The gate and the box understand networks

**Files:**
- Modify: `ops/src/vektr_ops/builds.py`, `manifest.py`, `gate.py`, `sync.py`, `cli.py`, `page.py`
- Test: `ops/tests/test_builds.py`, `test_manifest.py`, `test_gate.py`, `test_sync.py`, `test_cli.py`, `test_page.py`

**Interfaces:**
- Produces:
  - `builds.netcheck(binary: Path) -> bool` — runs `<binary> netcheck` with no input and a 60 s limit; true only for exit status 0 and output starting `netcheck ok`
  - `builds.net_hash(root: Path, commit: str) -> str | None` — first 16 hex of the SHA-256 of `engine/nets/default.vknn` at that commit (`git show`), `None` when the commit has no such file
  - `manifest.add_candidate(m, engine_commit, now, net: str | None = None) -> Version`
  - `gate.speed_ratio(store: Store, test: str) -> float | None` — the median of `meta["speed_ratio"]` over the counted batches of a test
  - each batch's `meta` gains `speed_ratio`: candidate bench speed ÷ champion bench speed on the machine that played, to 3 places
  - `gate.settled` adds `speed_ratio` to the verdict it writes when the test has one
  - `sync.follow(..., netcheck=builds.netcheck)` — a champion with a `net` must pass `netcheck` on this machine or the sync is refused; the speed rule becomes `speed < (1 − SPEED_MARGIN) · old_speed · ratio`, where `ratio` is `champion.gate["speed_ratio"]` when that gate was played against the build now in place, and 1.0 otherwise
  - `vektr-ops gate <ref>`: for a commit that has a network, builds it and refuses to register it (exit 2, `error: engine and trainer disagree on the network`) unless `netcheck` passes
  - the page's version table shows the network hash

- [ ] **Step 1: Write the failing tests**

```python
# test_builds.py
def test_netcheck_is_true_only_for_the_ok_line(tmp_path):
    assert netcheck(script(tmp_path, 'echo "netcheck ok 256 positions"')) is True
    assert netcheck(script(tmp_path, 'echo "mismatch x"; exit 1')) is False
    assert netcheck(script(tmp_path, 'cat >/dev/null')) is False      # an engine without the command
def test_net_hash_reads_the_file_at_that_commit(git_repo): ...       # None before the file exists
# test_manifest.py
def test_add_candidate_records_the_network(): ...
# test_gate.py
def test_batches_record_the_speed_ratio(...): ...                     # candidate 1.4M, champion 2.0M -> 0.7
def test_the_settled_verdict_carries_the_median_ratio(...): ...       # batches 0.68, 0.70, 0.90 -> 0.70
def test_a_test_with_no_recorded_ratio_settles_without_one(...): ...
# test_sync.py
def test_a_slower_network_build_is_accepted_when_the_gate_expected_it(...):
    # old 2.9M, new 2.0M, gate {"decision": "accept", "champion": old_id, "speed_ratio": 0.7}
def test_a_build_far_below_the_expected_ratio_is_refused(...):       # new 1.5M with ratio 0.7
def test_the_ratio_is_ignored_when_the_gate_was_against_another_build(...): ...
def test_a_champion_with_a_network_must_pass_netcheck(...):          # SyncRefused, symlink untouched
def test_rollback_to_a_champion_without_a_network_needs_no_netcheck(...): ...
# test_cli.py
def test_gate_refuses_a_candidate_that_fails_netcheck(...): ...       # exit 2, manifest unchanged
def test_gate_registers_the_network_hash(...): ...
```

- [ ] **Step 2: Run** `uv run --project ops pytest ops/tests` — Expected: the new tests FAIL. **Implement.** Run — Expected: all PASS.
- [ ] **Step 3: Commit** `feat(ops): gate and box checks for network versions`.

### Task 8: Burst instances

**Files:**
- Create: `ops/src/vektr_ops/runpod.py`, `ops/src/vektr_ops/burst.py`, `deploy/burst.sh`
- Modify: `ops/src/vektr_ops/cli.py` (command `burst`), `.env.example`, `README.md`, `docs/runbook.md`
- Test: `ops/tests/test_runpod.py`, `ops/tests/test_burst.py`

**Interfaces:**
- Produces:
  - `runpod.API = "https://api.runpod.io/v2"`; every call sends `Authorization: Bearer <key>` and takes `send=urllib.request.urlopen`
  - `runpod.create_pod(key: str, body: dict, send=...) -> dict` (the pod; `RunpodError(status, message)` otherwise), `runpod.get_pod(key, id, send=...) -> dict | None` (`None` for 404), `runpod.delete_pod(key, id, send=...) -> None` (204 and 404 both mean gone), `runpod.list_pods(key, send=...) -> list[dict]`
  - `burst.IMAGE = "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404"`, `burst.GPUS = ["NVIDIA RTX A4000", "NVIDIA GeForce RTX 3090", "NVIDIA RTX A5000", "NVIDIA GeForce RTX 4090"]`, `burst.MAX_RATE = 0.60`, `burst.SETUP_S = 1200`, `burst.NAME_PREFIX = "vektr64-"`
  - `burst.start_cmd(commit: str, hard_s: int) -> list[str]` — `["bash", "-c", script]`, where the script's first statement starts the watchdog `(sleep <hard_s>; runpodctl pod delete "$RUNPOD_POD_ID") &`, the second sets `trap 'runpodctl pod delete "$RUNPOD_POD_ID"' EXIT`, and only then fetches and runs `deploy/burst.sh` from `https://raw.githubusercontent.com/irom77/Vektr64/<commit>/deploy/burst.sh`
  - `burst.pod_body(run: str, commit: str, minutes: float, hf_token: str, gpu: str) -> dict` — `name`, `image`, `gpu: {"id": gpu, "count": 1}`, `cloud: "COMMUNITY"`, `disk: 20`, `env` (`HF_TOKEN`, `VEKTR_RUN`, `VEKTR_MINUTES`, `VEKTR_COMMIT`), `cmd`
  - `burst.launch(key: str, bodies: list[dict], *, max_rate: float = MAX_RATE, api=runpod) -> dict` — tries each body in turn; status 400 or 403 moves on to the next; a pod whose `cost` exceeds `max_rate` is deleted and the next is tried; raises `BurstFailed` when none is left
  - `burst.watch(key: str, pod_id: str, hard_s: int, *, api=runpod, sleep=time.sleep, clock=time.monotonic, poll_s: int = 30) -> str` — returns `"gone"` when the pod no longer exists; deletes it and returns `"deleted: <why>"` when its status is `EXITED` or `ERROR`, or when `hard_s + 300` seconds pass
  - `burst.kill_all(key: str, *, api=runpod) -> list[str]` — deletes every pod whose name starts with `NAME_PREFIX`; returns their ids
  - `vektr-ops burst --minutes 45 [--run ID] [--max-rate 0.60] [--yes]`, `vektr-ops burst --list`, `vektr-ops burst --kill`, `vektr-ops burst --local --minutes M` (runs `deploy/burst.sh` on this machine against the configured store, with no RunPod call)
  - `deploy/burst.sh` — with `VEKTR_LOCAL=1` it runs in the current checkout and uses `uv`; otherwise it clones the repository at `VEKTR_COMMIT`, installs `ops` and the trainer's non-PyTorch dependencies into the image's Python, sets `VEKTR_TRAINER_PYTHON=python3`; either way it ends with `vektr-ops train --run "$VEKTR_RUN" --minutes "$VEKTR_MINUTES"`

- [ ] **Step 1: Write the failing tests** (a fake `send` for `runpod`, a fake `api` object for `burst`)

```python
# test_runpod.py
def test_create_posts_json_with_a_bearer_header(): ...
def test_create_error_carries_status_and_message(): ...               # 400 -> RunpodError(400, ...)
def test_get_of_a_deleted_pod_is_none(): ...
def test_delete_treats_404_as_gone(): ...
# test_burst.py
def test_the_watchdog_and_trap_come_before_anything_else():
    script = start_cmd("abc123", 3900)[2]
    assert script.index("sleep 3900") < script.index("trap") < script.index("curl")
def test_the_token_is_only_in_env():
    body = pod_body("r1", "abc", 45, "hf_SECRET", GPUS[0])
    assert body["env"]["HF_TOKEN"] == "hf_SECRET" and "hf_SECRET" not in " ".join(body["cmd"]) + body["name"]
def test_the_next_gpu_is_tried_when_one_has_no_capacity(): ...        # first create raises 400
def test_a_rate_above_the_cap_is_deleted_and_not_used(): ...          # cost 0.90 -> delete called, BurstFailed
def test_insufficient_balance_stops_at_once(): ...                    # 402 -> BurstFailed, no further create
def test_watch_returns_when_the_pod_is_gone(): ...
def test_watch_deletes_a_pod_that_exited_or_errored(): ...
def test_watch_deletes_a_pod_that_outlives_the_hard_limit(): ...
def test_kill_all_touches_only_our_pods(): ...                        # "vektr64-r1" deleted, "other" kept
def test_burst_command_never_prints_a_token(cli_runner): ...          # neither HF_TOKEN nor RUNPOD_API_KEY in output
def test_burst_without_a_runpod_key_explains_what_to_set(cli_runner): ...   # exit 2
def test_burst_refuses_an_unpushed_commit(cli_runner): ...
@pytest.mark.slow
def test_local_dry_run_trains_and_uploads(tmp_path):
    # VEKTR_STORE with one 8,192-position shard; `vektr-ops burst --local --minutes 0.1 --run dry`
    # -> nets:runs/dry/net.vknn and run.json exist
```

- [ ] **Step 2: Run** `uv run --project ops pytest ops/tests/test_runpod.py ops/tests/test_burst.py` — Expected: FAIL. **Implement** `runpod.py`, `burst.py`, the `burst` command (it prints the GPU, the hourly rate and `worst case $<rate · hard_s / 3600>`, then asks to continue unless `--yes`; with no confirmation it deletes the pod it created) and `deploy/burst.sh`. Add `markers = ["slow"]` and `addopts = "-m 'not slow'"` to the ops pytest settings.
- [ ] **Step 3: Run** `uv run --project ops pytest ops/tests` — Expected: all PASS. Then `uv run --project ops pytest ops/tests -m slow` — Expected: the dry run PASSES (this is the design's "full burst dry run on CPU with a tiny dataset").
- [ ] **Step 4: Documentation.** `.env.example`: add `RUNPOD_API_KEY=` with where to create it (scope: pods read and write). `README.md`: add the Stage 3 plan link, the `trainer/` row, the trainer test command and the new commands. `docs/runbook.md`: a "Training a network" section — prepare data, start a burst, what the launcher prints, how to see and kill pods (`burst --list`, `burst --kill`), install the network, commit it on a branch, gate it, and what to do when a burst dies half-way (start it again with the same `--run`).
- [ ] **Step 5: Run everything.** `(cd engine && cargo test --release) && uv run --project trainer pytest trainer/tests && uv run --project ops pytest ops/tests` — Expected: all PASS.
- [ ] **Step 6: Commit** `feat(ops): rented GPU bursts that terminate themselves`.

---

## After the code is merged: the first burst and the first gated promotion

These steps spend money and change what the bot plays, so each needs the owner. They are run by hand with the commands above; no new code.

1. **Pull request 1** (Tasks 1–8) is reviewed and merged by the owner. On the box: `git pull`, then restart `vektr64-worker`. The bot keeps playing the Stage 1 champion.
2. **Owner:** create a RunPod account with a few dollars of credit, create an API key, and put `RUNPOD_API_KEY=` in the laptop's `.env`.
3. **Data:** `vektr-ops prep --positions 30000000` on the laptop (roughly 4 GB streamed, 30 shards uploaded to the private data repo).
4. **Proof that a pod removes itself:** `vektr-ops burst --minutes 1 --run smoke-<date>`. Expected: the launcher reports the pod gone within about 15 minutes and `vektr-ops burst --list` shows none. If the pod has to be deleted by the launcher instead of by itself, stop and fix that before the real run.
5. **The real burst:** `vektr-ops burst --minutes 45 --run first-<date>`. Afterwards `vektr-ops burst --list` must show no pods.
6. **Candidate:** on a new branch `net-first`, `vektr-ops install-net first-<date>`, run the engine tests, record the bench node count, commit the two network files and the `docs/bench.md` row, push.
7. **Gate:** `vektr-ops gate net-first`. The laptop and the box worker both contribute games. On `accept` the manifest promotes it, the box builds it, checks it and switches at its next sync; on `reject` the bot keeps the Stage 1 engine and the result is recorded.
8. **Pull request 2** (the network files) is opened with the gate result and merged by the owner. The anchor matches for the new champion start by themselves.
