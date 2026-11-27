# Stage 1 — Engine Plays on Lichess: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An original UCI chess engine (`vektr`) with a hand-written evaluation plays rated blitz on Lichess as Vektr64, unattended, from the always-on box.

**Architecture:** A Rust crate (`engine/`) holds search, evaluation, time management and the UCI loop on top of the `cozy-chess` move generator. A small Python CLI (`ops/`, `vektr-ops`) sizes the engine from the machine it is on, renders the `lichess-bot` config and launches the pinned `lichess-bot` bridge. A systemd unit keeps it running on the box.

**Tech Stack:** Rust 1.98 (edition 2021), `cozy-chess` 0.3.4, Python 3.12 managed by `uv`, `typer`, `psutil`, `pyyaml`, `pytest`, `lichess-bot` pinned at commit `a31b965a8eda79f3a8ec76b49ef9349fbf6c63bd`, systemd.

**Spec:** `docs/superpowers/specs/2026-11-27-vektr64-design.md` — this plan implements "Build order" stage 1 only. Stages 2–5 each get their own plan.

## Global Constraints

- Everything above move generation is original. No code or network from another engine.
- Single-threaded search. No variants, Chess960, tablebases or pondering.
- No pruning beyond alpha-beta, quiescence and move ordering in this stage. Null move, late-move reductions, aspiration windows and SEE wait for the stage 2 gate.
- No machine-specific config edits. Cores, available memory and CPU features are detected at run time.
- Secrets come from the environment or a git-ignored `.env`: `LICHESS_TOKEN` (stage 1 needs only this one). Never commit or log a token.
- Rated play: standard chess, blitz only, 3+2 and 5+3, against bots, one game at a time.
- Play machine: `irom-sei`, `ssh irom@100.114.82.19`, Ubuntu 24.04 x86-64, 4 cores / 8 threads, AVX2, about 2 GB RAM available. Rust and `uv` are not installed there yet.
- **The BOT account upgrade is irreversible. Task 8 stops and asks the owner before running it.**
- Work on a branch `stage1-engine-plays`, not `main`. Commit after every task.
- Scores are centipawns from the side to move. `MATE = 30000`; a mate in `n` plies scores `MATE - n`. `INF = 32000`.

## Review Focus

1. **Castling and promotion moves crossing the UCI boundary.** `cozy-chess` encodes castling as king-takes-rook (`e1h1`); Lichess sends and expects `e1g1`. Both directions must convert, and promotions (`e7e8q`) must round-trip. → Task 1 tests.
2. **`go` with almost no time, zero time or a negative clock.** Lichess can send `wtime` of a few milliseconds or below zero under lag. The engine must still print a legal `bestmove`. → Task 5 and Task 6 tests.
3. **Draw by repetition across the game history.** The moves in `position ... moves ...` are part of the repetition history; a winning engine must not walk into a threefold. → Task 3 tests.
4. **stdin closes or `quit` arrives mid-search.** The bridge kills engines; `vektr` must exit promptly and not hang or spin. → Task 6 tests.
5. **Mate scores stored in the transposition table.** A mate distance stored at one depth and read at another must be re-based on ply, or the engine reports wrong mates and shuffles. → Task 4 tests.

---

## File Structure

| File | Responsibility |
|---|---|
| `engine/Cargo.toml` | Crate `vektr`, lib + bin, release profile |
| `engine/src/lib.rs` | Module declarations, score constants |
| `engine/src/position.rs` | `Position`: board plus hash history; UCI move conversion; perft |
| `engine/src/eval.rs` | Hand-written tapered material + piece-square evaluation |
| `engine/src/search.rs` | Iterative deepening, alpha-beta, quiescence, limits, stop flag |
| `engine/src/tt.rs` | Transposition table |
| `engine/src/ordering.rs` | Move scoring: TT move, captures, killers, history |
| `engine/src/time.rs` | Soft and hard time limits from the clock |
| `engine/src/uci.rs` | UCI command loop and `go` parsing |
| `engine/src/bench.rs` | Fixed-depth benchmark over fixed positions |
| `engine/src/main.rs` | Argument dispatch: UCI loop or `bench` |
| `engine/tests/*.rs` | One integration test file per module |
| `ops/pyproject.toml` | `vektr-ops` package and entry point |
| `ops/src/vektr_ops/sizing.py` | Hash size from available memory |
| `ops/src/vektr_ops/botconfig.py` | Render `lichess-bot` config |
| `ops/src/vektr_ops/cli.py` | `vektr-ops play`, `vektr-ops status` |
| `ops/tests/*.py` | Unit tests |
| `bot/lichess-bot/` | Git submodule, pinned |
| `bot/config.template.yml` | Bridge settings with placeholders |
| `deploy/vektr64.service` | systemd unit |
| `deploy/install-box.sh` | One-time box setup |
| `.gitignore`, `.env.example` | Ignore rules and the token variable name |

---

### Task 1: Position, UCI moves and perft

**Files:**
- Create: `engine/Cargo.toml`, `engine/src/lib.rs`, `engine/src/position.rs`, `engine/src/main.rs` (empty `fn main() {}` for now), `.gitignore`
- Test: `engine/tests/position.rs`

**Interfaces:**
- Produces:
  - `pub const MATE: i32 = 30000; pub const INF: i32 = 32000; pub const MAX_PLY: usize = 128;` in `lib.rs`
  - `pub struct Position { pub board: cozy_chess::Board, /* hash history */ }`
  - `Position::startpos() -> Position`
  - `Position::from_fen(fen: &str) -> Result<Position, String>`
  - `Position::play_uci(&mut self, mv: &str) -> Result<(), String>` — rejects illegal moves
  - `Position::play(&mut self, mv: cozy_chess::Move)` — pushes the old hash to history
  - `Position::undo_to(&mut self, board: cozy_chess::Board)` — pops one history entry and restores the board (search keeps the previous board by value)
  - `Position::is_draw(&self, ply: usize) -> bool` — 50-move rule, insufficient material (bare kings, or one minor piece), or the current hash already appearing in history: once if that occurrence is within the last `ply` moves (inside the search), twice otherwise
  - `Position::move_to_uci(&self, mv: cozy_chess::Move) -> String`
  - `Position::legal_moves(&self) -> Vec<cozy_chess::Move>`
  - `pub fn perft(board: &cozy_chess::Board, depth: u32) -> u64`

- [ ] **Step 1: Create the crate.** `Cargo.toml`: package `vektr`, edition 2021, `[lib] name = "vektr"`, `[[bin]] name = "vektr"`, dependency `cozy-chess = "0.3.4"`, `[profile.release] lto = true, codegen-units = 1`. `.gitignore`: `target/`, `.env`, `.venv/`, `__pycache__/`, `bot/config.yml`. Read the `cozy-chess` 0.3.4 docs (`cargo doc -p cozy-chess --open` or docs.rs) for `Board`, `generate_moves`, `play_unchecked`, `hash`, `status`, `halfmove_clock` and `cozy_chess::util::{parse_uci_move, display_uci_move}` before writing code.

- [ ] **Step 2: Write the failing tests** in `engine/tests/position.rs`:

```rust
use vektr::position::{perft, Position};

const KIWIPETE: &str = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";

#[test] fn perft_startpos() {
    let p = Position::startpos();
    assert_eq!(perft(&p.board, 1), 20);
    assert_eq!(perft(&p.board, 4), 197_281);
    assert_eq!(perft(&p.board, 5), 4_865_609);
}
#[test] fn perft_kiwipete() {
    let p = Position::from_fen(KIWIPETE).unwrap();
    assert_eq!(perft(&p.board, 3), 97_862);
    assert_eq!(perft(&p.board, 4), 4_085_603);
}
#[test] fn castling_uses_standard_uci_both_ways() {
    let mut p = Position::from_fen(KIWIPETE).unwrap();
    let ucis: Vec<String> = p.legal_moves().iter().map(|m| p.move_to_uci(*m)).collect();
    assert!(ucis.contains(&"e1g1".to_string()));
    assert!(ucis.contains(&"e1c1".to_string()));
    assert!(!ucis.contains(&"e1h1".to_string()));
    p.play_uci("e1g1").unwrap();
    assert_eq!(p.board.to_string().split(' ').next().unwrap(),
               "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R4RK1");
}
#[test] fn promotion_round_trips() {
    let mut p = Position::from_fen("8/P6k/8/8/8/8/8/K7 w - - 0 1").unwrap();
    let ucis: Vec<String> = p.legal_moves().iter().map(|m| p.move_to_uci(*m)).collect();
    assert!(ucis.contains(&"a7a8q".to_string()) && ucis.contains(&"a7a8n".to_string()));
    p.play_uci("a7a8q").unwrap();
}
#[test] fn illegal_and_garbage_moves_are_rejected() {
    let mut p = Position::startpos();
    assert!(p.play_uci("e2e5").is_err());
    assert!(p.play_uci("zz").is_err());
    assert!(Position::from_fen("not a fen").is_err());
}
#[test] fn repetition_in_game_history_needs_three() {
    let mut p = Position::startpos();
    for m in ["g1f3", "g8f6", "f3g1", "f6g8"] { p.play_uci(m).unwrap(); }
    assert!(!p.is_draw(0));              // position seen twice in the game
    for m in ["g1f3", "g8f6", "f3g1", "f6g8"] { p.play_uci(m).unwrap(); }
    assert!(p.is_draw(0));               // third occurrence
}
#[test] fn repetition_inside_search_needs_two() {
    let mut p = Position::startpos();
    for m in ["g1f3", "g8f6", "f3g1", "f6g8"] { p.play_uci(m).unwrap(); }
    assert!(p.is_draw(4));               // all four moves were made by the search
}
#[test] fn fifty_move_rule_and_bare_kings() {
    assert!(Position::from_fen("8/8/8/4k3/8/8/4K2R/8 w - - 100 80").unwrap().is_draw(0));
    assert!(Position::from_fen("8/8/8/4k3/8/8/4K3/8 w - - 0 1").unwrap().is_draw(0));
    assert!(Position::from_fen("8/8/8/4k3/8/8/4K1N1/8 w - - 0 1").unwrap().is_draw(0));
}
```

- [ ] **Step 3: Run** `cd engine && cargo test --release --test position` — expected: compile failure, `position` module not found.
- [ ] **Step 4: Implement `position.rs`** to the Interfaces above. `perft` uses bulk counting at depth 1.
- [ ] **Step 5: Run** `cargo test --release --test position` — expected: 8 passed.
- [ ] **Step 6: Commit** `feat(engine): position, UCI move conversion, perft`.

---

### Task 2: Hand-written evaluation

**Files:**
- Create: `engine/src/eval.rs`
- Test: `engine/tests/eval.rs`

**Interfaces:**
- Consumes: `cozy_chess::Board`
- Produces: `pub fn evaluate(board: &cozy_chess::Board) -> i32` — centipawns from the side to move, always within `±10_000`.

Design: tapered evaluation. Middlegame and endgame material values and piece-square tables, blended by game phase (knight and bishop 1, rook 2, queen 4, total 24). Write the tables yourself: material `P 100, N 320, B 330, R 500, Q 900` (middlegame), `P 120, N 310, B 330, R 530, Q 950` (endgame); piece-square bonuses in the range −50..+50 expressing centralization, pawn advancement, king safety on the back rank in the middlegame and king centralization in the endgame. Black uses the vertically mirrored tables. No tempo bonus.

- [ ] **Step 1: Write the failing tests:**

```rust
use cozy_chess::Board;
use vektr::eval::evaluate;
fn b(fen: &str) -> Board { Board::from_fen(fen, false).unwrap() }

#[test] fn startpos_is_balanced() { assert_eq!(evaluate(&Board::default()), 0); }
#[test] fn score_is_from_side_to_move() {
    let w = b("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    let bl = b("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1");
    assert!(evaluate(&w) > 800);
    assert_eq!(evaluate(&w), -evaluate(&bl));
}
#[test] fn colour_mirrored_positions_score_the_same() {
    // same position with colours and ranks flipped, same side-to-move role
    let a = b("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3");
    let m = b("rnbqkb1r/pppp1ppp/5n2/4p3/4P3/2N5/PPPP1PPP/R1BQKBNR b KQkq - 2 3");
    assert_eq!(evaluate(&a), evaluate(&m));
}
#[test] fn centralized_knight_beats_rim_knight() {
    assert!(evaluate(&b("4k3/8/8/8/3N4/8/8/4K3 w - - 0 1"))
          > evaluate(&b("4k3/8/8/8/8/8/8/N3K3 w - - 0 1")));
}
#[test] fn endgame_king_prefers_the_centre() {
    assert!(evaluate(&b("7k/8/8/8/3K4/8/P7/8 w - - 0 1"))
          > evaluate(&b("7k/8/8/8/8/8/P7/K7 w - - 0 1")));
}
#[test] fn advanced_passed_pawn_is_worth_more() {
    assert!(evaluate(&b("4k3/P7/8/8/8/8/8/4K3 w - - 0 1"))
          > evaluate(&b("4k3/8/8/8/8/8/P7/4K3 w - - 0 1")));
}
```

- [ ] **Step 2: Run** `cargo test --release --test eval` — expected: compile failure.
- [ ] **Step 3: Implement `evaluate`.**
- [ ] **Step 4: Run** `cargo test --release --test eval` — expected: 6 passed.
- [ ] **Step 5: Commit** `feat(engine): tapered material and piece-square evaluation`.

---

### Task 3: Search core

**Files:**
- Create: `engine/src/search.rs`
- Test: `engine/tests/search.rs`

**Interfaces:**
- Consumes: `Position`, `evaluate`, `MATE`, `INF`, `MAX_PLY`
- Produces:

```rust
pub struct Limits {
    pub depth: Option<u32>,      // stop after completing this depth
    pub nodes: Option<u64>,      // abort once this many nodes are searched
    pub soft_ms: Option<u64>,    // do not start a new iteration after this
    pub hard_ms: Option<u64>,    // abort mid-search after this
}
impl Limits { pub fn depth(d: u32) -> Limits; pub fn infinite() -> Limits; }

pub struct SearchResult {
    pub best_move: Option<cozy_chess::Move>,  // None only when there are no legal moves
    pub score: i32,
    pub depth: u32,       // last fully completed depth
    pub nodes: u64,
    pub pv: Vec<cozy_chess::Move>,
}

pub struct Searcher { /* stop flag, node counter, start time; Task 4 adds TT and ordering state */ }
impl Searcher {
    pub fn new(hash_mb: usize) -> Searcher;       // hash_mb unused until Task 4
    pub fn stop_flag(&self) -> std::sync::Arc<std::sync::atomic::AtomicBool>;
    pub fn new_game(&mut self);
    pub fn search(&mut self, pos: &mut Position, limits: &Limits,
                  on_iter: &mut dyn FnMut(&SearchResult, u64 /*elapsed ms*/)) -> SearchResult;
}
```

Behaviour the tests do not fully pin:
- Iterative deepening from depth 1. `on_iter` is called after each completed depth.
- Negamax alpha-beta, fail-soft. At depth 0, quiescence: stand pat, then captures and queen promotions only; when in check, search all evasions and do not stand pat.
- Check extension: +1 depth when the side to move is in check.
- No legal moves: return `-(MATE - ply)` if in check, else 0. `pos.is_draw(ply)` at `ply > 0` returns 0.
- The stop flag and hard limit are polled every 2048 nodes. When an iteration is aborted, its result is discarded and the previous completed iteration is returned. **Depth 1 is never aborted**, so `best_move` is always a legal move when one exists.
- `search` clears the stop flag at entry.

- [ ] **Step 1: Write the failing tests:**

```rust
use vektr::{position::Position, search::{Limits, Searcher}, MATE};
fn best(fen: &str, depth: u32) -> (String, i32) {
    let mut p = Position::from_fen(fen).unwrap();
    let r = Searcher::new(16).search(&mut p, &Limits::depth(depth), &mut |_, _| {});
    (p.move_to_uci(r.best_move.unwrap()), r.score)
}
#[test] fn mate_in_one() {
    let (m, s) = best("6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1", 3);
    assert_eq!(m, "a1a8"); assert_eq!(s, MATE - 1);
}
#[test] fn mate_in_two() {
    let (m, s) = best("r1bq2r1/b4pk1/p1pp1p2/1p2pP2/1P2P1PB/3P4/1PPQ2P1/R3K2R w - - 0 1", 5);
    assert_eq!(m, "d2h6"); assert_eq!(s, MATE - 3);
}
#[test] fn sees_it_is_being_mated() {
    // only move Kb1, then ...Rg1#
    let (m, s) = best("6k1/8/8/8/8/6r1/7r/K7 w - - 0 1", 3);
    assert_eq!(m, "a1b1"); assert_eq!(s, -(MATE - 2));
}
#[test] fn wins_a_hanging_queen() {
    assert_eq!(best("4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1", 3).0, "d1d5");
}
#[test] fn quiescence_does_not_grab_a_defended_pawn() {
    // Qxd5 loses the queen to the e6 pawn; depth 1 must see it through quiescence
    assert_ne!(best("4k3/8/4p3/3p4/8/8/8/3QK3 w - - 0 1", 1).0, "d1d5");
}
#[test] fn stalemate_scores_zero_and_has_no_move() {
    let mut p = Position::from_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1").unwrap();
    let r = Searcher::new(16).search(&mut p, &Limits::depth(3), &mut |_, _| {});
    assert!(r.best_move.is_none()); assert_eq!(r.score, 0);
}
#[test] fn winning_side_avoids_threefold() {
    // White is a queen up. The position after Qe2 (black to move, Kg8) has already
    // occurred twice in the game, so Qd2-e2 now would be a threefold.
    let mut p = Position::from_fen("6k1/8/8/8/8/8/5Q2/6K1 w - - 0 1").unwrap();
    for m in ["f2e2","g8h8","e2f2","h8g8","f2e2","g8h8","e2d2","h8g8"] { p.play_uci(m).unwrap(); }
    assert!(!p.is_draw(0));
    let r = Searcher::new(16).search(&mut p, &Limits::depth(4), &mut |_, _| {});
    assert_ne!(p.move_to_uci(r.best_move.unwrap()), "d2e2");
    assert!(r.score > 500);
}
#[test] fn hard_limit_of_one_ms_still_returns_a_legal_move() {
    let mut p = Position::startpos();
    let l = Limits { depth: None, nodes: None, soft_ms: Some(0), hard_ms: Some(1) };
    let r = Searcher::new(16).search(&mut p, &l, &mut |_, _| {});
    assert!(p.legal_moves().contains(&r.best_move.unwrap()));
    assert!(r.depth >= 1);
}
#[test] fn node_limit_is_respected() {
    let mut p = Position::startpos();
    let l = Limits { depth: None, nodes: Some(20_000), soft_ms: None, hard_ms: None };
    let r = Searcher::new(16).search(&mut p, &l, &mut |_, _| {});
    assert!(r.nodes < 20_000 + 4096);
}
#[test] fn stop_flag_ends_an_infinite_search() {
    let mut s = Searcher::new(16);
    let flag = s.stop_flag();
    std::thread::spawn(move || { std::thread::sleep(std::time::Duration::from_millis(50));
        flag.store(true, std::sync::atomic::Ordering::Relaxed); });
    let t = std::time::Instant::now();
    let r = s.search(&mut Position::startpos(), &Limits::infinite(), &mut |_, _| {});
    assert!(t.elapsed().as_millis() < 1000); assert!(r.best_move.is_some());
}
#[test] fn pv_is_a_legal_line_starting_with_best_move() {
    let mut p = Position::startpos();
    let r = Searcher::new(16).search(&mut p, &Limits::depth(5), &mut |_, _| {});
    assert_eq!(r.pv[0], r.best_move.unwrap());
    for m in &r.pv { assert!(p.legal_moves().contains(m)); p.play(*m); }
}
```

If the mate-in-two FEN does not have `d2h6` as its unique mating key, verify with any reference tool and replace it with a position that does; keep the assertion form.

- [ ] **Step 2: Run** `cargo test --release --test search` — expected: compile failure.
- [ ] **Step 3: Implement `search.rs`.** Use a triangular PV table. Order moves captures-first for now (Task 4 replaces this).
- [ ] **Step 4: Run** `cargo test --release --test search` — expected: 11 passed.
- [ ] **Step 5: Commit** `feat(engine): iterative-deepening alpha-beta with quiescence`.

---

### Task 4: Transposition table and move ordering

**Files:**
- Create: `engine/src/tt.rs`, `engine/src/ordering.rs`
- Modify: `engine/src/search.rs`
- Test: `engine/tests/tt.rs`, additions to `engine/tests/search.rs`

**Interfaces:**
- Produces:

```rust
// tt.rs
#[derive(Clone, Copy, PartialEq, Debug)] pub enum Bound { Exact, Lower, Upper }
#[derive(Clone, Copy)] pub struct Entry { pub mv: Option<cozy_chess::Move>, pub score: i32, pub depth: u8, pub bound: Bound }
pub struct TT { /* power-of-two slot count */ }
impl TT {
    pub fn new(mb: usize) -> TT;             // largest power-of-two entry count that fits in `mb` MiB, at least 1024 entries
    pub fn clear(&mut self);
    pub fn probe(&self, hash: u64, ply: usize) -> Option<Entry>;   // score re-based to `ply`
    pub fn store(&mut self, hash: u64, ply: usize, e: Entry);      // score given relative to `ply`
    pub fn size_bytes(&self) -> usize;
}
// ordering.rs
pub struct History { /* killers[MAX_PLY][2], history[colour][from][to] */ }
impl History {
    pub fn new() -> History; pub fn clear(&mut self);
    pub fn score(&self, board: &cozy_chess::Board, mv: cozy_chess::Move,
                 tt_move: Option<cozy_chess::Move>, ply: usize) -> i32;
    pub fn on_cutoff(&mut self, board: &cozy_chess::Board, mv: cozy_chess::Move, ply: usize, depth: u32);
}
```

- Mate re-basing: scores with `|score| >= MATE - MAX_PLY` are stored as distance from the node (`score + ply` for wins, `score - ply` for losses) and reversed on probe.
- Store replaces always. The stored key is the full 64-bit hash; a probe with a different hash in the slot returns `None`.
- Ordering priority: TT move, then captures by most-valuable-victim / least-valuable-attacker, then queen promotions, then the two killers for this ply, then quiet moves by history score. `on_cutoff` updates killers and history only for quiet moves, with bonus `depth * depth`.
- Search uses a TT entry for a cutoff only at `ply > 0`, when `entry.depth >= depth`, and never when the position `is_draw`. `Searcher::new(hash_mb)` now allocates the table; `new_game` clears table and history.

- [ ] **Step 1: Write the failing tests** in `engine/tests/tt.rs`:

```rust
use vektr::{tt::{Bound, Entry, TT}, MATE};
#[test] fn stores_and_probes() {
    let mut t = TT::new(1);
    t.store(42, 0, Entry { mv: None, score: 17, depth: 5, bound: Bound::Exact });
    let e = t.probe(42, 0).unwrap();
    assert_eq!((e.score, e.depth, e.bound), (17, 5, Bound::Exact));
    assert!(t.probe(43, 0).is_none());
}
#[test] fn mate_scores_are_rebased_on_ply() {
    let mut t = TT::new(1);
    // at ply 4 we found "mate 3 plies from here": score MATE - 7
    t.store(7, 4, Entry { mv: None, score: MATE - 7, depth: 9, bound: Bound::Exact });
    assert_eq!(t.probe(7, 10).unwrap().score, MATE - 13);
    t.store(8, 4, Entry { mv: None, score: -(MATE - 7), depth: 9, bound: Bound::Exact });
    assert_eq!(t.probe(8, 10).unwrap().score, -(MATE - 13));
}
#[test] fn size_respects_the_budget() {
    let t = TT::new(16);
    assert!(t.size_bytes() <= 16 * 1024 * 1024 && t.size_bytes() > 4 * 1024 * 1024);
    assert!(TT::new(0).size_bytes() > 0);
}
```

Add to `engine/tests/search.rs`:

```rust
#[test] fn finds_the_fine_70_key_move() {
    // Lasker–Reichhelm: only Kb1 wins; needs a working transposition table to reach depth
    assert_eq!(best("8/k7/3p4/p2P1p2/P2P1P2/8/8/K7 w - - 0 1", 22).0, "a1b1");
}
#[test] fn reaches_depth_7_from_startpos_in_under_one_million_nodes() {
    let mut p = Position::startpos();
    let r = Searcher::new(16).search(&mut p, &Limits::depth(7), &mut |_, _| {});
    assert!(r.nodes < 1_000_000, "nodes = {}", r.nodes);
}
#[test] fn mate_distance_is_exact_with_a_warm_table() {
    let mut p = Position::from_fen("6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1").unwrap();
    let mut s = Searcher::new(16);
    s.search(&mut p, &Limits::depth(6), &mut |_, _| {});
    let r = s.search(&mut p, &Limits::depth(6), &mut |_, _| {});
    assert_eq!(r.score, MATE - 1);
}
```

- [ ] **Step 2: Run** `cargo test --release` — expected: compile failure for `tt`, and the three new search tests fail or time out.
- [ ] **Step 3: Implement `tt.rs` and `ordering.rs`, and wire both into `search.rs`.**
- [ ] **Step 4: Run** `cargo test --release` — expected: all tests in `position`, `eval`, `search` (14) and `tt` (3) pass. If the depth-7 node bound fails with correct ordering, record the measured count and set the bound 25% above it; do not weaken it further. If the Fine 70 test takes more than 20 seconds, the table is not working: fix that rather than lowering the depth.
- [ ] **Step 5: Commit** `feat(engine): transposition table and move ordering`.

---

### Task 5: Time management

**Files:**
- Create: `engine/src/time.rs`
- Test: `engine/tests/time.rs`

**Interfaces:**
- Produces:

```rust
pub struct Clock { pub remaining_ms: i64, pub inc_ms: i64, pub moves_to_go: Option<u32> }
/// Returns (soft_ms, hard_ms). Both are at least 1.
pub fn allocate(clock: &Clock, overhead_ms: i64) -> (u64, u64);
/// For `go movetime`.
pub fn fixed(movetime_ms: i64, overhead_ms: i64) -> (u64, u64);
```

Formula (integer arithmetic):
```
usable  = max(remaining_ms - overhead_ms, 1)
divisor = moves_to_go clamped to 2..=20, or 20 when absent
soft    = max(1, min(usable / divisor + inc_ms * 3 / 4, usable / 2))
hard    = max(soft, min(soft * 4, usable * 4 / 5))
fixed   = max(movetime_ms - overhead_ms, 1) for both
```

- [ ] **Step 1: Write the failing tests:**

```rust
use vektr::time::{allocate, fixed, Clock};
fn c(r: i64, i: i64, m: Option<u32>) -> Clock { Clock { remaining_ms: r, inc_ms: i, moves_to_go: m } }
#[test] fn blitz_3_plus_2_opening() { assert_eq!(allocate(&c(180_000, 2_000, None), 100), (10_495, 41_980)); }
#[test] fn low_on_time_with_increment() { assert_eq!(allocate(&c(1_000, 2_000, None), 100), (450, 720)); }
#[test] fn moves_to_go_is_used() { assert_eq!(allocate(&c(60_000, 0, Some(10)), 0), (6_000, 24_000)); }
#[test] fn zero_and_negative_clocks_give_one_ms() {
    assert_eq!(allocate(&c(0, 0, None), 100), (1, 1));
    assert_eq!(allocate(&c(-500, 2_000, None), 100), (1, 1));
    assert_eq!(allocate(&c(50, 0, None), 100), (1, 1));
}
#[test] fn hard_never_exceeds_usable_time() {
    for r in [10, 100, 1_000, 10_000, 300_000] { for i in [0, 2_000, 3_000] {
        let (s, h) = allocate(&c(r, i, None), 100);
        assert!(s <= h && (h as i64) <= (r - 100).max(1));
    }}
}
#[test] fn movetime() { assert_eq!(fixed(1_000, 100), (900, 900)); assert_eq!(fixed(50, 100), (1, 1)); }
```

- [ ] **Step 2: Run** `cargo test --release --test time` — expected: compile failure.
- [ ] **Step 3: Implement `time.rs`.**
- [ ] **Step 4: Run** `cargo test --release --test time` — expected: 6 passed.
- [ ] **Step 5: Commit** `feat(engine): time allocation`.

---

### Task 6: UCI loop and bench

**Files:**
- Create: `engine/src/uci.rs`, `engine/src/bench.rs`
- Modify: `engine/src/main.rs`
- Test: `engine/tests/uci.rs`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `pub fn uci::run(input: impl std::io::BufRead + Send + 'static, output: impl std::io::Write + Send + 'static)` — returns when `quit` is read or input ends.
  - `pub fn bench::run(depth: u32) -> (u64 /*nodes*/, u64 /*nps*/)`
  - Binary: `vektr` runs the UCI loop on stdin/stdout; `vektr bench [depth]` (default depth 8) prints exactly one final line `"{nodes} nodes {nps} nps"`.

UCI behaviour:
- `uci` → `id name Vektr64 0.1.0`, `id author Irek Romaniuk`, `option name Hash type spin default 64 min 1 max 4096`, `option name Move Overhead type spin default 100 min 0 max 5000`, `option name Threads type spin default 1 min 1 max 1`, `uciok`.
- `isready` → `readyok` (answered even while a search is running).
- `ucinewgame`, `setoption name <N> value <V>` (names case-insensitive; unknown options ignored), `position startpos|fen <fen> [moves ...]`.
- `go` accepts `wtime btime winc binc movestogo movetime depth nodes infinite`. Clock for the side to move goes through `time::allocate`; `movetime` through `time::fixed`. `go` with no arguments is `infinite`.
- The search runs on its own thread so `stop`, `isready` and `quit` are handled during a search. `stop` and `quit` set the stop flag. `quit` and end of input both wait for the search thread, then return.
- After each completed depth: `info depth D score cp X|mate N nodes N nps N time MS pv ...`. Mate score: `mate N` in moves, positive when the engine mates (`(MATE - score + 1) / 2`), negative when it is mated.
- Finish with `bestmove <uci>`. If there is no legal move: `bestmove 0000`.
- A malformed `position` or an illegal move in the list: print `info string error: <reason>`, keep the previous position, do not crash.
- Every line is flushed immediately.
- Bench positions: startpos, Kiwipete, `8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1`, `r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10`, each searched with a fresh `Searcher::new(16)`.

- [ ] **Step 1: Write the failing tests** in `engine/tests/uci.rs`. Helper `fn session(script: &str, wait_ms: u64) -> (String, std::process::ExitStatus)` spawns `env!("CARGO_BIN_EXE_vektr")`, writes `script`, sleeps `wait_ms`, closes stdin, and fails the test if the process has not exited within 3 seconds.

```rust
#[test] fn handshake() {
    let (out, st) = session("uci\nisready\nquit\n", 0);
    assert!(out.contains("id name Vektr64") && out.contains("option name Hash")
         && out.contains("option name Move Overhead") && out.contains("uciok") && out.contains("readyok"));
    assert!(st.success());
}
#[test] fn plays_a_legal_reply_with_castling_in_the_move_list() {
    let (out, _) = session("position startpos moves e2e4 e7e5 g1f3 b8c6 f1c4 f8c5 e1g1\ngo depth 4\n", 500);
    let bm = out.lines().rev().find(|l| l.starts_with("bestmove ")).unwrap();
    let mv = bm.split(' ').nth(1).unwrap();
    let mut p = vektr::position::Position::startpos();
    for m in ["e2e4","e7e5","g1f3","b8c6","f1c4","f8c5","e1g1"] { p.play_uci(m).unwrap(); }
    assert!(p.play_uci(mv).is_ok());
}
#[test] fn reports_mate_in_moves() {
    let (out, _) = session("position fen 6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1\ngo depth 4\n", 500);
    assert!(out.contains("score mate 1") && out.contains("bestmove a1a8"));
}
#[test] fn negative_and_tiny_clocks_still_answer() {
    for go in ["go wtime -20 btime 5000 winc 0 binc 0", "go wtime 3 btime 3", "go movetime 1"] {
        let (out, _) = session(&format!("position startpos\n{go}\n"), 300);
        assert!(out.lines().any(|l| l.starts_with("bestmove ") && !l.contains("0000")), "{go}");
    }
}
#[test] fn black_uses_its_own_clock() {
    let t = std::time::Instant::now();
    let (out, _) = session("position startpos moves e2e4\ngo wtime 600000 btime 400 winc 0 binc 0\n", 300);
    assert!(out.contains("bestmove ")); assert!(t.elapsed().as_millis() < 2000);
}
#[test] fn stop_ends_an_infinite_search() {
    let (out, _) = session("position startpos\ngo infinite\nisready\nstop\n", 300);
    assert!(out.contains("readyok") && out.contains("bestmove "));
}
#[test] fn closing_stdin_mid_search_exits() {
    let (_, st) = session("position startpos\ngo infinite\n", 200);   // helper enforces exit within 3 s
    assert!(st.success());
}
#[test] fn quit_mid_search_exits() {
    let (_, st) = session("position startpos\ngo infinite\nquit\n", 100);
    assert!(st.success());
}
#[test] fn bad_input_does_not_crash() {
    let (out, st) = session("position fen garbage\nposition startpos moves e2e5\nnonsense\nsetoption name Nope value 1\ngo depth 2\n", 300);
    assert!(out.contains("info string error") && out.contains("bestmove ") && st.success());
}
#[test] fn checkmated_position_answers_null_move() {
    let (out, _) = session("position fen R5k1/5ppp/8/8/8/8/8/4K3 b - - 0 1\ngo depth 2\n", 300);
    assert!(out.contains("bestmove 0000"));
}
#[test] fn hash_option_is_applied() {
    let (out, st) = session("setoption name Hash value 8\nisready\nposition startpos\ngo depth 5\n", 500);
    assert!(out.contains("readyok") && out.contains("bestmove ") && st.success());
}
#[test] fn bench_is_deterministic() {
    let run = || String::from_utf8(std::process::Command::new(env!("CARGO_BIN_EXE_vektr"))
        .args(["bench", "6"]).output().unwrap().stdout).unwrap();
    let (a, b) = (run(), run());
    let nodes = |s: &str| s.lines().last().unwrap().split(' ').next().unwrap().to_string();
    assert!(a.lines().last().unwrap().ends_with(" nps")); assert_eq!(nodes(&a), nodes(&b));
}
```

- [ ] **Step 2: Run** `cargo test --release --test uci` — expected: failures (binary prints nothing).
- [ ] **Step 3: Implement `uci.rs`, `bench.rs`, `main.rs`.** A reader thread forwards stdin lines over a channel; the main loop owns the `Position` and `Searcher` options; the search thread owns the `Searcher` for the duration of one `go` and hands it back.
- [ ] **Step 4: Run** `cargo test --release` — expected: every engine test passes. Then `cargo run --release -- bench` and write the node count into `docs/bench.md` as the stage 1 baseline.
- [ ] **Step 5: Commit** `feat(engine): UCI loop and bench`.

---

### Task 7: `vektr-ops play` and the Lichess bridge

**Files:**
- Create: `ops/pyproject.toml`, `ops/src/vektr_ops/__init__.py`, `ops/src/vektr_ops/sizing.py`, `ops/src/vektr_ops/botconfig.py`, `ops/src/vektr_ops/cli.py`, `bot/config.template.yml`, `.env.example`, `deploy/vektr64.service`, `deploy/install-box.sh`
- Add submodule: `bot/lichess-bot` → `https://github.com/lichess-bot-devs/lichess-bot` at `a31b965a8eda79f3a8ec76b49ef9349fbf6c63bd`
- Test: `ops/tests/test_sizing.py`, `ops/tests/test_botconfig.py`, `ops/tests/test_cli.py`

**Interfaces:**
- Consumes: the `vektr` binary at `engine/target/release/vektr`; UCI options `Hash`, `Move Overhead`.
- Produces:
  - `sizing.hash_mb(available_mb: int) -> int` — `available_mb // 16`, rounded down to a power of two, clamped to `16..=128`.
  - `botconfig.render(template: dict, *, engine_dir: str, engine_name: str, hash_mb: int) -> dict` — returns a new dict; never contains a token.
  - `botconfig.load_env(repo_root: Path) -> dict[str, str]` — process environment overlaid on `.env` (environment wins).
  - CLI `vektr-ops play [--dry-run]` and `vektr-ops status`.

Before writing `config.template.yml`, read `bot/lichess-bot/config.yml.default` and the wiki pages in the pinned submodule; use its key names exactly. The values this project fixes:

| Setting | Value |
|---|---|
| token | the placeholder from `config.yml.default`; the real one is passed as env `LICHESS_BOT_TOKEN` |
| engine protocol | `uci`; `ponder: false`; `uci_options`: `Hash` (rendered), `Move Overhead: 300`, `Threads: 1` |
| built-in opening books, online moves, tablebases | all disabled |
| `challenge.concurrency` | `1` |
| `challenge.variants` | `[standard]` |
| `challenge.time_controls` | `[blitz]` |
| `challenge.modes` | `[rated, casual]` |
| `challenge.accept_bot` | `true`; human challenges declined (`only_bot: true`) |
| `challenge` min/max initial and increment | initial 180–300 s, increment 2–3 s |
| `matchmaking.allow_matchmaking` | `true` |
| `matchmaking.challenge_variant` | `standard` |
| `matchmaking.challenge_initial_time` / `challenge_increment` | `[180, 300]` / `[2, 3]` |
| `matchmaking.challenge_mode` | `rated` |
| `matchmaking.challenge_timeout` | `35` (minutes idle before challenging: approximates the ~30 games/day budget; an exact budget is a stage 2 item) |
| `matchmaking.opponent_rating_difference` | `300` |
| `matchmaking.challenge_filter` | `coarse` |
| draw/resign offers | resign off; accept draws only via the bridge defaults |
| PGN | `pgn_directory: "game_records"` under `bot/`, added to `.gitignore` |

`vektr-ops play`:
1. Repo root = parent of `ops/`. Load env; if `LICHESS_TOKEN` is missing or empty, exit code 2 with `error: LICHESS_TOKEN is not set (put it in .env or the environment)`.
2. Run `cargo build --release` in `engine/`; exit non-zero with cargo's output if it fails.
3. `hash_mb(psutil.virtual_memory().available // 2**20)`, render the template, write `bot/config.yml` (mode 0600).
4. `--dry-run`: print the rendered config and exit 0 without starting the bridge.
5. Otherwise `os.execvpe` the bridge: `uv run --project bot/lichess-bot python lichess-bot.py --config ../config.yml` with cwd `bot/lichess-bot` and `LICHESS_BOT_TOKEN` set. If the pinned bridge has no `pyproject.toml`, `install-box.sh` creates `bot/lichess-bot/.venv` from its `requirements.txt` and `play` execs `.venv/bin/python` instead; pick whichever the pinned commit supports and state it in `README.md`.

`vektr-ops status`: prints `systemctl is-active vektr64` and, from `https://lichess.org/api/user/Vektr64` (no token needed), the blitz rating, rating deviation, rated game count, and whether a game is in progress. Network failure prints `lichess: unreachable` and exits 0.

`deploy/vektr64.service`: `User=irom`, `WorkingDirectory=/home/irom/Vektr64`, `EnvironmentFile=/home/irom/Vektr64/.env`, `ExecStart=/home/irom/.local/bin/uv run --project ops vektr-ops play`, `Restart=always`, `RestartSec=30`, `MemoryMax=600M`, `After=network-online.target`, `Wants=network-online.target`, `WantedBy=multi-user.target`.

`deploy/install-box.sh` (idempotent, run on the box from the repo root): install `rustup` stable (minimal profile) and `uv` if absent, `git submodule update --init`, `uv sync --project ops`, set up the bridge's Python environment, `cargo build --release`, `cargo test --release`, copy the unit to `/etc/systemd/system/` and `systemctl daemon-reload`. It does **not** enable or start the service and does not touch `.env`.

`.env.example`: `LICHESS_TOKEN=` with a comment: scope `bot:play`, create at lichess.org/account/oauth/token.

- [ ] **Step 1: Write the failing tests:**

```python
# ops/tests/test_sizing.py
import pytest
from vektr_ops.sizing import hash_mb
@pytest.mark.parametrize("avail,want", [(2048, 128), (1024, 64), (600, 32), (300, 16), (0, 16), (64000, 128)])
def test_hash_mb(avail, want): assert hash_mb(avail) == want

# ops/tests/test_botconfig.py
import copy, yaml
from pathlib import Path
from vektr_ops.botconfig import render, load_env
ROOT = Path(__file__).resolve().parents[2]
def tpl(): return yaml.safe_load((ROOT / "bot/config.template.yml").read_text())
def test_render_sets_engine_and_hash():
    t = tpl(); before = copy.deepcopy(t)
    c = render(t, engine_dir="/x/engine/target/release", engine_name="vektr", hash_mb=64)
    assert c["engine"]["dir"] == "/x/engine/target/release" and c["engine"]["name"] == "vektr"
    assert c["engine"]["uci_options"]["Hash"] == 64 and c["engine"]["uci_options"]["Threads"] == 1
    assert t == before
def test_template_pins_the_spec():
    c = tpl()
    assert c["challenge"]["concurrency"] == 1
    assert c["challenge"]["variants"] == ["standard"]
    assert c["challenge"]["time_controls"] == ["blitz"]
    assert c["matchmaking"]["allow_matchmaking"] is True
    assert c["matchmaking"]["challenge_mode"] == "rated"
    assert c["matchmaking"]["challenge_initial_time"] == [180, 300]
    assert c["matchmaking"]["challenge_increment"] == [2, 3]
def test_no_token_in_template_or_render():
    text = (ROOT / "bot/config.template.yml").read_text()
    assert "lip_" not in text
def test_env_overrides_dotenv(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("LICHESS_TOKEN=from_file\nOTHER=1\n")
    monkeypatch.setenv("LICHESS_TOKEN", "from_env")
    e = load_env(tmp_path); assert e["LICHESS_TOKEN"] == "from_env" and e["OTHER"] == "1"

# ops/tests/test_cli.py
from typer.testing import CliRunner
from vektr_ops.cli import app
def test_play_without_token_fails_clearly(monkeypatch, tmp_path):
    monkeypatch.delenv("LICHESS_TOKEN", raising=False)
    monkeypatch.setattr("vektr_ops.cli.repo_root", lambda: tmp_path)
    r = CliRunner().invoke(app, ["play", "--dry-run"])
    assert r.exit_code == 2 and "LICHESS_TOKEN is not set" in r.output
def test_dry_run_prints_config_without_token(monkeypatch):
    monkeypatch.setenv("LICHESS_TOKEN", "lip_secret_value")
    monkeypatch.setattr("vektr_ops.cli.build_engine", lambda root: None)
    r = CliRunner().invoke(app, ["play", "--dry-run"])
    assert r.exit_code == 0 and "Hash" in r.output and "lip_secret_value" not in r.output
```

- [ ] **Step 2: Run** `uv run --project ops pytest ops/tests -q` — expected: import errors.
- [ ] **Step 3: Add the submodule and write the package, template, unit file, install script and `.env.example`.** `cli.py` exposes module-level `repo_root()` and `build_engine(root)` so the tests can replace them.
- [ ] **Step 4: Run** `uv run --project ops pytest ops/tests -q` — expected: 12 passed. Then `LICHESS_TOKEN=x uv run --project ops vektr-ops play --dry-run` on the laptop — expected: a config with `Hash: 128` or lower and no token.
- [ ] **Step 5: Update `README.md`**: what Vektr64 is, the three commands (`install-box.sh`, `vektr-ops play`, `vektr-ops status`), where the token goes, link to the spec.
- [ ] **Step 6: Commit** `feat(ops): vektr-ops play/status, lichess-bot bridge, systemd unit`.

---

### Task 8: Deploy to the box and go live

This task changes things outside the repo. Steps marked **OWNER** stop and wait for the owner.

**Files:**
- Create: `docs/runbook.md` (what was done, how to restart, how to stop rated play, how to read logs)

- [ ] **Step 1: Push the branch** to `origin` (ask the owner first if the push has not been authorized in this session).
- [ ] **Step 2: Set up the box.** `ssh irom@100.114.82.19`, clone `https://github.com/irom77/Vektr64.git` to `/home/irom/Vektr64` on the branch, run `deploy/install-box.sh`. Expected: `cargo test --release` passes on the box. If Tailscale SSH asks for a browser check, give the owner the link.
- [ ] **Step 3: Record the box baseline.** `engine/target/release/vektr bench` on the box; add nodes and nps to `docs/bench.md` next to the laptop's. The node count must equal the laptop's.
- [ ] **Step 4: OWNER — token.** Ask the owner to create a token with scope `bot:play` and place it in `/home/irom/Vektr64/.env` on the box as `LICHESS_TOKEN=...` (`chmod 600`). The builder never sees or echoes the token.
- [ ] **Step 5: OWNER — irreversible upgrade.** Ask in so many words: "Upgrade Vektr64 to a BOT account now? This cannot be undone and the account can never play as a human again." Only on a yes, run the bridge's upgrade command on the box (`-u` flag of `lichess-bot.py`, with `LICHESS_BOT_TOKEN` taken from `.env`). Verify with `curl -s https://lichess.org/api/user/Vektr64` → `"title":"BOT"`.
- [ ] **Step 6: Casual smoke test.** Temporarily run `vektr-ops play` in the foreground with matchmaking off (`matchmaking.allow_matchmaking: false` in a local, uncommitted copy of the rendered config, or the bridge's equivalent flag). From the owner's own Lichess account or the Lichess site, challenge Vektr64 to one casual 3+2 game, or challenge a known always-on bot. Expected: the game completes with no illegal move, no time forfeit, no engine crash in the log; the engine's `info` lines show depth 8 or more in the middlegame.
- [ ] **Step 7: OWNER — rated play on.** With the owner's go-ahead: `sudo systemctl enable --now vektr64`. Expected: `systemctl is-active vektr64` → `active`; within an hour `vektr-ops status` shows at least one rated blitz game.
- [ ] **Step 8: Reliability checks on the live service.**
  - `sudo systemctl kill -s SIGKILL vektr64` → the service is `active` again within 60 seconds and resumes or re-enters play.
  - Drop the network for 60 seconds (`sudo tailscale down` is not enough — use `sudo ip link set <iface> down && sleep 60 && sudo ip link set <iface> up` from a local console or a `systemd-run` timer so the box recovers by itself) → the bridge reconnects without a restart. Skip this check if there is no safe way to do it remotely, and say so in the runbook.
  - After 20 rated games: zero losses on time caused by the engine exceeding its hard limit (inspect `bot/game_records` and the journal). Peak memory of the service stays under `MemoryMax`.
- [ ] **Step 9: Write `docs/runbook.md`, commit** `docs: stage 1 runbook and bench baselines`, and open a pull request to `main`.

---

## Done when

- All engine and ops tests pass on the laptop and on the box.
- Vektr64 holds the BOT title and has completed 20 rated blitz games from the box with no time forfeits and no crashes.
- The service survives a kill and a reboot without manual action.
- `docs/bench.md` holds the baseline node count, identical on both machines.

## Not in this plan

Gate runner, anchor ladder, external store, manifest, status page (stage 2). Network evaluation, trainer, bursts (stage 3). Self-play data generation and learning from games (stage 4). Automation and tuning (stage 5). Further search techniques, each of which needs the stage 2 gate.
