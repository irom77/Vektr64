//! Clock trajectory over a whole game. Slow and timing-dependent, so ignored by
//! default: run with `cargo test --release --test clock -- --ignored --nocapture`.

use std::time::Instant;

use vektr::position::Position;
use vektr::search::{Limits, Searcher};
use vektr::time::{allocate, Clock};

/// Self-play at 3+2 blitz scaled down fifteen-fold. Returns each side's clock in
/// milliseconds after each of its moves.
fn play(base_ms: i64, inc_ms: i64, max_moves: usize) -> [Vec<i64>; 2] {
    let mut pos = Position::startpos();
    let mut searchers = [Searcher::new(16), Searcher::new(16)];
    let mut clocks = [base_ms, base_ms];
    let mut trace = [Vec::new(), Vec::new()];
    for ply in 0..max_moves * 2 {
        let side = ply % 2;
        if pos.legal_moves().is_empty() || pos.is_draw(0) {
            break;
        }
        let clock = Clock { remaining_ms: clocks[side], inc_ms, moves_to_go: None };
        let (soft, hard) = allocate(&clock, 20);
        let limits = Limits { depth: None, nodes: None, soft_ms: Some(soft), hard_ms: Some(hard) };
        let start = Instant::now();
        let result = searchers[side].search(&mut pos, &limits, &mut |_, _| {});
        clocks[side] -= start.elapsed().as_millis() as i64;
        assert!(clocks[side] > 0, "side {side} lost on time at move {}", ply / 2 + 1);
        clocks[side] += inc_ms;
        trace[side].push(clocks[side]);
        pos.play(result.best_move.unwrap());
    }
    trace
}

#[test]
#[ignore]
fn blitz_clock_is_not_spent_in_the_opening() {
    let base = 12_000;
    let trace = play(base, 133, 60);
    for (side, clocks) in trace.iter().enumerate() {
        let sample: Vec<i64> = clocks.iter().step_by(5).copied().collect();
        println!("side {side}: {sample:?}");
        if let Some(&at_30) = clocks.get(29) {
            assert!(at_30 >= base * 8 / 100, "side {side} has {at_30} ms at move 30");
        }
        let low = *clocks.iter().min().unwrap();
        assert!(low >= base * 3 / 100, "side {side} dropped to {low} ms");
        // Nor should it hoard time: by move 30 a fair share must be spent.
        if let Some(&at_30) = clocks.get(29) {
            assert!(at_30 <= base * 70 / 100, "side {side} still has {at_30} ms at move 30");
        }
    }
}
