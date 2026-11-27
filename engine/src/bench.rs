//! A fixed workload whose node count identifies the search's behaviour.

use std::time::Instant;

use crate::position::Position;
use crate::search::{Limits, Searcher};

const POSITIONS: [&str; 4] = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
];

/// Searches each bench position to `depth` and returns `(nodes, nodes per second)`.
pub fn run(depth: u32) -> (u64, u64) {
    let start = Instant::now();
    let mut nodes = 0;
    for fen in POSITIONS {
        let mut pos = Position::from_fen(fen).expect("bench positions are valid");
        nodes += Searcher::new(16).search(&mut pos, &Limits::depth(depth), &mut |_, _| {}).nodes;
    }
    let micros = start.elapsed().as_micros().max(1) as u64;
    (nodes, nodes * 1_000_000 / micros)
}
