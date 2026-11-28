use vektr::{
    position::Position,
    search::{Limits, Searcher},
    MATE,
};

fn best(fen: &str, depth: u32) -> (String, i32) {
    let mut p = Position::from_fen(fen).unwrap();
    let r = Searcher::new(16).search(&mut p, &Limits::depth(depth), &mut |_, _| {});
    (p.move_to_uci(r.best_move.unwrap()), r.score)
}

#[test]
fn mate_in_one() {
    let (m, s) = best("6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1", 3);
    assert_eq!(m, "a1a8");
    assert_eq!(s, MATE - 1);
}

#[test]
fn mate_in_two() {
    let (m, s) = best("r1bq2r1/b4pk1/p1pp1p2/1p2pP2/1P2P1PB/3P4/1PPQ2P1/R3K2R w - - 0 1", 5);
    assert_eq!(m, "d2h6");
    assert_eq!(s, MATE - 3);
}

#[test]
fn sees_it_is_being_mated() {
    // only move Kb1, then ...Rg1#
    let (m, s) = best("6k1/8/8/8/8/6r1/7r/K7 w - - 0 1", 3);
    assert_eq!(m, "a1b1");
    assert_eq!(s, -(MATE - 2));
}

#[test]
fn wins_a_hanging_queen() {
    assert_eq!(best("4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1", 3).0, "d1d5");
}

#[test]
fn quiescence_does_not_grab_a_defended_pawn() {
    // Qxd5 loses the queen to the e6 pawn; depth 1 must see it through quiescence
    assert_ne!(best("4k3/8/4p3/3p4/8/8/8/3QK3 w - - 0 1", 1).0, "d1d5");
}

#[test]
fn stalemate_scores_zero_and_has_no_move() {
    let mut p = Position::from_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1").unwrap();
    let r = Searcher::new(16).search(&mut p, &Limits::depth(3), &mut |_, _| {});
    assert!(r.best_move.is_none());
    assert_eq!(r.score, 0);
}

#[test]
fn winning_side_avoids_threefold() {
    // White is a queen up. The position after Qe2 (black to move, Kg8) has already
    // occurred twice in the game, so Qd2-e2 now would be a threefold.
    let mut p = Position::from_fen("6k1/8/8/8/8/8/5Q2/6K1 w - - 0 1").unwrap();
    for m in ["f2e2", "g8h8", "e2f2", "h8g8", "f2e2", "g8h8", "e2d2", "h8g8"] {
        p.play_uci(m).unwrap();
    }
    assert!(!p.is_draw(0));
    let r = Searcher::new(16).search(&mut p, &Limits::depth(4), &mut |_, _| {});
    assert_ne!(p.move_to_uci(r.best_move.unwrap()), "d2e2");
    assert!(r.score > 500);
}

#[test]
fn hard_limit_of_one_ms_still_returns_a_legal_move() {
    let mut p = Position::startpos();
    let l = Limits { depth: None, nodes: None, soft_ms: Some(0), hard_ms: Some(1) };
    let r = Searcher::new(16).search(&mut p, &l, &mut |_, _| {});
    assert!(p.legal_moves().contains(&r.best_move.unwrap()));
    assert!(r.depth >= 1);
}

#[test]
fn node_limit_is_respected() {
    let mut p = Position::startpos();
    let l = Limits { depth: None, nodes: Some(20_000), soft_ms: None, hard_ms: None };
    let r = Searcher::new(16).search(&mut p, &l, &mut |_, _| {});
    assert!(r.nodes < 20_000 + 4096);
}

#[test]
fn stop_flag_ends_an_infinite_search() {
    let mut s = Searcher::new(16);
    let flag = s.stop_flag();
    std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(50));
        flag.store(true, std::sync::atomic::Ordering::Relaxed);
    });
    let t = std::time::Instant::now();
    let r = s.search(&mut Position::startpos(), &Limits::infinite(), &mut |_, _| {});
    assert!(t.elapsed().as_millis() < 1000);
    assert!(r.best_move.is_some());
}

#[test]
fn pv_is_a_legal_line_starting_with_best_move() {
    let mut p = Position::startpos();
    let r = Searcher::new(16).search(&mut p, &Limits::depth(5), &mut |_, _| {});
    assert_eq!(r.pv[0], r.best_move.unwrap());
    for m in &r.pv {
        assert!(p.legal_moves().contains(m));
        p.play(*m);
    }
}

#[test]
fn finds_the_fine_70_key_move() {
    // Lasker–Reichhelm: only Kb1 wins; needs a working transposition table to reach depth.
    // Every table size from 1 to 64 MiB settles on Kb1 by depth 30; shallower depths
    // flip between Kb1 and Kb2 depending on which entries were overwritten.
    let mut p = Position::from_fen("8/k7/3p4/p2P1p2/P2P1P2/8/8/K7 w - - 0 1").unwrap();
    let r = Searcher::new(64).search(&mut p, &Limits::depth(30), &mut |_, _| {});
    assert_eq!(p.move_to_uci(r.best_move.unwrap()), "a1b1");
    assert!(r.score > 250, "score = {}", r.score);
    // 20M with the hand-written evaluation; 98.2M measured with the bootstrap network.
    assert!(r.nodes < 150_000_000, "nodes = {}", r.nodes);
}

#[test]
fn reaches_depth_7_from_startpos_in_under_one_million_nodes() {
    let mut p = Position::startpos();
    let r = Searcher::new(16).search(&mut p, &Limits::depth(7), &mut |_, _| {});
    assert!(r.nodes < 1_000_000, "nodes = {}", r.nodes);
}

#[test]
fn mate_distance_is_exact_with_a_warm_table() {
    let mut p = Position::from_fen("6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1").unwrap();
    let mut s = Searcher::new(16);
    s.search(&mut p, &Limits::depth(6), &mut |_, _| {});
    let r = s.search(&mut p, &Limits::depth(6), &mut |_, _| {});
    assert_eq!(r.score, MATE - 1);
}

/// Plays `fen` with one long-lived searcher as White. Black replies with the move
/// leading to the most-repeated position, or a pseudo-random move when none repeats.
/// Returns the ply at which a position occurred for the third time, if any.
fn first_threefold(fen: &str, plies: usize, nodes: u64, seed: u64) -> Option<usize> {
    use std::collections::HashMap;
    let mut p = Position::from_fen(fen).unwrap();
    let mut s = Searcher::new(64);
    let mut rng = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut seen: HashMap<u64, u32> = HashMap::new();
    seen.insert(p.board.hash(), 1);
    for ply in 0..plies {
        let moves = p.legal_moves();
        if moves.is_empty() {
            return None;
        }
        let mv = if ply % 2 == 0 {
            let l = Limits { depth: None, nodes: Some(nodes), soft_ms: None, hard_ms: None };
            s.search(&mut p, &l, &mut |_, _| {}).best_move.unwrap()
        } else {
            let count = |m: &cozy_chess::Move| {
                let mut b = p.board.clone();
                b.play_unchecked(*m);
                seen.get(&b.hash()).copied().unwrap_or(0)
            };
            let best = moves.iter().copied().max_by_key(|m| count(m)).unwrap();
            if count(&best) > 0 {
                best
            } else {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                moves[(rng % moves.len() as u64) as usize]
            }
        };
        p.play(mv);
        let n = seen.entry(p.board.hash()).or_insert(0);
        *n += 1;
        if *n >= 3 {
            return Some(ply);
        }
    }
    None
}

#[test]
fn warm_table_does_not_walk_into_threefold_when_winning() {
    // White is a rook up in a blocked position and Black can only shuffle. With
    // entries left in the table from earlier moves, White used to repeat.
    let fen = "4k3/8/1p1p1p1p/pPpPpPpP/P1P1P1P1/8/8/R3K3 w - - 0 1";
    for seed in 1..=4 {
        assert_eq!(first_threefold(fen, 60, 300_000, seed), None, "seed {seed}");
    }
}

#[test]
fn a_position_scores_the_same_however_it_was_reached() {
    let mut played = Position::startpos();
    for m in ["e2e4", "e7e5", "g1f3", "b8c6"] {
        played.play_uci(m).unwrap();
    }
    let mut parsed =
        Position::from_fen("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3").unwrap();
    let a = Searcher::new(16).search(&mut played, &Limits::depth(5), &mut |_, _| {});
    let b = Searcher::new(16).search(&mut parsed, &Limits::depth(5), &mut |_, _| {});
    assert_eq!((a.score, a.nodes), (b.score, b.nodes));
}
