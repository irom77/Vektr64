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
