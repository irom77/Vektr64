use vektr::position::{perft, Position};

const KIWIPETE: &str = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";

#[test]
fn perft_startpos() {
    let p = Position::startpos();
    assert_eq!(perft(&p.board, 1), 20);
    assert_eq!(perft(&p.board, 4), 197_281);
    assert_eq!(perft(&p.board, 5), 4_865_609);
}

#[test]
fn perft_kiwipete() {
    let p = Position::from_fen(KIWIPETE).unwrap();
    assert_eq!(perft(&p.board, 3), 97_862);
    assert_eq!(perft(&p.board, 4), 4_085_603);
}

#[test]
fn castling_uses_standard_uci_both_ways() {
    let mut p = Position::from_fen(KIWIPETE).unwrap();
    let ucis: Vec<String> = p.legal_moves().iter().map(|m| p.move_to_uci(*m)).collect();
    assert!(ucis.contains(&"e1g1".to_string()));
    assert!(ucis.contains(&"e1c1".to_string()));
    assert!(!ucis.contains(&"e1h1".to_string()));
    p.play_uci("e1g1").unwrap();
    assert_eq!(
        p.board.to_string().split(' ').next().unwrap(),
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R4RK1"
    );
}

#[test]
fn promotion_round_trips() {
    let mut p = Position::from_fen("8/P6k/8/8/8/8/8/K7 w - - 0 1").unwrap();
    let ucis: Vec<String> = p.legal_moves().iter().map(|m| p.move_to_uci(*m)).collect();
    assert!(ucis.contains(&"a7a8q".to_string()) && ucis.contains(&"a7a8n".to_string()));
    p.play_uci("a7a8q").unwrap();
}

#[test]
fn illegal_and_garbage_moves_are_rejected() {
    let mut p = Position::startpos();
    assert!(p.play_uci("e2e5").is_err());
    assert!(p.play_uci("zz").is_err());
    assert!(Position::from_fen("not a fen").is_err());
}

#[test]
fn repetition_in_game_history_needs_three() {
    let mut p = Position::startpos();
    for m in ["g1f3", "g8f6", "f3g1", "f6g8"] {
        p.play_uci(m).unwrap();
    }
    assert!(!p.is_draw(0)); // position seen twice in the game
    for m in ["g1f3", "g8f6", "f3g1", "f6g8"] {
        p.play_uci(m).unwrap();
    }
    assert!(p.is_draw(0)); // third occurrence
}

#[test]
fn repetition_inside_search_needs_two() {
    let mut p = Position::startpos();
    for m in ["g1f3", "g8f6", "f3g1", "f6g8"] {
        p.play_uci(m).unwrap();
    }
    assert!(p.is_draw(4)); // all four moves were made by the search
}

#[test]
fn fifty_move_rule_and_bare_kings() {
    assert!(Position::from_fen("8/8/8/4k3/8/8/4K2R/8 w - - 100 80").unwrap().is_draw(0));
    assert!(Position::from_fen("8/8/8/4k3/8/8/4K3/8 w - - 0 1").unwrap().is_draw(0));
    assert!(Position::from_fen("8/8/8/4k3/8/8/4K1N1/8 w - - 0 1").unwrap().is_draw(0));
}
