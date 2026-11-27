use cozy_chess::Board;
use vektr::eval::evaluate;

fn b(fen: &str) -> Board {
    Board::from_fen(fen, false).unwrap()
}

#[test]
fn startpos_is_balanced() {
    assert_eq!(evaluate(&Board::default()), 0);
}

#[test]
fn score_is_from_side_to_move() {
    let w = b("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    let bl = b("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1");
    assert!(evaluate(&w) > 800);
    assert_eq!(evaluate(&w), -evaluate(&bl));
}

#[test]
fn colour_mirrored_positions_score_the_same() {
    // same position with colours and ranks flipped, same side-to-move role
    let a = b("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3");
    let m = b("rnbqkb1r/pppp1ppp/5n2/4p3/4P3/2N5/PPPP1PPP/R1BQKBNR b KQkq - 2 3");
    assert_eq!(evaluate(&a), evaluate(&m));
}

#[test]
fn centralized_knight_beats_rim_knight() {
    assert!(
        evaluate(&b("4k3/8/8/8/3N4/8/8/4K3 w - - 0 1"))
            > evaluate(&b("4k3/8/8/8/8/8/8/N3K3 w - - 0 1"))
    );
}

#[test]
fn endgame_king_prefers_the_centre() {
    assert!(
        evaluate(&b("7k/8/8/8/3K4/8/P7/8 w - - 0 1"))
            > evaluate(&b("7k/8/8/8/8/8/P7/K7 w - - 0 1"))
    );
}

#[test]
fn advanced_passed_pawn_is_worth_more() {
    assert!(
        evaluate(&b("4k3/P7/8/8/8/8/8/4K3 w - - 0 1"))
            > evaluate(&b("4k3/8/8/8/8/8/P7/4K3 w - - 0 1"))
    );
}
