use cozy_chess::{Board, Color, Move, Piece, Square};
use vektr::nnue::*;

const SPECIALS: [&str; 4] = [
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
    "1n2k3/P7/8/8/8/8/7p/4K1N1 w - - 0 1",
    "8/8/8/4k3/8/8/8/4K3 w - - 0 1",
];

fn legal(board: &Board) -> Vec<Move> {
    let mut v = Vec::new();
    board.generate_moves(|ml| {
        v.extend(ml);
        false
    });
    v
}

#[derive(Default)]
struct Seen {
    castle_short: bool,
    castle_long: bool,
    en_passant: bool,
    capture_promotion: bool,
    promotion: bool,
}

/// Walk 200 seeded random games from `fen`, checking every incremental update.
fn walk(fen: &str, net: &Network, seen: &mut Seen) {
    for seed in 1..=200u64 {
        let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
        let mut board = Board::from_fen(fen, false).unwrap();
        let mut acc = Accumulator::refresh(net, &board);
        for _ in 0..150 {
            let moves = legal(&board);
            if moves.is_empty() {
                break;
            }
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let mv = moves[(state % moves.len() as u64) as usize];
            let mover = board.side_to_move();
            let piece = board.piece_on(mv.from).unwrap();
            let target_empty = board.piece_on(mv.to).is_none();
            if piece == Piece::King && board.colors(mover).has(mv.to) {
                if mv.to.file() as usize > mv.from.file() as usize {
                    seen.castle_short = true;
                } else {
                    seen.castle_long = true;
                }
            }
            if piece == Piece::Pawn && target_empty && mv.from.file() != mv.to.file() {
                seen.en_passant = true;
            }
            if mv.promotion.is_some() {
                seen.promotion = true;
                if !target_empty {
                    seen.capture_promotion = true;
                }
            }
            let mut after = board.clone();
            after.play_unchecked(mv);
            let mut next = acc.clone();
            next.update_from(&acc, net, &board, &after);
            let fresh = Accumulator::refresh(net, &after);
            assert_eq!(next.white, fresh.white, "white half after {mv} from {board}");
            assert_eq!(next.black, fresh.black, "black half after {mv} from {board}");
            board = after;
            acc = next;
        }
    }
}

#[test]
fn embedded_network_agrees_with_the_trainer() {
    assert_eq!(netcheck(&Network::embedded(), DEFAULT_CHECK), Ok(256));
}

#[test]
fn incremental_update_equals_refresh_over_random_games() {
    let net = Network::embedded();
    let mut seen = Seen::default();
    walk(&Board::default().to_string(), &net, &mut seen);
    for fen in SPECIALS {
        walk(fen, &net, &mut seen);
    }
}

#[test]
fn each_special_move_is_actually_played() {
    let net = Network::embedded();
    let mut seen = Seen::default();
    for fen in SPECIALS {
        walk(fen, &net, &mut seen);
    }
    assert!(seen.castle_short, "O-O");
    assert!(seen.castle_long, "O-O-O");
    assert!(seen.en_passant, "e5xd6 e.p.");
    assert!(seen.promotion, "promotion");
    assert!(seen.capture_promotion, "a7xb8=Q");
    // and the named moves exist in their positions
    let b = Board::from_fen(SPECIALS[2], false).unwrap();
    assert!(legal(&b).iter().any(|m| m.from == Square::A7 && m.to == Square::B8 && m.promotion == Some(Piece::Queen)));
    let b = Board::from_fen(SPECIALS[1], false).unwrap();
    assert!(legal(&b).iter().any(|m| m.from == Square::E5 && m.to == Square::D6));
}

#[test]
fn simd_and_scalar_give_the_same_score() {
    let net = Network::embedded();
    for line in DEFAULT_CHECK.lines() {
        let fen = line.rsplit_once(';').unwrap().0;
        let board = parse_fen(fen).unwrap();
        let acc = Accumulator::refresh(&net, &board);
        for stm in [Color::White, Color::Black] {
            assert_eq!(net.evaluate(&acc, stm), net.evaluate_scalar(&acc, stm), "{fen}");
        }
    }
    let mut state = 0x1234_5678_9abc_def1u64;
    let mut acc = Accumulator::refresh(&net, &Board::default());
    for _ in 0..10_000 {
        for i in 0..HIDDEN {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            acc.white[i] = (state % 900) as i16 - 300;
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            acc.black[i] = (state % 900) as i16 - 300;
        }
        for stm in [Color::White, Color::Black] {
            assert_eq!(net.evaluate(&acc, stm), net.evaluate_scalar(&acc, stm));
        }
    }
}

#[test]
fn colour_mirrored_positions_score_the_same() {
    let net = Network::embedded();
    let a = Board::from_fen("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", false).unwrap();
    let m = Board::from_fen("rnbqkb1r/pppp1ppp/5n2/4p3/4P3/2N5/PPPP1PPP/R1BQKBNR b KQkq - 2 3", false).unwrap();
    assert_eq!(net.evaluate_board(&a), net.evaluate_board(&m));
}

#[test]
fn bad_network_bytes_are_errors_not_panics() {
    let good = DEFAULT_NET.to_vec();
    assert!(Network::from_bytes(&good).is_ok());
    let mut cases: Vec<(&str, Vec<u8>)> = vec![
        ("empty", vec![]),
        ("100 bytes", good[..100].to_vec()),
        ("one byte short", good[..good.len() - 1].to_vec()),
    ];
    let mut long = good.clone();
    long.push(0);
    cases.push(("one byte long", long));
    let mut m = good.clone();
    m[0] = b'X';
    cases.push(("wrong magic", m));
    let mut m = good.clone();
    m[4..8].copy_from_slice(&2u32.to_le_bytes());
    cases.push(("version 2", m));
    let mut m = good.clone();
    m[8..12].copy_from_slice(&256u32.to_le_bytes());
    cases.push(("hidden 256", m));
    let mut m = good.clone();
    m[12..14].copy_from_slice(&506i16.to_le_bytes());
    cases.push(("ft weight 506", m));
    let mut m = good.clone();
    let out_w = 12 + 768 * 128 * 2 + 128 * 2;
    m[out_w..out_w + 2].copy_from_slice(&128i16.to_le_bytes());
    cases.push(("output weight 128", m));
    for (name, bytes) in cases {
        assert!(Network::from_bytes(&bytes).is_err(), "{name} must be an error");
    }
}

#[test]
fn netcheck_reports_a_disagreement() {
    assert!(netcheck(&Network::embedded(), "8/8/8/4k3/8/8/8/4K3 w - -;12345").is_err());
}

#[test]
fn netcheck_with_no_positions_is_an_error() {
    for check in ["", "\n  \n"] {
        let err = netcheck(&Network::embedded(), check).unwrap_err();
        assert!(err.iter().any(|l| l.contains("check file has no positions")), "{err:?}");
    }
}

#[test]
fn the_embedded_network_knows_material() {
    let net = Network::embedded();
    let no_queen = parse_fen("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
    assert!(net.evaluate_board(&no_queen) > 300);
    assert!(net.evaluate_board(&Board::default()).abs() < 150);
}
