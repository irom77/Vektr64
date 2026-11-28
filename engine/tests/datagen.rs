use vektr::datagen::{encode, load_starts, play_game, Rng, Settings, RECORD_BYTES};
use vektr::search::Searcher;
use vektr::{MATE, MAX_PLY};

use cozy_chess::Board;

fn reference() -> String {
    format!("0c004401fc02{}dbff0100", "0003".repeat(29))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn board(fen: &str) -> Board {
    Board::from_fen(fen, false).unwrap()
}

fn settings(starts: Vec<String>, start_share: u32) -> Settings {
    Settings { nodes: 300, starts, start_share }
}

#[test]
fn encode_matches_the_defined_record() {
    let b = board("4k3/8/8/8/8/8/4P3/4K3 b - - 0 1");
    let rec = encode(&b, -37, 2);
    assert_eq!(rec.len(), RECORD_BYTES);
    assert_eq!(hex(&rec), reference());
}

#[test]
fn encode_clips_the_score_and_turns_the_result() {
    let w = board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1");
    let rec = encode(&w, 9999, 2);
    assert_eq!(i16::from_le_bytes([rec[64], rec[65]]), 3000);
    assert_eq!(rec[66], 0);
    assert_eq!(rec[67], 2);

    let b = board("4k3/8/8/8/8/8/4P3/4K3 b - - 0 1");
    let rec = encode(&b, -9999, 2);
    assert_eq!(i16::from_le_bytes([rec[64], rec[65]]), -3000);
    assert_eq!(rec[66], 1);
    assert_eq!(rec[67], 0);
    assert_eq!(encode(&b, 0, 1)[67], 1);
}

#[test]
fn rng_repeats_and_stays_in_range() {
    let mut a = Rng::new(7);
    let mut b = Rng::new(7);
    for _ in 0..100 {
        assert_eq!(a.next(), b.next());
    }
    let mut z = Rng::new(0);
    assert!((0..10).any(|_| z.next() != 0));
    let mut r = Rng::new(1);
    let mut seen = [false; 5];
    for _ in 0..1000 {
        let v = r.below(5);
        assert!(v < 5);
        seen[v] = true;
    }
    assert!(seen.iter().all(|&s| s));
}

#[test]
fn load_starts_keeps_only_playable_positions() {
    let text = "garbage\n\n4k3/8/8/8/8/8/4P3/4K3 w - - 0 1\n7k/5Q2/6K1/8/8/8/8/8 b - - 0 1\n";
    let starts = load_starts(text);
    assert_eq!(starts, vec!["4k3/8/8/8/8/8/4P3/4K3 w - - 0 1".to_string()]);
}

#[test]
fn a_game_gives_quiet_positions_and_a_result() {
    let s = settings(vec![], 0);
    let mut searcher = Searcher::new(1);
    let game = (1..=20u64)
        .find_map(|seed| play_game(&mut searcher, &mut Rng::new(seed), &s))
        .expect("some seed gives a game");
    assert!(game.result <= 2);
    assert!(!game.samples.is_empty());
    for sample in &game.samples {
        assert!(sample.board.checkers().is_empty());
        assert!(sample.score.abs() < MATE - MAX_PLY as i32);
    }
}

#[test]
fn the_same_seed_plays_the_same_game() {
    let s = settings(vec![], 0);
    let run = || {
        let mut searcher = Searcher::new(1);
        play_game(&mut searcher, &mut Rng::new(3), &s)
    };
    let (a, b) = (run(), run());
    match (a, b) {
        (Some(a), Some(b)) => {
            assert_eq!(a.result, b.result);
            assert_eq!(a.samples.len(), b.samples.len());
            for (x, y) in a.samples.iter().zip(&b.samples) {
                assert_eq!(encode(&x.board, x.score, a.result), encode(&y.board, y.score, b.result));
            }
        }
        (None, None) => {}
        _ => panic!("same seed, different outcome"),
    }
}

#[test]
fn a_crushing_start_is_adjudicated_as_a_win() {
    let s = settings(vec!["4k3/8/8/8/8/8/8/QQQQK3 w - - 0 1".to_string()], 100);
    let mut searcher = Searcher::new(1);
    let mut wins = 0;
    for seed in 1..=10u64 {
        if let Some(game) = play_game(&mut searcher, &mut Rng::new(seed), &s) {
            assert_ne!(game.result, 0, "seed {seed} lost");
            assert!(game.samples.len() < 400, "seed {seed} ran to the ply cap");
            if game.result == 2 {
                wins += 1;
            }
        }
    }
    assert!(wins >= 8, "only {wins} wins");
}

#[test]
fn unplayable_starts_give_no_game() {
    let s = settings(vec!["7k/5Q2/6K1/8/8/8/8/8 b - - 0 1".to_string()], 100);
    let mut searcher = Searcher::new(1);
    assert!(play_game(&mut searcher, &mut Rng::new(1), &s).is_none());
}
