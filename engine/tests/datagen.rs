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

mod command {
    use std::path::PathBuf;
    use std::process::Command;

    use vektr::datagen::{run, Settings, RECORD_BYTES};
    use vektr::nnue::Network;

    fn settings() -> Settings {
        Settings { nodes: 200, starts: vec![], start_share: 0 }
    }

    fn bytes(positions: u64, threads: usize, seed: u64) -> (Vec<u8>, vektr::datagen::Summary) {
        let mut out: Vec<u8> = Vec::new();
        let summary = run(&settings(), Network::embedded(), positions, threads, seed, &mut out).unwrap();
        (out, summary)
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("vektr-datagen-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn run_writes_exactly_the_requested_positions() {
        let (out, s) = bytes(200, 1, 1);
        assert_eq!(out.len(), 200 * RECORD_BYTES);
        assert_eq!(s.positions, 200);
        assert_eq!(s.white_wins + s.draws + s.black_wins, s.games);
    }

    #[test]
    fn single_thread_runs_repeat() {
        let (a, _) = bytes(100, 1, 5);
        let (b, _) = bytes(100, 1, 5);
        let (c, _) = bytes(100, 1, 6);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn threads_share_one_target() {
        let (out, _) = bytes(300, 3, 1);
        assert_eq!(out.len(), 300 * RECORD_BYTES);
        for rec in out.chunks(RECORD_BYTES) {
            let idx: Vec<u16> = (0..32).map(|i| u16::from_le_bytes([rec[2 * i], rec[2 * i + 1]])).collect();
            let pads = idx.iter().position(|&v| v == 768).unwrap_or(32);
            assert!(idx[..pads].windows(2).all(|w| w[0] < w[1]));
            assert!(idx[pads..].iter().all(|&v| v == 768));
            assert!(rec[66] <= 1);
            assert!(rec[67] <= 2);
        }
    }

    fn vektr() -> Command {
        Command::new(env!("CARGO_BIN_EXE_vektr"))
    }

    #[test]
    fn the_command_writes_the_file_and_leaves_no_part() {
        let dir = scratch("ok");
        let file = dir.join("a.vkd");
        let o = vektr()
            .args(["datagen", "--out", file.to_str().unwrap(), "--positions", "100", "--nodes", "200", "--seed", "7"])
            .output()
            .unwrap();
        assert_eq!(o.status.code(), Some(0));
        assert_eq!(std::fs::metadata(&file).unwrap().len(), 6800);
        assert!(!dir.join("a.vkd.part").exists());
        let stdout = String::from_utf8(o.stdout).unwrap();
        assert!(stdout.lines().last().unwrap().starts_with("datagen: "));
    }

    #[test]
    fn the_command_skips_bad_start_lines() {
        let dir = scratch("starts");
        let file = dir.join("a.vkd");
        let starts = dir.join("starts.txt");
        std::fs::write(&starts, "garbage\n4k3/8/8/8/8/8/8/QQQQK3 w - - 0 1\n").unwrap();
        let o = vektr()
            .args(["datagen", "--out", file.to_str().unwrap(), "--positions", "100", "--nodes", "200"])
            .args(["--starts", starts.to_str().unwrap()])
            .output()
            .unwrap();
        assert_eq!(o.status.code(), Some(0));
        assert_eq!(std::fs::metadata(&file).unwrap().len(), 6800);
    }

    #[test]
    fn a_missing_network_fails_without_an_output_file() {
        let dir = scratch("nonet");
        let file = dir.join("a.vkd");
        let none = dir.join("none.vknn");
        let o = vektr()
            .args(["datagen", "--out", file.to_str().unwrap(), "--positions", "100", "--net", none.to_str().unwrap()])
            .output()
            .unwrap();
        assert_eq!(o.status.code(), Some(1));
        assert!(!file.exists());
        assert!(!dir.join("a.vkd.part").exists());
    }

    #[test]
    fn missing_arguments_print_usage() {
        let o = vektr().arg("datagen").output().unwrap();
        assert_eq!(o.status.code(), Some(2));
        assert!(!o.stderr.is_empty());
    }
}
