use std::io::{Read, Write};
use std::process::{Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

/// Runs the engine, feeds it `script`, waits `wait_ms`, then closes its input.
/// Fails the test if the engine has not exited three seconds later.
fn session(script: &str, wait_ms: u64) -> (String, ExitStatus) {
    let mut child = Command::new(env!("CARGO_BIN_EXE_vektr"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = child.stdout.take().unwrap();
    let reader = std::thread::spawn(move || {
        let mut out = String::new();
        stdout.read_to_string(&mut out).unwrap();
        out
    });
    // The engine may already have quit; a broken pipe here is not a failure.
    let _ = stdin.write_all(script.as_bytes());
    let _ = stdin.flush();
    std::thread::sleep(Duration::from_millis(wait_ms));
    drop(stdin);
    let deadline = Instant::now() + Duration::from_secs(3);
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if Instant::now() > deadline {
            child.kill().unwrap();
            panic!("engine did not exit within 3 seconds of its input closing");
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    (reader.join().unwrap(), status)
}

#[test]
fn handshake() {
    let (out, st) = session("uci\nisready\nquit\n", 0);
    assert!(
        out.contains("id name Vektr64")
            && out.contains("option name Hash")
            && out.contains("option name Move Overhead")
            && out.contains("uciok")
            && out.contains("readyok")
    );
    assert!(st.success());
}

#[test]
fn plays_a_legal_reply_with_castling_in_the_move_list() {
    let (out, _) =
        session("position startpos moves e2e4 e7e5 g1f3 b8c6 f1c4 f8c5 e1g1\ngo depth 4\n", 500);
    let bm = out.lines().rev().find(|l| l.starts_with("bestmove ")).unwrap();
    let mv = bm.split(' ').nth(1).unwrap();
    let mut p = vektr::position::Position::startpos();
    for m in ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "e1g1"] {
        p.play_uci(m).unwrap();
    }
    assert!(p.play_uci(mv).is_ok());
}

#[test]
fn reports_mate_in_moves() {
    let (out, _) = session("position fen 6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1\ngo depth 4\n", 500);
    assert!(out.contains("score mate 1") && out.contains("bestmove a1a8"));
}

#[test]
fn negative_and_tiny_clocks_still_answer() {
    for go in ["go wtime -20 btime 5000 winc 0 binc 0", "go wtime 3 btime 3", "go movetime 1"] {
        let (out, _) = session(&format!("position startpos\n{go}\n"), 300);
        assert!(out.lines().any(|l| l.starts_with("bestmove ") && !l.contains("0000")), "{go}");
    }
}

#[test]
fn black_uses_its_own_clock() {
    let t = Instant::now();
    let (out, _) =
        session("position startpos moves e2e4\ngo wtime 600000 btime 400 winc 0 binc 0\n", 300);
    assert!(out.contains("bestmove "));
    assert!(t.elapsed().as_millis() < 2000);
}

#[test]
fn stop_ends_an_infinite_search() {
    let (out, _) = session("position startpos\ngo infinite\nisready\nstop\n", 300);
    assert!(out.contains("readyok") && out.contains("bestmove "));
}

#[test]
fn closing_stdin_mid_search_exits() {
    let (_, st) = session("position startpos\ngo infinite\n", 200); // helper enforces exit within 3 s
    assert!(st.success());
}

#[test]
fn quit_mid_search_exits() {
    let (_, st) = session("position startpos\ngo infinite\nquit\n", 100);
    assert!(st.success());
}

#[test]
fn bad_input_does_not_crash() {
    let (out, st) = session(
        "position fen garbage\nposition startpos moves e2e5\nnonsense\nsetoption name Nope value 1\ngo depth 2\n",
        300,
    );
    assert!(out.contains("info string error") && out.contains("bestmove ") && st.success());
}

#[test]
fn checkmated_position_answers_null_move() {
    let (out, _) = session("position fen R5k1/5ppp/8/8/8/8/8/4K3 b - - 0 1\ngo depth 2\n", 300);
    assert!(out.contains("bestmove 0000"));
}

#[test]
fn hash_option_is_applied() {
    let (out, st) =
        session("setoption name Hash value 8\nisready\nposition startpos\ngo depth 5\n", 500);
    assert!(out.contains("readyok") && out.contains("bestmove ") && st.success());
}

#[test]
fn bench_is_deterministic() {
    let run = || {
        String::from_utf8(
            Command::new(env!("CARGO_BIN_EXE_vektr")).args(["bench", "6"]).output().unwrap().stdout,
        )
        .unwrap()
    };
    let (a, b) = (run(), run());
    let nodes = |s: &str| s.lines().last().unwrap().split(' ').next().unwrap().to_string();
    assert!(a.lines().last().unwrap().ends_with(" nps"));
    assert_eq!(nodes(&a), nodes(&b));
}

fn last_cp(out: &str) -> i32 {
    let line = out.lines().rev().find(|l| l.starts_with("info depth") && l.contains(" score cp ")).unwrap();
    let mut it = line.split(' ').skip_while(|&t| t != "cp");
    it.nth(1).unwrap().parse().unwrap()
}

/// The embedded network with its output bias raised by 255 * 64, which moves every
/// evaluation (from the side to move's view) up by 400 centipawns.
fn raised_net_file(tag: &str) -> String {
    let mut bytes = vektr::nnue::DEFAULT_NET.to_vec();
    let n = bytes.len();
    let bias = i32::from_le_bytes(bytes[n - 4..].try_into().unwrap()) + 255 * 64;
    bytes[n - 4..].copy_from_slice(&bias.to_le_bytes());
    let path = std::env::temp_dir().join(format!("vektr-test-{}-{tag}.vknn", std::process::id()));
    std::fs::write(&path, bytes).unwrap();
    path.to_str().unwrap().to_string()
}

#[test]
fn evalfile_is_advertised() {
    let (out, _) = session("uci\nquit\n", 0);
    assert!(out.contains("option name EvalFile type string"));
}

#[test]
fn a_missing_evalfile_is_reported_and_the_engine_still_moves() {
    let (out, _) = session(
        "setoption name EvalFile value /no/such/file\nposition startpos\ngo depth 2\n",
        500,
    );
    assert!(out.lines().any(|l| l.starts_with("info string error")));
    assert!(out.lines().any(|l| l.starts_with("bestmove ") && !l.contains("0000")));
}

#[test]
fn a_truncated_evalfile_is_reported_and_the_engine_still_moves() {
    let path = std::env::temp_dir().join(format!("vektr-test-{}-trunc.vknn", std::process::id()));
    std::fs::write(&path, &vektr::nnue::DEFAULT_NET[..1000]).unwrap();
    let (out, _) = session(
        &format!("setoption name EvalFile value {}\nposition startpos\ngo depth 2\n", path.display()),
        500,
    );
    let _ = std::fs::remove_file(&path);
    assert!(out.lines().any(|l| l.starts_with("info string error")));
    assert!(out.lines().any(|l| l.starts_with("bestmove ") && !l.contains("0000")));
}

// Depth 2 rather than 1: the leaf after two plies has the root's side to move, so the raised
// bias raises the root score; at depth 1 the leaf belongs to the opponent and it would lower it.
#[test]
fn a_loaded_evalfile_changes_the_score() {
    let (base, _) = session("position startpos\ngo depth 2\n", 500);
    let path = raised_net_file("score");
    let (raised, _) = session(
        &format!("setoption name EvalFile value {path}\nposition startpos\ngo depth 2\n"),
        500,
    );
    let _ = std::fs::remove_file(&path);
    let diff = last_cp(&raised) - last_cp(&base);
    assert!((350..=450).contains(&diff), "score moved by {diff}");
}

#[test]
fn the_loaded_network_survives_a_hash_change() {
    let (base, _) = session("position startpos\ngo depth 2\n", 500);
    let path = raised_net_file("hash");
    let (raised, _) = session(
        &format!(
            "setoption name EvalFile value {path}\nsetoption name Hash value 8\nposition startpos\ngo depth 2\n"
        ),
        500,
    );
    let _ = std::fs::remove_file(&path);
    let diff = last_cp(&raised) - last_cp(&base);
    assert!((350..=450).contains(&diff), "score moved by {diff}");
}
