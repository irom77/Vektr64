//! The UCI protocol loop: text commands in, moves out.

use std::collections::VecDeque;
use std::io::{BufRead, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};

use cozy_chess::Color;

use crate::nnue::Network;

use crate::position::Position;
use crate::search::{Limits, SearchResult, Searcher};
use crate::time::{self, Clock};
use crate::{MATE, MAX_PLY};

const NAME: &str = concat!("Vektr64 ", env!("CARGO_PKG_VERSION"));
const AUTHOR: &str = "Irek Romaniuk";
const DEFAULT_HASH_MB: usize = 64;
const DEFAULT_OVERHEAD_MS: i64 = 100;

enum Msg {
    Line(String),
    /// Input closed.
    Eof,
    /// The search thread finished and hands the searcher back.
    Done(Box<Searcher>),
}

type Out = Arc<Mutex<Box<dyn Write + Send>>>;

fn say(out: &Out, line: &str) {
    let mut out = out.lock().unwrap();
    // The GUI going away is not something the engine can report or recover from.
    let _ = writeln!(out, "{line}");
    let _ = out.flush();
}

/// `cp X` for ordinary scores, `mate N` (in moves) for forced mates.
fn format_score(score: i32) -> String {
    if score.abs() >= MATE - MAX_PLY as i32 {
        let moves = (MATE - score.abs() + 1) / 2;
        format!("mate {}", if score > 0 { moves } else { -moves })
    } else {
        format!("cp {score}")
    }
}

fn info_line(pos: &Position, r: &SearchResult, ms: u64) -> String {
    let mut line = format!(
        "info depth {} score {} nodes {} nps {} time {} pv",
        r.depth,
        format_score(r.score),
        r.nodes,
        r.nodes * 1000 / ms.max(1),
        ms
    );
    let mut walk = pos.clone();
    for &mv in &r.pv {
        line.push(' ');
        line.push_str(&walk.move_to_uci(mv));
        walk.play(mv);
    }
    line
}

struct Engine {
    out: Out,
    tx: Sender<Msg>,
    rx: Receiver<Msg>,
    /// Lines that arrived while waiting for a search to finish.
    backlog: VecDeque<String>,
    pos: Position,
    /// `None` while the search thread has it.
    searcher: Option<Box<Searcher>>,
    stop: Arc<AtomicBool>,
    overhead_ms: i64,
    net: Arc<Network>,
    input_closed: bool,
}

impl Engine {
    /// Blocks until the search thread, if any, hands the searcher back.
    fn wait_for_search(&mut self) {
        while self.searcher.is_none() {
            match self.rx.recv() {
                Ok(Msg::Done(s)) => self.searcher = Some(s),
                Ok(Msg::Line(l)) => self.backlog.push_back(l),
                Ok(Msg::Eof) | Err(_) => {
                    self.input_closed = true;
                    self.stop.store(true, Ordering::Relaxed);
                }
            }
        }
    }

    fn stop_search(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        self.wait_for_search();
    }

    fn set_position(&mut self, args: &[&str]) -> Result<(), String> {
        let moves_at = args.iter().position(|&t| t == "moves").unwrap_or(args.len());
        let mut pos = match args.first() {
            Some(&"startpos") => Position::startpos(),
            Some(&"fen") => Position::from_fen(&args[1..moves_at].join(" "))?,
            _ => return Err("position needs 'startpos' or 'fen'".to_string()),
        };
        for mv in args.iter().skip(moves_at + 1) {
            pos.play_uci(mv)?;
        }
        self.pos = pos;
        Ok(())
    }

    fn set_option(&mut self, args: &[&str]) {
        let value_at = args.iter().position(|&t| t == "value").unwrap_or(args.len());
        let name = args.get(1..value_at).unwrap_or(&[]).join(" ").to_lowercase();
        let value = args.get(value_at + 1..).unwrap_or(&[]).join(" ");
        match name.as_str() {
            "hash" => {
                if let Ok(mb) = value.parse::<usize>() {
                    let mut searcher = Searcher::new(mb.clamp(1, 4096));
                    searcher.set_network(self.net.clone());
                    self.stop = searcher.stop_flag();
                    self.searcher = Some(Box::new(searcher));
                }
            }
            "evalfile" => match Network::load(&value) {
                Ok(net) => {
                    let net = Arc::new(net);
                    if let Some(s) = self.searcher.as_mut() {
                        s.set_network(net.clone());
                    }
                    self.net = net;
                }
                Err(e) => say(&self.out, &format!("info string error: {e}")),
            },
            "move overhead" => {
                if let Ok(ms) = value.parse::<i64>() {
                    self.overhead_ms = ms.clamp(0, 5000);
                }
            }
            _ => {}
        }
    }

    fn limits(&self, args: &[&str]) -> Limits {
        let mut limits = Limits::infinite();
        let (mut wtime, mut btime, mut winc, mut binc) = (None, None, 0, 0);
        let (mut moves_to_go, mut movetime) = (None, None);
        let mut it = args.iter();
        while let Some(&key) = it.next() {
            let mut num = || it.next().and_then(|v| v.parse::<i64>().ok());
            match key {
                "wtime" => wtime = num(),
                "btime" => btime = num(),
                "winc" => winc = num().unwrap_or(0),
                "binc" => binc = num().unwrap_or(0),
                "movestogo" => moves_to_go = num().map(|n| n.clamp(1, 1000) as u32),
                "movetime" => movetime = num(),
                "depth" => limits.depth = num().map(|n| n.clamp(1, MAX_PLY as i64) as u32),
                "nodes" => limits.nodes = num().map(|n| n.max(1) as u64),
                _ => {}
            }
        }
        let (remaining, inc) = match self.pos.board.side_to_move() {
            Color::White => (wtime, winc),
            Color::Black => (btime, binc),
        };
        let budget = if let Some(ms) = movetime {
            Some(time::fixed(ms, self.overhead_ms))
        } else {
            remaining.map(|remaining_ms| {
                time::allocate(&Clock { remaining_ms, inc_ms: inc, moves_to_go }, self.overhead_ms)
            })
        };
        if let Some((soft, hard)) = budget {
            limits.soft_ms = Some(soft);
            limits.hard_ms = Some(hard);
        }
        limits
    }

    fn go(&mut self, args: &[&str]) {
        let limits = self.limits(args);
        let mut searcher = self.searcher.take().expect("go is only called when idle");
        let mut pos = self.pos.clone();
        let out = self.out.clone();
        let tx = self.tx.clone();
        // Cleared here, not in the search thread, so a `stop` sent right after
        // `go` cannot be lost.
        self.stop.store(false, Ordering::Relaxed);
        std::thread::spawn(move || {
            let root = pos.clone();
            let result =
                searcher.search(&mut pos, &limits, &mut |r, ms| say(&out, &info_line(&root, r, ms)));
            let best = result.best_move.map_or("0000".to_string(), |mv| root.move_to_uci(mv));
            say(&out, &format!("bestmove {best}"));
            let _ = tx.send(Msg::Done(searcher));
        });
    }

    /// Handles one command. Returns false when the engine should exit.
    fn handle(&mut self, line: &str) -> bool {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        let Some(&command) = tokens.first() else { return true };
        let args = &tokens[1..];
        match command {
            "uci" => {
                say(&self.out, &format!("id name {NAME}"));
                say(&self.out, &format!("id author {AUTHOR}"));
                say(
                    &self.out,
                    &format!("option name Hash type spin default {DEFAULT_HASH_MB} min 1 max 4096"),
                );
                say(
                    &self.out,
                    &format!(
                        "option name Move Overhead type spin default {DEFAULT_OVERHEAD_MS} min 0 max 5000"
                    ),
                );
                say(&self.out, "option name EvalFile type string default <embedded>");
                say(&self.out, "option name Threads type spin default 1 min 1 max 1");
                say(&self.out, "uciok");
            }
            "isready" => say(&self.out, "readyok"),
            "stop" => self.stop_search(),
            "quit" => {
                self.stop_search();
                return false;
            }
            "ucinewgame" => {
                self.stop_search();
                if let Some(s) = self.searcher.as_mut() {
                    s.new_game();
                }
            }
            "setoption" => {
                self.stop_search();
                self.set_option(args);
            }
            "position" => {
                self.stop_search();
                if let Err(e) = self.set_position(args) {
                    say(&self.out, &format!("info string error: {e}"));
                }
            }
            "go" => {
                self.stop_search();
                self.go(args);
            }
            _ => say(&self.out, &format!("info string error: unknown command {command}")),
        }
        true
    }
}

/// Runs the UCI loop until `quit` is read or the input ends.
pub fn run(input: impl BufRead + Send + 'static, output: impl Write + Send + 'static) {
    let (tx, rx) = channel();
    let reader_tx = tx.clone();
    std::thread::spawn(move || {
        for line in input.lines() {
            let Ok(line) = line else { break };
            if reader_tx.send(Msg::Line(line)).is_err() {
                return;
            }
        }
        let _ = reader_tx.send(Msg::Eof);
    });

    let searcher = Searcher::new(DEFAULT_HASH_MB);
    let mut engine = Engine {
        out: Arc::new(Mutex::new(Box::new(output))),
        tx,
        rx,
        backlog: VecDeque::new(),
        pos: Position::startpos(),
        stop: searcher.stop_flag(),
        searcher: Some(Box::new(searcher)),
        overhead_ms: DEFAULT_OVERHEAD_MS,
        net: Network::embedded(),
        input_closed: false,
    };

    loop {
        let line = if let Some(line) = engine.backlog.pop_front() {
            line
        } else if engine.input_closed {
            break;
        } else {
            match engine.rx.recv() {
                Ok(Msg::Line(line)) => line,
                Ok(Msg::Done(s)) => {
                    engine.searcher = Some(s);
                    continue;
                }
                Ok(Msg::Eof) | Err(_) => break,
            }
        };
        if !engine.handle(&line) {
            return;
        }
    }
    engine.stop_search();
}
