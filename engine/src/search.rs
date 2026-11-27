//! Iterative-deepening alpha-beta search with quiescence.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use cozy_chess::{Board, Move, Piece, Square};

use crate::eval::evaluate;
use crate::ordering::History;
use crate::position::Position;
use crate::tt::{Bound, Entry, TT};
use crate::{INF, MATE, MAX_PLY};

/// What ends a search. Every field that is `None` places no limit.
#[derive(Clone, Debug, Default)]
pub struct Limits {
    /// Stop after completing this depth.
    pub depth: Option<u32>,
    /// Abort once this many nodes are searched.
    pub nodes: Option<u64>,
    /// Do not start a new iteration after this many milliseconds.
    pub soft_ms: Option<u64>,
    /// Abort mid-search after this many milliseconds.
    pub hard_ms: Option<u64>,
}

impl Limits {
    pub fn depth(d: u32) -> Limits {
        Limits { depth: Some(d), ..Limits::default() }
    }

    pub fn infinite() -> Limits {
        Limits::default()
    }
}

#[derive(Clone, Debug)]
pub struct SearchResult {
    /// `None` only when the position has no legal moves.
    pub best_move: Option<Move>,
    pub score: i32,
    /// Last fully completed depth.
    pub depth: u32,
    pub nodes: u64,
    pub pv: Vec<Move>,
}

const MAX_MOVES: usize = 256;
const NO_MOVE: Move = Move { from: Square::A1, to: Square::A1, promotion: None };

/// Fixed-capacity move list with a score per move, picked best-first.
pub(crate) struct MoveList {
    moves: [Move; MAX_MOVES],
    scores: [i32; MAX_MOVES],
    len: usize,
}

impl MoveList {
    fn new() -> MoveList {
        MoveList { moves: [NO_MOVE; MAX_MOVES], scores: [0; MAX_MOVES], len: 0 }
    }

    fn push(&mut self, mv: Move, score: i32) {
        self.moves[self.len] = mv;
        self.scores[self.len] = score;
        self.len += 1;
    }

    /// Moves the best remaining move into slot `i` and returns it.
    fn pick(&mut self, i: usize) -> Move {
        let mut best = i;
        for j in i + 1..self.len {
            if self.scores[j] > self.scores[best] {
                best = j;
            }
        }
        self.moves.swap(i, best);
        self.scores.swap(i, best);
        self.moves[i]
    }
}

/// True for a move that takes a piece, including en passant.
pub(crate) fn is_capture(board: &Board, mv: Move) -> bool {
    if board.colors(!board.side_to_move()).has(mv.to) {
        return true;
    }
    // A pawn changing file onto an empty square is an en passant capture.
    board.piece_on(mv.from) == Some(Piece::Pawn) && mv.from.file() != mv.to.file()
}

/// Most valuable victim first, least valuable attacker as the tie-break.
fn capture_score(board: &Board, mv: Move) -> i32 {
    let victim = board.piece_on(mv.to).unwrap_or(Piece::Pawn) as i32;
    let attacker = board.piece_on(mv.from).unwrap_or(Piece::Pawn) as i32;
    victim * 8 - attacker
}

pub struct Searcher {
    stop: Arc<AtomicBool>,
    nodes: u64,
    start: Instant,
    hard_ms: Option<u64>,
    node_limit: Option<u64>,
    root_depth: u32,
    aborted: bool,
    pv: Vec<Vec<Move>>,
    tt: TT,
    history: History,
}

impl Searcher {
    pub fn new(hash_mb: usize) -> Searcher {
        Searcher {
            stop: Arc::new(AtomicBool::new(false)),
            nodes: 0,
            start: Instant::now(),
            hard_ms: None,
            node_limit: None,
            root_depth: 0,
            aborted: false,
            pv: (0..=MAX_PLY).map(|_| Vec::with_capacity(MAX_PLY)).collect(),
            tt: TT::new(hash_mb),
            history: History::new(),
        }
    }

    /// Setting this flag ends the running search. The caller clears it before
    /// starting the next one.
    pub fn stop_flag(&self) -> Arc<AtomicBool> {
        self.stop.clone()
    }

    /// Forgets everything learned from earlier positions.
    pub fn new_game(&mut self) {
        self.tt.clear();
        self.history.clear();
    }

    /// Searches `pos`, calling `on_iter` with the result and elapsed
    /// milliseconds after each completed depth.
    pub fn search(
        &mut self,
        pos: &mut Position,
        limits: &Limits,
        on_iter: &mut dyn FnMut(&SearchResult, u64),
    ) -> SearchResult {
        self.start = Instant::now();
        self.nodes = 0;
        self.aborted = false;
        self.hard_ms = limits.hard_ms;
        self.node_limit = limits.nodes;

        let in_check = !pos.board.checkers().is_empty();
        let mut result = SearchResult {
            best_move: None,
            score: if in_check { -MATE } else { 0 },
            depth: 0,
            nodes: 0,
            pv: Vec::new(),
        };
        if pos.legal_moves().is_empty() {
            return result;
        }

        let max_depth = limits.depth.unwrap_or(MAX_PLY as u32 - 1).clamp(1, MAX_PLY as u32 - 1);
        for depth in 1..=max_depth {
            self.root_depth = depth;
            let score = self.negamax(pos, depth as i32, 0, -INF, INF);
            if self.aborted {
                break;
            }
            result = SearchResult {
                best_move: self.pv[0].first().copied(),
                score,
                depth,
                nodes: self.nodes,
                pv: self.pv[0].clone(),
            };
            let elapsed = self.start.elapsed().as_millis() as u64;
            on_iter(&result, elapsed);
            if limits.soft_ms.is_some_and(|soft| elapsed >= soft) {
                break;
            }
        }
        result.nodes = self.nodes;
        result
    }

    /// Polls the limits. The first iteration is never interrupted, so a search
    /// always has a move to return.
    fn should_abort(&mut self) -> bool {
        if self.aborted {
            return true;
        }
        if self.root_depth > 1 && self.nodes & 2047 == 0 {
            let out_of_time =
                self.hard_ms.is_some_and(|hard| self.start.elapsed().as_millis() as u64 >= hard);
            let out_of_nodes = self.node_limit.is_some_and(|limit| self.nodes >= limit);
            if out_of_time || out_of_nodes || self.stop.load(Ordering::Relaxed) {
                self.aborted = true;
            }
        }
        self.aborted
    }

    fn negamax(&mut self, pos: &mut Position, mut depth: i32, ply: usize, mut alpha: i32, beta: i32) -> i32 {
        self.pv[ply].clear();
        if ply > 0 && pos.is_draw(ply) {
            return 0;
        }
        if ply >= MAX_PLY - 1 {
            return evaluate(&pos.board);
        }
        let in_check = !pos.board.checkers().is_empty();
        if in_check {
            depth += 1;
        }
        if depth <= 0 {
            return self.quiescence(pos, ply, alpha, beta);
        }
        self.nodes += 1;
        if self.should_abort() {
            return 0;
        }

        let board = pos.board.clone();
        let hash = board.hash();
        let mut tt_move = None;
        if let Some(entry) = self.tt.probe(hash, ply) {
            tt_move = entry.mv;
            if ply > 0 && entry.depth as i32 >= depth {
                let usable = match entry.bound {
                    Bound::Exact => true,
                    Bound::Lower => entry.score >= beta,
                    Bound::Upper => entry.score <= alpha,
                };
                if usable {
                    return entry.score;
                }
            }
        }

        let mut list = MoveList::new();
        board.generate_moves(|moves| {
            for mv in moves {
                list.push(mv, self.history.score(&board, mv, tt_move, ply));
            }
            false
        });
        if list.len == 0 {
            return if in_check { -(MATE - ply as i32) } else { 0 };
        }

        let alpha_start = alpha;
        let mut best_move = None;
        let mut best = -INF;
        for i in 0..list.len {
            let mv = list.pick(i);
            pos.play(mv);
            let score = -self.negamax(pos, depth - 1, ply + 1, -beta, -alpha);
            pos.undo_to(board.clone());
            if self.aborted {
                return 0;
            }
            if score > best {
                best = score;
                if score > alpha {
                    alpha = score;
                    let (head, tail) = self.pv.split_at_mut(ply + 1);
                    let line = &mut head[ply];
                    line.clear();
                    line.push(mv);
                    line.extend_from_slice(&tail[0]);
                    best_move = Some(mv);
                    if score >= beta {
                        self.history.on_cutoff(&board, mv, ply, depth as u32);
                        break;
                    }
                }
            }
        }

        let bound = if best >= beta {
            Bound::Lower
        } else if best > alpha_start {
            Bound::Exact
        } else {
            Bound::Upper
        };
        self.tt.store(hash, ply, Entry { mv: best_move, score: best, depth: depth.min(255) as u8, bound });
        best
    }

    /// Resolves captures so the static evaluation is only trusted in quiet positions.
    fn quiescence(&mut self, pos: &mut Position, ply: usize, mut alpha: i32, beta: i32) -> i32 {
        self.pv[ply].clear();
        self.nodes += 1;
        if self.should_abort() {
            return 0;
        }
        if ply >= MAX_PLY - 1 {
            return evaluate(&pos.board);
        }

        let board = pos.board.clone();
        let in_check = !board.checkers().is_empty();
        let mut best = -INF;
        if !in_check {
            best = evaluate(&board);
            if best >= beta {
                return best;
            }
            alpha = alpha.max(best);
        }

        let mut list = MoveList::new();
        let mut any_legal = false;
        board.generate_moves(|moves| {
            for mv in moves {
                any_legal = true;
                let capture = is_capture(&board, mv);
                let queening = mv.promotion == Some(Piece::Queen);
                if in_check || capture || queening {
                    let score = if capture { 1000 + capture_score(&board, mv) } else { 0 };
                    list.push(mv, score + if queening { 500 } else { 0 });
                }
            }
            false
        });
        if in_check && !any_legal {
            return -(MATE - ply as i32);
        }

        for i in 0..list.len {
            let mv = list.pick(i);
            pos.play(mv);
            let score = -self.quiescence(pos, ply + 1, -beta, -alpha);
            pos.undo_to(board.clone());
            if self.aborted {
                return 0;
            }
            if score > best {
                best = score;
                if score > alpha {
                    alpha = score;
                    if score >= beta {
                        break;
                    }
                }
            }
        }
        best
    }
}
