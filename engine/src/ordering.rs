//! Move ordering: which moves to try first so cutoffs come early.

use cozy_chess::{Board, Move, Piece};

use crate::search::is_capture;
use crate::MAX_PLY;

const TT_MOVE: i32 = 1_000_000;
const CAPTURE: i32 = 100_000;
const QUEEN_PROMOTION: i32 = 90_000;
const KILLER: [i32; 2] = [80_000, 79_000];
const HISTORY_MAX: i32 = 70_000;

/// What earlier parts of the search learned about quiet moves.
pub struct History {
    /// Two quiet moves per ply that recently caused a cutoff there.
    killers: [[Option<Move>; 2]; MAX_PLY],
    /// Cutoff counts by side to move, from-square and to-square.
    history: Box<[[[i32; 64]; 64]; 2]>,
}

impl Default for History {
    fn default() -> Self {
        History::new()
    }
}

impl History {
    pub fn new() -> History {
        History { killers: [[None; 2]; MAX_PLY], history: Box::new([[[0; 64]; 64]; 2]) }
    }

    pub fn clear(&mut self) {
        *self = History::new();
    }

    /// Higher is searched earlier: table move, captures by most valuable victim
    /// and least valuable attacker, queen promotions, killers, then history.
    pub fn score(&self, board: &Board, mv: Move, tt_move: Option<Move>, ply: usize) -> i32 {
        if tt_move == Some(mv) {
            return TT_MOVE;
        }
        let queening = if mv.promotion == Some(Piece::Queen) { QUEEN_PROMOTION } else { 0 };
        if is_capture(board, mv) {
            let victim = board.piece_on(mv.to).unwrap_or(Piece::Pawn) as i32;
            let attacker = board.piece_on(mv.from).unwrap_or(Piece::Pawn) as i32;
            return CAPTURE + queening + victim * 8 - attacker;
        }
        if queening != 0 {
            return queening;
        }
        if ply < MAX_PLY {
            for (slot, bonus) in self.killers[ply].iter().zip(KILLER) {
                if *slot == Some(mv) {
                    return bonus;
                }
            }
        }
        self.history[board.side_to_move() as usize][mv.from as usize][mv.to as usize]
    }

    /// Records a move that caused a beta cutoff. Only quiet moves are remembered.
    pub fn on_cutoff(&mut self, board: &Board, mv: Move, ply: usize, depth: u32) {
        if is_capture(board, mv) || mv.promotion.is_some() {
            return;
        }
        if ply < MAX_PLY && self.killers[ply][0] != Some(mv) {
            self.killers[ply][1] = self.killers[ply][0];
            self.killers[ply][0] = Some(mv);
        }
        let cell = &mut self.history[board.side_to_move() as usize][mv.from as usize][mv.to as usize];
        *cell = (*cell + (depth * depth) as i32).min(HISTORY_MAX);
    }
}
