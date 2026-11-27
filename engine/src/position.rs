//! A board together with the hashes of the positions that led to it.

use cozy_chess::{util, Board, Move, Piece};

#[derive(Clone)]
pub struct Position {
    pub board: Board,
    /// Hashes of every earlier position, oldest first. Excludes the current one.
    history: Vec<u64>,
}

impl Position {
    pub fn startpos() -> Position {
        Position { board: Board::default(), history: Vec::new() }
    }

    pub fn from_fen(fen: &str) -> Result<Position, String> {
        let board = Board::from_fen(fen.trim(), false).map_err(|e| format!("bad fen: {e}"))?;
        Ok(Position { board, history: Vec::new() })
    }

    /// Plays a move given in standard UCI notation (castling as `e1g1`).
    pub fn play_uci(&mut self, mv: &str) -> Result<(), String> {
        let parsed = util::parse_uci_move(&self.board, mv)
            .map_err(|e| format!("bad move {mv}: {e}"))?;
        if !self.board.is_legal(parsed) {
            return Err(format!("illegal move {mv}"));
        }
        self.play(parsed);
        Ok(())
    }

    /// Plays a legal move and records the position it left.
    pub fn play(&mut self, mv: Move) {
        self.history.push(self.board.hash());
        self.board.play_unchecked(mv);
    }

    /// Reverses one `play`, given the board as it was before that move.
    pub fn undo_to(&mut self, board: Board) {
        self.history.pop();
        self.board = board;
    }

    /// Draw by the fifty-move rule, insufficient material, or repetition.
    ///
    /// `ply` is how many of the most recent moves were made by the search. One
    /// earlier occurrence inside the search is treated as a draw; positions from
    /// the actual game must have occurred twice before.
    pub fn is_draw(&self, ply: usize) -> bool {
        if self.board.halfmove_clock() >= 100 || self.insufficient_material() {
            return true;
        }
        let hash = self.board.hash();
        let reach = (self.board.halfmove_clock() as usize).min(self.history.len());
        let mut seen = 0;
        let mut back = 2;
        while back <= reach {
            if self.history[self.history.len() - back] == hash {
                if back <= ply {
                    return true;
                }
                seen += 1;
                if seen == 2 {
                    return true;
                }
            }
            back += 2;
        }
        false
    }

    fn insufficient_material(&self) -> bool {
        let b = &self.board;
        if !(b.pieces(Piece::Pawn) | b.pieces(Piece::Rook) | b.pieces(Piece::Queen)).is_empty() {
            return false;
        }
        (b.pieces(Piece::Knight) | b.pieces(Piece::Bishop)).len() <= 1
    }

    /// Standard UCI notation for a move in this position.
    pub fn move_to_uci(&self, mv: Move) -> String {
        util::display_uci_move(&self.board, mv).to_string()
    }

    pub fn legal_moves(&self) -> Vec<Move> {
        let mut out = Vec::with_capacity(48);
        self.board.generate_moves(|moves| {
            out.extend(moves);
            false
        });
        out
    }
}

/// Counts leaf positions `depth` plies below `board`.
pub fn perft(board: &Board, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    let mut nodes = 0;
    board.generate_moves(|moves| {
        if depth == 1 {
            nodes += moves.len() as u64;
        } else {
            for mv in moves {
                let mut next = board.clone();
                next.play_unchecked(mv);
                nodes += perft(&next, depth - 1);
            }
        }
        false
    });
    nodes
}
