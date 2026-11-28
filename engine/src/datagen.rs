//! Self-play games at a fixed node count, and their encoding as training records.
//!
//! One record is 68 bytes, little-endian: 32 x u16 White-view feature indices in
//! ascending order padded with 768, an i16 score for the side to move (clipped to
//! +-3000), a u8 side to move (0 white, 1 black) and a u8 result for the side to
//! move (0 loss, 1 draw, 2 win).

use cozy_chess::{Board, Color, Piece};

use crate::nnue::features;
use crate::position::Position;
use crate::search::{Limits, Searcher};
use crate::{MATE, MAX_PLY};

pub const RECORD_BYTES: usize = 68;

const SCORE_CLIP: i32 = 3000;
const PAD: u16 = 768;
const MAX_GAME_PLIES: usize = 400;
const OPENING_SCORE_LIMIT: i32 = 600;
const ADJUDICATE_SCORE: i32 = 2000;
const ADJUDICATE_SEARCHES: u32 = 5;

/// splitmix64.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed)
    }

    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A value in `0..n`; `n` must be positive.
    pub fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

pub struct Settings {
    /// Nodes per move.
    pub nodes: u64,
    /// FENs accepted by `load_starts`.
    pub starts: Vec<String>,
    /// Percent of games that begin from `starts`.
    pub start_share: u32,
}

/// A quiet position and the search score for its side to move.
pub struct Sample {
    pub board: Board,
    pub score: i32,
}

pub struct Game {
    pub samples: Vec<Sample>,
    /// White's view: 0 loss, 1 draw, 2 win.
    pub result: u8,
}

/// The lines of `text` that parse as a position with at least one legal move.
pub fn load_starts(text: &str) -> Vec<String> {
    text.lines()
        .map(str::trim)
        .filter(|line| {
            Position::from_fen(line).map(|p| !p.legal_moves().is_empty()).unwrap_or(false)
        })
        .map(str::to_string)
        .collect()
}

/// Encodes a position as one 68-byte record. `white_result` is White's view of
/// the game result (0 loss, 1 draw, 2 win).
pub fn encode(board: &Board, score: i32, white_result: u8) -> [u8; RECORD_BYTES] {
    let mut indices: Vec<u16> = Vec::with_capacity(32);
    for color in Color::ALL {
        for piece in Piece::ALL {
            for sq in board.colored_pieces(color, piece) {
                indices.push(features(color, piece, sq as usize).0 as u16);
            }
        }
    }
    indices.sort_unstable();
    indices.resize(32, PAD);

    let mut out = [0u8; RECORD_BYTES];
    for (i, idx) in indices.iter().enumerate() {
        out[2 * i..2 * i + 2].copy_from_slice(&idx.to_le_bytes());
    }
    let clipped = score.clamp(-SCORE_CLIP, SCORE_CLIP) as i16;
    out[64..66].copy_from_slice(&clipped.to_le_bytes());
    let white_to_move = board.side_to_move() == Color::White;
    out[66] = if white_to_move { 0 } else { 1 };
    out[67] = if white_to_move { white_result } else { 2 - white_result };
    out
}

/// Plays a random legal move; `None` if the position is over.
fn random_move(pos: &mut Position, rng: &mut Rng) -> Option<()> {
    if pos.is_draw(0) {
        return None;
    }
    let moves = pos.legal_moves();
    if moves.is_empty() {
        return None;
    }
    pos.play(moves[rng.below(moves.len())]);
    Some(())
}

fn is_capture_or_promotion(board: &Board, mv: cozy_chess::Move) -> bool {
    if mv.promotion.is_some() {
        return true;
    }
    let mover = board.side_to_move();
    if board.colors(!mover).has(mv.to) {
        return true;
    }
    // En passant: a pawn changing file onto an empty square.
    board.piece_on(mv.from) == Some(Piece::Pawn) && mv.from.file() != mv.to.file()
}

/// Plays one game against itself. `None` if the opening is unusable.
pub fn play_game(searcher: &mut Searcher, rng: &mut Rng, settings: &Settings) -> Option<Game> {
    let from_starts = !settings.starts.is_empty() && (rng.below(100) as u32) < settings.start_share;
    let mut pos;
    if from_starts {
        let fen = &settings.starts[rng.below(settings.starts.len())];
        pos = Position::from_fen(fen).ok()?;
        for _ in 0..2 {
            random_move(&mut pos, rng)?;
        }
    } else {
        pos = Position::startpos();
        let plies = 8 + (rng.next() & 1) as usize;
        for _ in 0..plies {
            random_move(&mut pos, rng)?;
        }
    }
    if pos.is_draw(0) || pos.legal_moves().is_empty() {
        return None;
    }

    searcher.new_game();
    let limits = Limits { nodes: Some(settings.nodes), ..Limits::default() };
    let mut samples = Vec::new();
    let mut streak = 0u32;
    let mut streak_white_ahead = false;
    let mut first_search = true;

    let white_result = 'game: {
        for _ in 0..MAX_GAME_PLIES {
            let white_to_move = pos.board.side_to_move() == Color::White;
            if pos.legal_moves().is_empty() {
                if pos.board.checkers().is_empty() {
                    break 'game 1;
                }
                break 'game if white_to_move { 0 } else { 2 };
            }
            if pos.is_draw(0) {
                break 'game 1;
            }

            let result = searcher.search(&mut pos, &limits, &mut |_, _| {});
            let best = result.best_move?;
            let score = result.score;

            if first_search && !from_starts && score.abs() > OPENING_SCORE_LIMIT {
                return None;
            }
            first_search = false;

            if pos.board.checkers().is_empty()
                && !is_capture_or_promotion(&pos.board, best)
                && score.abs() < MATE - MAX_PLY as i32
            {
                samples.push(Sample { board: pos.board.clone(), score });
            }

            if score.abs() >= ADJUDICATE_SCORE {
                let white_ahead = (score > 0) == white_to_move;
                if streak > 0 && white_ahead != streak_white_ahead {
                    streak = 0;
                }
                streak_white_ahead = white_ahead;
                streak += 1;
                if streak >= ADJUDICATE_SEARCHES {
                    break 'game if white_ahead { 2 } else { 0 };
                }
            } else {
                streak = 0;
            }

            pos.play(best);
        }
        1
    };
    Some(Game { samples, result: white_result })
}
