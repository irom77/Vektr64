//! Hand-written evaluation: material plus piece-square bonuses, blended between
//! a middlegame and an endgame set by how much material is left.

use cozy_chess::{Board, Color, Piece};

const PIECES: [Piece; 6] =
    [Piece::Pawn, Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen, Piece::King];

const MATERIAL_MG: [i32; 6] = [100, 320, 330, 500, 900, 0];
const MATERIAL_EG: [i32; 6] = [120, 310, 330, 530, 950, 0];

/// Contribution of each piece type to the game phase; the full board sums to 24.
const PHASE: [i32; 6] = [0, 1, 1, 2, 4, 0];
const PHASE_MAX: i32 = 24;

// Tables are written as the board looks from White's side: first row is rank 8.

#[rustfmt::skip]
const PAWN_MG: [i32; 64] = [
      0,   0,   0,   0,   0,   0,   0,   0,
     40,  45,  45,  50,  50,  45,  45,  40,
     15,  20,  25,  35,  35,  25,  20,  15,
      5,   8,  12,  28,  28,  12,   8,   5,
      0,   2,   8,  24,  24,   4,   2,   0,
      4,   0,  -4,   6,   6,  -8,   0,   4,
      4,   8,   8, -22, -22,  12,   8,   4,
      0,   0,   0,   0,   0,   0,   0,   0,
];

#[rustfmt::skip]
const PAWN_EG: [i32; 64] = [
      0,   0,   0,   0,   0,   0,   0,   0,
     50,  50,  50,  50,  50,  50,  50,  50,
     36,  36,  34,  32,  32,  34,  36,  36,
     22,  20,  18,  16,  16,  18,  20,  22,
     12,  10,   8,   6,   6,   8,  10,  12,
      4,   4,   2,   2,   2,   2,   4,   4,
      0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,
];

#[rustfmt::skip]
const KNIGHT: [i32; 64] = [
    -50, -38, -28, -26, -26, -28, -38, -50,
    -36, -18,   0,   4,   4,   0, -18, -36,
    -26,   4,  16,  20,  20,  16,   4, -26,
    -24,   6,  20,  26,  26,  20,   6, -24,
    -24,   4,  18,  24,  24,  18,   4, -24,
    -28,   2,  12,  16,  16,  12,   2, -28,
    -36, -20,   0,   4,   4,   0, -20, -36,
    -50, -34, -28, -26, -26, -28, -34, -50,
];

#[rustfmt::skip]
const BISHOP: [i32; 64] = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,   2,   0,   0,   0,   0,   2, -10,
    -10,   0,   8,  10,  10,   8,   0, -10,
    -10,   6,   8,  14,  14,   8,   6, -10,
    -10,   2,  12,  14,  14,  12,   2, -10,
    -10,  12,  10,   8,   8,  10,  12, -10,
    -10,  10,   0,   2,   2,   0,  10, -10,
    -20, -10, -14, -10, -10, -14, -10, -20,
];

#[rustfmt::skip]
const ROOK: [i32; 64] = [
      4,   6,   8,  10,  10,   8,   6,   4,
     16,  20,  22,  24,  24,  22,  20,  16,
     -4,   0,   2,   4,   4,   2,   0,  -4,
     -6,   0,   0,   2,   2,   0,   0,  -6,
     -6,   0,   0,   2,   2,   0,   0,  -6,
     -8,   0,   0,   2,   2,   0,   0,  -8,
     -8,  -2,   0,   4,   4,   0,  -2,  -8,
     -4,  -2,   4,  10,  10,   6,  -2,  -4,
];

#[rustfmt::skip]
const QUEEN: [i32; 64] = [
    -18, -10,  -8,  -4,  -4,  -8, -10, -18,
    -10,   0,   2,   2,   2,   2,   0, -10,
     -8,   2,   6,   6,   6,   6,   2,  -8,
     -4,   2,   6,   8,   8,   6,   2,  -4,
     -4,   2,   6,   8,   8,   6,   2,  -4,
     -8,   4,   6,   6,   6,   6,   4,  -8,
    -10,   0,   4,   2,   2,   2,   0, -10,
    -18, -10,  -8,  -2,  -4,  -8, -10, -18,
];

#[rustfmt::skip]
const KING_MG: [i32; 64] = [
    -50, -50, -50, -50, -50, -50, -50, -50,
    -50, -50, -50, -50, -50, -50, -50, -50,
    -44, -48, -50, -50, -50, -50, -48, -44,
    -36, -42, -46, -50, -50, -46, -42, -36,
    -26, -32, -38, -44, -44, -38, -32, -26,
    -10, -16, -24, -30, -30, -24, -16, -10,
     16,  14,  -6, -18, -18,  -6,  14,  16,
     22,  34,  18,  -8,   0,  -4,  36,  24,
];

#[rustfmt::skip]
const KING_EG: [i32; 64] = [
    -50, -34, -24, -18, -18, -24, -34, -50,
    -30, -12,  -2,   4,   4,  -2, -12, -30,
    -22,   0,  18,  24,  24,  18,   0, -22,
    -18,   6,  26,  36,  36,  26,   6, -18,
    -18,   6,  26,  36,  36,  26,   6, -18,
    -22,   0,  18,  24,  24,  18,   0, -22,
    -30, -12,  -2,   4,   4,  -2, -12, -30,
    -50, -34, -24, -18, -18, -24, -34, -50,
];

const TABLE_MG: [&[i32; 64]; 6] = [&PAWN_MG, &KNIGHT, &BISHOP, &ROOK, &QUEEN, &KING_MG];
const TABLE_EG: [&[i32; 64]; 6] = [&PAWN_EG, &KNIGHT, &BISHOP, &ROOK, &QUEEN, &KING_EG];

/// Static score in centipawns from the point of view of the side to move.
pub fn evaluate(board: &Board) -> i32 {
    let mut mg = 0;
    let mut eg = 0;
    let mut phase = 0;
    for (i, &piece) in PIECES.iter().enumerate() {
        for (color, sign) in [(Color::White, 1), (Color::Black, -1)] {
            for sq in board.colored_pieces(color, piece) {
                let file = sq.file() as usize;
                let rank = sq.rank() as usize;
                // Row 0 of a table is the piece's own eighth rank.
                let row = if color == Color::White { 7 - rank } else { rank };
                let idx = row * 8 + file;
                mg += sign * (MATERIAL_MG[i] + TABLE_MG[i][idx]);
                eg += sign * (MATERIAL_EG[i] + TABLE_EG[i][idx]);
                phase += PHASE[i];
            }
        }
    }
    let phase = phase.min(PHASE_MAX);
    let white = (mg * phase + eg * (PHASE_MAX - phase)) / PHASE_MAX;
    if board.side_to_move() == Color::White {
        white
    } else {
        -white
    }
}
