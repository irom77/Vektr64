//! Vektr64 chess engine.

pub mod position;

/// Score of delivering mate at the root. A mate in `n` plies scores `MATE - n`.
pub const MATE: i32 = 30000;
/// Larger than any reachable score.
pub const INF: i32 = 32000;
/// Maximum search depth in plies.
pub const MAX_PLY: usize = 128;
pub mod eval;
pub mod search;
