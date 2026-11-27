//! How long to think about one move.

/// The clock of the side to move, as the GUI reports it.
pub struct Clock {
    pub remaining_ms: i64,
    pub inc_ms: i64,
    pub moves_to_go: Option<u32>,
}

/// Returns `(soft_ms, hard_ms)`: no new search iteration starts after the soft
/// limit, and the search is cut off at the hard limit. Both are at least 1.
///
/// The target for a move is a thirtieth of the clock plus half the increment. An
/// iteration started just before the soft limit usually runs several times longer,
/// so the soft limit is half the target and the hard limit twice it; the average
/// then lands near the target and the clock settles well above zero.
pub fn allocate(clock: &Clock, overhead_ms: i64) -> (u64, u64) {
    let usable = (clock.remaining_ms - overhead_ms).max(1);
    let divisor = clock.moves_to_go.map_or(30, |m| m.clamp(2, 30)) as i64;
    let inc = clock.inc_ms.max(0);
    let target = (usable / divisor + inc / 2).min(usable / 3);
    let soft = (target / 2).max(1);
    let hard = (target * 2).min(usable / 2).max(soft);
    (soft as u64, hard as u64)
}

/// Limits for `go movetime`: use the whole time, less the overhead.
pub fn fixed(movetime_ms: i64, overhead_ms: i64) -> (u64, u64) {
    let t = (movetime_ms - overhead_ms).max(1) as u64;
    (t, t)
}
