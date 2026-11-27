//! How long to think about one move.

/// The clock of the side to move, as the GUI reports it.
pub struct Clock {
    pub remaining_ms: i64,
    pub inc_ms: i64,
    pub moves_to_go: Option<u32>,
}

/// Returns `(soft_ms, hard_ms)`: no new search iteration starts after the soft
/// limit, and the search is cut off at the hard limit. Both are at least 1.
pub fn allocate(clock: &Clock, overhead_ms: i64) -> (u64, u64) {
    let usable = (clock.remaining_ms - overhead_ms).max(1);
    let divisor = clock.moves_to_go.map_or(20, |m| m.clamp(2, 20)) as i64;
    let inc = clock.inc_ms.max(0);
    let soft = (usable / divisor + inc * 3 / 4).min(usable / 2).max(1);
    let hard = (soft * 4).min(usable * 4 / 5).max(soft);
    (soft as u64, hard as u64)
}

/// Limits for `go movetime`: use the whole time, less the overhead.
pub fn fixed(movetime_ms: i64, overhead_ms: i64) -> (u64, u64) {
    let t = (movetime_ms - overhead_ms).max(1) as u64;
    (t, t)
}
