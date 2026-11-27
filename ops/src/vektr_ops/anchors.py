"""Anchor ladder: place a version on one absolute scale by playing fixed-strength opponents.

Stockfish is used here only as a yardstick. It never plays on Lichess for this account.
"""

import math
from pathlib import Path

from vektr_ops import gate, tools
from vektr_ops.builds import bench_nps, ensure_build
from vektr_ops.manifest import Version, load, save
from vektr_ops.match import Engine, Interrupted, run_batch
from vektr_ops.stats import expected_score
from vektr_ops.store import Store

ANCHORS = (1500, 1800, 2100, 2400)
GAMES_PER_ANCHOR = 200
LOW, HIGH = 0.0, 4000.0

Result = tuple[float, int, int, int]  # anchor rating, wins, draws, losses


def rating(results: list[Result]) -> float | None:
    """The rating at which the expected points against the anchors equal the points scored."""
    games = sum(w + d + l for _, w, d, l in results)
    if games == 0:
        return None
    points = sum(w + 0.5 * d for _, w, d, _ in results)

    def surplus(r: float) -> float:
        return sum((w + d + l) * expected_score(r - a) for a, w, d, l in results) - points

    if surplus(HIGH) <= 0:
        return HIGH
    if surplus(LOW) >= 0:
        return LOW
    low, high = LOW, HIGH
    for _ in range(60):
        mid = (low + high) / 2
        if surplus(mid) < 0:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def margin(results: list[Result]) -> float:
    """Half-width of the 95% interval of `rating`, from the spread of the game scores."""
    r = rating(results)
    if r is None:
        return 0.0
    variance = slope = 0.0
    for a, w, d, l in results:
        n = w + d + l
        if n == 0:
            continue
        mean = (w + 0.5 * d) / n
        variance += w * (1 - mean) ** 2 + d * (0.5 - mean) ** 2 + l * mean**2
        e = expected_score(r - a)
        slope += n * math.log(10) / 400 * e * (1 - e)
    return 1.96 * math.sqrt(variance) / slope if slope > 0 else 0.0


def tally(store: Store, version_id: str) -> dict[int, list[int]]:
    """Wins, draws and losses so far against each anchor."""
    totals = {anchor: [0, 0, 0] for anchor in ANCHORS}
    for record in gate.batches(store, f"anchor/{version_id}"):
        anchor = record.get("meta", {}).get("anchor")
        if anchor in totals:
            for i, key in enumerate(("wins", "draws", "losses")):
                totals[anchor][i] += record[key]
    return totals


def estimate(totals: dict[int, list[int]]) -> dict:
    results = [(float(a), w, d, l) for a, (w, d, l) in totals.items()]
    r = rating(results)
    return {
        "elo": None if r is None else round(r, 1),
        "margin": round(margin(results), 1),
        "games": sum(sum(t) for t in totals.values()),
    }


def anchor_step(store: Store, root: Path, version: Version, runner=None) -> dict | None:
    """Play one batch against the least-played anchor; record the rating once all are done.

    Returns the running estimate, or None when the version is already measured.
    """
    root = Path(root)
    m = load(store)
    version = m.get(version.id)
    if version.anchor is not None:
        return None

    totals = tally(store, version.id)
    anchor = min(ANCHORS, key=lambda a: sum(totals[a]))
    if sum(totals[anchor]) < GAMES_PER_ANCHOR:
        binary = ensure_build(root, version.engine_commit)
        here_nps = bench_nps(binary)
        name = f"{anchor}-{gate.new_batch_name()}"
        ours = Engine("vektr", binary, dict(gate.ENGINE_OPTIONS))
        theirs = Engine(
            f"sf{anchor}",
            tools.ensure(tools.STOCKFISH, root / "tools"),
            {
                "Threads": "1",
                "Hash": "16",
                "UCI_LimitStrength": "true",
                "UCI_Elo": str(anchor),
                "Move Overhead": "10",
            },
        )
        try:
            played = gate.play(
                root, ours, theirs, here_nps, version.box_nps or here_nps, runner or run_batch, name
            )
        except Interrupted:
            raise KeyboardInterrupt from None  # a partial anchor batch is simply not counted
        if played is None:
            gate.time.sleep(gate.BUSY_WAIT_S)
            return estimate(totals)
        result, pgn, meta = played
        gate.upload_named(
            store, f"anchor/{version.id}", name, result, pgn, {**meta, "anchor": anchor}
        )
        totals = tally(store, version.id)

    current = estimate(totals)
    if all(sum(totals[a]) >= GAMES_PER_ANCHOR for a in ANCHORS):
        m = load(store)
        m.get(version.id).anchor = current
        save(store, m)
    return current
