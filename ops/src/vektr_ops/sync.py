"""The play machine follows the manifest: build the champion, check it, switch to it."""

import os
import subprocess
import time
from pathlib import Path

from vektr_ops.builds import bench_nps, ensure_build
from vektr_ops.manifest import load, save
from vektr_ops.store import Store

SPEED_MARGIN = 0.10


class SyncRefused(Exception):
    """The champion in the manifest cannot be put into play on this machine."""


def current_id(root: Path) -> str | None:
    """The version id `builds/current` points at, if any."""
    link = Path(root) / "builds" / "current"
    return os.readlink(link).rstrip("/").split("/")[-1] if link.is_symlink() else None


def switch(root: Path, id: str) -> None:
    """Point `builds/current` at `builds/<id>` in one atomic rename."""
    builds = Path(root) / "builds"
    temp = builds / "current.new"
    temp.unlink(missing_ok=True)
    temp.symlink_to(id)
    os.replace(temp, builds / "current")


def engine_dir(root: Path) -> Path:
    """Where the bridge finds the engine: the followed champion, else the working tree."""
    current = Path(root) / "builds" / "current"
    return current if (current / "vektr").exists() else Path(root) / "engine" / "target" / "release"


def engine_starts(binary: Path) -> bool:
    """True when the binary answers `uci` with `uciok` within five seconds."""
    try:
        done = subprocess.run(
            [str(binary)], input="uci\nquit\n", capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return "uciok" in done.stdout


def follow(store: Store, root: Path, build, bench, starts, record_speed: bool) -> str:
    m = load(store)
    champion = m.champion_version()
    if champion is None:
        raise SyncRefused("the manifest has no champion")
    playing = current_id(root)
    if playing == champion.id:
        return "unchanged"

    binary = build(root, champion.engine_commit)
    if not starts(binary):
        raise SyncRefused(f"{champion.id} does not start")
    speed = bench(binary)
    old_binary = root / "builds" / playing / "vektr" if playing else None
    known = {v.id: v for v in m.versions}
    rolling_back = playing in known and known[playing].parent == champion.id
    if old_binary is not None and old_binary.exists() and not rolling_back:
        old_speed = bench(old_binary)
        if speed < (1 - SPEED_MARGIN) * old_speed:
            raise SyncRefused(
                f"{champion.id} is slower than {playing}: {speed} against {old_speed} nodes per second"
            )
    switch(root, champion.id)

    if record_speed and champion.box_nps is None:
        m = load(store)
        m.get(champion.id).box_nps = speed
        save(store, m)
    return f"switched to {champion.id}"


def sync(
    store: Store,
    root: Path,
    *,
    build=ensure_build,
    bench=bench_nps,
    starts=engine_starts,
    record_speed: bool = True,
) -> str:
    """Bring `builds/current` in line with the manifest's champion.

    On any failure the current engine is left in place, the reason is written to
    `state/sync-error`, and the error is raised again.
    """
    root = Path(root)
    note = root / "state" / "sync-error"
    try:
        outcome = follow(store, root, build, bench, starts, record_speed)
    except Exception as error:
        note.parent.mkdir(parents=True, exist_ok=True)
        note.write_text(f"{type(error).__name__}: {error}"[:2000])
        raise
    note.unlink(missing_ok=True)
    return outcome


def upload_games(store: Store, root: Path, *, min_age_s: float = 1800) -> int:
    """Copy game records the store does not have yet; returns how many.

    Records touched within `min_age_s` are left for the next run: the game may still be on.
    """
    records = Path(root) / "bot" / "game_records"
    if not records.is_dir():
        return 0
    have = set(store.list("data", "games/lichess/"))
    sent = 0
    for record in sorted(records.glob("*.pgn")):
        path = f"games/lichess/{record.name}"
        if path not in have and time.time() - record.stat().st_mtime >= min_age_s:
            store.write_bytes("data", path, record.read_bytes())
            sent += 1
    return sent
