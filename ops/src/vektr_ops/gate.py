"""The strength gate: a candidate plays the champion until a sequential test decides."""

import secrets
import socket
import statistics
import sys
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

import psutil

from vektr_ops import tools
from vektr_ops.builds import bench_nps, ensure_build
from vektr_ops.manifest import Version, load, promote, reject, update
from vektr_ops.match import (
    PAIRS_PER_BATCH,
    Engine,
    Interrupted,
    bot_here,
    concurrency,
    run_batch,
    scaled_tc,
)
from vektr_ops.results import BatchResult
from vektr_ops.stats import Penta, add, decide, elo_estimate, llr
from vektr_ops.store import Store, StoreUnavailable, read_json, write_json

GAME_CAP = 40_000
ELO0 = 0.0
ELO1 = 5.0
ENGINE_OPTIONS = {"Hash": "16", "Threads": "1", "Move Overhead": "10"}
BUSY_WAIT_S = 60
DISCARD_LIMIT = 5  # this many unusable batches and none usable: the candidate is rejected


def test_id(champion_id: str, candidate_id: str) -> str:
    return f"{champion_id}__{candidate_id}"


def batch_name(host: str, now: datetime, rand: str) -> str:
    return f"{host}-{now.astimezone(timezone.utc):%Y%m%dT%H%M%SZ}-{rand}"


def new_batch_name() -> str:
    return batch_name(socket.gethostname(), datetime.now(timezone.utc), secrets.token_hex(2))


def machine() -> tuple[int, int]:
    """Physical cores and available memory in MiB."""
    cores = psutil.cpu_count(logical=False) or psutil.cpu_count() or 1
    return cores, psutil.virtual_memory().available // 2**20


def upload_named(
    store: Store, folder: str, name: str, result: BatchResult, pgn: str, meta: dict
) -> None:
    """Write one batch under `folder`; an invalid one goes to `folder/discarded/`."""
    base = f"{folder}/{name}" if result.valid else f"{folder}/discarded/{name}"
    body = {
        "penta": list(result.penta),
        "wins": result.wins,
        "draws": result.draws,
        "losses": result.losses,
        "games": result.games,
        "bad": result.bad,
        "meta": meta,
    }
    write_json(store, "data", f"{base}.json", body)
    store.write_bytes("data", f"{base}.pgn", pgn.encode())


def upload_batch(
    store: Store, test: str, name: str, result: BatchResult, pgn: str, meta: dict
) -> None:
    upload_named(store, f"gate/{test}", name, result, pgn, meta)


def batches(store: Store, folder: str) -> list[dict]:
    """Every counted batch record directly inside `folder` (not the discarded ones)."""
    prefix = f"{folder}/"
    records = []
    for path in store.list("data", prefix):
        if path.endswith(".json") and "/" not in path[len(prefix) :]:
            record = read_json(store, "data", path)
            if record is not None:
                records.append({**record, "path": path})
    return records


def pooled(store: Store, test: str) -> tuple[Penta, int]:
    """Pair counts and games summed over every machine's batches for one test."""
    penta: Penta = (0, 0, 0, 0, 0)
    games = 0
    for record in batches(store, f"gate/{test}"):
        penta = add(penta, tuple(record["penta"]))  # type: ignore[arg-type]
        games += record["games"]
    return penta, games


def speed_ratio(store: Store, test: str) -> float | None:
    """Median candidate-to-champion search speed over a test's counted batches, if any recorded it."""
    ratios = [
        r["meta"]["speed_ratio"]
        for r in batches(store, f"gate/{test}")
        if isinstance(r.get("meta", {}).get("speed_ratio"), (int, float))
    ]
    return round(statistics.median(ratios), 3) if ratios else None


def verdict(penta: Penta, games: int) -> dict:
    """Where a test stands: accept, reject or continue, with the numbers behind it."""
    value = llr(penta, ELO0, ELO1)
    decision, reason = decide(value), "sprt"
    if decision == "continue" and games >= GAME_CAP:
        decision, reason = "reject", "cap"
    elo, margin = elo_estimate(penta)
    return {
        "decision": decision,
        "reason": reason,
        "llr": round(value, 3),
        "penta": list(penta),
        "games": games,
        "elo": round(elo, 1),
        "margin": round(margin, 1),
        "elo0": ELO0,
        "elo1": ELO1,
    }


def settle(store: Store, candidate_id: str, champion_id: str, result: dict) -> bool:
    """Write a final verdict into the manifest: promote on accept, reject otherwise.

    The verdict only stands against the champion it was played against. Returns False, and
    writes nothing, when the champion has changed or the candidate is already settled.
    """
    final = {
        **result,
        "champion": champion_id,
        "finished": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    def change(m) -> bool:
        version = m.get(candidate_id)
        if version.status != "candidate" or m.champion != champion_id:
            return False
        if final["decision"] == "accept":
            version.gate = final
            promote(m, candidate_id)
        else:
            reject(m, candidate_id, final)
        return True

    return update(store, change)


def settled(store: Store, candidate_id: str, champion_id: str, result: dict) -> dict:
    """Settle a decided test; if the champion moved on meanwhile, the test simply stays open."""
    ratio = speed_ratio(store, test_id(champion_id, candidate_id))
    if ratio is not None:
        result = {**result, "speed_ratio": ratio}
    if settle(store, candidate_id, champion_id, result):
        return result
    version = load(store).get(candidate_id)
    if version.status != "candidate":
        return version.gate or {"decision": version.status}
    return {**result, "decision": "continue", "reason": "champion changed"}


def play(
    root: Path, a: Engine, b: Engine, here_nps: int, box_nps: int, runner, name: str,
    ratio: float | None = None,
):
    """Run one batch sized for this machine; returns (result, pgn, meta) or None if busy."""
    cores, available_mb = machine()
    with_bot = bot_here()
    jobs = concurrency(cores, available_mb, with_bot)
    if jobs == 0:
        return None
    tc = scaled_tc(box_nps, here_nps)
    seed = zlib.crc32(name.encode())
    meta = {"host": socket.gethostname(), "tc": tc, "jobs": jobs, "here_nps": here_nps, "seed": seed}
    if ratio is not None:
        meta["speed_ratio"] = ratio
    kwargs = dict(
        tc=tc,
        book=tools.ensure(tools.BOOK, root / "tools"),
        pairs=PAIRS_PER_BATCH,
        jobs=jobs,
        seed=seed,
        workdir=root / "state" / "work" / name,
        low_priority=with_bot,
    )
    result, pgn = runner(tools.ensure(tools.FASTCHESS, root / "tools"), a, b, **kwargs)
    return result, pgn, meta


def step(store: Store, root: Path, candidate: Version, runner=None) -> dict:
    """Play one batch for one candidate, pool it, and settle the test if it is decided."""
    root = Path(root)
    m = load(store)
    version = m.get(candidate.id)
    if version.status != "candidate":
        return version.gate or {"decision": version.status}
    champion = m.champion_version()
    if champion is None:
        raise RuntimeError("the manifest has no champion to test against")
    test = test_id(champion.id, version.id)

    current = verdict(*pooled(store, test))
    if current["decision"] != "continue":  # another machine played the deciding batch
        return settled(store, version.id, champion.id, current)

    champion_bin = ensure_build(root, champion.engine_commit)
    candidate_bin = ensure_build(root, version.engine_commit)
    here_nps = bench_nps(champion_bin)
    ratio = round(bench_nps(candidate_bin) / here_nps, 3)
    name = new_batch_name()
    a = Engine("candidate", candidate_bin, dict(ENGINE_OPTIONS))
    b = Engine("champion", champion_bin, dict(ENGINE_OPTIONS))
    try:
        played = play(root, a, b, here_nps, champion.box_nps or here_nps, runner or run_batch, name, ratio)
    except Interrupted as stopped:
        if stopped.result.games:
            upload_batch(store, test, name, stopped.result, stopped.pgn, {"interrupted": True})
        raise KeyboardInterrupt from None
    if played is None:
        time.sleep(BUSY_WAIT_S)
        return current
    result, pgn, meta = played
    upload_batch(store, test, name, result, pgn, meta)

    current = verdict(*pooled(store, test))
    if current["decision"] == "continue" and not result.valid and current["games"] == 0:
        discarded = sum(
            path.endswith(".json") for path in store.list("data", f"gate/{test}/discarded/")
        )
        if discarded >= DISCARD_LIMIT:
            current = {**current, "decision": "reject", "reason": "invalid", "discarded": discarded}
    if current["decision"] != "continue":
        return settled(store, version.id, champion.id, current)
    return current


def attempt(what: str, job) -> bool:
    """Run one piece of worker work; a fault in it is reported and the worker carries on."""
    try:
        return bool(job())
    except StoreUnavailable:
        raise
    except Exception as error:
        print(f"{what}: {type(error).__name__}: {str(error)[-500:]}", file=sys.stderr, flush=True)
        return False


def work(store: Store, root: Path, *, once: bool = False, idle_s: int = 300) -> None:
    """Worker loop: advance every open test by a batch, then any anchor work that is due."""
    from vektr_ops.anchors import anchor_step

    while True:
        busy = False
        try:
            for candidate in load(store).candidates():
                busy |= attempt(candidate.id, lambda: step(store, root, candidate))
            champion = load(store).champion_version()
            if champion is not None:
                busy |= attempt(champion.id, lambda: anchor_step(store, root, champion) is not None)
        except StoreUnavailable:
            if once:
                raise
        if once:
            return
        if not busy:
            time.sleep(idle_s)
