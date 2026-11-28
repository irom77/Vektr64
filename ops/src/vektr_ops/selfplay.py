"""Self-play data: run the engine's `datagen` with the champion's network and upload one shard."""

import gzip
import secrets
import shutil
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from vektr_ops import manifest
from vektr_ops.builds import ensure_build
from vektr_ops.store import Store

SHARD_POSITIONS = 500_000
NODES = 5000
PER_NET = 120
PREFIX = "selfplay/"
RECORD_BYTES = 68
SUFFIX = ".vkd.gz"
TIMEOUT_S = 3 * 3600


class DatagenFailed(Exception):
    """The engine did not produce a usable shard."""


def shard_path(net: str, host: str, now: datetime, rand: str) -> str:
    """Store path of a shard; names sort by time, then name the network that played it."""
    return f"{PREFIX}{now.strftime('%Y%m%dT%H%M%SZ')}-{net}-{host}-{rand}{SUFFIX}"


def net_of(path: str) -> str:
    """The network part of a shard path (the host may itself contain hyphens)."""
    return path.removeprefix(PREFIX).split("-")[1]


def shards_for(store: Store, net: str) -> list[str]:
    return [p for p in store.list("data", PREFIX) if p.endswith(SUFFIX) and net_of(p) == net]


def _local_name(path: str) -> str:
    return path.replace("/", "__")


def flush(store: Store, root: Path) -> int:
    """Upload the finished shards an earlier failure left in `state/datagen/`; returns how many."""
    folder = Path(root) / "state" / "datagen"
    uploaded = 0
    for shard in sorted(folder.glob(f"*{SUFFIX}")):
        store.write_bytes("data", shard.name.replace("__", "/"), shard.read_bytes(), overwrite=True)
        shard.unlink()
        uploaded += 1
    return uploaded


def _champion_net(store: Store, root: Path) -> tuple[str, Path | None]:
    """The champion's network hash and a local file holding it; ("none", None) without one."""
    champion = manifest.load(store).champion_version()
    if champion is None:
        raise DatagenFailed("there is no champion")
    if champion.net is None:
        return "none", None
    local = Path(root) / "state" / "nets" / f"{champion.net}.vknn"
    if not local.exists():
        data = store.read_bytes("nets", f"nets/{champion.net}.vknn")
        if data is None:
            raise DatagenFailed(f"network {champion.net} is not in the store")
        local.parent.mkdir(parents=True, exist_ok=True)
        part = local.with_name(local.name + ".part")
        part.write_bytes(data)
        part.replace(local)
    return champion.net, local


def _starts_file(store: Store, root: Path, folder: Path) -> Path | None:
    """All lost-game positions in one file; None when the store has none."""
    cache = Path(root) / "state" / "experience"
    lines: list[str] = []
    for path in store.list("data", "experience/"):
        if not path.endswith(".fens"):
            continue
        local = cache / path.removeprefix("experience/")
        if not local.exists():
            data = store.read_bytes("data", path)
            if data is None:
                continue
            cache.mkdir(parents=True, exist_ok=True)
            part = local.with_name(local.name + ".part")
            part.write_bytes(data)
            part.replace(local)
        lines += [line for line in local.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return None
    target = folder / "starts.txt"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def generate(
    store: Store,
    root: Path,
    *,
    commit: str,
    threads: int,
    positions: int = SHARD_POSITIONS,
    build=ensure_build,
    run=subprocess.run,
) -> str:
    """Play self-play games with the champion's network and upload one shard; returns its store path."""
    root = Path(root)
    flush(store, root)
    folder = root / "state" / "datagen"
    folder.mkdir(parents=True, exist_ok=True)
    for stale in [*folder.glob("*.part"), *folder.glob("*.vkd")]:  # left by a killed engine
        stale.unlink()
    net, net_file = _champion_net(store, root)
    binary = build(root, commit)
    starts = _starts_file(store, root, folder)
    out = folder / "shard.vkd"
    command = [str(binary), "datagen", "--out", str(out), "--positions", str(positions),
               "--threads", str(threads), "--nodes", str(NODES), "--seed", str(secrets.randbits(63))]
    if net_file is not None:
        command += ["--net", str(net_file)]
    if starts is not None:
        command += ["--starts", str(starts)]
    try:
        done = run(command, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise DatagenFailed(f"engine did not finish: {error}") from error
    if done.returncode != 0:
        raise DatagenFailed(f"engine exited {done.returncode}: {(done.stderr or '')[-500:]}{(done.stdout or '')[-500:]}")
    if not out.is_file():
        raise DatagenFailed("engine wrote no output")
    if out.stat().st_size != positions * RECORD_BYTES:
        raise DatagenFailed(f"output has {out.stat().st_size} bytes, expected {positions * RECORD_BYTES}")
    path = shard_path(net, socket.gethostname(), datetime.now(timezone.utc), secrets.token_hex(2))
    target = folder / _local_name(path)
    part = target.with_name(target.name + ".part")
    with open(out, "rb") as source, gzip.open(part, "wb") as packed:
        shutil.copyfileobj(source, packed)
    part.replace(target)
    out.unlink()
    store.write_bytes("data", path, target.read_bytes())
    target.unlink()
    return path
