"""Running the trainer project from `vektr-ops`, and moving its inputs and results through the store."""

import hashlib
import json
import subprocess
import time
from pathlib import Path

from vektr_ops.store import Store


def trainer_cmd(env: dict, module: str, *args: str) -> list[str]:
    """Command that runs a trainer module: the python named by `VEKTR_TRAINER_PYTHON`, else `uv run`."""
    python = env.get("VEKTR_TRAINER_PYTHON")
    if python:
        return [python, "-m", module, *args]
    return ["uv", "run", "--project", "trainer", "python", "-m", module, *args]


def pull_shards(store: Store, dest: Path, limit: int | None = None) -> list[str]:
    """Fetch training shards (the first `limit`, by path) into `dest`; returns their store paths.

    Local names are the store path with `/` as `__`, so they sort like the paths and never collide;
    files in `dest` that are not selected are removed, because the trainer reads every shard there.
    """
    names = sorted(n for n in store.list("data", "bootstrap/") if n.endswith(".npz"))
    names = names[:limit]
    dest.mkdir(parents=True, exist_ok=True)
    wanted = {n.replace("/", "__") for n in names}
    for old in dest.iterdir():
        if old.name not in wanted:
            old.unlink()
    for name in names:
        target = dest / name.replace("/", "__")
        if target.exists():
            continue
        data = store.read_bytes("data", name)
        if data is None:
            raise FileNotFoundError(f"data:{name} disappeared from the store")
        part = target.with_name(target.name + ".part")
        part.write_bytes(data)
        part.replace(target)
    return names


def _mtime(path: Path) -> int | None:
    try:
        return path.stat().st_mtime_ns
    except FileNotFoundError:
        return None


def run_training(
    store: Store,
    root: Path,
    run: str,
    *,
    minutes: float,
    shards: int | None = None,
    env: dict,
    start=subprocess.Popen,
    poll_s: float = 15,
) -> dict:
    """Train a network, keeping the checkpoint and the results in `nets:runs/<run>/`."""
    work = root / "state" / "train" / run
    out = work / "out"
    out.mkdir(parents=True, exist_ok=True)
    prefix = f"runs/{run}/"
    paths = pull_shards(store, work / "shards", shards)

    checkpoint = out / "checkpoint.pt"
    saved = store.read_bytes("nets", prefix + "checkpoint.pt")
    if saved is not None:
        checkpoint.write_bytes(saved)
    uploaded = _mtime(checkpoint)

    def upload_checkpoint() -> None:
        nonlocal uploaded
        seen = _mtime(checkpoint)
        if seen is not None and seen != uploaded:
            store.write_bytes("nets", prefix + "checkpoint.pt", checkpoint.read_bytes(), overwrite=True)
            uploaded = seen

    cmd = trainer_cmd(
        env, "vektr_trainer.train",
        "--shards", str(work / "shards"), "--out", str(out), "--minutes", str(minutes),
    )
    log_path = work / "trainer.log"
    with open(log_path, "wb") as log:
        proc = start(cmd, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        while (code := proc.poll()) is None:
            upload_checkpoint()
            time.sleep(poll_s)
    upload_checkpoint()
    if code != 0:
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-30:])
        raise RuntimeError(f"trainer exited with status {code}:\n{tail}")

    record = json.loads((out / "run.json").read_text())
    record["shards"] = paths
    (out / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    for name in ("net.vknn", "net.check", "run.json"):
        store.write_bytes("nets", prefix + name, (out / name).read_bytes(), overwrite=True)
    store.write_bytes("nets", f"nets/{record['net']}.vknn", (out / "net.vknn").read_bytes(), overwrite=True)
    return record


class NetCheckFailed(Exception):
    """The installed network did not pass the engine's check; the old network is back in place."""


def install_net(store: Store, root: Path, run: str, check=None) -> str:
    """Make a run's network the engine's default; returns the network hash.

    `check(root)` runs with the new files in place (the engine compiles them in) and returns the
    failure lines, empty when all is well. On failure the previous files are restored.
    """
    net = store.read_bytes("nets", f"runs/{run}/net.vknn")
    netcheck = store.read_bytes("nets", f"runs/{run}/net.check")
    if net is None or netcheck is None:
        raise KeyError(f"run {run!r} has no network in the store")
    target = root / "engine" / "nets"
    target.mkdir(parents=True, exist_ok=True)
    files = {target / "default.vknn": net, target / "default.check": netcheck}
    before = {p: p.read_bytes() if p.exists() else None for p in files}
    for path, data in files.items():
        path.write_bytes(data)
    try:
        problems = check(root) if check else []
    except BaseException:
        problems = None
        raise
    finally:
        if problems is None or problems:
            for path, data in before.items():
                path.unlink(missing_ok=True) if data is None else path.write_bytes(data)
    if problems:
        raise NetCheckFailed("\n".join(problems))
    return hashlib.sha256(net).hexdigest()[:16]
