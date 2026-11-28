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
    """Fetch training shards (the first `limit`, by path) into `dest`; returns their store paths."""
    names = sorted(n for n in store.list("data", "bootstrap/") if n.endswith(".npz"))
    names = names[:limit]
    dest.mkdir(parents=True, exist_ok=True)
    for name in names:
        target = dest / Path(name).name
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


def install_net(store: Store, root: Path, run: str) -> str:
    """Make a run's network the engine's default; returns the network hash."""
    net = store.read_bytes("nets", f"runs/{run}/net.vknn")
    check = store.read_bytes("nets", f"runs/{run}/net.check")
    if net is None or check is None:
        raise KeyError(f"run {run!r} has no network in the store")
    target = root / "engine" / "nets"
    target.mkdir(parents=True, exist_ok=True)
    (target / "default.vknn").write_bytes(net)
    (target / "default.check").write_bytes(check)
    return hashlib.sha256(net).hexdigest()[:16]
