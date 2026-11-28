import hashlib
import json
import os
from pathlib import Path

import pytest

from vektr_ops.store import LocalStore
from vektr_ops.training import install_net, pull_shards, run_training, trainer_cmd


def store_with_shards(tmp_path, n=3):
    store = LocalStore(tmp_path / "store")
    for i in range(n):
        store.write_bytes("data", f"bootstrap/lichess-eval-20261101/shard-{i:05d}.npz", f"s{i}".encode())
    return store


def test_shards_are_pulled_once_and_limited(tmp_path):
    store, dest = store_with_shards(tmp_path), tmp_path / "shards"
    paths = pull_shards(store, dest, limit=2)
    assert paths == [f"bootstrap/lichess-eval-20261101/shard-{i:05d}.npz" for i in range(2)]
    assert sorted(p.name for p in dest.iterdir()) == ["shard-00000.npz", "shard-00001.npz"]
    (dest / "shard-00000.npz").write_bytes(b"local")
    pull_shards(store, dest, limit=2)
    assert (dest / "shard-00000.npz").read_bytes() == b"local"  # not fetched again
    assert len(pull_shards(store, dest)) == 3


class FakeProc:
    """A trainer whose successive `poll()` calls run the scripted steps; the last one returns the exit code."""

    def __init__(self, out, steps, code):
        self.out, self.steps, self.code = out, list(steps), code

    def poll(self):
        if self.steps:
            self.steps.pop(0)(self.out)
            return None
        return self.code


def starter(steps, code=0, seen=None):
    def start(cmd, cwd=None, stdout=None, stderr=None):
        out = cmd[cmd.index("--out") + 1]
        if seen is not None:
            seen.append((cmd, Path(out, "checkpoint.pt").exists()))
        if stdout is not None:
            stdout.write(b"trainer output line\n")
        return FakeProc(Path(out), steps, code)

    return start


def write_checkpoint(data, mtime):
    def step(out):
        (out / "checkpoint.pt").write_bytes(data)
        os.utime(out / "checkpoint.pt", ns=(mtime, mtime))

    return step


def finish(out):
    (out / "checkpoint.pt").write_bytes(b"final")
    (out / "net.vknn").write_bytes(b"network")
    (out / "net.check").write_bytes(b"check")
    (out / "run.json").write_text(json.dumps({"net": "abc123", "steps": 5}))


def train(tmp_path, store, start, **kw):
    return run_training(store, tmp_path, "r1", minutes=1, env={"VEKTR_TRAINER_PYTHON": "py"}, start=start, poll_s=0, **kw)


def test_results_are_uploaded_after_a_clean_run(tmp_path):
    store = store_with_shards(tmp_path, 2)
    record = train(tmp_path, store, starter([finish]), shards=2)
    shards = [f"bootstrap/lichess-eval-20261101/shard-{i:05d}.npz" for i in range(2)]
    assert record["shards"] == shards
    assert store.read_bytes("nets", "runs/r1/checkpoint.pt") == b"final"
    assert store.read_bytes("nets", "runs/r1/net.vknn") == b"network"
    assert store.read_bytes("nets", "runs/r1/net.check") == b"check"
    assert json.loads(store.read_bytes("nets", "runs/r1/run.json"))["shards"] == shards
    assert store.read_bytes("nets", "nets/abc123.vknn") == b"network"


def test_a_changed_checkpoint_is_uploaded_while_training_runs(tmp_path):
    store, seen = store_with_shards(tmp_path), {}

    def look(key):
        return lambda out: seen.update({key: store.read_bytes("nets", "runs/r1/checkpoint.pt")})

    steps = [write_checkpoint(b"one", 10**18), look("mid"), write_checkpoint(b"two", 2 * 10**18), look("later")]
    train(tmp_path, store, starter(steps + [finish]))
    assert seen == {"mid": b"one", "later": b"two"}


def test_an_existing_checkpoint_is_restored_before_training_starts(tmp_path):
    store, seen = store_with_shards(tmp_path), []
    store.write_bytes("nets", "runs/r1/checkpoint.pt", b"saved")
    train(tmp_path, store, starter([finish], seen=seen))
    assert seen[0][1] is True
    assert "--minutes" in seen[0][0] and seen[0][0][:3] == ["py", "-m", "vektr_trainer.train"]


def test_a_failed_run_keeps_its_checkpoint_and_raises(tmp_path):
    store = store_with_shards(tmp_path)
    with pytest.raises(RuntimeError, match="trainer output line"):
        train(tmp_path, store, starter([write_checkpoint(b"partial", 10**18)], code=1))
    assert store.read_bytes("nets", "runs/r1/checkpoint.pt") == b"partial"
    assert store.read_bytes("nets", "runs/r1/net.vknn") is None


def test_install_net_writes_both_files_and_returns_the_hash(tmp_path):
    store = LocalStore(tmp_path / "store")
    store.write_bytes("nets", "runs/r1/net.vknn", b"network")
    store.write_bytes("nets", "runs/r1/net.check", b"check")
    assert install_net(store, tmp_path, "r1") == hashlib.sha256(b"network").hexdigest()[:16]
    assert (tmp_path / "engine/nets/default.vknn").read_bytes() == b"network"
    assert (tmp_path / "engine/nets/default.check").read_bytes() == b"check"


def test_install_net_for_an_unknown_run_is_an_error(tmp_path):
    with pytest.raises(KeyError):
        install_net(LocalStore(tmp_path / "store"), tmp_path, "nope")
