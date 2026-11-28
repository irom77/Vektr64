import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from vektr_ops.store import LocalStore, StoreUnavailable
from vektr_ops.training import NetCheckFailed, install_net, pull_shards, run_training, trainer_cmd


def store_with_shards(tmp_path, n=3):
    store = LocalStore(tmp_path / "store")
    for i in range(n):
        store.write_bytes("data", f"bootstrap/lichess-eval-20261101/shard-{i:05d}.npz", f"s{i}".encode())
    return store


def test_shards_are_pulled_once_and_limited(tmp_path):
    store, dest = store_with_shards(tmp_path), tmp_path / "shards"
    paths = pull_shards(store, dest, limit=2)
    assert paths == [f"bootstrap/lichess-eval-20261101/shard-{i:05d}.npz" for i in range(2)]
    local = [f"bootstrap__lichess-eval-20261101__shard-{i:05d}.npz" for i in range(3)]
    assert sorted(p.name for p in dest.iterdir()) == local[:2]
    (dest / local[0]).write_bytes(b"local")
    pull_shards(store, dest, limit=2)
    assert (dest / local[0]).read_bytes() == b"local"  # not fetched again
    assert len(pull_shards(store, dest)) == 3


def test_shards_outside_the_selection_are_removed(tmp_path):
    store, dest = store_with_shards(tmp_path), tmp_path / "shards"
    pull_shards(store, dest)
    pull_shards(store, dest, limit=1)
    assert len(list(dest.iterdir())) == 1


def test_same_shard_name_in_two_datasets_does_not_collide(tmp_path):
    store, dest = LocalStore(tmp_path / "store"), tmp_path / "shards"
    store.write_bytes("data", "bootstrap/a/shard-00000.npz", b"A")
    store.write_bytes("data", "bootstrap/b/shard-00000.npz", b"B")
    pull_shards(store, dest)
    assert sorted(p.read_bytes() for p in dest.iterdir()) == [b"A", b"B"]


def selfplay_store(tmp_path, n=5, boot=2):
    store = store_with_shards(tmp_path, boot)
    for i in range(n):
        store.write_bytes("data", f"selfplay/2026112{i}T000000Z-net{i % 2}-host-abcd.vkd.gz", f"v{i}".encode())
    return store


def test_pull_shards_adds_the_newest_selfplay(tmp_path):
    store, dest = selfplay_store(tmp_path), tmp_path / "shards"
    paths = pull_shards(store, dest, selfplay=3)
    boot = [f"bootstrap/lichess-eval-20261101/shard-{i:05d}.npz" for i in range(2)]
    newest = [f"selfplay/2026112{i}T000000Z-net{i % 2}-host-abcd.vkd.gz" for i in (2, 3, 4)]
    assert paths == boot + newest
    assert sorted(p.name for p in dest.iterdir()) == sorted(n.replace("/", "__") for n in paths)
    for n in paths:
        assert (dest / n.replace("/", "__")).read_bytes() == store.read_bytes("data", n)


def test_selfplay_zero_takes_none(tmp_path):
    store, dest = selfplay_store(tmp_path), tmp_path / "shards"
    paths = pull_shards(store, dest, selfplay=0)
    assert all(p.startswith("bootstrap/") for p in paths) and len(paths) == 2
    assert len(list(dest.iterdir())) == 2


def test_shards_no_longer_selected_are_removed(tmp_path):
    store, dest = selfplay_store(tmp_path), tmp_path / "shards"
    pull_shards(store, dest, selfplay=5)
    assert len(list(dest.iterdir())) == 7
    pull_shards(store, dest, selfplay=2)
    assert len(list(dest.iterdir())) == 4


class FakeProc:
    """A trainer whose successive `poll()` calls run the scripted steps; the last one returns the exit code."""

    def __init__(self, out, steps, code, on_exit):
        self.out, self.steps, self.code, self.on_exit = out, list(steps), code, on_exit

    def poll(self):
        if self.steps:
            self.steps.pop(0)(self.out)
            return None
        if self.on_exit:
            self.on_exit(self.out)
        return self.code


def starter(steps, code=0, seen=None, on_exit=None):
    def start(cmd, cwd=None, stdout=None, stderr=None):
        out = cmd[cmd.index("--out") + 1]
        if seen is not None:
            seen.append((cmd, Path(out, "checkpoint.pt").exists()))
        if stdout is not None:
            stdout.write(b"trainer output line\n")
        return FakeProc(Path(out), steps, code, on_exit)

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


class CountingStore(LocalStore):
    """Records every checkpoint upload."""

    def __init__(self, root):
        super().__init__(root)
        self.checkpoint_writes = []

    def write_bytes(self, repo, path, data, **kw):
        if path.endswith("checkpoint.pt"):
            self.checkpoint_writes.append(data)
        super().write_bytes(repo, path, data, **kw)


def finish_keep(out):
    """A clean finish whose checkpoint is the one already written."""
    (out / "net.vknn").write_bytes(b"network")
    (out / "net.check").write_bytes(b"check")
    (out / "run.json").write_text(json.dumps({"net": "abc123"}))


def train(tmp_path, store, start, **kw):
    return run_training(store, tmp_path, "r1", minutes=1, env={"VEKTR_TRAINER_PYTHON": "py"}, start=start, poll_s=0, **kw)


def test_results_are_uploaded_after_a_clean_run(tmp_path):
    store = store_with_shards(tmp_path, 2)
    record = train(tmp_path, store, starter([], on_exit=finish), shards=2)
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
    train(tmp_path, store, starter(steps, on_exit=finish))
    assert seen == {"mid": b"one", "later": b"two"}


def test_an_existing_checkpoint_is_restored_before_training_starts(tmp_path):
    store, seen = store_with_shards(tmp_path), []
    store = CountingStore(tmp_path / "store")
    store.write_bytes("nets", "runs/r1/checkpoint.pt", b"saved")
    store.checkpoint_writes.clear()
    started = []

    def look(out):
        started.append((out / "checkpoint.pt").read_bytes())

    train(tmp_path, store, starter([look, look], seen=seen, on_exit=finish_keep))
    assert started == [b"saved", b"saved"]
    assert store.checkpoint_writes == []  # restored file is not uploaded again
    assert seen[0][1] is True
    assert "--minutes" in seen[0][0] and seen[0][0][:3] == ["py", "-m", "vektr_trainer.train"]


def test_an_unchanged_checkpoint_is_not_uploaded_again_and_tmp_never(tmp_path):
    store = CountingStore(tmp_path / "store")
    for i in range(2):
        store.write_bytes("data", f"bootstrap/d/shard-{i:05d}.npz", b"x")

    def tmp(out):
        (out / "checkpoint.pt.tmp").write_bytes(b"half")

    steps = [write_checkpoint(b"one", 10**18), tmp, lambda out: None, lambda out: None]
    train(tmp_path, store, starter(steps, on_exit=finish_keep))
    assert store.checkpoint_writes == [b"one"]
    assert not [n for n in store.list("nets", "") if n.endswith(".tmp")]


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


def seeded_engine(tmp_path):
    nets = tmp_path / "engine" / "nets"
    nets.mkdir(parents=True)
    (nets / "default.vknn").write_bytes(b"old net")
    (nets / "default.check").write_bytes(b"old check")
    store = LocalStore(tmp_path / "store")
    store.write_bytes("nets", "runs/r1/net.vknn", b"network")
    store.write_bytes("nets", "runs/r1/net.check", b"check")
    return store, nets


def test_a_failed_check_restores_the_engine_network(tmp_path):
    store, nets = seeded_engine(tmp_path)
    seen = []

    def check(root):
        seen.append((nets / "default.vknn").read_bytes())  # the engine is built with the new files
        return ["mismatch at position 3"]

    with pytest.raises(NetCheckFailed, match="mismatch at position 3"):
        install_net(store, tmp_path, "r1", check=check)
    assert seen == [b"network"]
    assert (nets / "default.vknn").read_bytes() == b"old net"
    assert (nets / "default.check").read_bytes() == b"old check"


def test_a_passing_check_keeps_the_new_network(tmp_path):
    store, nets = seeded_engine(tmp_path)
    install_net(store, tmp_path, "r1", check=lambda root: [])
    assert (nets / "default.vknn").read_bytes() == b"network"


class FlakyStore(LocalStore):
    """Fails the first `fails` writes to paths ending in `suffix` with StoreUnavailable."""

    def __init__(self, root, suffix, fails):
        super().__init__(root)
        self.suffix, self.fails, self.attempts = suffix, fails, 0

    def write_bytes(self, repo, path, data, **kw):
        if path.endswith(self.suffix):
            self.attempts += 1
            if self.attempts <= self.fails:
                raise StoreUnavailable("hub down")
        super().write_bytes(repo, path, data, **kw)


def test_a_failed_checkpoint_upload_in_the_poll_loop_is_retried_at_the_next_poll(tmp_path, capsys):
    store = FlakyStore(tmp_path / "store", "checkpoint.pt", 1)
    for i in range(2):
        store.write_bytes("data", f"bootstrap/d/shard-{i:05d}.npz", b"x")
    steps = [write_checkpoint(b"one", 10**18), lambda out: None]
    train(tmp_path, store, starter(steps, on_exit=finish_keep))
    assert "hub down" in capsys.readouterr().err
    assert store.read_bytes("nets", "runs/r1/checkpoint.pt") == b"one"


def test_final_uploads_are_retried_with_backoff(tmp_path):
    store = FlakyStore(tmp_path / "store", "net.vknn", 2)
    for i in range(2):
        store.write_bytes("data", f"bootstrap/d/shard-{i:05d}.npz", b"x")
    sleeps = []
    train(tmp_path, store, starter([], on_exit=finish), sleep=sleeps.append)
    assert store.read_bytes("nets", "runs/r1/net.vknn") == b"network"
    assert len([s for s in sleeps if s]) == 2  # two backoff waits, none at poll_s=0


def test_final_uploads_that_keep_failing_raise_after_five_tries(tmp_path):
    store = FlakyStore(tmp_path / "store", "run.json", 10**6)
    for i in range(2):
        store.write_bytes("data", f"bootstrap/d/shard-{i:05d}.npz", b"x")
    sleeps = []
    with pytest.raises(StoreUnavailable):
        train(tmp_path, store, starter([], on_exit=finish), sleep=sleeps.append)
    assert store.attempts == 5


def test_the_train_command_reports_a_store_failure_in_one_line(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from vektr_ops import cli, training

    def boom(*a, **k):
        raise StoreUnavailable("hub down")

    monkeypatch.setattr(cli, "the_store", lambda root: None)
    monkeypatch.setattr(cli, "load_env", lambda root: {})
    monkeypatch.setattr(training, "run_training", boom)
    result = CliRunner().invoke(cli.app, ["train", "--run", "r", "--minutes", "1"])
    assert result.exit_code == 1
    assert "hub down" in result.output and "Traceback" not in result.output


class Killable(FakeProc):
    def __init__(self, *a, hang=False, **k):
        super().__init__(*a, **k)
        self.calls, self.hang = [], hang

    def terminate(self):
        self.calls.append("terminate")

    def kill(self):
        self.calls.append("kill")

    def wait(self, timeout=None):
        self.calls.append("wait")
        if self.hang and "kill" not in self.calls:
            raise subprocess.TimeoutExpired("py", timeout)


@pytest.mark.parametrize("exc", [KeyboardInterrupt, RuntimeError])
def test_an_interrupt_terminates_the_trainer(tmp_path, exc):
    store = store_with_shards(tmp_path)
    procs = []

    def start(cmd, cwd=None, stdout=None, stderr=None):
        procs.append(Killable(Path(cmd[cmd.index("--out") + 1]), [boom], 0, None))
        return procs[0]

    def boom(out):
        raise exc("stop")

    with pytest.raises(exc):
        train(tmp_path, store, start)
    assert procs[0].calls[0] == "terminate" and "kill" not in procs[0].calls


def test_a_trainer_that_ignores_terminate_is_killed(tmp_path):
    store = store_with_shards(tmp_path)
    procs = []

    def start(cmd, cwd=None, stdout=None, stderr=None):
        procs.append(Killable(Path(cmd[cmd.index("--out") + 1]), [boom], 0, None, hang=True))
        return procs[0]

    def boom(out):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        train(tmp_path, store, start)
    assert procs[0].calls[0] == "terminate" and procs[0].calls[-1] == "wait" and "kill" in procs[0].calls


def test_the_run_record_lists_selfplay_shards(tmp_path):
    store = selfplay_store(tmp_path)
    record = train(tmp_path, store, starter([], on_exit=finish), selfplay=2)
    assert record["shards"][-2:] == [f"selfplay/2026112{i}T000000Z-net{i % 2}-host-abcd.vkd.gz" for i in (3, 4)]
    assert len(record["shards"]) == 4


def test_a_resumed_run_says_it_uses_the_pinned_shards(tmp_path, capsys):
    store = selfplay_store(tmp_path)
    train(tmp_path, store, starter([], on_exit=finish), selfplay=2)
    assert "resuming" not in capsys.readouterr().err
    train(tmp_path, store, starter([], on_exit=finish), selfplay=0)
    assert "resuming with 4 pinned shards" in capsys.readouterr().err


def test_a_resumed_run_keeps_its_shard_selection(tmp_path):
    store = selfplay_store(tmp_path)
    first = train(tmp_path, store, starter([], on_exit=finish), selfplay=2)
    store.write_bytes("data", "selfplay/20261129T000000Z-net0-host-abcd.vkd.gz", b"newer")
    seen = []
    again = train(tmp_path, store, starter([], seen=seen, on_exit=finish), selfplay=2)
    assert again["shards"] == first["shards"]
    assert not any("20261129" in p.name for p in (tmp_path / "state/train/r1/shards").iterdir())
    other = run_training(store, tmp_path, "r2", minutes=1, env={"VEKTR_TRAINER_PYTHON": "py"},
                         start=starter([], on_exit=finish), poll_s=0, selfplay=2)
    assert "selfplay/20261129T000000Z-net0-host-abcd.vkd.gz" in other["shards"]


def test_a_pinned_shard_missing_from_the_store_stops_before_training(tmp_path):
    store = selfplay_store(tmp_path)
    train(tmp_path, store, starter([], on_exit=finish), selfplay=2)
    (tmp_path / "store/data/selfplay/20261124T000000Z-net0-host-abcd.vkd.gz").unlink()
    started = []
    with pytest.raises(FileNotFoundError, match="20261124"):
        train(tmp_path, store, starter([], seen=started, on_exit=finish), selfplay=2)
    assert started == []
