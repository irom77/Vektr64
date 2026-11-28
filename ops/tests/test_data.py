from pathlib import Path

from vektr_ops.data import run_prep, source_tag
from vektr_ops.store import LocalStore
from vektr_ops.training import trainer_cmd

TAG = "lichess-eval-20261101"


def fake_head(stamp):
    class Response:
        headers = {"Last-Modified": stamp}

        def close(self):
            pass

    return lambda request, **kwargs: Response()


def test_source_tag_comes_from_last_modified():
    assert source_tag("u", fetch=fake_head("Sun, 01 Nov 2026 17:23:27 GMT")) == TAG


def fake_runner(calls):
    """Writes the shards the command asks for and announces them like the trainer does."""

    def run(cmd):
        calls.append(cmd)
        arg = lambda name: cmd[cmd.index(name) + 1]
        out, skip, positions = Path(arg("--out")), int(arg("--skip")), int(arg("--positions"))
        out.mkdir(parents=True, exist_ok=True)
        for i in range(skip, positions // 1_000_000):
            path = out / f"shard-{i:05d}.npz"
            path.write_bytes(f"shard {i}".encode())
            yield f"SHARD {path}\n"

    return run


def tagged(monkeypatch):
    monkeypatch.setattr("vektr_ops.data.source_tag", lambda url, fetch=None: TAG)


def test_shards_are_uploaded_under_the_tag_and_local_copies_removed(tmp_path, monkeypatch):
    tagged(monkeypatch)
    store, calls = LocalStore(tmp_path / "store"), []
    assert run_prep(store, tmp_path, positions=2_000_000, runner=fake_runner(calls)) == 2
    assert store.list("data", f"bootstrap/{TAG}/") == [
        f"bootstrap/{TAG}/shard-00000.npz",
        f"bootstrap/{TAG}/shard-00001.npz",
    ]
    assert store.read_bytes("data", f"bootstrap/{TAG}/shard-00001.npz") == b"shard 1"
    assert not list((tmp_path / "state").rglob("*.npz"))
    assert calls[0][calls[0].index("--skip") + 1] == "0"


def test_a_second_run_skips_what_the_store_already_has(tmp_path, monkeypatch):
    tagged(monkeypatch)
    store, calls = LocalStore(tmp_path / "store"), []
    run_prep(store, tmp_path, positions=2_000_000, runner=fake_runner(calls))
    assert run_prep(store, tmp_path, positions=3_000_000, runner=fake_runner(calls)) == 1
    assert calls[1][calls[1].index("--skip") + 1] == "2"
    assert len(store.list("data", f"bootstrap/{TAG}/")) == 3


def test_trainer_cmd_uses_the_named_python_when_set():
    assert trainer_cmd({"VEKTR_TRAINER_PYTHON": "python3"}, "m", "--x")[:3] == ["python3", "-m", "m"]
    assert trainer_cmd({}, "m")[:4] == ["uv", "run", "--project", "trainer"]
