import gzip
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from vektr_ops import selfplay
from vektr_ops.manifest import load, save
from vektr_ops.selfplay import DatagenFailed, RECORD_BYTES, generate
from vektr_ops.store import LocalStore, StoreUnavailable

NET = "4a10db9d9f3f084c"


def champion_store(tmp_path, net=NET, net_bytes=b"netbytes"):
    from helpers import store_with_champion

    store = store_with_champion(tmp_path / "store")
    m = load(store)
    m.champion_version().net = net
    save(store, m)
    if net and net_bytes is not None:
        store.write_bytes("nets", f"nets/{net}.vknn", net_bytes)
    return store


def fake_run(commands=None, size=None, status=0, stderr="", write=True):
    def run(command, **kw):
        if commands is not None:
            commands.append(command)
        out = Path(command[command.index("--out") + 1])
        if write:
            n = int(command[command.index("--positions") + 1]) * RECORD_BYTES if size is None else size
            out.write_bytes(bytes(n))
        return subprocess.CompletedProcess(command, status, stdout="", stderr=stderr)

    return run


def build(root, commit):
    return Path("/fake/vektr")


def go(store, root, run, positions=10, threads=2, build=build):
    return generate(store, root, commit="a" * 40, threads=threads, positions=positions, build=build, run=run)


def test_shard_path_sorts_by_time_and_names_the_network():
    t = datetime(2026, 11, 28, 17, 1, 2, tzinfo=timezone.utc)
    p = selfplay.shard_path(NET, "irom-SEi", t, "9f3c")
    assert p == "selfplay/20261128T170102Z-4a10db9d9f3f084c-irom-SEi-9f3c.vkd.gz"
    assert selfplay.net_of(p) == NET
    later = selfplay.shard_path(NET, "a", datetime(2026, 11, 29, tzinfo=timezone.utc), "0000")
    assert later > p


def test_generate_uploads_one_gzipped_shard(tmp_path):
    store = champion_store(tmp_path)
    path = go(store, tmp_path, fake_run())
    assert path.startswith("selfplay/") and f"-{NET}-" in path
    assert len(gzip.decompress(store.read_bytes("data", path))) == 10 * RECORD_BYTES
    assert list((tmp_path / "state" / "datagen").iterdir()) == []
    assert selfplay.shards_for(store, NET) == [path]
    assert selfplay.shards_for(store, "other") == []


def test_the_champions_network_is_given_to_the_engine(tmp_path):
    store = champion_store(tmp_path)
    commands, seen = [], {}

    def run(command, **kw):
        seen["net"] = Path(command[command.index("--net") + 1]).read_bytes()
        assert kw["stdin"] == subprocess.DEVNULL
        return fake_run(commands)(command, **kw)

    go(store, tmp_path, run)
    c = commands[0]
    assert seen["net"] == b"netbytes"
    assert c[1] == "datagen"
    for flag, value in [("--nodes", "5000"), ("--positions", "10"), ("--threads", "2")]:
        assert c[c.index(flag) + 1] == value
    assert "--starts" not in c


def test_a_champion_without_a_network_uses_the_built_in_one(tmp_path):
    store = champion_store(tmp_path, net=None)
    commands = []
    path = go(store, tmp_path, fake_run(commands))
    assert "--net" not in commands[0] and "-none-" in path


def test_a_network_missing_from_the_store_refuses(tmp_path):
    store = champion_store(tmp_path, net_bytes=None)
    commands = []
    with pytest.raises(DatagenFailed):
        go(store, tmp_path, fake_run(commands))
    assert commands == []


def test_lost_game_positions_become_the_starts_file(tmp_path):
    store = champion_store(tmp_path)
    store.write_bytes("data", "experience/game one.fens", b"fen a\nfen b\n")
    store.write_bytes("data", "experience/game two.fens", b"fen c\n")
    contents = {}

    def run(command, **kw):
        contents["starts"] = Path(command[command.index("--starts") + 1]).read_text().splitlines()
        return fake_run()(command, **kw)

    go(store, tmp_path, run)
    assert contents["starts"] == ["fen a", "fen b", "fen c"]
    commands = []
    go(champion_store(tmp_path / "other"), tmp_path / "other", fake_run(commands))
    assert "--starts" not in commands[0]


def test_a_wrong_sized_output_is_not_uploaded(tmp_path):
    store = champion_store(tmp_path)
    before = store.list("data", "")
    with pytest.raises(DatagenFailed):
        go(store, tmp_path, fake_run(size=100))
    assert store.list("data", "") == before


def test_no_output_is_not_uploaded(tmp_path):
    store = champion_store(tmp_path)
    with pytest.raises(DatagenFailed):
        go(store, tmp_path, fake_run(write=False))
    assert store.list("data", selfplay.PREFIX) == []


def test_an_engine_failure_carries_its_output(tmp_path):
    store = champion_store(tmp_path)
    with pytest.raises(DatagenFailed, match="boom"):
        go(store, tmp_path, fake_run(status=1, stderr="boom", write=False))


def test_a_stale_part_file_is_cleared_and_never_uploaded(tmp_path):
    store = champion_store(tmp_path)
    folder = tmp_path / "state" / "datagen"
    folder.mkdir(parents=True)
    (folder / "shard.vkd.part").write_bytes(b"half")
    (folder / "shard.vkd").write_bytes(b"old")
    go(store, tmp_path, fake_run())
    assert list(folder.iterdir()) == []
    assert len(store.list("data", selfplay.PREFIX)) == 1
    assert selfplay.flush(store, tmp_path) == 0


def test_an_upload_failure_keeps_the_shard_for_flush(tmp_path):
    good = champion_store(tmp_path)

    class Down:
        def __getattr__(self, name):
            return getattr(good, name)

        def write_bytes(self, *a, **k):
            raise StoreUnavailable("down")

    with pytest.raises(StoreUnavailable):
        go(Down(), tmp_path, fake_run())
    folder = tmp_path / "state" / "datagen"
    kept = list(folder.glob("*.vkd.gz"))
    assert len(kept) == 1
    assert selfplay.flush(good, tmp_path) == 1
    assert len(good.list("data", selfplay.PREFIX)) == 1
    assert list(folder.iterdir()) == []
