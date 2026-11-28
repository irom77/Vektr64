import itertools

import numpy as np
import pytest
import torch

from vektr_trainer import vknn
from vektr_trainer.features import EMPTY, fen_features
from vektr_trainer.prep import NO_RESULT, write_shard
from vektr_trainer.train import Config, train

VALUES = {"p": 100, "n": 300, "b": 300, "r": 500, "q": 900}


def material(fen):
    fields = fen.split()
    total = sum(VALUES.get(c.lower(), 0) * (1 if c.isupper() else -1) for c in fields[0])
    return -total if fields[1] == "b" else total


def make_shards(directory, n=8192):
    fens = vknn.check_fens()
    features = np.full((n, 32), EMPTY, dtype=np.uint16)
    stm = np.zeros(n, dtype=np.uint8)
    score = np.zeros(n, dtype=np.int16)
    for i, fen in zip(range(n), itertools.cycle(fens)):
        idx, side = fen_features(fen)
        features[i, : len(idx)] = idx
        stm[i] = side
        score[i] = material(fen)
    write_shard(directory / "shard.npz", features, stm, score, np.full(n, NO_RESULT, dtype=np.uint8))
    return directory


def fake_clock():
    counter = itertools.count()
    return lambda: float(next(counter))


@pytest.fixture(scope="module")
def shards(tmp_path_factory):
    return make_shards(tmp_path_factory.mktemp("shards"))


def cfg(minutes, **kw):
    return Config(minutes=minutes, batch=512, **kw)


def test_training_lowers_the_validation_loss(shards, tmp_path):
    first = train(shards, tmp_path / "a", cfg(0.2, checkpoint_s=5), device="cpu", clock=fake_clock())
    longer = train(shards, tmp_path / "b", cfg(2.0, checkpoint_s=5), device="cpu", clock=fake_clock())
    assert longer["val_loss"] < first["val_loss"]


def test_outputs_are_a_valid_network_with_a_matching_check_file(shards, tmp_path):
    info = train(shards, tmp_path, cfg(0.1), device="cpu", clock=fake_clock())
    net = vknn.read(tmp_path / "net.vknn")
    assert info["net"] == vknn.net_hash(tmp_path / "net.vknn")
    assert all(vknn.evaluate(net, fen) == cp for fen, cp in vknn.read_check(tmp_path / "net.check"))
    assert set(info) >= {"net", "positions", "steps", "epochs", "elapsed_s", "train_loss", "val_loss", "device", "config"}


def test_a_second_run_resumes_from_the_checkpoint(shards, tmp_path):
    a = train(shards, tmp_path, cfg(0.5), device="cpu", clock=fake_clock())
    b = train(shards, tmp_path, cfg(1.0), device="cpu", clock=fake_clock())
    assert b["steps"] > a["steps"] and b["elapsed_s"] >= 60
    c = train(shards, tmp_path, cfg(1.0), device="cpu", clock=fake_clock())
    assert c["steps"] == b["steps"]


def test_a_damaged_checkpoint_is_an_error_not_a_silent_restart(shards, tmp_path):
    (tmp_path / "checkpoint.pt").write_bytes(b"half")
    with pytest.raises(Exception):
        train(shards, tmp_path, cfg(0.1), device="cpu", clock=fake_clock())


def test_no_temporary_checkpoint_is_left_behind(shards, tmp_path):
    train(shards, tmp_path, cfg(0.1, checkpoint_s=2), device="cpu", clock=fake_clock())
    assert sorted(p.name for p in tmp_path.iterdir()) == ["checkpoint.pt", "net.check", "net.vknn", "run.json"]


def test_no_shards_is_an_error(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError):
        train(tmp_path / "empty", tmp_path / "out", cfg(0.1), device="cpu", clock=fake_clock())
