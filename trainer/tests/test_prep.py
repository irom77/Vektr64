import json

import numpy as np
import pytest

from vektr_trainer.features import EMPTY
from vektr_trainer.prep import NO_RESULT, parse_line, prep, read_shard, write_shard


def line(fen, *evals):
    return json.dumps({"fen": fen, "evals": list(evals)}).encode()


def ev(depth, **pv):
    return {"depth": depth, "knodes": 1, "pvs": [pv]}


QUIET = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -"


def test_deepest_evaluation_first_line_is_used():
    got = parse_line(line(QUIET, ev(20, cp=10, line="e2e4"), ev(36, cp=31, line="d2d4"), ev(30, cp=5, line="g1f3")))
    assert got[2] == 31 and got[1] == 0 and len(got[0]) == 32


def test_score_is_turned_to_the_side_to_move():
    black = "7r/1p3k2/p1bPR3/5p2/2B2P1p/8/PP4P1/3K4 b - -"
    assert parse_line(line(black, ev(30, cp=69, line="f7g7")))[2] == -69


def test_scores_are_clipped():
    assert parse_line(line(QUIET, ev(30, cp=9000, line="e2e4")))[2] == 3000
    assert parse_line(line(QUIET, ev(30, cp=-9000, line="e2e4")))[2] == -3000


@pytest.mark.parametrize(
    "raw",
    [
        b"not json",
        b"{}",
        line(QUIET),
        line(QUIET, ev(30, mate=3, line="e2e4")),
        line("8/8/8 w - -", ev(30, cp=1, line="e2e4")),
        line("4k3/8/8/3p4/4P3/8/8/4K3 w - -", ev(30, cp=50, line="e4d5")),  # best move captures
        line("4k3/8/8/3pP3/8/8/8/4K3 w - d6", ev(30, cp=50, line="e5d6")),  # en passant capture
    ],
)
def test_unusable_lines_are_skipped(raw):
    assert parse_line(raw) is None


def test_castling_as_king_takes_own_rook_is_not_a_capture():
    assert parse_line(line("r3k2r/8/8/8/8/8/8/R3K2R w KQkq -", ev(30, cp=0, line="e1h1"))) is not None


def test_a_line_without_a_move_list_is_kept():
    assert parse_line(line(QUIET, {"depth": 30, "knodes": 1, "pvs": [{"cp": 12}]}))[2] == 12


def test_shard_round_trip(tmp_path):
    features = np.full((2, 32), EMPTY, dtype=np.uint16)
    features[0, :3] = [1, 2, 700]
    features[1, :1] = [5]
    path = tmp_path / "s.npz"
    write_shard(
        path,
        features,
        np.array([0, 1], dtype=np.uint8),
        np.array([-5, 3000], dtype=np.int16),
        np.array([2, NO_RESULT], dtype=np.uint8),
    )
    got = read_shard(path)
    assert got["features"].dtype == np.uint16 and got["features"].shape == (2, 32)
    assert got["stm"].dtype == np.uint8 and got["score"].dtype == np.int16 and got["result"].dtype == np.uint8
    assert (got["features"] == features).all() and (got["features"][0, 3:] == EMPTY).all()
    assert got["stm"].tolist() == [0, 1] and got["score"].tolist() == [-5, 3000]
    assert got["result"].tolist() == [2, 255]
    assert [p.name for p in tmp_path.iterdir()] == ["s.npz"]


def test_prep_writes_full_shards_and_can_skip(tmp_path):
    lines = [line(QUIET, ev(30, cp=i, line="e2e4")) for i in range(7)] + [b"junk"]
    assert prep(lines, tmp_path / "a", positions=6, shard_positions=2) == 3
    assert prep(lines, tmp_path / "b", positions=6, skip_shards=1, shard_positions=2) == 2
    assert not (tmp_path / "b" / "shard-00000.npz").exists()
    for name in ("shard-00001.npz", "shard-00002.npz"):
        assert (read_shard(tmp_path / "a" / name)["score"] == read_shard(tmp_path / "b" / name)["score"]).all()


def test_prep_announces_each_shard(tmp_path):
    seen = []
    lines = [line(QUIET, ev(30, cp=i, line="e2e4")) for i in range(4)]
    prep(lines, tmp_path, positions=4, shard_positions=2, on_shard=seen.append)
    assert [p.name for p in seen] == ["shard-00000.npz", "shard-00001.npz"]


def test_input_that_ends_early_leaves_no_partial_shard(tmp_path):
    assert prep([line(QUIET, ev(30, cp=1, line="e2e4"))] * 3, tmp_path, positions=4, shard_positions=2) == 1
    assert not (tmp_path / "shard-00001.npz").exists()


def test_positions_must_be_whole_shards(tmp_path):
    with pytest.raises(ValueError):
        prep([], tmp_path, positions=3, shard_positions=2)
