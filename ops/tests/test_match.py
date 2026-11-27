from pathlib import Path

from vektr_ops.match import Engine, concurrency, fastchess_args, scaled_tc


def test_concurrency_rules():
    assert concurrency(4, 2000, bot_here=True) == 2
    assert concurrency(2, 8000, bot_here=True) == 0
    assert concurrency(11, 15000, bot_here=False) == 10
    assert concurrency(11, 600, bot_here=False) == 2
    assert concurrency(4, 300, bot_here=True) == 1
    assert concurrency(1, 100, bot_here=False) == 1


def test_time_control_is_scaled_to_box_speed():
    assert scaled_tc(2_850_000, 2_850_000) == "8.00+0.080"
    assert scaled_tc(2_850_000, 5_000_000) == "4.56+0.046"


def test_fastchess_arguments():
    a = Engine("candidate", Path("/b/new/vektr"), {"Hash": "16", "Move Overhead": "10"})
    b = Engine("champion", Path("/b/old/vektr"), {"Hash": "16", "Move Overhead": "10"})
    args = fastchess_args(
        Path("/t/fastchess"),
        a,
        b,
        tc="8.00+0.080",
        book=Path("/t/8moves_v3.pgn"),
        pairs=50,
        jobs=2,
        seed=7,
        pgn_out=Path("/w/out.pgn"),
    )
    joined = " ".join(args)
    assert args[0] == "/t/fastchess"
    assert "cmd=/b/new/vektr name=candidate option.Hash=16" in joined
    assert "option.Move Overhead=10" in args  # one argument, space kept
    assert "-rounds 50 -games 2 -repeat" in joined
    assert "-concurrency 2" in joined and "-srand 7" in joined
    assert "file=/t/8moves_v3.pgn format=pgn order=random" in joined
    assert "tc=8.00+0.080" in joined and "-recover" in args
    assert "file=/w/out.pgn" in joined
