import numpy as np
import pytest

from vektr_trainer.features import feature, fen_features
from vektr_trainer.vknn import (
    QNet,
    check_fens,
    evaluate,
    random_net,
    read,
    read_check,
    write,
    write_check,
)

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def zero_net(out_b=0):
    return QNet(np.zeros((768, 128), np.int16), np.zeros(128, np.int16), np.zeros(256, np.int16), out_b)


def mirror(fen):  # swap colours and flip ranks; the side to move swaps too
    board, stm = fen.split()[:2]
    return "/".join(r.swapcase() for r in reversed(board.split("/"))) + (" b" if stm == "w" else " w") + " - -"


def test_output_bias_alone_gives_the_scale():
    assert evaluate(zero_net(255 * 64), START) == 400


def test_division_truncates_toward_zero():
    assert evaluate(zero_net(-1), START) == 0 and evaluate(zero_net(-41), START) == -1


def test_one_feature_by_hand():
    net = zero_net()
    net.ft_w[feature(0, 5, 4), 0] = 100  # white king e1, first hidden unit
    net.out_w[0] = 64  # "us" half
    net.out_w[128] = -64  # "them" half
    # Only the white king is on the board (Black's own king would put the same feature in the "them" view). White to move: us sees it (100² · 64), them does not.
    assert evaluate(net, "8/8/8/8/8/8/8/4K3 w - -") == (100 * 100 * 64 // 255) * 400 // (255 * 64)
    assert evaluate(net, "8/8/8/8/8/8/8/4K3 b - -") == -((100 * 100 * 64 // 255) * 400 // (255 * 64))


def test_mirrored_positions_score_the_same():
    net = random_net(7)
    for fen in check_fens():
        assert evaluate(net, fen) == evaluate(net, mirror(fen))


def test_write_read_round_trip(tmp_path):
    net = random_net(3)
    write(tmp_path / "n.vknn", net)
    back = read(tmp_path / "n.vknn")
    assert (tmp_path / "n.vknn").stat().st_size == 197_392
    assert (back.ft_w == net.ft_w).all() and (back.out_w == net.out_w).all() and back.out_b == net.out_b
    assert (back.ft_b == net.ft_b).all()


@pytest.mark.parametrize("damage", ["truncate", "magic", "version", "hidden", "range"])
def test_read_rejects_bad_files(tmp_path, damage):
    path = tmp_path / "n.vknn"
    write(path, random_net(3))
    data = bytearray(path.read_bytes())
    if damage == "truncate":
        data = data[:-1]
    elif damage == "magic":
        data[0:4] = b"XXXX"
    elif damage == "version":
        data[4:8] = (2).to_bytes(4, "little")
    elif damage == "hidden":
        data[8:12] = (64).to_bytes(4, "little")
    else:  # first ft_w entry beyond 505
        data[12:14] = (506).to_bytes(2, "little", signed=True)
    path.write_bytes(bytes(data))
    with pytest.raises(ValueError):
        read(path)


def test_check_fens_are_256_valid_positions_with_both_sides_to_move():
    fens = check_fens()
    assert len(fens) == 256 and {fen_features(f)[1] for f in fens} == {0, 1}


def test_check_file_round_trip(tmp_path):
    net = random_net(5)
    write_check(tmp_path / "c", net)
    assert read_check(tmp_path / "c") == [(f, evaluate(net, f)) for f in check_fens()]
