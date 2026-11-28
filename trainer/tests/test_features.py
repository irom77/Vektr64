import pytest

from vektr_trainer.features import EMPTY, feature, fen_features, flip

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_startpos_has_32_features_and_white_to_move():
    indices, stm = fen_features(START)
    assert len(indices) == 32 and stm == 0 and indices == sorted(indices)


def test_known_indices():
    indices, _ = fen_features(START)
    assert feature(0, 0, 8) == 8 and 8 in indices  # white pawn a2
    assert feature(1, 5, 60) == 764 and 764 in indices  # black king e8


def test_flip_is_the_black_view_and_its_own_inverse():
    assert flip(8) == 432  # white pawn a2 seen by Black
    assert all(flip(flip(i)) == i for i in range(768)) and flip(EMPTY) == EMPTY


def test_lichess_four_field_fen_and_black_to_move():
    assert fen_features("7r/1p3k2/p1bPR3/5p2/2B2P1p/8/PP4P1/3K4 b - -")[1] == 1


@pytest.mark.parametrize(
    "fen",
    [
        "8/8/8 w - -",
        "9/8/8/8/8/8/8/8 w - -",
        "x7/8/8/8/8/8/8/8 w - -",
        "QQQQQQQQ/QQQQQQQQ/QQQQQQQQ/QQQQQQQQ/QQQQQQQQ/8/8/8 w - -",
        "",
    ],
)
def test_bad_boards_are_rejected(fen):
    with pytest.raises(ValueError):
        fen_features(fen)
