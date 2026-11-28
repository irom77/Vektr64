"""Feature indices from a FEN (White's view unless flipped)."""

INPUTS = 768
HIDDEN = 128
EMPTY = 768  # padding index meaning "no feature"
MAX_PIECES = 32

_PIECES = {"p": 0, "n": 1, "b": 2, "r": 3, "q": 4, "k": 5}


def feature(colour: int, piece: int, square: int) -> int:
    """White-view index of a piece; colour 0 = white, square a1 = 0 ... h8 = 63."""
    return colour * 384 + piece * 64 + square


def flip(index: int) -> int:
    """The same feature as seen by Black (colours swapped, ranks mirrored)."""
    if index == EMPTY:
        return EMPTY
    colour, rest = divmod(index, 384)
    piece, square = divmod(rest, 64)
    return feature(1 - colour, piece, square ^ 56)


def fen_features(fen: str) -> tuple[list[int], int]:
    """White-view feature indices (ascending) and side to move (0 white, 1 black)."""
    fields = fen.split()
    if not fields:
        raise ValueError("empty FEN")
    ranks = fields[0].split("/")
    if len(ranks) != 8:
        raise ValueError(f"board needs 8 ranks: {fen!r}")
    indices: list[int] = []
    for row, rank in enumerate(ranks):
        file = 0
        for ch in rank:
            if ch.isdigit() and ch != "0" and ch != "9":
                file += int(ch)
            elif ch.lower() in _PIECES:
                if file >= 8:
                    raise ValueError(f"rank too long: {fen!r}")
                indices.append(feature(int(ch.islower()), _PIECES[ch.lower()], (7 - row) * 8 + file))
                file += 1
            else:
                raise ValueError(f"unknown board character {ch!r}: {fen!r}")
        if file != 8:
            raise ValueError(f"rank is not 8 files wide: {fen!r}")
    if len(indices) > MAX_PIECES:
        raise ValueError(f"more than {MAX_PIECES} pieces: {fen!r}")
    stm = fields[1] if len(fields) > 1 else "w"
    if stm not in ("w", "b"):
        raise ValueError(f"bad side to move: {fen!r}")
    return sorted(indices), int(stm == "b")
