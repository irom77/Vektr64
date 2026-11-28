"""Lichess evaluation lines -> training shards.

Each input line is one JSON object of the Lichess evaluation database. The deepest evaluation's
first line gives the score; forced mates and positions whose best move is a capture are dropped.
"""

import argparse
import io
import json
import os
import sys
import tempfile
import urllib.request
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np

from vektr_trainer.features import EMPTY, MAX_PIECES, fen_features

SHARD_POSITIONS = 1_000_000
SCORE_CLIP = 3000
NO_RESULT = 255


def _board(placement: str) -> dict[tuple[int, int], str]:
    """Piece letters by (file, rank) from the board field of a FEN."""
    squares: dict[tuple[int, int], str] = {}
    for row, rank in enumerate(placement.split("/")):
        file = 0
        for ch in rank:
            if ch.isdigit():
                file += int(ch)
            else:
                squares[(file, 7 - row)] = ch
                file += 1
    return squares


def _is_capture(placement: str, move: str) -> bool:
    """True when the move takes a piece of the other colour, or is a pawn changing file onto an empty square."""
    board = _board(placement)
    origin = (ord(move[0]) - 97, int(move[1]) - 1)
    target = (ord(move[2]) - 97, int(move[3]) - 1)
    mover = board.get(origin)
    if mover is None:
        return False
    victim = board.get(target)
    if victim is not None:
        return victim.isupper() != mover.isupper()  # king takes own rook is castling
    return mover.lower() == "p" and origin[0] != target[0]


def parse_line(line: bytes) -> tuple[list[int], int, int] | None:
    """(White-view indices, side to move, centipawns for the side to move), or None if unusable."""
    try:
        record = json.loads(line)
        fen = record["fen"]
        best = max(record["evals"], key=lambda e: e["depth"])["pvs"][0]
        if "cp" not in best:
            return None
        moves = best.get("line", "").split()
        if moves and _is_capture(fen.split()[0], moves[0]):
            return None
        indices, stm = fen_features(fen)
        score = int(best["cp"])
    except (ValueError, KeyError, TypeError, IndexError, AttributeError):
        return None
    if stm == 1:
        score = -score
    return indices, stm, max(-SCORE_CLIP, min(SCORE_CLIP, score))


def write_shard(path: Path, features: np.ndarray, stm: np.ndarray, score: np.ndarray, result: np.ndarray) -> None:
    """Write a shard atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp = tempfile.mkstemp(dir=path.parent, suffix=".part")
    try:
        with os.fdopen(handle, "wb") as out:
            np.savez_compressed(
                out,
                features=features.astype(np.uint16),
                stm=stm.astype(np.uint8),
                score=score.astype(np.int16),
                result=result.astype(np.uint8),
            )
        os.replace(temp, path)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise


def read_shard(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {name: data[name] for name in data.files}


def prep(
    lines: Iterable[bytes],
    out: Path,
    *,
    positions: int,
    skip_shards: int = 0,
    shard_positions: int = SHARD_POSITIONS,
    on_shard: Callable[[Path], None] | None = None,
) -> int:
    """Write `shard-NNNNN.npz` for the first `positions` usable lines, except the first `skip_shards`.

    Returns the shards written. Input that ends early leaves the incomplete shard unwritten.
    """
    if positions % shard_positions:
        raise ValueError(f"positions ({positions}) must be a multiple of {shard_positions}")
    total = positions // shard_positions
    out = Path(out)
    features = np.full((shard_positions, MAX_PIECES), EMPTY, dtype=np.uint16)
    stm = np.zeros(shard_positions, dtype=np.uint8)
    score = np.zeros(shard_positions, dtype=np.int16)
    result = np.full(shard_positions, NO_RESULT, dtype=np.uint8)
    filled, index, written = 0, 0, 0
    for raw in lines:
        if index >= total:
            break
        parsed = parse_line(raw)
        if parsed is None:
            continue
        indices, side, cp = parsed
        features[filled] = EMPTY
        features[filled, : len(indices)] = indices
        stm[filled], score[filled] = side, cp
        filled += 1
        if filled == shard_positions:
            if index >= skip_shards:
                path = out / f"shard-{index:05d}.npz"
                write_shard(path, features, stm, score, result)
                written += 1
                if on_shard:
                    on_shard(path)
            index, filled = index + 1, 0
    return written


def stream_lines(url: str) -> Iterable[bytes]:
    """Decompressed lines of a `.jsonl.zst` URL, streamed."""
    import zstandard

    request = urllib.request.Request(url, headers={"User-Agent": "vektr-trainer"})
    with urllib.request.urlopen(request, timeout=60) as response:
        with zstandard.ZstdDecompressor().stream_reader(response) as reader:
            yield from io.BufferedReader(reader, 1 << 20)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--positions", required=True, type=int)
    parser.add_argument("--skip", type=int, default=0)
    args = parser.parse_args(argv)
    shards = prep(
        stream_lines(args.url),
        args.out,
        positions=args.positions,
        skip_shards=args.skip,
        on_shard=lambda path: print(f"SHARD {path}", flush=True),
    )
    print(f"wrote {shards} shards", file=sys.stderr)


if __name__ == "__main__":
    main()
