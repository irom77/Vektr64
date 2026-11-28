"""Self-play position records: 68 bytes, little-endian, no header; `.vkd`, or gzip as `.vkd.gz`.

Layout (the engine writes the same): 32 x u16 White-view feature indices padded with 768,
i16 score for the side to move, u8 side to move, u8 result for the side to move (0 loss, 1 draw, 2 win).
The result therefore means what the `result` array of an `.npz` shard means.
"""

import gzip
import zlib
from pathlib import Path

import numpy as np

from vektr_trainer.features import EMPTY, MAX_PIECES

RECORD = np.dtype([("features", "<u2", (MAX_PIECES,)), ("score", "<i2"), ("stm", "u1"), ("result", "u1")])
RECORD_BYTES = 68
assert RECORD.itemsize == RECORD_BYTES


def read_records(path: Path) -> dict[str, np.ndarray]:
    path = Path(path)
    try:
        opener = gzip.open if path.name.endswith(".gz") else open
        with opener(path, "rb") as handle:
            raw = handle.read()
    except (EOFError, OSError, zlib.error) as error:
        raise ValueError(f"{path}: unreadable record file ({error})") from error
    if len(raw) % RECORD_BYTES:
        raise ValueError(f"{path}: {len(raw)} bytes is not a multiple of {RECORD_BYTES}")
    rows = np.frombuffer(raw, dtype=RECORD)
    if rows["features"].size and rows["features"].max() > EMPTY:
        raise ValueError(f"{path}: feature index above {EMPTY}")
    if rows["stm"].size and rows["stm"].max() > 1:
        raise ValueError(f"{path}: side to move above 1")
    if rows["result"].size and rows["result"].max() > 2:
        raise ValueError(f"{path}: result above 2")
    return {name: np.ascontiguousarray(rows[name]) for name in RECORD.names}


def write_records(path: Path, features, stm, score, result) -> None:
    path = Path(path)
    rows = np.zeros(len(stm), dtype=RECORD)
    rows["features"] = features
    rows["score"] = score
    rows["stm"] = stm
    rows["result"] = result
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "wb") as out:
        out.write(rows.tobytes())
