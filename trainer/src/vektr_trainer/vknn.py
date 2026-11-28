"""Network file format (VKNN v1), quantised network and the reference integer evaluation."""

import argparse
import hashlib
import os
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .features import HIDDEN, INPUTS, flip, fen_features

QA = 255
QB = 64
SCALE = 400

MAGIC = b"VKNN"
VERSION = 1
FILE_SIZE = 12 + INPUTS * HIDDEN * 2 + HIDDEN * 2 + 2 * HIDDEN * 2 + 4  # 197,392
MAX_FT = 505
MAX_OUT_W = 127


@dataclass
class QNet:
    ft_w: np.ndarray  # int16 (768, 128), feature-major
    ft_b: np.ndarray  # int16 (128,)
    out_w: np.ndarray  # int16 (256,)
    out_b: int


def _validate(net: QNet) -> None:
    if np.abs(net.ft_w.astype(np.int32)).max() > MAX_FT or np.abs(net.ft_b.astype(np.int32)).max() > MAX_FT:
        raise ValueError(f"feature weights outside +-{MAX_FT}")
    if np.abs(net.out_w.astype(np.int32)).max() > MAX_OUT_W:
        raise ValueError(f"output weights outside +-{MAX_OUT_W}")


def write(path: Path, net: QNet) -> None:
    """Write atomically: a temporary file in the same directory, then rename."""
    _validate(net)
    path = Path(path)
    data = b"".join(
        [
            MAGIC,
            struct.pack("<II", VERSION, HIDDEN),
            net.ft_w.astype("<i2").tobytes(),
            net.ft_b.astype("<i2").tobytes(),
            net.out_w.astype("<i2").tobytes(),
            struct.pack("<i", int(net.out_b)),
        ]
    )
    assert len(data) == FILE_SIZE
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def read(path: Path) -> QNet:
    data = Path(path).read_bytes()
    if len(data) < 12 or data[:4] != MAGIC:
        raise ValueError("not a VKNN file")
    version, hidden = struct.unpack("<II", data[4:12])
    if version != VERSION:
        raise ValueError(f"unsupported VKNN version {version}")
    if hidden != HIDDEN:
        raise ValueError(f"unsupported hidden size {hidden}")
    if len(data) != FILE_SIZE:
        raise ValueError(f"wrong file length {len(data)}, expected {FILE_SIZE}")
    pos = 12
    ft_w = np.frombuffer(data, "<i2", INPUTS * HIDDEN, pos).reshape(INPUTS, HIDDEN).astype(np.int16)
    pos += INPUTS * HIDDEN * 2
    ft_b = np.frombuffer(data, "<i2", HIDDEN, pos).astype(np.int16)
    pos += HIDDEN * 2
    out_w = np.frombuffer(data, "<i2", 2 * HIDDEN, pos).astype(np.int16)
    pos += 2 * HIDDEN * 2
    (out_b,) = struct.unpack("<i", data[pos:])
    net = QNet(ft_w, ft_b, out_w, out_b)
    _validate(net)
    return net


def random_net(seed: int) -> QNet:
    rng = np.random.default_rng(seed)
    ft_w = rng.integers(-64, 65, (INPUTS, HIDDEN)).astype(np.int16)
    ft_b = rng.integers(-64, 65, HIDDEN).astype(np.int16)
    out_w = rng.integers(-127, 128, 2 * HIDDEN).astype(np.int16)
    out_b = int(rng.integers(-20000, 20001))
    return QNet(ft_w, ft_b, out_w, out_b)


def net_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def _div(a: int, b: int) -> int:
    """Integer division truncating toward zero (exact for any size)."""
    q = abs(a) // b
    return q if a >= 0 else -q


def evaluate(net: QNet, fen: str) -> int:
    """Centipawns from the side to move's point of view (not clamped, as in the file spec)."""
    indices, stm = fen_features(fen)
    white = [int(v) for v in net.ft_b]
    black = [int(v) for v in net.ft_b]
    for f in indices:
        w_row = net.ft_w[f].tolist()
        b_row = net.ft_w[flip(f)].tolist()
        white = [a + b for a, b in zip(white, w_row)]
        black = [a + b for a, b in zip(black, b_row)]
    us, them = (white, black) if stm == 0 else (black, white)
    out_w = net.out_w.tolist()
    total = 0
    for j in range(HIDDEN):
        total += min(max(us[j], 0), QA) ** 2 * out_w[j]
        total += min(max(them[j], 0), QA) ** 2 * out_w[HIDDEN + j]
    return _div((_div(total, QA) + int(net.out_b)) * SCALE, QA * QB)


def check_fens() -> list[str]:
    text = (Path(__file__).parent / "check_fens.txt").read_text()
    return [line for line in text.splitlines() if line]


def write_check(path: Path, net: QNet) -> None:
    lines = [f"{fen};{evaluate(net, fen)}" for fen in check_fens()]
    Path(path).write_text("\n".join(lines) + "\n")


def read_check(path: Path) -> list[tuple[str, int]]:
    out = []
    for line in Path(path).read_text().splitlines():
        if line:
            fen, cp = line.rsplit(";", 1)
            out.append((fen, int(cp)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(prog="vektr_trainer.vknn")
    sub = parser.add_subparsers(dest="command", required=True)
    rnd = sub.add_parser("random", help="write a seeded random network and its check file")
    rnd.add_argument("--seed", type=int, required=True)
    rnd.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    net = random_net(args.seed)
    write(args.out / "default.vknn", net)
    write_check(args.out / "default.check", net)
    print(f"wrote {args.out / 'default.vknn'} ({net_hash(args.out / 'default.vknn')})")


if __name__ == "__main__":
    main()
