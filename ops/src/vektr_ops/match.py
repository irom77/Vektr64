"""Size and run one batch of games between two engines with fastchess."""

import os
import signal
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from vektr_ops.results import BatchResult, read_pgn

BASE_S = 8.0  # gate time control at box speed: seconds on the clock...
INC_S = 0.08  # ...plus seconds per move
PAIRS_PER_BATCH = 50
MB_PER_GAME = 256
SERVICE = "vektr64"


@dataclass
class Engine:
    name: str
    cmd: Path
    options: dict[str, str] = field(default_factory=dict)


def concurrency(physical_cores: int, available_mb: int, bot_here: bool) -> int:
    """Parallel games for this machine; 0 means "do not play here now"."""
    if bot_here:
        if physical_cores < 4:
            return 0
        jobs = 2
    else:
        jobs = physical_cores - 1
    return max(1, min(jobs, available_mb // MB_PER_GAME))


def scaled_tc(box_nps: int, here_nps: int) -> str:
    """The gate time control shortened or stretched so this machine thinks like the box."""
    factor = box_nps / here_nps
    return f"{BASE_S * factor:.2f}+{INC_S * factor:.3f}"


def bot_here() -> bool:
    """True when the Lichess bot service is running on this machine."""
    try:
        done = subprocess.run(["systemctl", "is-active", SERVICE], capture_output=True, text=True)
    except FileNotFoundError:
        return False
    return done.stdout.strip() == "active"


def fastchess_args(
    fastchess: Path,
    a: Engine,
    b: Engine,
    *,
    tc: str,
    book: Path,
    pairs: int,
    jobs: int,
    seed: int,
    pgn_out: Path,
) -> list[str]:
    """The fastchess command line: each opening is played twice with colours swapped."""
    args = [str(fastchess)]
    for engine in (a, b):
        args += ["-engine", f"cmd={engine.cmd}", f"name={engine.name}"]
        args += [f"option.{key}={value}" for key, value in engine.options.items()]
    args += ["-each", f"tc={tc}", "proto=uci"]
    args += ["-openings", f"file={book}", "format=pgn", "order=random", "-srand", str(seed)]
    args += ["-rounds", str(pairs), "-games", "2", "-repeat"]
    args += ["-concurrency", str(jobs), "-recover"]
    args += ["-draw", "movenumber=40", "movecount=8", "score=10"]
    args += ["-resign", "movecount=3", "score=600"]
    args += ["-pgnout", f"file={pgn_out}"]
    return args


def run_batch(
    fastchess: Path,
    a: Engine,
    b: Engine,
    *,
    tc: str,
    book: Path,
    pairs: int,
    jobs: int,
    seed: int,
    workdir: Path,
    low_priority: bool,
) -> tuple[BatchResult, str]:
    """Play one batch; returns engine `a`'s result and the PGN, also when interrupted."""
    workdir.mkdir(parents=True, exist_ok=True)
    pgn_out = workdir / "games.pgn"
    pgn_out.unlink(missing_ok=True)
    args = fastchess_args(
        fastchess, a, b, tc=tc, book=book, pairs=pairs, jobs=jobs, seed=seed, pgn_out=pgn_out
    )
    with open(workdir / "fastchess.log", "w") as log:
        process = subprocess.Popen(
            args,
            cwd=workdir,
            stdout=log,
            stderr=subprocess.STDOUT,
            preexec_fn=(lambda: os.nice(19)) if low_priority else None,
        )
        try:
            process.wait()
        except KeyboardInterrupt:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
            text = pgn_out.read_text() if pgn_out.exists() else ""
            raise Interrupted(read_pgn(text, a.name), text) from None
    text = pgn_out.read_text() if pgn_out.exists() else ""
    return read_pgn(text, a.name), text


class Interrupted(Exception):
    """A batch was stopped part-way; carries what had been played."""

    def __init__(self, result: BatchResult, pgn: str) -> None:
        super().__init__("batch interrupted")
        self.result = result
        self.pgn = pgn
