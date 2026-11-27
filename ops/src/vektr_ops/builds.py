"""Engine binaries built from a given commit, kept side by side in `builds/<id>/`."""

import re
import shutil
import subprocess
from pathlib import Path

from vektr_ops.manifest import version_id

BENCH = re.compile(r"^(\d+) nodes (\d+) nps\s*$")


class BuildFailed(Exception):
    """The engine could not be built from the requested commit."""


def parse_bench(line: str) -> tuple[int, int]:
    """Nodes and nodes per second from a `vektr bench` line."""
    match = BENCH.match(line.strip().splitlines()[-1] if line.strip() else "")
    if not match:
        raise ValueError(f"not a bench line: {line!r}")
    return int(match.group(1)), int(match.group(2))


def bench_nps(binary: Path) -> int:
    """Search speed of a build on this machine, from its fixed-depth bench."""
    done = subprocess.run([str(binary), "bench", "8"], capture_output=True, text=True, timeout=600)
    return parse_bench(done.stdout)[1]


def ensure_build(root: Path, commit: str) -> Path:
    """The engine binary for `commit`, building it once from a clean export of that commit."""
    home = Path(root) / "builds" / version_id(commit)
    binary = home / "vektr"
    if binary.exists():
        return binary
    source = home / "src"
    try:
        source.mkdir(parents=True, exist_ok=True)
        archive = subprocess.run(
            ["git", "archive", commit, "engine"], cwd=root, capture_output=True
        )
        if archive.returncode != 0:
            raise BuildFailed(archive.stderr.decode(errors="replace").strip())
        subprocess.run(["tar", "-x", "-C", str(source)], input=archive.stdout, check=True)
        built = subprocess.run(
            ["cargo", "build", "--release"], cwd=source / "engine", capture_output=True, text=True
        )
        if built.returncode != 0:
            raise BuildFailed(built.stdout + built.stderr)
        temp = home / "vektr.part"
        shutil.copy2(source / "engine" / "target" / "release" / "vektr", temp)
        temp.replace(binary)
        shutil.rmtree(source, ignore_errors=True)
        return binary
    except BaseException:
        shutil.rmtree(home, ignore_errors=True)
        raise
