"""Engine binaries built from a given commit, kept side by side in `builds/<id>/`."""

import fcntl
import hashlib
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from vektr_ops.manifest import version_id

RETRY_AFTER_S = 3600
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


def netcheck(binary: Path) -> bool:
    """True when the build's compiled-in network matches its check file.

    The command gets no input: an engine without `netcheck` falls into its UCI loop and is
    ended by the empty stdin (or the time limit), and never prints the ok line.
    """
    try:
        done = subprocess.run(
            [str(binary), "netcheck"], input="", capture_output=True, text=True, timeout=60
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    ok = re.match(r"netcheck ok (\d+) positions", done.stdout)
    return done.returncode == 0 and ok is not None and int(ok.group(1)) > 0


def net_hash(root: Path, commit: str) -> str | None:
    """First 16 hex of the SHA-256 of the default network at `commit`; None if it has none."""
    shown = subprocess.run(
        ["git", "show", f"{commit}:engine/nets/default.vknn"], cwd=root, capture_output=True
    )
    if shown.returncode != 0:
        return None
    return hashlib.sha256(shown.stdout).hexdigest()[:16]


def ensure_build(root: Path, commit: str) -> Path:
    """The engine binary for `commit`, building it once from a clean export of that commit.

    One process builds a given commit at a time, in a scratch directory of its own, so a
    failed or interrupted build never touches a finished binary. A commit that failed to
    build is not tried again for `RETRY_AFTER_S`; the reason is kept in `builds/<id>.failed`.
    """
    builds = Path(root) / "builds"
    id = version_id(commit)
    home = builds / id
    binary = home / "vektr"
    if binary.exists():
        return binary
    builds.mkdir(parents=True, exist_ok=True)
    note = builds / f"{id}.failed"
    with open(builds / f"{id}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if binary.exists():  # another process built it while this one waited
            return binary
        if note.exists() and time.time() - note.stat().st_mtime < RETRY_AFTER_S:
            raise BuildFailed(f"failed earlier, not retried yet: {note.read_text()}")
        scratch = Path(tempfile.mkdtemp(dir=builds, prefix=f"{id}.build-"))
        try:
            compile_into(root, commit, scratch)
            home.mkdir(exist_ok=True)
            temp = home / "vektr.part"
            shutil.copy2(scratch / "engine" / "target" / "release" / "vektr", temp)
            temp.replace(binary)
            note.unlink(missing_ok=True)
            return binary
        except BuildFailed as error:
            note.write_text(str(error)[-2000:])
            raise
        finally:
            shutil.rmtree(scratch, ignore_errors=True)


def compile_into(root: Path, commit: str, scratch: Path) -> None:
    """Export the commit's `engine/` into `scratch` and build it there."""
    known = subprocess.run(["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=root, capture_output=True)
    if known.returncode != 0:  # promoted elsewhere: this checkout has not fetched it yet
        subprocess.run(["git", "fetch", "--quiet", "origin"], cwd=root, capture_output=True)
    archive = subprocess.run(["git", "archive", commit, "engine"], cwd=root, capture_output=True)
    if archive.returncode != 0:
        raise BuildFailed(archive.stderr.decode(errors="replace").strip())
    unpacked = subprocess.run(["tar", "-x", "-C", str(scratch)], input=archive.stdout, capture_output=True)
    if unpacked.returncode != 0:
        raise BuildFailed(unpacked.stderr.decode(errors="replace").strip())
    built = subprocess.run(
        ["cargo", "build", "--release"], cwd=scratch / "engine", capture_output=True, text=True
    )
    if built.returncode != 0:
        raise BuildFailed(built.stdout + built.stderr)
