"""Training data: run the preparation and keep its shards in the store."""

import shutil
import subprocess
import urllib.request
from collections.abc import Callable, Iterable
from email.utils import parsedate_to_datetime
from pathlib import Path

from vektr_ops.botconfig import load_env
from vektr_ops.store import Store
from vektr_ops.training import trainer_cmd

SOURCE = "https://database.lichess.org/lichess_db_eval.jsonl.zst"


def source_tag(url: str, fetch=urllib.request.urlopen) -> str:
    """`lichess-eval-YYYYMMDD` from the file's `Last-Modified` header."""
    response = fetch(url, timeout=30)  # only the headers are read; closing drops the body
    try:
        stamp = response.headers["Last-Modified"]
    finally:
        response.close()
    return "lichess-eval-" + parsedate_to_datetime(stamp).strftime("%Y%m%d")


def subprocess_runner(cwd: Path) -> Callable[[list[str]], Iterable[str]]:
    """A runner that yields the command's output lines as they appear and fails on a non-zero exit."""

    def run(cmd: list[str]) -> Iterable[str]:
        with subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, text=True) as proc:
            yield from proc.stdout
            if proc.wait() != 0:
                raise subprocess.CalledProcessError(proc.returncode, cmd)

    return run


def run_prep(
    store: Store, root: Path, *, positions: int, url: str = SOURCE, runner=None
) -> int:
    """Prepare shards not yet in the store and upload each as it is announced; returns shards uploaded."""
    tag = source_tag(url)
    prefix = f"bootstrap/{tag}/"
    have = len([n for n in store.list("data", prefix) if n.endswith(".npz")])
    out = root / "state" / "prep" / tag
    cmd = trainer_cmd(
        load_env(root), "vektr_trainer.prep",
        "--url", url, "--out", str(out), "--positions", str(positions), "--skip", str(have),
    )
    uploaded = 0
    for text in (runner or subprocess_runner(root))(cmd):
        if not text.startswith("SHARD "):
            continue
        path = Path(text[len("SHARD "):].strip())
        store.write_bytes("data", prefix + path.name, path.read_bytes())
        path.unlink()
        uploaded += 1
    shutil.rmtree(out, ignore_errors=True)
    return uploaded
