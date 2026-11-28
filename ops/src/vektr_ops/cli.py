"""`vektr-ops`: run and inspect the Vektr64 bot."""

import json
import os
import subprocess
import urllib.request
from pathlib import Path

import psutil
import typer
import yaml

from vektr_ops import gate as gate_mod
from vektr_ops import manifest
from vektr_ops.botconfig import load_env, render
from vektr_ops.sizing import hash_mb
from vektr_ops.store import Store, StoreUnavailable, open_store
from vektr_ops.sync import engine_dir

ACCOUNT = "Vektr64"
SERVICE = "vektr64"

app = typer.Typer(no_args_is_help=True, add_completion=False)


def repo_root() -> Path:
    """The repository root: the parent of `ops/`."""
    return Path(__file__).resolve().parents[3]


def build_engine(root: Path) -> None:
    """Compile the engine in release mode; exits with cargo's output on failure."""
    result = subprocess.run(
        ["cargo", "build", "--release"], cwd=root / "engine", capture_output=True, text=True
    )
    if result.returncode != 0:
        typer.echo(result.stdout + result.stderr, err=True)
        raise typer.Exit(result.returncode)


def resolve_commit(root: Path, ref: str) -> str:
    """The full commit id a git ref names; exits if git does not know it."""
    done = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if done.returncode != 0:
        typer.echo(f"error: {ref!r} is not a commit in this repository")
        raise typer.Exit(2)
    return done.stdout.strip()


def is_pushed(root: Path, commit: str) -> bool:
    """True when some branch on `origin` contains the commit, so other machines can build it."""
    done = subprocess.run(
        ["git", "branch", "-r", "--contains", commit], cwd=root, capture_output=True, text=True
    )
    return done.returncode == 0 and bool(done.stdout.strip())


def the_store(root: Path) -> Store:
    """The configured store; exits with an explanation when there is none."""
    try:
        return open_store(load_env(root))
    except StoreUnavailable as error:
        typer.echo(f"error: {error}")
        raise typer.Exit(2)


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def progress_line(v: dict) -> str:
    if "llr" not in v:
        return f"{v.get('decision', '?')} ({v.get('reason', '')})"
    return f"games={v['games']} llr={v['llr']:+.2f} elo={v['elo']:+.1f} ±{v['margin']:.1f}"


@app.command()
def gate(
    ref: str = typer.Argument(None, help="Git ref of the candidate engine."),
    work: bool = typer.Option(False, "--work", help="Run as a worker on every open test."),
    max_batches: int = typer.Option(0, "--max-batches", help="Stop after this many batches."),
) -> None:
    """Test a candidate engine against the champion until the result is decided."""
    root = repo_root()
    store = the_store(root)
    if work:
        gate_mod.work(store, root)
        return
    if ref is None:
        typer.echo("error: give a git ref to test, or --work")
        raise typer.Exit(2)
    commit = resolve_commit(root, ref)
    if not is_pushed(root, commit):
        typer.echo(f"error: {commit[:12]} is not on origin; push it first so every machine can build it")
        raise typer.Exit(2)
    if manifest.load(store).champion is None:
        typer.echo("error: there is no champion yet; run `vektr-ops promote <ref> --bootstrap`")
        raise typer.Exit(2)
    candidate = manifest.update(store, lambda m: manifest.add_candidate(m, commit, utc_now()))
    if candidate.status == "champion":
        typer.echo(f"{candidate.id} is already the champion")
        return
    if candidate.status != "candidate":
        reason = (candidate.gate or {}).get("reason", "")
        typer.echo(f"{candidate.id} is already {candidate.status} ({reason})")
        return
    typer.echo(f"testing {candidate.id} against champion {manifest.load(store).champion}")
    played = 0
    while True:
        v = gate_mod.step(store, root, candidate)
        played += 1
        typer.echo(progress_line(v))
        if v["decision"] != "continue":
            typer.echo(f"decision: {v['decision']} ({v.get('reason', '')})")
            return
        if max_batches and played >= max_batches:
            typer.echo("stopped: batch limit reached, test still open")
            return


@app.command()
def anchors(
    id: str = typer.Argument(None, help="Version id; the champion when omitted."),
) -> None:
    """Measure a version against fixed-strength opponents to place it on the rating scale."""
    from vektr_ops.anchors import anchor_step

    root = repo_root()
    store = the_store(root)
    m = manifest.load(store)
    try:
        version = m.get(id) if id else m.champion_version()
    except KeyError:
        typer.echo(f"error: no version {id!r} in the manifest")
        raise typer.Exit(2)
    if version is None:
        typer.echo("error: there is no champion yet; run `vektr-ops promote <ref> --bootstrap`")
        raise typer.Exit(2)
    while (current := anchor_step(store, root, version)) is not None:
        elo = "?" if current["elo"] is None else f"{current['elo']:.0f}"
        typer.echo(f"games={current['games']} rating={elo} ±{current['margin']:.0f}")
    done = manifest.load(store).get(version.id).anchor
    typer.echo(f"anchor rating of {version.id}: {done['elo']:.0f} ±{done['margin']:.0f}")


@app.command()
def promote(
    ref: str = typer.Argument(..., help="Version id, or a git ref with --bootstrap."),
    bootstrap: bool = typer.Option(False, "--bootstrap", help="Register the first champion."),
) -> None:
    """Make a version the champion. Needs a passed gate, except for the very first version."""
    root = repo_root()
    store = the_store(root)
    commit = resolve_commit(root, ref) if bootstrap else None

    def change(m) -> str:
        if bootstrap:
            version = manifest.add_candidate(m, commit, utc_now())
            manifest.promote(m, version.id, bootstrap=True)
        else:
            manifest.promote(m, ref)
        return m.champion

    try:
        champion = manifest.update(store, change)
    except KeyError:
        typer.echo(f"error: no version {ref!r} in the manifest")
        raise typer.Exit(2)
    except manifest.NotPromotable as error:
        typer.echo(f"error: {error}")
        raise typer.Exit(2)
    typer.echo(f"champion: {champion}")


@app.command()
def reject(id: str = typer.Argument(..., help="Version id of an open candidate.")) -> None:
    """Withdraw a candidate: its test stops on every machine and it is never played."""
    store = the_store(repo_root())

    def change(m) -> None:
        if m.get(id).status != "candidate":
            raise manifest.NotPromotable(f"{id} is {m.get(id).status}, not an open candidate")
        record = {"decision": "reject", "reason": "withdrawn", "games": 0, "finished": utc_now()}
        manifest.reject(m, id, record)

    try:
        manifest.update(store, change)
    except KeyError:
        typer.echo(f"error: no version {id!r} in the manifest")
        raise typer.Exit(2)
    except manifest.NotPromotable as error:
        typer.echo(f"error: {error}")
        raise typer.Exit(2)
    typer.echo(f"rejected: {id}")


@app.command()
def rollback() -> None:
    """Restore the previous champion. The box switches back at its next sync."""
    store = the_store(repo_root())
    try:
        restored = manifest.update(store, manifest.rollback)
    except manifest.NotPromotable as error:
        typer.echo(f"error: {error}")
        raise typer.Exit(2)
    typer.echo(f"champion: {restored.id}")


@app.command()
def sync() -> None:
    """On the play machine: follow the manifest's champion and send game records to the store."""
    from vektr_ops import sync as sync_mod
    from vektr_ops.match import bot_here

    root = repo_root()
    store = the_store(root)
    try:
        typer.echo(sync_mod.sync(store, root, record_speed=bot_here()))
    except Exception as error:
        typer.echo(f"sync failed: {type(error).__name__}: {error}")
        refresh_page(root)
        raise typer.Exit(1)
    try:
        typer.echo(f"game records uploaded: {sync_mod.upload_games(store, root)}")
    except StoreUnavailable as error:
        typer.echo(f"game records not uploaded: {error}")
    finally:
        refresh_page(root)


def refresh_page(root: Path) -> None:
    """Regenerate the page after a sync; a page problem never fails the sync."""
    try:
        write_page(root, root / "site")
    except Exception as error:  # the page is a convenience, the sync is not
        typer.echo(f"page not updated: {type(error).__name__}: {error}")


@app.command()
def play(
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the bridge config and exit."),
) -> None:
    """Build the engine and play on Lichess through the lichess-bot bridge."""
    root = repo_root()
    env = load_env(root)
    token = env.get("LICHESS_TOKEN", "")
    if not token:
        typer.echo("error: LICHESS_TOKEN is not set (put it in .env or the environment)")
        raise typer.Exit(2)

    directory = engine_dir(root)
    if directory == root / "engine" / "target" / "release":
        build_engine(root)

    template = yaml.safe_load((root / "bot" / "config.template.yml").read_text())
    available_mb = psutil.virtual_memory().available // 2**20
    config = render(
        template,
        engine_dir=str(directory),
        engine_name="vektr",
        hash_mb=hash_mb(available_mb),
    )
    text = yaml.safe_dump(config, sort_keys=False)
    config_path = root / "bot" / "config.yml"
    config_path.write_text(text)
    config_path.chmod(0o600)

    if dry_run:
        typer.echo(text)
        return

    bridge = root / "bot" / "lichess-bot"
    python = bridge / ".venv" / "bin" / "python"
    if not python.exists():
        typer.echo(f"error: {python} is missing; run deploy/install-box.sh first")
        raise typer.Exit(1)
    os.chdir(bridge)
    os.execve(
        python,
        [str(python), "lichess-bot.py", "--config", "../config.yml"],
        {**env, "LICHESS_BOT_TOKEN": token},
    )


def store_lines(root: Path) -> None:
    """Status lines that come from the store and from this machine's sync state."""
    from vektr_ops.sync import current_id

    note = root / "state" / "sync-error"
    try:
        m = manifest.load(open_store(load_env(root)))
    except StoreUnavailable as error:
        configured = "HF_TOKEN or VEKTR_STORE" not in str(error)
        typer.echo("store: unreachable" if configured else "store: not configured")
    else:
        champion = m.champion_version()
        typer.echo(f"champion: {champion.id if champion else 'none'}")
        typer.echo(f"playing: {current_id(root) or 'working-tree build'}")
        if champion and champion.anchor and champion.anchor.get("elo") is not None:
            a = champion.anchor
            typer.echo(f"anchor rating: {a['elo']:.0f} ±{a['margin']:.0f}")
        else:
            typer.echo("anchor rating: not measured yet")
        gated = [v for v in m.versions if v.gate]
        if gated:
            last = gated[-1]
            typer.echo(f"last gate: {last.id} {last.gate.get('decision')} ({last.gate.get('games')} games)")
        else:
            typer.echo("last gate: none")
        open_tests = len(m.candidates())
        if open_tests:
            typer.echo(f"open tests: {open_tests}")
    if note.is_file():
        typer.echo(f"sync error: {note.read_text().strip()}")


def write_page(root: Path, out: Path) -> Path:
    """Generate the status page into `out/index.html`."""
    from vektr_ops import page as page_mod

    try:
        store = open_store(load_env(root))
    except StoreUnavailable:
        store = None
    out.mkdir(parents=True, exist_ok=True)
    target = out / "index.html"
    temp = out / "index.html.part"
    temp.write_text(page_mod.render(page_mod.collect(store, root)))
    temp.replace(target)
    return target


@app.command()
def page(
    out: Path = typer.Option(None, "--out", help="Directory to write; default is site/."),
) -> None:
    """Generate the static status page."""
    root = repo_root()
    typer.echo(write_page(root, out or root / "site"))


@app.command()
def status() -> None:
    """Show whether the service is running and how the account stands on Lichess."""
    try:
        active = subprocess.run(
            ["systemctl", "is-active", SERVICE], capture_output=True, text=True
        ).stdout.strip()
    except FileNotFoundError:
        active = "unknown (no systemctl)"
    typer.echo(f"service: {active or 'unknown'}")
    store_lines(repo_root())

    try:
        request = urllib.request.Request(
            f"https://lichess.org/api/user/{ACCOUNT}", headers={"Accept": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            user = json.load(response)
    except (OSError, ValueError):
        typer.echo("lichess: unreachable")
        return

    blitz = user.get("perfs", {}).get("blitz", {})
    rated = user.get("count", {}).get("rated", 0)
    typer.echo(f"title: {user.get('title', 'none')}")
    typer.echo(
        f"blitz: {blitz.get('rating', '?')} "
        f"(deviation {blitz.get('rd', '?')}, {blitz.get('games', 0)} games"
        f"{', provisional' if blitz.get('prov') else ''})"
    )
    typer.echo(f"rated games: {rated}")
    typer.echo(f"in a game now: {'yes' if user.get('playing') else 'no'}")
