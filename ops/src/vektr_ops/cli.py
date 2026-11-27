"""`vektr-ops`: run and inspect the Vektr64 bot."""

import json
import os
import subprocess
import urllib.request
from pathlib import Path

import psutil
import typer
import yaml

from vektr_ops.botconfig import load_env, render
from vektr_ops.sizing import hash_mb

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

    build_engine(root)

    template = yaml.safe_load((root / "bot" / "config.template.yml").read_text())
    available_mb = psutil.virtual_memory().available // 2**20
    config = render(
        template,
        engine_dir=str(root / "engine" / "target" / "release"),
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
