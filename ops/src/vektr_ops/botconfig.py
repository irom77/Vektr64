"""Build the lichess-bot config and read secrets from the environment."""

import copy
import os
from pathlib import Path


def render(template: dict, *, engine_dir: str, engine_name: str, hash_mb: int) -> dict:
    """Fill the machine-specific values into a copy of the template.

    The token is never written here; the bridge reads it from LICHESS_BOT_TOKEN.
    """
    config = copy.deepcopy(template)
    config["engine"]["dir"] = engine_dir
    config["engine"]["name"] = engine_name
    config["engine"]["uci_options"]["Hash"] = hash_mb
    return config


def load_env(repo_root: Path) -> dict[str, str]:
    """Variables from `<repo_root>/.env`, overridden by the process environment."""
    env: dict[str, str] = {}
    dotenv = repo_root / ".env"
    if dotenv.is_file():
        for line in dotenv.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            env[key.strip()] = value.strip().strip("\"'")
    env.update(os.environ)
    return env
