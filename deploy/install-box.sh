#!/usr/bin/env bash
# One-time (and repeatable) setup of a machine that will run the bot.
# Run from the repository root. Does not start the service and does not touch .env.
set -euo pipefail

cd "$(dirname "$0")/.."
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

if ! command -v cc >/dev/null; then
    echo "A C toolchain is needed to build Rust programs: sudo apt-get install -y build-essential" >&2
    exit 1
fi
if ! command -v cargo >/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
fi
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

git submodule update --init

uv sync --project ops
uv venv --allow-existing bot/lichess-bot/.venv
uv pip install --python bot/lichess-bot/.venv/bin/python -r bot/lichess-bot/requirements.txt

(cd engine && cargo build --release && cargo test --release)

if [ "${1:-}" != "--no-service" ]; then
    sudo cp deploy/vektr64.service /etc/systemd/system/vektr64.service
    sudo systemctl daemon-reload
    echo "Service installed but not started. When .env holds the token: sudo systemctl enable --now vektr64"
fi
