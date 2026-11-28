#!/usr/bin/env bash
# Trains a network and uploads it to the store. Runs inside a rented pod (fetched and started by
# the pod's command), or on this machine with VEKTR_LOCAL=1 for a dry run.
# Needs VEKTR_RUN and VEKTR_MINUTES; in a pod also VEKTR_COMMIT and HF_TOKEN.
set -euo pipefail

: "${VEKTR_RUN:?VEKTR_RUN is not set}"
: "${VEKTR_MINUTES:?VEKTR_MINUTES is not set}"

if [ "${VEKTR_LOCAL:-}" = "1" ]; then
    cd "$(dirname "$0")/.."
    exec uv run --project ops vektr-ops train --run "$VEKTR_RUN" --minutes "$VEKTR_MINUTES"
fi

: "${VEKTR_COMMIT:?VEKTR_COMMIT is not set}"
export PIP_BREAK_SYSTEM_PACKAGES=1 PIP_DISABLE_PIP_VERSION_CHECK=1
command -v git >/dev/null || { apt-get update -qq && apt-get install -y -qq git; }

# RunPod restarts a container whose command ended, so this may not be the first run here.
[ -d /workspace/Vektr64/.git ] || git clone -q https://github.com/irom77/Vektr64.git /workspace/Vektr64
cd /workspace/Vektr64
git checkout -q "$VEKTR_COMMIT"

# The image already has PyTorch (with CUDA); install the rest, editable so the repository is the root.
python3 -m pip install -q -e ops
python3 -m pip install -q --no-deps -e trainer
python3 -m pip install -q numpy zstandard
export VEKTR_TRAINER_PYTHON=python3

vektr-ops train --run "$VEKTR_RUN" --minutes "$VEKTR_MINUTES"
