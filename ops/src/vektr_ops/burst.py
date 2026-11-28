"""Burst instances: rent a GPU pod that trains a network and always ends itself.

A pod is deleted by four independent means, so a lost connection or a crashed script cannot
leave one running: the watchdog and the EXIT trap inside the pod, `watch` on the launcher, and
`kill_all` by hand.
"""

import shlex
import time

from vektr_ops import runpod

IMAGE = "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404"
GPUS = ["NVIDIA RTX A4000", "NVIDIA GeForce RTX 3090", "NVIDIA RTX A5000", "NVIDIA GeForce RTX 4090"]
MAX_RATE = 0.60  # USD per hour
SETUP_S = 1200  # allowance for the image pull, clone and dependency install
NAME_PREFIX = "vektr64-"
REPO_RAW = "https://raw.githubusercontent.com/irom77/Vektr64"
DELETE = 'runpodctl pod delete "$RUNPOD_POD_ID"'


class BurstFailed(Exception):
    """No pod could be started, or none within the price cap."""


def hard_limit(minutes: float) -> int:
    """Seconds after which the pod removes itself, whatever it is doing."""
    return SETUP_S + int(minutes * 60)


def start_cmd(commit: str, hard_s: int) -> list[str]:
    """The container command. The watchdog and the trap are armed before anything can fail."""
    url = f"{REPO_RAW}/{commit}/deploy/burst.sh"
    script = "\n".join(
        [
            f"({'sleep ' + str(hard_s)}; {DELETE}) &",
            f"trap '{DELETE}' EXIT",
            "set -eu",
            f"curl -fsSL {shlex.quote(url)} -o /tmp/burst.sh",
            "bash /tmp/burst.sh",
        ]
    )
    return ["bash", "-c", script]


def pod_body(run: str, commit: str, minutes: float, hf_token: str, gpu: str, cloud: str = "SECURE") -> dict:
    """The create request. The token travels only in `env`, never in the name or the command."""
    return {
        "name": f"{NAME_PREFIX}{run}",
        "image": IMAGE,
        "gpu": {"id": gpu, "count": 1},
        "cloud": cloud,
        "disk": 20,
        "env": {
            "HF_TOKEN": hf_token,
            "VEKTR_RUN": run,
            "VEKTR_MINUTES": str(minutes),
            "VEKTR_COMMIT": commit,
        },
        "cmd": start_cmd(commit, hard_limit(minutes)),
    }


def launch(key: str, bodies: list[dict], *, max_rate: float = MAX_RATE, api=runpod) -> dict:
    """Create the first pod that gets capacity at an acceptable price; returns it.

    No capacity or no access on one GPU (400, 403) moves on to the next; any other refusal
    (no balance, a bad request) stops at once. A pod above `max_rate` is deleted, not used.
    """
    notes: list[str] = []
    for body in bodies:
        gpu = body["gpu"]["id"]
        try:
            pod = api.create_pod(key, body)
            pod_id = pod["id"]
        except runpod.RunpodError as error:
            if error.status in (400, 403):
                notes.append(f"{gpu}: {error.message}")
                continue
            raise BurstFailed(f"{gpu}: {error}") from error
        except (OSError, ValueError, KeyError, TypeError) as error:
            # The request may have been accepted: a pod could exist whose id we never saw.
            raise BurstFailed(
                f"{gpu}: the create call failed ({type(error).__name__}: {error}) and a pod may be running; "
                "run `vektr-ops burst --kill` and check the RunPod console"
            ) from error
        cost = pod.get("cost")
        if isinstance(cost, (int, float)) and cost <= max_rate:
            return pod
        try:
            api.delete_pod(key, pod_id)
        except Exception as error:
            raise BurstFailed(
                f"pod {pod_id} costs {cost} USD/h, above the cap, and could not be deleted ({error}); "
                "run `vektr-ops burst --kill`"
            ) from error
        notes.append(f"{gpu}: rate {cost} USD/h is above the cap of {max_rate}")
    raise BurstFailed("no pod started:\n" + "\n".join(notes))


def watch(
    key: str,
    pod_id: str,
    hard_s: int,
    *,
    api=runpod,
    sleep=time.sleep,
    clock=time.monotonic,
    poll_s: int = 30,
) -> str:
    """Wait until the pod is gone; delete it if it stops or overstays. Returns what happened."""
    limit = clock() + hard_s + 300
    while True:
        try:
            pod = api.get_pod(key, pod_id)
        except (runpod.RunpodError, OSError):
            pod = {"status": "UNKNOWN"}  # a failed poll says nothing; the deadline still applies
        if pod is None or pod.get("status") == "TERMINATED":
            return "gone"
        status = pod.get("status")
        why = f"status {status}" if status in ("EXITED", "ERROR") else None
        if why is None and clock() >= limit:
            why = f"still there after {hard_s + 300} s"
        if why:
            api.delete_pod(key, pod_id)
            return f"deleted: {why}"
        sleep(poll_s)


def kill_all(key: str, *, api=runpod) -> list[str]:
    """Delete every pod this tool created (by name prefix); returns their ids."""
    killed = []
    for pod in api.list_pods(key):
        if str(pod.get("name", "")).startswith(NAME_PREFIX):
            api.delete_pod(key, pod["id"])
            killed.append(pod["id"])
    return killed
