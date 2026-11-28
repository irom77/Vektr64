"""A thin client for the RunPod REST API (v2): just the pod calls the burst launcher needs."""

import json
import urllib.error
import urllib.request

API = "https://api.runpod.io/v2"


class RunpodError(Exception):
    """RunPod refused a call: the HTTP status and what it said."""

    def __init__(self, status: int, message: str):
        super().__init__(f"RunPod answered {status}: {message}")
        self.status = status
        self.message = message


def _call(key: str, method: str, path: str, body: dict | None, send, *, gone_ok: bool = False):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        API + path,
        data=data,
        method=method,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with send(request, timeout=60) as response:
            raw = response.read()
    except urllib.error.HTTPError as error:
        if gone_ok and error.code == 404:
            return None
        text = error.read().decode(errors="replace") if hasattr(error, "read") else ""
        raise RunpodError(error.code, _message(text) or str(error.reason)) from None
    return json.loads(raw) if raw else None


def _message(text: str) -> str:
    try:
        parsed = json.loads(text)
    except ValueError:
        return text.strip()[:300]
    if isinstance(parsed, dict):
        return str(parsed.get("error") or parsed.get("message") or text)[:300]
    return text[:300]


def create_pod(key: str, body: dict, send=urllib.request.urlopen) -> dict:
    """Create a pod; returns it (with `id` and `cost`). Raises `RunpodError` on a refusal."""
    return _call(key, "POST", "/pods", body, send)


def get_pod(key: str, id: str, send=urllib.request.urlopen) -> dict | None:
    """The pod, or None when it no longer exists."""
    return _call(key, "GET", f"/pods/{id}", None, send, gone_ok=True)


def delete_pod(key: str, id: str, send=urllib.request.urlopen) -> None:
    """Delete a pod. Already gone (404) counts as deleted."""
    _call(key, "DELETE", f"/pods/{id}", None, send, gone_ok=True)


def list_pods(key: str, send=urllib.request.urlopen) -> list[dict]:
    return _call(key, "GET", "/pods", None, send) or []
