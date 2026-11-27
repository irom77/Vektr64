"""The external store: a local directory in tests, Hugging Face Hub in production."""

import json
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Protocol

REPOS = ("nets", "data")


class StoreUnavailable(Exception):
    """The store is not configured or cannot be reached right now."""


class Store(Protocol):
    def read_bytes(self, repo: str, path: str) -> bytes | None: ...

    def write_bytes(self, repo: str, path: str, data: bytes, *, overwrite: bool = False) -> None: ...

    def list(self, repo: str, prefix: str) -> list[str]: ...


def checked(repo: str, path: str) -> str:
    """Validate a repo name and a relative path; returns the normalised path."""
    if repo not in REPOS:
        raise ValueError(f"unknown repo {repo!r}")
    parts = PurePosixPath(path).parts
    if not parts or path.startswith("/") or ".." in parts:
        raise ValueError(f"bad store path {path!r}")
    return str(PurePosixPath(*parts))


def read_json(store: Store, repo: str, path: str) -> dict | None:
    raw = store.read_bytes(repo, path)
    return None if raw is None else json.loads(raw)


def write_json(store: Store, repo: str, path: str, obj: dict, *, overwrite: bool = False) -> None:
    data = json.dumps(obj, indent=1, sort_keys=True).encode()
    store.write_bytes(repo, path, data, overwrite=overwrite)


class LocalStore:
    """Files at `root/<repo>/<path>`."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def read_bytes(self, repo: str, path: str) -> bytes | None:
        target = self.root / repo / checked(repo, path)
        return target.read_bytes() if target.is_file() else None

    def write_bytes(self, repo: str, path: str, data: bytes, *, overwrite: bool = False) -> None:
        target = self.root / repo / checked(repo, path)
        if target.exists() and not overwrite:
            raise FileExistsError(f"{repo}:{path} already exists")
        target.parent.mkdir(parents=True, exist_ok=True)
        handle, temp = tempfile.mkstemp(dir=target.parent)
        with os.fdopen(handle, "wb") as out:
            out.write(data)
        os.replace(temp, target)

    def list(self, repo: str, prefix: str) -> list[str]:
        if repo not in REPOS:
            raise ValueError(f"unknown repo {repo!r}")
        base = self.root / repo
        names = (str(p.relative_to(base).as_posix()) for p in base.rglob("*") if p.is_file())
        return sorted(n for n in names if n.startswith(prefix))


class HubStore:
    """Two private Hugging Face repos: `<user>/vektr64-nets` and `<user>/vektr64-data`."""

    KINDS = {"nets": "model", "data": "dataset"}

    def __init__(self, token: str) -> None:
        self.token = token
        self._api = None
        self._user: str | None = None
        self._ready: set[str] = set()

    def _repo(self, repo: str) -> tuple[object, str, str]:
        """The API client, the repo id and its type; creates the repo once if needed."""
        from huggingface_hub import HfApi

        if repo not in REPOS:
            raise ValueError(f"unknown repo {repo!r}")
        if self._api is None:
            self._api = HfApi(token=self.token)
        if self._user is None:
            self._user = self._api.whoami()["name"]
        repo_id, kind = f"{self._user}/vektr64-{repo}", self.KINDS[repo]
        if repo not in self._ready:
            self._api.create_repo(repo_id, repo_type=kind, private=True, exist_ok=True)
            self._ready.add(repo)
        return self._api, repo_id, kind

    def read_bytes(self, repo: str, path: str) -> bytes | None:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError

        path = checked(repo, path)
        try:
            _, repo_id, kind = self._repo(repo)
            local = hf_hub_download(repo_id, path, repo_type=kind, token=self.token)
            return Path(local).read_bytes()
        except EntryNotFoundError:
            return None
        except Exception as error:  # network, auth and server faults all mean "not now"
            raise StoreUnavailable(f"{type(error).__name__}: {error}") from error

    def write_bytes(self, repo: str, path: str, data: bytes, *, overwrite: bool = False) -> None:
        path = checked(repo, path)
        try:
            api, repo_id, kind = self._repo(repo)
            exists = not overwrite and api.file_exists(repo_id, path, repo_type=kind)
        except Exception as error:
            raise StoreUnavailable(f"{type(error).__name__}: {error}") from error
        if exists:
            raise FileExistsError(f"{repo}:{path} already exists")
        try:
            api.upload_file(
                path_or_fileobj=data,
                path_in_repo=path,
                repo_id=repo_id,
                repo_type=kind,
                commit_message=f"write {path}",
            )
        except Exception as error:
            raise StoreUnavailable(f"{type(error).__name__}: {error}") from error

    def list(self, repo: str, prefix: str) -> list[str]:
        try:
            api, repo_id, kind = self._repo(repo)
            names = api.list_repo_files(repo_id, repo_type=kind)
        except ValueError:
            raise
        except Exception as error:
            raise StoreUnavailable(f"{type(error).__name__}: {error}") from error
        return sorted(n for n in names if n.startswith(prefix))


def open_store(env: dict[str, str]) -> Store:
    """The store this machine is configured for: `VEKTR_STORE` wins over `HF_TOKEN`."""
    if env.get("VEKTR_STORE"):
        return LocalStore(Path(env["VEKTR_STORE"]))
    if env.get("HF_TOKEN"):
        return HubStore(env["HF_TOKEN"])
    raise StoreUnavailable("set HF_TOKEN or VEKTR_STORE")
