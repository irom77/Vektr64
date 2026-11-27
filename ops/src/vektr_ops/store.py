"""The external store: a local directory in tests, Hugging Face Hub in production."""

import fcntl
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Protocol

REPOS = ("nets", "data")


class StoreUnavailable(Exception):
    """The store is not configured or cannot be reached right now."""


class StoreConflict(Exception):
    """The file changed in the store since it was read; read it again and redo the change."""


class Store(Protocol):
    def read_bytes(self, repo: str, path: str) -> bytes | None: ...

    def read_versioned(self, repo: str, path: str) -> tuple[bytes | None, str]:
        """The file and a token for the state it was read in, to hand back as `expect`."""
        ...

    def write_bytes(
        self, repo: str, path: str, data: bytes, *, overwrite: bool = False, expect: str | None = None
    ) -> None:
        """Write a file. With `expect`, raises `StoreConflict` unless the state is still that one."""
        ...

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

    @staticmethod
    def _token(data: bytes | None) -> str:
        return "absent" if data is None else hashlib.sha256(data).hexdigest()

    @contextmanager
    def _locked(self, repo: str):
        self.root.mkdir(parents=True, exist_ok=True)
        with open(self.root / f".lock-{repo}", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def read_versioned(self, repo: str, path: str) -> tuple[bytes | None, str]:
        data = self.read_bytes(repo, path)
        return data, self._token(data)

    def write_bytes(
        self, repo: str, path: str, data: bytes, *, overwrite: bool = False, expect: str | None = None
    ) -> None:
        target = self.root / repo / checked(repo, path)
        with self._locked(repo):
            if expect is not None:
                if self._token(self.read_bytes(repo, path)) != expect:
                    raise StoreConflict(f"{repo}:{path} changed since it was read")
            elif target.exists() and not overwrite:
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

    def _download(self, repo: str, path: str, revision: str | None) -> bytes | None:
        import huggingface_hub
        from huggingface_hub.errors import RemoteEntryNotFoundError

        try:
            _, repo_id, kind = self._repo(repo)
            local = huggingface_hub.hf_hub_download(
                repo_id, path, repo_type=kind, token=self.token, revision=revision
            )
            return Path(local).read_bytes()
        except RemoteEntryNotFoundError:  # the Hub answered: no such file
            return None
        except Exception as error:  # network, auth and server faults all mean "not now"
            raise StoreUnavailable(f"{type(error).__name__}: {error}") from error

    def read_bytes(self, repo: str, path: str) -> bytes | None:
        return self._download(repo, checked(repo, path), None)

    def read_versioned(self, repo: str, path: str) -> tuple[bytes | None, str]:
        """The file as of the repo's newest commit, asked for fresh so no cached copy is used."""
        path = checked(repo, path)
        try:
            api, repo_id, kind = self._repo(repo)
            head = api.repo_info(repo_id, repo_type=kind).sha
        except Exception as error:
            raise StoreUnavailable(f"{type(error).__name__}: {error}") from error
        return self._download(repo, path, head), head

    def write_bytes(
        self, repo: str, path: str, data: bytes, *, overwrite: bool = False, expect: str | None = None
    ) -> None:
        from huggingface_hub.errors import HfHubHTTPError

        path = checked(repo, path)
        try:
            api, repo_id, kind = self._repo(repo)
            exists = (
                expect is None and not overwrite and api.file_exists(repo_id, path, repo_type=kind)
            )
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
                parent_commit=expect,
            )
        except HfHubHTTPError as error:
            status = getattr(getattr(error, "response", None), "status_code", None)
            if expect is not None and status in (409, 412):
                raise StoreConflict(f"{repo}:{path} changed since it was read") from error
            raise StoreUnavailable(f"{type(error).__name__}: {error}") from error
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
