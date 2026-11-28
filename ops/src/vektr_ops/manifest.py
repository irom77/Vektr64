"""The version manifest: which engine builds exist, and which one plays."""

from dataclasses import asdict, dataclass, field

import json

from vektr_ops.store import Store, StoreConflict

PATH = "manifest.json"
SCHEMA = 1


class NotPromotable(Exception):
    """The requested promotion or rollback is not allowed."""


@dataclass
class Version:
    id: str
    engine_commit: str
    net: str | None = None
    parent: str | None = None
    status: str = "candidate"  # candidate, champion, rejected or retired
    created: str = ""
    gate: dict | None = None
    anchor: dict | None = None
    box_nps: int | None = None


@dataclass
class Manifest:
    champion: str | None = None
    versions: list[Version] = field(default_factory=list)
    # The store state this was loaded from; never saved. A manifest is saved once per load.
    revision: str | None = field(default=None, compare=False)

    def get(self, id: str) -> Version:
        for version in self.versions:
            if version.id == id:
                return version
        raise KeyError(id)

    def champion_version(self) -> Version | None:
        return self.get(self.champion) if self.champion else None

    def candidates(self) -> list[Version]:
        return [v for v in self.versions if v.status == "candidate"]


def load(store: Store) -> Manifest:
    """Read the newest manifest; an empty one when the store has none yet."""
    raw, revision = store.read_versioned("nets", PATH)
    if raw is None:
        return Manifest(revision=revision)
    body = json.loads(raw)
    versions = [Version(**v) for v in body.get("versions", [])]
    return Manifest(body.get("champion"), versions, revision)


def save(store: Store, m: Manifest) -> None:
    """Write the manifest; `StoreConflict` if someone else wrote it since `m` was loaded."""
    body = {"schema": SCHEMA, "champion": m.champion, "versions": [asdict(v) for v in m.versions]}
    data = json.dumps(body, indent=1, sort_keys=True).encode()
    store.write_bytes("nets", PATH, data, overwrite=True, expect=m.revision)


def update(store: Store, change, attempts: int = 5):
    """Load, apply `change(manifest)`, save; start over if another machine wrote in between.

    Returns what `change` returned. Nothing is saved if `change` raises or changes nothing.
    """
    for attempt in range(attempts):
        m = load(store)
        before = (m.champion, [asdict(v) for v in m.versions])
        result = change(m)
        if before == (m.champion, [asdict(v) for v in m.versions]):
            return result
        try:
            save(store, m)
        except StoreConflict:
            if attempt == attempts - 1:
                raise
            continue
        return result


def version_id(engine_commit: str, net: str | None = None) -> str:
    """A version is named by its engine commit (a network hash joins it in stage 3)."""
    return engine_commit[:12]


def add_candidate(m: Manifest, engine_commit: str, now: str) -> Version:
    """Register a commit as a candidate; returns the existing version if already known."""
    id = version_id(engine_commit)
    try:
        return m.get(id)
    except KeyError:
        version = Version(id=id, engine_commit=engine_commit, created=now)
        m.versions.append(version)
        return version


def promote(m: Manifest, id: str, *, bootstrap: bool = False) -> None:
    """Make a version the champion. Needs an accepted gate, or `bootstrap` with no champion."""
    version = m.get(id)
    if m.champion == id:
        return
    if bootstrap:
        if m.champion is not None:
            raise NotPromotable("bootstrap is only for the first version; a champion exists")
    elif not version.gate or version.gate.get("decision") != "accept":
        raise NotPromotable(f"{id} has not passed the gate")
    old = m.champion_version()
    if old is not None:
        old.status = "retired"
        version.parent = old.id
    version.status = "champion"
    m.champion = id


def reject(m: Manifest, id: str, gate: dict) -> None:
    """Record a failed gate. The version is kept for reference and never played."""
    version = m.get(id)
    if version.status != "candidate":
        return
    version.gate = gate
    version.status = "rejected"


def rollback(m: Manifest) -> Version:
    """Restore the champion's parent as champion."""
    current = m.champion_version()
    if current is None or current.parent is None:
        raise NotPromotable("there is no previous champion to roll back to")
    previous = m.get(current.parent)
    current.status = "retired"
    previous.status = "champion"
    m.champion = previous.id
    return previous
