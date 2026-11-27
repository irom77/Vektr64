"""The version manifest: which engine builds exist, and which one plays."""

from dataclasses import asdict, dataclass, field

from vektr_ops.store import Store, read_json, write_json

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
    """Read the manifest; an empty one when the store has none yet."""
    raw = read_json(store, "nets", PATH)
    if raw is None:
        return Manifest()
    return Manifest(raw.get("champion"), [Version(**v) for v in raw.get("versions", [])])


def save(store: Store, m: Manifest) -> None:
    body = {"schema": SCHEMA, "champion": m.champion, "versions": [asdict(v) for v in m.versions]}
    write_json(store, "nets", PATH, body, overwrite=True)


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
