"""Shared builders for ops tests: stores with a champion and candidates."""

from vektr_ops.manifest import Manifest, add_candidate, load, promote, save
from vektr_ops.store import LocalStore

A, B, C = "a" * 40, "b" * 40, "c" * 40


def store_with_champion(path, commit=A, box_nps=None):
    s = LocalStore(path)
    m = Manifest(None, [])
    add_candidate(m, commit, "2026-11-27T00:00:00Z")
    promote(m, commit[:12], bootstrap=True)
    m.get(commit[:12]).box_nps = box_nps
    save(s, m)
    return s


def store_with_champion_only(tmp_path):
    s = store_with_champion(tmp_path / "store", A, 2_850_000)
    return s, load(s)


def store_with_champion_and_candidate(tmp_path):
    s = store_with_champion(tmp_path / "store", A, 2_850_000)
    m = load(s)
    add_candidate(m, B, "2026-11-28T00:00:00Z")
    save(s, m)
    return s, m


def promote_accepted(s, commit):
    m = load(s)
    v = add_candidate(m, commit, "2026-11-28T00:00:00Z")
    v.gate = {"decision": "accept", "llr": 3.0}
    promote(m, v.id)
    save(s, m)
