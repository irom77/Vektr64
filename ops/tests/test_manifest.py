import copy

import pytest

from vektr_ops.manifest import (
    Manifest,
    NotPromotable,
    add_candidate,
    load,
    promote,
    reject,
    rollback,
    save,
)
from vektr_ops.store import LocalStore

A, B, C = "a" * 40, "b" * 40, "c" * 40
ACCEPT = {"decision": "accept", "llr": 3.0}


def boot():
    m = Manifest(None, [])
    add_candidate(m, A, "t0")
    promote(m, A[:12], bootstrap=True)
    return m


def test_empty_store_gives_empty_manifest(tmp_path):
    assert load(LocalStore(tmp_path)) == Manifest(None, [])


def test_save_and_load_round_trip(tmp_path):
    s = LocalStore(tmp_path)
    m = boot()
    save(s, m)
    assert load(s) == m


def test_bootstrap_only_when_there_is_no_champion():
    m = boot()
    add_candidate(m, B, "t1")
    with pytest.raises(NotPromotable):
        promote(m, B[:12], bootstrap=True)


def test_promotion_needs_an_accepted_gate():
    m = boot()
    v = add_candidate(m, B, "t1")
    with pytest.raises(NotPromotable):
        promote(m, v.id)
    v.gate = {"decision": "reject"}
    with pytest.raises(NotPromotable):
        promote(m, v.id)


def test_promote_retires_the_old_champion_and_links_parent():
    m = boot()
    v = add_candidate(m, B, "t1")
    v.gate = ACCEPT
    promote(m, v.id)
    assert m.champion == B[:12] and v.status == "champion" and v.parent == A[:12]
    assert m.get(A[:12]).status == "retired"


def test_rollback_restores_the_parent():
    m = boot()
    v = add_candidate(m, B, "t1")
    v.gate = ACCEPT
    promote(m, v.id)
    restored = rollback(m)
    assert restored.id == A[:12] and m.champion == A[:12]
    assert restored.status == "champion" and m.get(B[:12]).status == "retired"


def test_rollback_without_a_parent_fails():
    with pytest.raises(NotPromotable):
        rollback(boot())


def test_reject_records_the_gate_and_never_plays():
    m = boot()
    v = add_candidate(m, B, "t1")
    reject(m, v.id, {"decision": "reject"})
    assert v.status == "rejected" and m.champion == A[:12] and m.candidates() == []


def test_adding_the_same_commit_twice_is_one_version():
    m = boot()
    assert add_candidate(m, A, "t9") is m.get(A[:12]) and len(m.versions) == 1


def test_promoting_the_champion_again_changes_nothing():
    m = boot()
    before = copy.deepcopy(m)
    promote(m, A[:12], bootstrap=True)
    assert m == before
