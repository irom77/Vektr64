import pytest
from helpers import A, B, promote_accepted, store_with_champion

from vektr_ops.builds import BuildFailed
from vektr_ops.manifest import load, rollback, save
from vektr_ops.store import LocalStore, StoreUnavailable
from vektr_ops.sync import SyncRefused, current_id, engine_dir, sync, upload_games


def fake_build(root, commit):
    p = root / "builds" / commit[:12] / "vektr"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")
    return p


def run(s, root, build=fake_build, bench=lambda p: 2_850_000, starts=lambda p: True, netcheck=lambda p: True):
    return sync(s, root, build=build, bench=bench, starts=starts, netcheck=netcheck)


def test_first_sync_switches_to_the_champion(tmp_path):
    s = store_with_champion(tmp_path / "store", A)
    assert run(s, tmp_path) == f"switched to {A[:12]}"
    assert current_id(tmp_path) == A[:12]
    assert load(s).get(A[:12]).box_nps == 2_850_000


def test_sync_is_idempotent(tmp_path):
    s = store_with_champion(tmp_path / "store", A)
    run(s, tmp_path)
    assert run(s, tmp_path) == "unchanged"


def test_recorded_box_speed_is_not_overwritten(tmp_path):
    s = store_with_champion(tmp_path / "store", A, box_nps=1_000_000)
    run(s, tmp_path)
    assert load(s).get(A[:12]).box_nps == 1_000_000


def test_slow_build_is_refused(tmp_path):
    s = store_with_champion(tmp_path / "store", A)
    run(s, tmp_path)
    promote_accepted(s, B)
    speeds = {A[:12]: 2_850_000, B[:12]: 2_500_000}
    with pytest.raises(SyncRefused, match="slower"):
        run(s, tmp_path, bench=lambda p: speeds[p.parent.name])
    assert current_id(tmp_path) == A[:12]
    assert "slower" in (tmp_path / "state" / "sync-error").read_text()


def test_slightly_slower_build_is_allowed(tmp_path):
    s = store_with_champion(tmp_path / "store", A)
    run(s, tmp_path)
    promote_accepted(s, B)
    speeds = {A[:12]: 2_850_000, B[:12]: 2_600_000}
    assert run(s, tmp_path, bench=lambda p: speeds[p.parent.name]) == f"switched to {B[:12]}"


def test_build_that_does_not_start_is_refused(tmp_path):
    s = store_with_champion(tmp_path / "store", A)
    run(s, tmp_path)
    promote_accepted(s, B)
    with pytest.raises(SyncRefused, match="does not start"):
        run(s, tmp_path, starts=lambda p: p.parent.name != B[:12])
    assert current_id(tmp_path) == A[:12]


def test_build_failure_leaves_the_current_engine(tmp_path):
    s = store_with_champion(tmp_path / "store", A)
    run(s, tmp_path)
    promote_accepted(s, B)

    def broken(root, commit):
        raise BuildFailed("cargo said no")

    with pytest.raises(BuildFailed):
        run(s, tmp_path, build=broken)
    assert current_id(tmp_path) == A[:12]
    assert "cargo said no" in (tmp_path / "state" / "sync-error").read_text()


def test_unreachable_store_leaves_the_current_engine(tmp_path):
    s = store_with_champion(tmp_path / "store", A)
    run(s, tmp_path)

    class Down:
        def read_versioned(self, *a):
            raise StoreUnavailable("offline")

    with pytest.raises(StoreUnavailable):
        run(Down(), tmp_path)
    assert current_id(tmp_path) == A[:12]


def test_rollback_switches_back_even_to_a_slower_build(tmp_path):
    s = store_with_champion(tmp_path / "store", A)
    run(s, tmp_path)
    promote_accepted(s, B)
    speeds = {A[:12]: 2_000_000, B[:12]: 2_850_000}
    run(s, tmp_path, bench=lambda p: speeds[p.parent.name])
    m = load(s)
    rollback(m)
    save(s, m)

    def no_build(root, commit):
        assert (root / "builds" / commit[:12] / "vektr").exists()
        return root / "builds" / commit[:12] / "vektr"

    out = run(s, tmp_path, build=no_build, bench=lambda p: speeds[p.parent.name])
    assert out == f"switched to {A[:12]}" and current_id(tmp_path) == A[:12]


def test_success_clears_an_old_error(tmp_path):
    s = store_with_champion(tmp_path / "store", A)
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "sync-error").write_text("old")
    run(s, tmp_path)
    assert not (tmp_path / "state" / "sync-error").exists()


def test_no_champion_is_refused(tmp_path):
    with pytest.raises(SyncRefused, match="no champion"):
        run(LocalStore(tmp_path / "store"), tmp_path)


def test_games_are_uploaded_once(tmp_path):
    s = LocalStore(tmp_path / "store")
    rec = tmp_path / "bot" / "game_records"
    rec.mkdir(parents=True)
    (rec / "a.pgn").write_text("1. e4")
    assert upload_games(s, tmp_path) == 0  # just written: the game may still be in progress
    assert upload_games(s, tmp_path, min_age_s=0) == 1 and upload_games(s, tmp_path, min_age_s=0) == 0
    assert s.read_bytes("data", "games/lichess/a.pgn") == b"1. e4"


def test_engine_dir_falls_back_to_the_working_tree(tmp_path):
    assert engine_dir(tmp_path) == tmp_path / "engine" / "target" / "release"
    s = store_with_champion(tmp_path / "store", A)
    run(s, tmp_path)
    assert engine_dir(tmp_path) == tmp_path / "builds" / "current"


def test_build_that_left_no_binary_is_not_switched_to(tmp_path):
    s = store_with_champion(tmp_path / "store", A)
    run(s, tmp_path)
    promote_accepted(s, B)
    with pytest.raises(SyncRefused, match="no engine binary"):
        run(s, tmp_path, build=lambda root, commit: root / "builds" / commit[:12] / "vektr")
    assert current_id(tmp_path) == A[:12]


def test_current_pointing_at_a_missing_build_is_repaired(tmp_path):
    import shutil

    s = store_with_champion(tmp_path / "store", A)
    run(s, tmp_path)
    shutil.rmtree(tmp_path / "builds" / A[:12])
    assert run(s, tmp_path) == f"switched to {A[:12]}"
    assert (tmp_path / "builds" / "current" / "vektr").exists()


def with_ratio(s, ratio, champion=A[:12]):
    m = load(s)
    v = m.get(B[:12])
    v.gate = {"decision": "accept", "champion": champion, **({"speed_ratio": ratio} if ratio else {})}
    save(s, m)


def slower_network_setup(tmp_path, ratio, champion=A[:12]):
    s = store_with_champion(tmp_path / "store", A)
    run(s, tmp_path)
    promote_accepted(s, B)
    with_ratio(s, ratio, champion)
    return s


def run_speeds(s, root, new, **kw):
    speeds = {A[:12]: 2_900_000, B[:12]: new}
    return run(s, root, bench=lambda p: speeds[p.parent.name], **kw)


def test_a_slower_network_build_is_accepted_when_the_gate_expected_it(tmp_path):
    s = slower_network_setup(tmp_path, 0.7)
    assert run_speeds(s, tmp_path, 2_000_000) == f"switched to {B[:12]}"


def test_a_build_far_below_the_expected_ratio_is_refused(tmp_path):
    s = slower_network_setup(tmp_path, 0.7)
    with pytest.raises(SyncRefused, match="slower"):
        run_speeds(s, tmp_path, 1_500_000)
    assert current_id(tmp_path) == A[:12]


def test_the_ratio_is_ignored_when_the_gate_was_against_another_build(tmp_path):
    s = slower_network_setup(tmp_path, 0.7, champion="f" * 12)
    with pytest.raises(SyncRefused, match="slower"):
        run_speeds(s, tmp_path, 2_000_000)


def netted(s):
    m = load(s)
    m.get(B[:12]).net = "0123456789abcdef"
    save(s, m)


def test_a_champion_with_a_network_must_pass_netcheck(tmp_path):
    s = store_with_champion(tmp_path / "store", A)
    run(s, tmp_path)
    promote_accepted(s, B)
    netted(s)
    with pytest.raises(SyncRefused, match="netcheck"):
        run(s, tmp_path, netcheck=lambda p: False)
    assert current_id(tmp_path) == A[:12]
    assert run(s, tmp_path, netcheck=lambda p: True) == f"switched to {B[:12]}"


def test_rollback_to_a_champion_without_a_network_needs_no_netcheck(tmp_path):
    s = store_with_champion(tmp_path / "store", A)
    run(s, tmp_path)
    promote_accepted(s, B)
    netted(s)
    run(s, tmp_path, netcheck=lambda p: True)
    m = load(s)
    rollback(m)
    save(s, m)

    def boom(p):
        raise AssertionError("no netcheck for a build without a network")

    assert run(s, tmp_path, netcheck=boom) == f"switched to {A[:12]}"
