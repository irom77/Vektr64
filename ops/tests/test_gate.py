from datetime import datetime, timezone
from pathlib import Path

import pytest
from helpers import A, B, store_with_champion_and_candidate, store_with_champion_only

from vektr_ops import gate
from vektr_ops.gate import batch_name, pooled, step, upload_batch, verdict, work
from vektr_ops.manifest import load, save
from vektr_ops.results import BatchResult
from vektr_ops.store import LocalStore


@pytest.fixture(autouse=True)
def no_real_tools(monkeypatch):
    monkeypatch.setattr(gate, "ensure_build", lambda root, commit: Path(root) / commit[:12])
    monkeypatch.setattr(gate, "bench_nps", lambda binary: 2_850_000)
    monkeypatch.setattr(gate.tools, "ensure", lambda tool, tools_dir: Path(tools_dir) / tool.member)
    monkeypatch.setattr(gate, "bot_here", lambda: False)


def result(penta, bad=0):
    games = 2 * sum(penta)
    return BatchResult(penta, 0, games, 0, games, bad)


def test_names():
    assert gate.test_id("aaa", "bbb") == "aaa__bbb"
    when = datetime(2026, 11, 28, 9, 5, 7, tzinfo=timezone.utc)
    assert batch_name("box", when, "1f2e") == "box-20261128T090507Z-1f2e"


def test_pooling_sums_valid_batches_from_all_machines(tmp_path):
    s = LocalStore(tmp_path)
    upload_batch(s, "a__b", "box-1", result((1, 4, 10, 6, 2)), "pgn", {})
    upload_batch(s, "a__b", "laptop-1", result((1, 4, 10, 6, 2)), "pgn", {})
    assert pooled(s, "a__b") == ((2, 8, 20, 12, 4), 92)


def test_invalid_batch_is_kept_but_not_counted(tmp_path):
    s = LocalStore(tmp_path)
    upload_batch(s, "a__b", "box-1", result((0, 0, 50, 0, 0), bad=5), "pgn", {})
    assert pooled(s, "a__b") == ((0, 0, 0, 0, 0), 0)
    assert s.list("data", "gate/a__b/discarded/") != []


def test_same_batch_name_cannot_be_uploaded_twice(tmp_path):
    s = LocalStore(tmp_path)
    upload_batch(s, "a__b", "box-1", result((0, 0, 1, 0, 0)), "pgn", {})
    with pytest.raises(FileExistsError):
        upload_batch(s, "a__b", "box-1", result((0, 0, 1, 0, 0)), "pgn", {})


def test_verdict_accepts_rejects_and_continues():
    assert verdict((100, 400, 1000, 480, 120), 4200)["decision"] == "accept"
    assert verdict((200, 600, 1000, 400, 100), 4600)["decision"] == "reject"
    assert verdict((5, 20, 50, 20, 5), 200)["decision"] == "continue"


def test_game_cap_rejects():
    v = verdict((5, 20, 50, 20, 5), 40_000)
    assert v["decision"] == "reject" and v["reason"] == "cap"


def test_undecided_test_continues_below_the_cap():
    assert verdict((5, 20, 50, 20, 5), 39_900)["decision"] == "continue"


def test_step_promotes_on_accept(tmp_path):
    s, m = store_with_champion_and_candidate(tmp_path)
    runner = lambda *a, **k: (result((100, 400, 1000, 480, 120)), "pgn")
    v = step(s, tmp_path, m.get(B[:12]), runner=runner)
    after = load(s)
    assert v["decision"] == "accept" and after.champion == B[:12]
    assert after.get(B[:12]).gate["decision"] == "accept"


def test_step_rejects_and_keeps_the_champion(tmp_path):
    s, m = store_with_champion_and_candidate(tmp_path)
    runner = lambda *a, **k: (result((200, 600, 1000, 400, 100)), "pgn")
    step(s, tmp_path, m.get(B[:12]), runner=runner)
    after = load(s)
    assert after.champion == A[:12] and after.get(B[:12]).status == "rejected"


def test_step_continues_while_undecided(tmp_path):
    s, m = store_with_champion_and_candidate(tmp_path)
    v = step(s, tmp_path, m.get(B[:12]), runner=lambda *a, **k: (result((5, 20, 50, 20, 5)), "pgn"))
    assert v["decision"] == "continue" and load(s).get(B[:12]).status == "candidate"


def test_step_on_an_already_decided_candidate_plays_nothing(tmp_path):
    s, m = store_with_champion_and_candidate(tmp_path)
    runner = lambda *a, **k: (result((100, 400, 1000, 480, 120)), "pgn")
    step(s, tmp_path, m.get(B[:12]), runner=runner)

    def boom(*a, **k):
        raise AssertionError("must not play")

    assert step(s, tmp_path, load(s).get(B[:12]), runner=boom)["decision"] == "accept"


def test_step_finishes_a_test_another_machine_already_decided(tmp_path):
    s, m = store_with_champion_and_candidate(tmp_path)
    upload_batch(s, gate.test_id(A[:12], B[:12]), "laptop-1", result((100, 400, 1000, 480, 120)), "p", {})

    def boom(*a, **k):
        raise AssertionError("must not play")

    assert step(s, tmp_path, m.get(B[:12]), runner=boom)["decision"] == "accept"
    assert load(s).champion == B[:12]


def test_step_passes_scaled_time_control_and_low_priority(tmp_path, monkeypatch):
    s, m = store_with_champion_and_candidate(tmp_path)
    monkeypatch.setattr(gate, "bench_nps", lambda binary: 5_000_000)
    monkeypatch.setattr(gate, "bot_here", lambda: True)
    monkeypatch.setattr(gate, "machine", lambda: (4, 2000))
    seen = {}

    def runner(fastchess, a, b, **k):
        seen.update(k, a=a.name, b=b.name)
        return result((5, 20, 50, 20, 5)), "pgn"

    step(s, tmp_path, m.get(B[:12]), runner=runner)
    assert seen["tc"] == "4.56+0.046" and seen["jobs"] == 2 and seen["low_priority"] is True
    assert (seen["a"], seen["b"]) == ("candidate", "champion")


def test_work_with_nothing_open_plays_nothing(tmp_path, monkeypatch):
    s, m = store_with_champion_only(tmp_path)
    m.champion_version().anchor = {"elo": 1900.0, "margin": 30.0, "games": 800}
    save(s, m)
    monkeypatch.setattr(gate, "run_batch", lambda *a, **k: 1 / 0)
    work(s, tmp_path, once=True, idle_s=0)


def test_work_measures_an_unmeasured_champion(tmp_path, monkeypatch):
    s, _ = store_with_champion_only(tmp_path)
    calls = []
    monkeypatch.setattr(
        "vektr_ops.anchors.anchor_step", lambda store, root, v: calls.append(v.id) or {"games": 100}
    )
    work(s, tmp_path, once=True, idle_s=0)
    assert calls == [A[:12]]


def test_work_advances_every_open_test(tmp_path, monkeypatch):
    s, _ = store_with_champion_and_candidate(tmp_path)
    monkeypatch.setattr("vektr_ops.anchors.anchor_step", lambda store, root, v: None)
    monkeypatch.setattr(gate, "run_batch", lambda *a, **k: (result((5, 20, 50, 20, 5)), "pgn"))
    work(s, tmp_path, once=True, idle_s=0)
    assert pooled(s, gate.test_id(A[:12], B[:12]))[1] == 200


def add(s, commit):
    from vektr_ops.manifest import add_candidate

    m = load(s)
    add_candidate(m, commit, "2026-11-28T00:00:00Z")
    save(s, m)
    return load(s).get(commit[:12])


def test_accept_records_the_champion_it_was_played_against(tmp_path):
    s, m = store_with_champion_and_candidate(tmp_path)
    step(s, tmp_path, m.get(B[:12]), runner=lambda *a, **k: (result((100, 400, 1000, 480, 120)), "p"))
    assert load(s).get(B[:12]).gate["champion"] == A[:12]


def test_verdict_against_a_replaced_champion_is_not_applied(tmp_path):
    from helpers import C, promote_accepted

    s, _ = store_with_champion_and_candidate(tmp_path)
    c = add(s, C)

    def runner(*a, **k):  # another machine promotes B while this batch is being played
        promote_accepted(s, B)
        return result((100, 400, 1000, 480, 120)), "pgn"

    v = step(s, tmp_path, c, runner=runner)
    after = load(s)
    assert v["decision"] == "continue"
    assert after.champion == B[:12] and after.get(C[:12]).status == "candidate"


def test_worker_survives_a_candidate_that_does_not_build(tmp_path, monkeypatch):
    from helpers import C

    from vektr_ops.builds import BuildFailed

    s, _ = store_with_champion_and_candidate(tmp_path)
    add(s, C)

    def build(root, commit):
        if commit == B:
            raise BuildFailed("error[E0425]: cannot find value")
        return Path(root) / commit[:12]

    monkeypatch.setattr(gate, "ensure_build", build)
    monkeypatch.setattr("vektr_ops.anchors.anchor_step", lambda store, root, v: None)
    monkeypatch.setattr(gate, "run_batch", lambda *a, **k: (result((5, 20, 50, 20, 5)), "pgn"))
    work(s, tmp_path, once=True, idle_s=0)
    assert pooled(s, gate.test_id(A[:12], C[:12]))[1] == 200
    assert load(s).get(B[:12]).status == "candidate"  # a local build fault rejects nothing


def test_candidate_with_only_discarded_batches_is_rejected(tmp_path):
    s, m = store_with_champion_and_candidate(tmp_path)
    runner = lambda *a, **k: (result((0, 0, 50, 0, 0), bad=10), "pgn")
    for _ in range(gate.DISCARD_LIMIT - 1):
        assert step(s, tmp_path, load(s).get(B[:12]), runner=runner)["decision"] == "continue"
    v = step(s, tmp_path, load(s).get(B[:12]), runner=runner)
    after = load(s).get(B[:12])
    assert v["decision"] == "reject" and v["reason"] == "invalid"
    assert after.status == "rejected" and load(s).champion == A[:12]


def test_discarded_batches_do_not_reject_once_some_are_counted(tmp_path):
    s, m = store_with_champion_and_candidate(tmp_path)
    step(s, tmp_path, m.get(B[:12]), runner=lambda *a, **k: (result((5, 20, 50, 20, 5)), "pgn"))
    bad = lambda *a, **k: (result((0, 0, 50, 0, 0), bad=10), "pgn")
    for _ in range(gate.DISCARD_LIMIT + 1):
        step(s, tmp_path, load(s).get(B[:12]), runner=bad)
    assert load(s).get(B[:12]).status == "candidate"
