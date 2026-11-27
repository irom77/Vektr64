from pathlib import Path

import pytest
from helpers import store_with_champion_only
from pytest import approx

from vektr_ops import anchors, gate
from vektr_ops.anchors import anchor_step, margin, rating
from vektr_ops.manifest import load
from vektr_ops.results import BatchResult


@pytest.fixture(autouse=True)
def no_real_tools(monkeypatch):
    monkeypatch.setattr(anchors, "ensure_build", lambda root, commit: Path(root) / commit[:12])
    monkeypatch.setattr(anchors, "bench_nps", lambda binary: 2_850_000)
    monkeypatch.setattr(gate.tools, "ensure", lambda tool, tools_dir: Path(tools_dir) / tool.member)
    monkeypatch.setattr(gate, "bot_here", lambda: False)


def test_even_score_equals_the_anchor():
    assert rating([(2000, 50, 0, 50)]) == approx(2000, abs=0.5)


def test_three_quarters_is_191_above():
    assert rating([(1800, 75, 0, 25)]) == approx(1990.85, abs=0.5)


def test_several_anchors_are_combined():
    assert rating([(1800, 75, 0, 25), (2181.7, 25, 0, 75)]) == approx(1990.85, abs=1.0)


def test_draws_count_as_half():
    assert rating([(2000, 0, 100, 0)]) == approx(2000, abs=0.5)


def test_no_games_gives_no_rating():
    assert rating([]) is None and rating([(1500, 0, 0, 0)]) is None


def test_all_wins_is_clamped_not_infinite():
    assert rating([(2400, 10, 0, 0)]) == 4000
    assert rating([(1500, 0, 0, 10)]) == 0


def test_margin_shrinks_with_more_games():
    assert 0 < margin([(2000, 200, 0, 200)]) < margin([(2000, 20, 0, 20)])


def test_anchor_step_records_the_rating_when_complete(tmp_path):
    s, _ = store_with_champion_only(tmp_path)
    seen = []

    def runner(fastchess, a, b, **k):
        seen.append((b.name, b.options.get("UCI_Elo"), b.options.get("UCI_LimitStrength")))
        return BatchResult((0, 0, 50, 0, 0), 50, 0, 50, 100, 0), "pgn"

    while anchor_step(s, tmp_path, load(s).champion_version(), runner=runner) is not None:
        pass
    a = load(s).champion_version().anchor
    assert a["games"] == 800 and a["elo"] == approx(1950, abs=1)
    assert len(seen) == 8 and ("sf1500", "1500", "true") in seen and ("sf2400", "2400", "true") in seen


def test_discarded_anchor_batches_do_not_count(tmp_path):
    s, _ = store_with_champion_only(tmp_path)
    bad = lambda *a, **k: (BatchResult((0, 0, 50, 0, 0), 50, 0, 50, 100, 9), "pgn")
    out = anchor_step(s, tmp_path, load(s).champion_version(), runner=bad)
    assert out["games"] == 0 and load(s).champion_version().anchor is None
