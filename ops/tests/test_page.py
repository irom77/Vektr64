from helpers import A, store_with_champion

from vektr_ops.manifest import Version
from vektr_ops.page import PageData, collect, line_chart, render


def data(**over):
    base = dict(
        generated="2026-11-28 10:00 UTC",
        blitz_rating=1850,
        blitz_rd=62,
        blitz_games=73,
        history=[("2026-11-27", 3000), ("2026-11-28", 1850)],
        versions=[],
        champion=None,
        playing_id=None,
        sync_error=None,
    )
    return PageData(**{**base, **over})


def test_page_is_self_contained():
    html = render(data())
    assert "<script" not in html and "http://" not in html and "https://" not in html
    assert "<svg" in html and "1850" in html


def test_eligibility_shows_both_thresholds():
    html = render(data(blitz_games=12, blitz_rd=140))
    assert "12 / 50" in html and "140 / 75" in html and "not yet eligible" in html.lower()
    ok = render(data()).lower()
    assert "eligible" in ok and "not yet eligible" not in ok


def test_gate_history_lists_every_decision():
    vs = [
        Version("a" * 12, "a" * 40, None, None, "champion", "t0", None,
                {"elo": 1950.0, "margin": 30.0, "games": 800}, 2850000),
        Version("b" * 12, "b" * 40, None, None, "rejected", "t1",
                {"decision": "reject", "llr": -2.97, "games": 5200, "elo": -1.2, "margin": 6.0},
                None, None),
    ]
    html = render(data(versions=vs, champion="a" * 12, playing_id="a" * 12))
    assert "bbbbbbbbbbbb" in html and "reject" in html and "5200" in html and "1950" in html


def test_mismatch_between_champion_and_playing_is_called_out():
    html = render(data(champion="a" * 12, playing_id="b" * 12))
    assert "not the champion" in html
    assert "not the champion" not in render(data(champion="a" * 12, playing_id="a" * 12))


def test_sync_error_is_shown_and_escaped():
    assert "&lt;b&gt;boom" in render(data(sync_error="<b>boom"))


def test_empty_account_renders():
    html = render(data(blitz_rating=None, blitz_rd=None, blitz_games=0, history=[]))
    assert "No data yet." in html


def test_line_chart_needs_two_points():
    assert line_chart([("a", 1.0)]) == "<p>No data yet.</p>"
    assert line_chart([("a", 1.0), ("b", 2.0)]).startswith("<svg")


def test_line_chart_survives_a_flat_series():
    assert line_chart([("2026-11-27", 1500.0), ("2026-11-28", 1500.0)]).startswith("<svg")


def test_collect_reads_lichess_and_the_manifest(tmp_path):
    s = store_with_champion(tmp_path / "store", A)

    def fetch_json(url):
        if url.endswith("/rating-history"):
            return [{"name": "Bullet", "points": []},
                    {"name": "Blitz", "points": [[2026, 10, 27, 3000], [2026, 10, 28, 1900]]}]
        return {"perfs": {"blitz": {"rating": 1900, "rd": 80, "games": 40}}}

    d = collect(s, tmp_path, fetch_json)
    assert d.history == [("2026-11-27", 3000), ("2026-11-28", 1900)]  # months are zero-based
    assert (d.blitz_rating, d.blitz_rd, d.blitz_games) == (1900, 80, 40)
    assert d.champion == A[:12] and d.playing_id is None


def test_collect_survives_lichess_and_store_being_down(tmp_path):
    def fetch_json(url):
        raise OSError("offline")

    d = collect(None, tmp_path, fetch_json)
    assert d.history == [] and d.blitz_rating is None and d.versions == []
