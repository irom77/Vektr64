"""The static status page: rating history, eligibility, the anchor curve, gate decisions."""

import html
import json
import math
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

from vektr_ops.manifest import Version, load
from vektr_ops.store import Store, StoreUnavailable
from vektr_ops.sync import current_id

ACCOUNT = "Vektr64"
MIN_GAMES = 50
MAX_RD = 75
LICHESS = "https://lichess.org/api/user/" + ACCOUNT


@dataclass
class PageData:
    generated: str
    blitz_rating: int | None
    blitz_rd: int | None
    blitz_games: int
    history: list[tuple[str, int]]
    versions: list[Version] = field(default_factory=list)
    champion: str | None = None
    playing_id: str | None = None
    sync_error: str | None = None


def fetch_json(url: str):
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.load(response)


def collect(store: Store | None, root: Path, fetch_json=fetch_json) -> PageData:
    """Gather everything the page shows; anything unreachable is simply left empty."""
    rating = rd = None
    games = 0
    history: list[tuple[str, int]] = []
    try:
        blitz = fetch_json(LICHESS).get("perfs", {}).get("blitz", {})
        rating, rd, games = blitz.get("rating"), blitz.get("rd"), blitz.get("games", 0)
        for perf in fetch_json(LICHESS + "/rating-history"):
            if perf.get("name") == "Blitz":
                history = [(f"{y:04d}-{m + 1:02d}-{d:02d}", r) for y, m, d, r in perf["points"]]
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        pass

    versions: list[Version] = []
    champion = None
    if store is not None:
        try:
            m = load(store)
            versions, champion = m.versions, m.champion
        except StoreUnavailable:
            pass

    note = Path(root) / "state" / "sync-error"
    return PageData(
        generated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        blitz_rating=rating,
        blitz_rd=rd,
        blitz_games=games,
        history=history,
        versions=versions,
        champion=champion,
        playing_id=current_id(Path(root)),
        sync_error=note.read_text() if note.is_file() else None,
    )


def ticks(low: float, high: float, count: int = 4) -> list[float]:
    """Round axis values covering [low, high]; always at least two."""
    span = max(high - low, 1.0)
    step = 10 ** math.floor(math.log10(span / count))
    for factor in (1, 2, 5, 10):
        if span / (step * factor) <= count:
            step *= factor
            break
    step = max(step, 1)
    first = math.floor(low / step) * step
    last = math.ceil(high / step) * step
    if last <= first:
        last = first + step
    return [first + i * step for i in range(round((last - first) / step) + 1)]


def positions(labels: list[str]) -> list[float]:
    """Horizontal position 0..1 of each point: by date when labels are dates, else evenly."""
    try:
        days = [date.fromisoformat(label).toordinal() for label in labels]
        span = days[-1] - days[0]
        if span > 0 and days == sorted(days):
            return [(d - days[0]) / span for d in days]
    except ValueError:
        pass
    return [i / (len(labels) - 1) for i in range(len(labels))]


def line_chart(points: list[tuple[str, float]], *, width: int = 720, height: int = 240) -> str:
    """A single-series line chart as inline SVG; hovering a point shows its value."""
    if len(points) < 2:
        return "<p>No data yet.</p>"
    left, right, top, bottom = 48, 56, 12, 28
    plot_w, plot_h = width - left - right, height - top - bottom
    values = [v for _, v in points]
    grid = ticks(min(values), max(values))
    low, high = grid[0], grid[-1]
    xs = [left + p * plot_w for p in positions([label for label, _ in points])]
    ys = [top + (high - v) / (high - low) * plot_h for v in values]

    out = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Line chart, {len(points)} points, latest {values[-1]:.0f}">'
    ]
    for value in grid:
        y = top + (high - value) / (high - low) * plot_h
        out.append(f'<line class="grid" x1="{left}" x2="{left + plot_w}" y1="{y:.1f}" y2="{y:.1f}"/>')
        out.append(f'<text class="tick" x="{left - 8}" y="{y + 4:.1f}" text-anchor="end">{value:.0f}</text>')
    path = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(zip(xs, ys)))
    out.append(f'<path class="series" d="{path}"/>')
    show_dots = len(points) <= 60
    for (label, value), x, y in zip(points, xs, ys):
        tip = f"<title>{html.escape(str(label))}: {value:.0f}</title>"
        if show_dots:
            out.append(f'<circle class="dot" cx="{x:.1f}" cy="{y:.1f}" r="4"/>')
        out.append(f'<circle class="hit" cx="{x:.1f}" cy="{y:.1f}" r="12">{tip}</circle>')
    out.append(
        f'<text class="value" x="{xs[-1] + 8:.1f}" y="{ys[-1] + 4:.1f}">{values[-1]:.0f}</text>'
    )
    first, last = html.escape(str(points[0][0])), html.escape(str(points[-1][0]))
    out.append(f'<text class="tick" x="{left}" y="{height - 6}">{first}</text>')
    out.append(f'<text class="tick" x="{left + plot_w}" y="{height - 6}" text-anchor="end">{last}</text>')
    out.append("</svg>")
    return "".join(out)


STYLE = """
:root{color-scheme:light dark;--surface:#ffffff;--text:#191919;--muted:#666666;--series:#257de0;
--rule:#e4e4e4;--warn:#a34a00}
@media (prefers-color-scheme:dark){:root{--surface:#191919;--text:#efefef;--muted:#9c9c9c;
--series:#338ced;--rule:#383838;--warn:#f18a59}}
body{margin:0;background:var(--surface);color:var(--text);
font:16px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:760px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:1.5rem;margin:0 0 4px}h2{font-size:1.1rem;margin:36px 0 8px}
.muted,footer{color:var(--muted);font-size:.875rem}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:20px 0}
.tile{border:1px solid var(--rule);border-radius:8px;padding:12px}
.tile b{display:block;font-size:1.6rem;line-height:1.2}
.notice{border-left:4px solid var(--warn);padding:8px 12px;margin:16px 0}
svg{width:100%;height:auto;display:block}
.grid{stroke:var(--rule);stroke-width:1}.tick{fill:var(--muted);font-size:12px}
.series{fill:none;stroke:var(--series);stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.dot{fill:var(--series);stroke:var(--surface);stroke-width:2}.hit{fill:transparent}
.value{fill:var(--text);font-size:13px;font-weight:600}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:.9rem}
th,td{text-align:left;padding:6px 10px 6px 0;border-bottom:1px solid var(--rule);white-space:nowrap}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
code{font-size:.85em}
"""


def esc(value) -> str:
    return html.escape(str(value))


def signed(value: float) -> str:
    return f"{value:+.1f}"


def render(data: PageData) -> str:
    """The whole page as one self-contained HTML document."""
    eligible = (
        data.blitz_rd is not None and data.blitz_games >= MIN_GAMES and data.blitz_rd <= MAX_RD
    )
    rating = "–" if data.blitz_rating is None else esc(data.blitz_rating)
    rd = "–" if data.blitz_rd is None else esc(data.blitz_rd)
    parts = [
        "<!doctype html><html lang=en><meta charset=utf-8>",
        '<meta name=viewport content="width=device-width,initial-scale=1">',
        f"<title>Vektr64 status</title><style>{STYLE}</style><main>",
        "<h1>Vektr64</h1>",
        f'<p class=muted>Generated {esc(data.generated)}</p>',
        "<div class=tiles>",
        f"<div class=tile>Blitz rating<b>{rating}</b></div>",
        f"<div class=tile>Rated blitz games<b>{esc(data.blitz_games)} / {MIN_GAMES}</b></div>",
        f"<div class=tile>Rating deviation<b>{rd} / {MAX_RD}</b></div>",
        f"<div class=tile>Leaderboard<b>{'Eligible' if eligible else 'Not yet eligible'}</b></div>",
        "</div>",
    ]
    if data.sync_error:
        parts.append(f"<p class=notice>Last sync failed: {esc(data.sync_error)}</p>")
    if data.champion and data.playing_id != data.champion:
        playing = esc(data.playing_id) if data.playing_id else "the working-tree build"
        parts.append(
            f"<p class=notice>The engine in play (<code>{playing}</code>) is not the champion "
            f"(<code>{esc(data.champion)}</code>).</p>"
        )

    parts.append("<h2>Lichess blitz rating</h2>")
    parts.append(line_chart([(label, float(value)) for label, value in data.history]))

    measured = [v for v in data.versions if v.anchor and v.anchor.get("elo") is not None]
    parts.append("<h2>Strength by version</h2>")
    parts.append(
        "<p class=muted>Rating against fixed-strength opponents, one point per measured version."
        " This scale is not the Lichess scale.</p>"
    )
    if measured:
        if len(measured) > 1:
            parts.append(line_chart([(v.id, float(v.anchor["elo"])) for v in measured]))
        parts.append(
            "<div class=scroll><table><tr><th>Version<th>Network<th class=num>Rating<th class=num>± margin"
            "<th class=num>Games<th>Status"
        )
        for v in measured:
            parts.append(
                f"<tr><td><code>{esc(v.id)}</code><td><code>{esc(v.net or '–')}</code>"
                f"<td class=num>{v.anchor['elo']:.0f}"
                f"<td class=num>{v.anchor.get('margin', 0):.0f}<td class=num>{esc(v.anchor.get('games', ''))}"
                f"<td>{esc(v.status)}"
            )
        parts.append("</table></div>")
    else:
        parts.append("<p>No data yet.</p>")

    gated = [v for v in data.versions if v.gate]
    parts.append("<h2>Gate decisions</h2>")
    if gated:
        parts.append(
            "<div class=scroll><table><tr><th>Version<th>Network<th>Decision<th class=num>Games"
            "<th class=num>Elo<th class=num>± margin<th>Finished"
        )
        for v in reversed(gated):
            g = v.gate
            elo = signed(g["elo"]) if isinstance(g.get("elo"), (int, float)) else "–"
            margin = f"{g['margin']:.1f}" if isinstance(g.get("margin"), (int, float)) else "–"
            parts.append(
                f"<tr><td><code>{esc(v.id)}</code><td><code>{esc(v.net or '–')}</code>"
                f"<td>{esc(g.get('decision', ''))}"
                f"<td class=num>{esc(g.get('games', ''))}<td class=num>{elo}<td class=num>{margin}"
                f"<td>{esc(g.get('finished', v.created))}"
            )
        parts.append("</table></div>")
    else:
        parts.append("<p>No candidate has been tested yet.</p>")

    parts.append("</main></html>")
    return "".join(parts)
