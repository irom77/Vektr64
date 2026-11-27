"""Read a match PGN into results scored per opening pair."""

import re
from dataclasses import dataclass

from vektr_ops.stats import Penta

TAG = re.compile(r'^\[(\w+)\s+"(.*)"\]\s*$')
BAD_ENDINGS = ("time", "illegal", "stall", "disconnect", "abandon", "crash")
POINTS = {"1-0": (2, 0), "0-1": (0, 2), "1/2-1/2": (1, 1)}  # half-points: white, black


@dataclass
class BatchResult:
    """One batch from the subject's point of view; `games` counts whole pairs only."""

    penta: Penta
    wins: int
    draws: int
    losses: int
    games: int
    bad: int

    @property
    def valid(self) -> bool:
        """False when there is nothing to count or more than 1% of games ended badly."""
        return self.games > 0 and self.bad * 100 <= self.games


def headers(text: str) -> list[dict[str, str]]:
    """The tag section of every game in the text, in order."""
    games: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for line in text.splitlines():
        match = TAG.match(line.strip())
        if match:
            if current is None:
                current = {}
                games.append(current)
            current[match.group(1)] = match.group(2)
        elif line.strip():
            current = None  # movetext: the next tag starts a new game
    return games


def read_pgn(text: str, subject: str) -> BatchResult:
    """Score the games of `subject`, grouped into opening pairs by the Round tag."""
    rounds: dict[str, list[dict[str, str]]] = {}
    for tags in headers(text):
        rounds.setdefault(tags.get("Round", "?"), []).append(tags)

    penta = [0, 0, 0, 0, 0]
    wins = draws = losses = bad = 0
    for pair in rounds.values():
        if len(pair) != 2 or any(g.get("Result") not in POINTS for g in pair):
            continue
        first, second = pair
        swapped = (
            first.get("White") == second.get("Black") and first.get("Black") == second.get("White")
        )
        if not swapped or subject not in (first.get("White"), first.get("Black")):
            continue
        total = 0
        for g in pair:
            white, black = POINTS[g["Result"]]
            mine = white if g["White"] == subject else black
            total += mine
            wins += mine == 2
            draws += mine == 1
            losses += mine == 0
            ending = g.get("Termination", "").lower()
            bad += any(word in ending for word in BAD_ENDINGS)
        penta[total] += 1

    games = 2 * sum(penta)
    return BatchResult(tuple(penta), wins, draws, losses, games, bad)  # type: ignore[arg-type]
