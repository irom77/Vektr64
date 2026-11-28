"""Starting positions for self-play, taken from the bot's lost Lichess games."""

import io
import time
from pathlib import Path

import chess.pgn

from vektr_ops.results import headers
from vektr_ops.store import Store

ACCOUNT = "Vektr64"
FIRST_PLY = 12


def lost_starts(pgn: str, account: str = ACCOUNT) -> list[str]:
    """FENs of the main-line positions from ply FIRST_PLY up to, not including, the final one.

    Empty unless the game is standard chess from the initial position, ended normally,
    and `account` lost it. Any record that cannot be read gives an empty list.
    """
    try:
        tags = headers(pgn)
        if len(tags) != 1:
            return []
        tags = tags[0]
        loser = {"1-0": tags.get("Black"), "0-1": tags.get("White")}.get(tags.get("Result", ""))
        if (
            loser != account
            or tags.get("Variant", "Standard") != "Standard"
            or "FEN" in tags
            or tags.get("Termination") != "Normal"
        ):
            return []
        game = chess.pgn.read_game(io.StringIO(pgn))
        if game is None or game.errors:
            return []
        board = game.board()
        fens: list[str] = []
        for move in game.mainline_moves():
            board.push(move)
            fens.append(board.fen())
        return fens[FIRST_PLY - 1 : -1]
    except Exception:
        return []


def upload_experience(store: Store, root: Path, *, min_age_s: float = 1800) -> int:
    """Write `experience/<stem>.fens` for each lost game the store has none for; returns how many.

    Records touched within `min_age_s` are left for the next run: the game may still be on.
    """
    records = Path(root) / "bot" / "game_records"
    if not records.is_dir():
        return 0
    have = set(store.list("data", "experience/"))
    written = 0
    for record in sorted(records.glob("*.pgn")):
        path = f"experience/{record.stem}.fens"
        if path in have or time.time() - record.stat().st_mtime < min_age_s:
            continue
        try:
            starts = lost_starts(record.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError):
            continue
        if starts:
            store.write_bytes("data", path, ("\n".join(starts) + "\n").encode())
            written += 1
    return written
