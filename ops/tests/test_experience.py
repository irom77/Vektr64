import os

import chess
import chess.pgn

from vektr_ops.experience import FIRST_PLY, lost_starts, upload_experience
from vektr_ops.store import LocalStore

# 30 plies: a knight shuffle (Nf3 Nf6 Ng1 Ng8, repeated) after 1. e4 e5 ... all legal.
MOVES = "e4 e5 Nf3 Nf6 Ng1 Ng8 Nf3 Nf6 Ng1 Ng8 Nf3 Nf6 Ng1 Ng8 Nf3 Nf6 Ng1 Ng8 Nf3 Nf6 Ng1 Ng8 Nf3 Nf6 Ng1 Ng8 Nf3 Nf6 Ng1 Ng8".split()


def headers(white="x", black="Vektr64", result="1-0", variant="Standard", termination="Normal", extra=""):
    return (
        f'[White "{white}"]\n[Black "{black}"]\n[Result "{result}"]\n[Variant "{variant}"]\n'
        f'[Termination "{termination}"]\n[TimeControl "180+2"]\n{extra}\n'
    )


def movetext(moves, variations=True):
    out = []
    for i, san in enumerate(moves):
        ply = i + 1
        number = f"{(ply + 1) // 2}." if ply % 2 else f"{ply // 2}..."
        out.append(f"{number} {san} {{ [%clk 0:03:00] }}")
        if variations and ply % 2 == 0:
            # an alternative black move that is legal and differs from the main line
            out.append(f"( {number} a6 {{ [%eval 0.35,9] }} )")
    return " ".join(out)


def record(moves=MOVES, result="1-0", **kw):
    return headers(result=result, **kw) + movetext(moves) + f" {result}\n"


LOSS = record()


def main_line_fens(moves):
    board = chess.Board()
    fens = []
    for san in moves:
        board.push_san(san)
        fens.append(board.fen())
    return fens


def test_a_loss_gives_every_position_from_ply_12():
    starts = lost_starts(LOSS)
    assert len(starts) == 30 - FIRST_PLY == 18
    for fen in starts:
        chess.Board(fen)
    assert starts == main_line_fens(MOVES)[FIRST_PLY - 1 : 29]


def test_variations_are_not_followed():
    # the variation 1... a6 style alternatives put a pawn on a6; the main line never does
    for fen in lost_starts(LOSS):
        assert chess.Board(fen).piece_at(chess.A6) is None


def test_wins_draws_and_forfeits_give_nothing():
    assert lost_starts(record(result="0-1")) == []
    assert lost_starts(record(result="1/2-1/2")) == []
    assert lost_starts(record(termination="Time forfeit")) == []
    assert lost_starts(record(white="Vektr64", black="x", result="1-0")) == []


def test_other_variants_and_set_up_positions_give_nothing():
    assert lost_starts(record(variant="Chess960")) == []
    fen = '[FEN "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"]\n'
    assert lost_starts(record(extra=fen)) == []


def test_short_empty_and_broken_records_give_nothing():
    assert lost_starts(record(MOVES[:8])) == []
    assert lost_starts(headers()) == []
    assert lost_starts("not a pgn") == []
    assert lost_starts("") == []
    assert lost_starts(record(MOVES[:15] + ["Qh7"])) == []  # illegal move


def make_records(tmp_path):
    records = tmp_path / "bot" / "game_records"
    records.mkdir(parents=True)
    old = records / "octarine-bot vs Vektr64 - ua63qkWw.pgn"
    old.write_text(LOSS)
    (records / "Vektr64 vs y - win.pgn").write_text(record(result="0-1"))
    fresh = records / "z vs Vektr64 - fresh.pgn"
    fresh.write_text(LOSS)
    past = fresh.stat().st_mtime - 7200
    for p in (old, records / "Vektr64 vs y - win.pgn"):
        os.utime(p, (past, past))
    return records


def test_upload_writes_one_file_per_lost_game_once(tmp_path):
    make_records(tmp_path)
    store = LocalStore(tmp_path / "store")
    assert upload_experience(store, tmp_path) == 1
    data = store.read_bytes("data", "experience/octarine-bot vs Vektr64 - ua63qkWw.fens")
    assert len(data.decode().splitlines()) == 18
    assert store.list("data", "experience/") == ["experience/octarine-bot vs Vektr64 - ua63qkWw.fens"]
    assert upload_experience(store, tmp_path) == 0


def test_a_broken_record_does_not_stop_the_others(tmp_path):
    records = make_records(tmp_path)
    garbage = records / "a garbage.pgn"
    garbage.write_bytes(b"\xff\xfe\x00 [Result\n")
    past = garbage.stat().st_mtime - 7200
    os.utime(garbage, (past, past))
    assert upload_experience(LocalStore(tmp_path / "store"), tmp_path) == 1
