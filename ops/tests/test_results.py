from vektr_ops.results import read_pgn


def game(round_, white, black, result, termination="normal"):
    return (
        f'[Event "t"]\n[Round "{round_}"]\n[White "{white}"]\n[Black "{black}"]\n'
        f'[Result "{result}"]\n[Termination "{termination}"]\n\n1. e4 {result}\n\n'
    )


def test_pairs_are_scored_for_the_subject():
    text = game(1, "candidate", "champion", "1-0") + game(1, "champion", "candidate", "1/2-1/2")
    r = read_pgn(text, "candidate")
    assert r.penta == (0, 0, 0, 1, 0)
    assert (r.wins, r.draws, r.losses, r.games) == (1, 1, 0, 2)


def test_black_win_counts_for_black():
    text = game(1, "champion", "candidate", "0-1") + game(1, "candidate", "champion", "1-0")
    assert read_pgn(text, "candidate").penta == (0, 0, 0, 0, 1)


def test_half_finished_pair_is_dropped():
    text = (
        game(1, "candidate", "champion", "1-0")
        + game(1, "champion", "candidate", "0-1")
        + game(2, "candidate", "champion", "1-0")
    )
    r = read_pgn(text, "candidate")
    assert r.games == 2 and sum(r.penta) == 1


def test_unfinished_game_is_not_a_result():
    text = game(1, "candidate", "champion", "*") + game(1, "champion", "candidate", "1-0")
    assert read_pgn(text, "candidate").games == 0


def test_time_forfeits_make_a_batch_invalid():
    text = game(1, "candidate", "champion", "0-1", "time forfeit") + game(
        1, "champion", "candidate", "1-0"
    )
    r = read_pgn(text, "candidate")
    assert r.bad == 1 and not r.valid


def test_one_bad_game_in_two_hundred_is_tolerated():
    text = "".join(
        game(n, "candidate", "champion", "1/2-1/2") + game(n, "champion", "candidate", "1/2-1/2")
        for n in range(1, 100)
    )
    text += game(100, "candidate", "champion", "0-1", "illegal move") + game(
        100, "champion", "candidate", "1/2-1/2"
    )
    r = read_pgn(text, "candidate")
    assert r.games == 200 and r.bad == 1 and r.valid


def test_empty_text_is_invalid():
    assert not read_pgn("", "candidate").valid
