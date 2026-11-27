"""Sequential test and Elo arithmetic for matches scored in opening pairs."""

import math

# Counts of opening pairs in which the subject scored 0, 1/2, 1, 1 1/2 and 2 points.
Penta = tuple[int, int, int, int, int]

PAIR_SCORES = (0.0, 0.25, 0.5, 0.75, 1.0)


def expected_score(elo: float) -> float:
    """Expected score per game for a player `elo` points stronger than the opponent."""
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def score_to_elo(score: float) -> float:
    """Inverse of `expected_score`, clamped away from the infinite ends."""
    score = min(max(score, 1e-6), 1.0 - 1e-6)
    return -400.0 * math.log10(1.0 / score - 1.0)


def moments(penta: Penta) -> tuple[int, float, float]:
    """Number of pairs, mean pair score and its variance."""
    n = sum(penta)
    if n == 0:
        return 0, 0.5, 0.0
    mean = sum(c * s for c, s in zip(penta, PAIR_SCORES)) / n
    var = sum(c * (s - mean) ** 2 for c, s in zip(penta, PAIR_SCORES)) / n
    return n, mean, var


def llr(penta: Penta, elo0: float = 0.0, elo1: float = 5.0) -> float:
    """Log-likelihood ratio of "gain is elo1" against "gain is elo0"."""
    n, mean, var = moments(penta)
    if n == 0 or var == 0.0:
        return 0.0
    s0, s1 = expected_score(elo0), expected_score(elo1)
    return n * (s1 - s0) * (2.0 * mean - s0 - s1) / (2.0 * var)


def decide(llr_value: float, alpha: float = 0.05, beta: float = 0.05) -> str:
    """`accept`, `reject` or `continue` for a running log-likelihood ratio."""
    if llr_value >= math.log((1.0 - beta) / alpha):
        return "accept"
    if llr_value <= math.log(beta / (1.0 - alpha)):
        return "reject"
    return "continue"


def elo_estimate(penta: Penta) -> tuple[float, float]:
    """Elo difference and the half-width of its 95% interval."""
    n, mean, var = moments(penta)
    if n == 0:
        return 0.0, 0.0
    spread = 1.96 * math.sqrt(var / n)
    return score_to_elo(mean), (score_to_elo(mean + spread) - score_to_elo(mean - spread)) / 2.0


def add(a: Penta, b: Penta) -> Penta:
    """Element-wise sum of two pair counts."""
    return tuple(x + y for x, y in zip(a, b))  # type: ignore[return-value]
