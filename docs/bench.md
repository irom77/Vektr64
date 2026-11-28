# Bench baselines

`vektr bench` searches four fixed positions to depth 8 with a fresh 16 MiB table each.
The node count is a fingerprint of the search: it must be identical on every machine
and changes only when search or evaluation changes. Speed (nps) depends on the machine.

| Version | Change | Nodes | Laptop nps | Box nps |
|---|---|---|---|---|
| 0.1.0 (stage 1) | alpha-beta, quiescence, table, ordering, hand-written eval | 38,791,222 | 4.4–5.5M | 2.85M |
| 0.2.0 (stage 3) | NNUE evaluation, bootstrap network f4e77e4c4ed3f87f, not gated | 34,234,013 | 3.4–3.7M | not measured |

## Anchor ratings

Each champion plays 200 games against Stockfish 19 limited to each of four `UCI_Elo`
settings, at the gate time control (8+0.08 at box speed). The rating is the single value
that best explains all four results. This scale is its own: it is not the Lichess scale,
and it is only good for comparing one version of Vektr64 with another.

| Version | Engine | Anchor rating | Games | vs 1500 (W/D/L) | vs 1800 | vs 2100 | vs 2400 |
|---|---|---|---|---|---|---|---|
| be104b19b30e | 0.1.0 (stage 1) | 2167 ±33 | 1000 | 283/1/16 | 143/6/51 | 185/21/94 | 58/18/124 |

The 1500 and 2100 anchors got 300 games each because the laptop and the box both started a
batch for them at the same moment; extra games are counted, not thrown away.
