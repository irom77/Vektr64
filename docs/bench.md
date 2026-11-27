# Bench baselines

`vektr bench` searches four fixed positions to depth 8 with a fresh 16 MiB table each.
The node count is a fingerprint of the search: it must be identical on every machine
and changes only when search or evaluation changes. Speed (nps) depends on the machine.

| Version | Change | Nodes | Laptop nps | Box nps |
|---|---|---|---|---|
| 0.1.0 (stage 1) | alpha-beta, quiescence, table, ordering, hand-written eval | 38,791,222 | 5.5M | — |
