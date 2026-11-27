# Stage 2: Measuring Tools Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give Vektr64 the tools that decide whether a change is an improvement: a statistical gate between two engine builds, an absolute Elo yardstick, an external store with a version manifest, promotion and rollback on the box, and a status page.

**Architecture:** Everything is new Python in `ops/` plus systemd units; the engine is not changed. `fastchess` plays the games; our own code pairs the results, computes the sequential test, and pools batches from any machine through a small `Store` interface (a local directory in tests, Hugging Face Hub in production). The box follows the manifest's champion by building that commit into `builds/<id>/` and atomically switching a `builds/current` symlink that the bridge starts each game from.

**Tech Stack:** Python 3.12, `uv`, typer, psutil, `huggingface_hub`, pytest; `fastchess` v1.8.3-alpha, Stockfish `sf_19` (yardstick only), the `8moves_v3` opening book; systemd.

**Spec:** `docs/superpowers/specs/2026-11-27-vektr64-design.md`

## Global Constraints

- Work on branch `stage2-measuring-tools`, cut from `main`.
- No file under `engine/` changes. `vektr bench 8` must still print `38791222 nodes`.
- Secrets (`LICHESS_TOKEN`, `HF_TOKEN`) come from the environment or the git-ignored `.env`. They are never written to git, to the store, to logs or to the page.
- Ops tests never touch the network, Lichess, Hugging Face or a real `fastchess`: they use `LocalStore` in `tmp_path` and scripted inputs.
- Nothing is promoted unless its gate decision is `accept`. The only exception is the very first version, registered with `promote --bootstrap` when the manifest has no champion.
- Gate bounds: accept at +5 Elo, reject at 0 Elo, 5% error rates (`elo0=0`, `elo1=5`, `alpha=beta=0.05`).
- On a machine that is also running the `vektr64` service: at most 2 parallel games, at nice 19, and none at all if it has fewer than 4 physical cores.
- Every job sizes itself from detected cores and available memory; nothing is edited per machine.
- Time every measurement with a monotonic clock (`time.monotonic`); the laptop's wall clock steps.
- Data-repo paths are append-only: a write to an existing path fails.
- Commit messages end with `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>`.

## Decisions this plan makes (owner can overturn)

- **Version id** is the engine commit's first 12 hex characters (a network hash is appended in stage 3). The design says "engine git tag"; a commit id is used instead so the box never needs git push rights.
- **Gate time control** is 8 s + 0.08 s per move at box speed, scaled on other machines. **Game cap** 20,000; hitting it rejects the candidate.
- **Opening book** `8moves_v3.pgn` (balanced), each opening played twice with colours swapped.
- **Anchors** are Stockfish `sf_19` limited to 1500, 1800, 2100 and 2400 Elo, 200 games against each.
- **Box check margin**: a new build is refused if its bench speed is more than 10% below the build it replaces.
- **Eligibility line on the page**: 50 rated blitz games and rating deviation at or below 75.
- **Status page** is served from the box on port 8064.

## Review Focus

1. A batch played on a loaded machine has time forfeits or engine crashes: it must be discarded, not counted. (Task 2)
2. Two machines upload batches for the same test at the same moment: both are counted exactly once and neither overwrites the other. (Tasks 3, 7)
3. The store is unreachable while the box syncs: the bot keeps playing its current build and the symlink is untouched. (Task 9)
4. The champion commit does not build, or fails the box check: the symlink is untouched and `status` shows the error. (Task 9)
5. A batch is interrupted part-way: half-finished opening pairs are dropped and only whole pairs are counted. (Task 2)

## File Structure

| File | Responsibility |
|---|---|
| `ops/src/vektr_ops/stats.py` | Sequential test and Elo arithmetic. Pure functions |
| `ops/src/vektr_ops/results.py` | Read a fastchess PGN into paired results |
| `ops/src/vektr_ops/store.py` | `Store` interface, `LocalStore`, `HubStore`, `open_store` |
| `ops/src/vektr_ops/manifest.py` | Version records, promote, reject, rollback |
| `ops/src/vektr_ops/tools.py` | Fetch pinned fastchess, Stockfish and the book into `tools/` |
| `ops/src/vektr_ops/builds.py` | Build the engine at a commit into `builds/<id>/`, measure bench speed |
| `ops/src/vektr_ops/match.py` | Size and run one fastchess batch |
| `ops/src/vektr_ops/gate.py` | Pool batches, decide, drive a test to its end, worker loop |
| `ops/src/vektr_ops/anchors.py` | Anchor matches and the absolute Elo estimate |
| `ops/src/vektr_ops/sync.py` | Box follows the manifest: build, check, switch, upload game records |
| `ops/src/vektr_ops/page.py` | Render the static status page |
| `ops/src/vektr_ops/cli.py` | New commands: `gate`, `anchors`, `promote`, `rollback`, `sync`, `page`; extended `status`, `play` |
| `deploy/vektr64-sync.{service,timer}`, `deploy/vektr64-worker.service`, `deploy/vektr64-page.service` | Box units |

`.gitignore` gains `tools/`, `builds/`, `site/`, `state/`.

---

### Task 1: Sequential test arithmetic

**Files:**
- Create: `ops/src/vektr_ops/stats.py`
- Test: `ops/tests/test_stats.py`

**Interfaces:**
- Produces:
  - `Penta = tuple[int, int, int, int, int]` — counts of opening pairs scoring 0, ½, 1, 1½, 2 points for the candidate.
  - `expected_score(elo: float) -> float`
  - `llr(penta: Penta, elo0: float = 0.0, elo1: float = 5.0) -> float`
  - `decide(llr_value: float, alpha: float = 0.05, beta: float = 0.05) -> str` — `"accept"`, `"reject"` or `"continue"`.
  - `elo_estimate(penta: Penta) -> tuple[float, float]` — Elo and 95% margin.
  - `add(a: Penta, b: Penta) -> Penta`

- [ ] **Step 1: Write the failing tests**

```python
from pytest import approx
from vektr_ops.stats import add, decide, elo_estimate, expected_score, llr

def test_expected_score_is_logistic():
    assert expected_score(0) == approx(0.5)
    assert expected_score(190.8485) == approx(0.75, abs=1e-6)

def test_llr_reference_values():
    assert llr((10, 40, 100, 60, 20)) == approx(1.14, abs=1e-3)
    assert llr((20, 60, 100, 40, 10)) == approx(-1.3457, abs=1e-3)
    assert llr((100, 400, 1000, 480, 120)) == approx(3.0951, abs=1e-3)

def test_llr_is_zero_without_information():
    assert llr((0, 0, 0, 0, 0)) == 0.0
    assert llr((0, 0, 30, 0, 0)) == 0.0          # zero variance

def test_decide_uses_the_five_percent_bounds():
    assert decide(2.95) == "accept"
    assert decide(-2.95) == "reject"
    assert decide(2.94) == "continue"

def test_elo_estimate_reference():
    elo, margin = elo_estimate((10, 40, 100, 60, 20))
    assert elo == approx(30.29, abs=0.01)
    assert margin == approx(21.8, abs=0.3)

def test_add_sums_counts():
    assert add((1, 2, 3, 4, 5), (5, 4, 3, 2, 1)) == (6, 6, 6, 6, 6)
```

- [ ] **Step 2: Run** `uv run --project ops pytest ops/tests/test_stats.py -q` — Expected: FAIL, `ModuleNotFoundError: vektr_ops.stats`.

- [ ] **Step 3: Implement `stats.py`.** The one formula the tests do not fully determine:

```
pair scores s = (0, .25, .5, .75, 1);  N = sum(penta)
mu  = sum(n_i * s_i) / N;   var = sum(n_i * (s_i - mu)^2) / N
s0 = expected_score(elo0);  s1 = expected_score(elo1)
llr = N * (s1 - s0) * (2*mu - s0 - s1) / (2 * var)      (0.0 when N == 0 or var == 0)
bounds: accept at llr >= ln((1-beta)/alpha), reject at llr <= ln(beta/(1-alpha))
margin: half the width of the Elo interval for mu ± 1.96*sqrt(var/N)
```

- [ ] **Step 4: Run** the same command — Expected: 6 passed.
- [ ] **Step 5: Commit** `feat(ops): sequential test arithmetic`.

---

### Task 2: Reading match results

**Files:**
- Create: `ops/src/vektr_ops/results.py`
- Test: `ops/tests/test_results.py`

**Interfaces:**
- Consumes: `Penta` from Task 1.
- Produces:
  - `@dataclass BatchResult: penta: Penta; wins: int; draws: int; losses: int; games: int; bad: int` — all from the point of view of the engine named `subject`; `games` counts only games in whole pairs; `bad` counts games that ended by time forfeit, illegal move, crash, stall or disconnect.
  - `read_pgn(text: str, subject: str) -> BatchResult`
  - `BatchResult.valid -> bool` — false when `bad * 100 > games` or `games == 0`.

- [ ] **Step 1: Write the failing tests.** Build PGN text with a helper `game(round, white, black, result, termination="normal")` that emits only the header tags `Round`, `White`, `Black`, `Result`, `Termination` and a one-move body.

```python
def test_pairs_are_scored_for_the_subject():
    text = game(1, "candidate", "champion", "1-0") + game(1, "champion", "candidate", "1/2-1/2")
    r = read_pgn(text, "candidate")
    assert r.penta == (0, 0, 0, 1, 0) and (r.wins, r.draws, r.losses, r.games) == (1, 1, 0, 2)

def test_black_win_counts_for_black():
    text = game(1, "champion", "candidate", "0-1") + game(1, "candidate", "champion", "1-0")
    assert read_pgn(text, "candidate").penta == (0, 0, 0, 0, 1)

def test_half_finished_pair_is_dropped():
    text = game(1, "candidate", "champion", "1-0") + game(1, "champion", "candidate", "0-1") \
         + game(2, "candidate", "champion", "1-0")
    r = read_pgn(text, "candidate")
    assert r.games == 2 and sum(r.penta) == 1

def test_unfinished_game_is_not_a_result():
    text = game(1, "candidate", "champion", "*") + game(1, "champion", "candidate", "1-0")
    assert read_pgn(text, "candidate").games == 0

def test_time_forfeits_make_a_batch_invalid():
    text = game(1, "candidate", "champion", "0-1", "time forfeit") + game(1, "champion", "candidate", "1-0")
    r = read_pgn(text, "candidate")
    assert r.bad == 1 and not r.valid

def test_one_bad_game_in_two_hundred_is_tolerated():
    text = "".join(game(n, "candidate", "champion", "1/2-1/2") + game(n, "champion", "candidate", "1/2-1/2")
                   for n in range(1, 100))
    text += game(100, "candidate", "champion", "0-1", "illegal move") + game(100, "champion", "candidate", "1/2-1/2")
    r = read_pgn(text, "candidate")
    assert r.games == 200 and r.bad == 1 and r.valid

def test_empty_text_is_invalid():
    assert not read_pgn("", "candidate").valid
```

- [ ] **Step 2: Run** `uv run --project ops pytest ops/tests/test_results.py -q` — Expected: FAIL, module missing.
- [ ] **Step 3: Implement.** Parse header tags with a regular expression, no PGN library. Games are grouped by the `Round` tag; a group counts only when it holds exactly two finished games with colours swapped. A termination is bad when its lower-cased text contains any of `time`, `illegal`, `stall`, `disconnect`, `abandon`, `crash`.
- [ ] **Step 4: Run** — Expected: 7 passed.
- [ ] **Step 5: Commit** `feat(ops): read paired results from match PGN`.

---

### Task 3: The store

**Files:**
- Create: `ops/src/vektr_ops/store.py`
- Modify: `ops/pyproject.toml` (add `huggingface_hub>=0.26`), `.env.example` (add `HF_TOKEN=` with a comment: write-scope token from https://huggingface.co/settings/tokens), `.gitignore`
- Test: `ops/tests/test_store.py`

**Interfaces:**
- Produces:
  - `class Store(Protocol)` with `read_bytes(repo: str, path: str) -> bytes | None`, `write_bytes(repo: str, path: str, data: bytes, *, overwrite: bool = False) -> None` (raises `FileExistsError` when the path exists and `overwrite` is false), `list(repo: str, prefix: str) -> list[str]` (sorted full paths).
  - `repo` is `"nets"` or `"data"`; anything else raises `ValueError`.
  - `read_json(store, repo, path) -> dict | None`, `write_json(store, repo, path, obj, *, overwrite=False) -> None`
  - `class LocalStore(root: Path)` — files at `root/<repo>/<path>`.
  - `class HubStore(token: str)` — `nets` is the private model repo `<user>/vektr64-nets`, `data` the private dataset repo `<user>/vektr64-data`, where `<user>` comes from `HfApi.whoami`; both are created private on first use if missing.
  - `class StoreUnavailable(Exception)` — raised by `HubStore` for any network or server failure.
  - `open_store(env: dict[str, str]) -> Store` — `VEKTR_STORE` (a directory) wins; else `HF_TOKEN`; else raises `StoreUnavailable("set HF_TOKEN or VEKTR_STORE")`.

- [ ] **Step 1: Write the failing tests**

```python
def test_round_trip(tmp_path):
    s = LocalStore(tmp_path)
    s.write_bytes("data", "gate/t/a.json", b"{}")
    assert s.read_bytes("data", "gate/t/a.json") == b"{}"
    assert s.read_bytes("data", "gate/t/missing.json") is None

def test_existing_path_is_not_overwritten(tmp_path):
    s = LocalStore(tmp_path)
    s.write_bytes("data", "x", b"1")
    with pytest.raises(FileExistsError):
        s.write_bytes("data", "x", b"2")
    s.write_bytes("nets", "manifest.json", b"1")
    s.write_bytes("nets", "manifest.json", b"2", overwrite=True)
    assert s.read_bytes("nets", "manifest.json") == b"2"

def test_list_is_sorted_and_prefixed(tmp_path):
    s = LocalStore(tmp_path)
    for name in ("gate/t/b.json", "gate/t/a.json", "gate/u/c.json"):
        s.write_bytes("data", name, b"")
    assert s.list("data", "gate/t/") == ["gate/t/a.json", "gate/t/b.json"]

def test_unknown_repo_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        LocalStore(tmp_path).read_bytes("secrets", "x")

def test_path_cannot_escape_the_store(tmp_path):
    with pytest.raises(ValueError):
        LocalStore(tmp_path).write_bytes("data", "../../etc/x", b"")

def test_open_store_prefers_the_local_directory(tmp_path):
    assert isinstance(open_store({"VEKTR_STORE": str(tmp_path), "HF_TOKEN": "x"}), LocalStore)

def test_open_store_without_settings_explains(tmp_path):
    with pytest.raises(StoreUnavailable, match="HF_TOKEN or VEKTR_STORE"):
        open_store({})
```

- [ ] **Step 2: Run** `uv run --project ops pytest ops/tests/test_store.py -q` — Expected: FAIL, module missing.
- [ ] **Step 3: Implement.** `LocalStore` writes to a temporary file in the same directory and renames. `HubStore` uses `HfApi.upload_file`, `hf_hub_download` and `list_repo_files`; a missing file returns `None`; an existing path without `overwrite` raises `FileExistsError`; every other library or network error becomes `StoreUnavailable`. `HubStore` has no unit test; Task 11 exercises it for real.
- [ ] **Step 4: Run** — Expected: 7 passed. Then `uv run --project ops pytest ops/tests -q` — Expected: all pass.
- [ ] **Step 5: Commit** `feat(ops): store interface with local and Hugging Face backends`.

---

### Task 4: The manifest

**Files:**
- Create: `ops/src/vektr_ops/manifest.py`
- Test: `ops/tests/test_manifest.py`

**Interfaces:**
- Consumes: `Store`, `read_json`, `write_json` (Task 3).
- Produces:
  - `@dataclass Version: id: str; engine_commit: str; net: str | None; parent: str | None; status: str; created: str; gate: dict | None; anchor: dict | None; box_nps: int | None`
  - `@dataclass Manifest: champion: str | None; versions: list[Version]` with `get(id) -> Version` (raises `KeyError`), `champion_version() -> Version | None`, `candidates() -> list[Version]`.
  - `load(store) -> Manifest` (an empty manifest when the file is missing), `save(store, m) -> None` — at `nets:manifest.json`, `{"schema": 1, ...}`.
  - `version_id(engine_commit: str, net: str | None = None) -> str` — `engine_commit[:12]`.
  - `add_candidate(m, engine_commit, now: str) -> Version` — returns the existing version if the id is already there.
  - `promote(m, id, *, bootstrap: bool = False) -> None`
  - `reject(m, id, gate: dict) -> None`
  - `rollback(m) -> Version` — returns the restored champion.
  - `class NotPromotable(Exception)`

- [ ] **Step 1: Write the failing tests**

```python
A, B, C = "a" * 40, "b" * 40, "c" * 40
ACCEPT = {"decision": "accept", "llr": 3.0}

def boot():
    m = Manifest(None, [])
    add_candidate(m, A, "t0"); promote(m, A[:12], bootstrap=True)
    return m

def test_empty_store_gives_empty_manifest(tmp_path):
    assert load(LocalStore(tmp_path)) == Manifest(None, [])

def test_save_and_load_round_trip(tmp_path):
    s = LocalStore(tmp_path); m = boot(); save(s, m)
    assert load(s) == m

def test_bootstrap_only_when_there_is_no_champion():
    m = boot(); add_candidate(m, B, "t1")
    with pytest.raises(NotPromotable):
        promote(m, B[:12], bootstrap=True)

def test_promotion_needs_an_accepted_gate():
    m = boot(); v = add_candidate(m, B, "t1")
    with pytest.raises(NotPromotable):
        promote(m, v.id)
    v.gate = {"decision": "reject"}
    with pytest.raises(NotPromotable):
        promote(m, v.id)

def test_promote_retires_the_old_champion_and_links_parent():
    m = boot(); v = add_candidate(m, B, "t1"); v.gate = ACCEPT; promote(m, v.id)
    assert m.champion == B[:12] and v.status == "champion" and v.parent == A[:12]
    assert m.get(A[:12]).status == "retired"

def test_rollback_restores_the_parent():
    m = boot(); v = add_candidate(m, B, "t1"); v.gate = ACCEPT; promote(m, v.id)
    restored = rollback(m)
    assert restored.id == A[:12] and m.champion == A[:12]
    assert restored.status == "champion" and m.get(B[:12]).status == "retired"

def test_rollback_without_a_parent_fails():
    with pytest.raises(NotPromotable):
        rollback(boot())

def test_reject_records_the_gate_and_never_plays():
    m = boot(); v = add_candidate(m, B, "t1"); reject(m, v.id, {"decision": "reject"})
    assert v.status == "rejected" and m.champion == A[:12] and m.candidates() == []

def test_adding_the_same_commit_twice_is_one_version():
    m = boot(); assert add_candidate(m, A, "t9") is m.get(A[:12]) and len(m.versions) == 1

def test_promoting_the_champion_again_changes_nothing():
    m = boot(); before = copy.deepcopy(m); promote(m, A[:12], bootstrap=True)
    assert m == before
```

- [ ] **Step 2: Run** `uv run --project ops pytest ops/tests/test_manifest.py -q` — Expected: FAIL, module missing.
- [ ] **Step 3: Implement.** Statuses are exactly `candidate`, `champion`, `rejected`, `retired`.
- [ ] **Step 4: Run** — Expected: 10 passed.
- [ ] **Step 5: Commit** `feat(ops): version manifest with promote, reject and rollback`.

---

### Task 5: Pinned tools and engine builds

**Files:**
- Create: `ops/src/vektr_ops/tools.py`, `ops/src/vektr_ops/builds.py`
- Test: `ops/tests/test_tools.py`, `ops/tests/test_builds.py`

**Interfaces:**
- Produces (`tools.py`):
  - `@dataclass Tool: name: str; url: str; sha256: str; member: str` and the constants `FASTCHESS`, `STOCKFISH`, `BOOK`:
    - `https://github.com/Disservin/fastchess/releases/download/v1.8.3-alpha/fastchess-linux-x86-64.tar`
    - `https://github.com/official-stockfish/Stockfish/releases/download/sf_19/stockfish-linux-x86-64-universal.tar.gz`
    - `https://github.com/official-stockfish/books/raw/37751b08db745aa02babfa74056dd1168b9da803/8moves_v3.pgn.zip`
  - `ensure(tool: Tool, tools_dir: Path, fetch=urllib.request.urlopen) -> Path` — the path of the extracted `member`; downloads only when it is missing; raises `ValueError("checksum mismatch for <name>")` and leaves nothing behind when the hash differs.
- Produces (`builds.py`):
  - `ensure_build(root: Path, commit: str) -> Path` — the engine binary at `root/builds/<id>/vektr`, built with `git archive <commit> engine | tar -x` into `root/builds/<id>/src` and `cargo build --release`; returns at once if the binary exists; raises `BuildFailed` with cargo's output otherwise and removes the half-made directory.
  - `bench_nps(binary: Path) -> int` — runs `<binary> bench 8`, parses `"{nodes} nodes {nps} nps"`.
  - `parse_bench(line: str) -> tuple[int, int]`
  - `class BuildFailed(Exception)`

- [ ] **Step 1: Write the failing tests**

```python
def test_parse_bench():
    assert parse_bench("38791222 nodes 2850000 nps\n") == (38791222, 2850000)

def test_parse_bench_rejects_other_output():
    with pytest.raises(ValueError):
        parse_bench("error: no such command")

def test_ensure_downloads_once_and_verifies(tmp_path):
    payload = make_tar({"fastchess": b"#!/bin/sh\n"})          # test helper
    tool = Tool("fastchess", "https://example.invalid/f.tar", hashlib.sha256(payload).hexdigest(), "fastchess")
    calls = []
    def fetch(url): calls.append(url); return io.BytesIO(payload)
    path = ensure(tool, tmp_path, fetch)
    assert path.read_bytes() == b"#!/bin/sh\n" and os.access(path, os.X_OK)
    ensure(tool, tmp_path, fetch)
    assert len(calls) == 1

def test_ensure_refuses_a_wrong_checksum(tmp_path):
    tool = Tool("fastchess", "https://example.invalid/f.tar", "0" * 64, "fastchess")
    with pytest.raises(ValueError, match="checksum mismatch for fastchess"):
        ensure(tool, tmp_path, lambda url: io.BytesIO(b"junk"))
    assert list(tmp_path.iterdir()) == []

def test_ensure_build_reuses_an_existing_binary(tmp_path):
    binary = tmp_path / "builds" / ("a" * 12) / "vektr"
    binary.parent.mkdir(parents=True); binary.write_bytes(b"x")
    assert ensure_build(tmp_path, "a" * 40) == binary        # no git, no cargo needed
```

- [ ] **Step 2: Run** `uv run --project ops pytest ops/tests/test_tools.py ops/tests/test_builds.py -q` — Expected: FAIL, modules missing.
- [ ] **Step 3: Implement both modules.** Archives may be `.tar`, `.tar.gz` or `.zip`; `member` is matched by file name anywhere in the archive.
- [ ] **Step 4: Pin the checksums.** Download each of the three URLs once, compute `sha256sum`, and write the values into the constants. Confirm the Stockfish archive's executable name and set `STOCKFISH.member` to it.
- [ ] **Step 5: Run** the tests — Expected: 5 passed. Then, for real:

```bash
uv run --project ops python -c "
from pathlib import Path; from vektr_ops import tools, builds
r = Path('.').resolve()
for t in (tools.FASTCHESS, tools.STOCKFISH, tools.BOOK): print(tools.ensure(t, r / 'tools'))
b = builds.ensure_build(r, '$(git rev-parse HEAD)'); print(b, builds.bench_nps(b))"
```

Expected: three paths under `tools/`, a path under `builds/`, and a speed in the millions. `tools/fastchess --version` prints `1.8.3`.
- [ ] **Step 6: Commit** `feat(ops): pinned match tools and per-commit engine builds`.

---

### Task 6: Running one batch

**Files:**
- Create: `ops/src/vektr_ops/match.py`
- Test: `ops/tests/test_match.py`

**Interfaces:**
- Consumes: `read_pgn`, `BatchResult` (Task 2).
- Produces:
  - `BASE_S = 8.0`, `INC_S = 0.08`, `PAIRS_PER_BATCH = 50`
  - `concurrency(physical_cores: int, available_mb: int, bot_here: bool) -> int` — with the bot: 2 when `physical_cores >= 4`, else 0; without: `physical_cores - 1`; then capped at `available_mb // 256`; never below 1 unless the bot rule gave 0.
  - `scaled_tc(box_nps: int, here_nps: int) -> str` — `BASE_S` and `INC_S` multiplied by `box_nps / here_nps`, formatted `"{base:.2f}+{inc:.3f}"`.
  - `@dataclass Engine: name: str; cmd: Path; options: dict[str, str]`
  - `fastchess_args(fastchess: Path, a: Engine, b: Engine, *, tc: str, book: Path, pairs: int, jobs: int, seed: int, pgn_out: Path) -> list[str]`
  - `run_batch(fastchess, a, b, *, tc, book, pairs, jobs, seed, workdir: Path, low_priority: bool) -> tuple[BatchResult, str]` — the result for engine `a` and the PGN text.
  - `bot_here() -> bool` — true when `systemctl is-active vektr64` says `active`.

- [ ] **Step 1: Write the failing tests**

```python
def test_concurrency_rules():
    assert concurrency(4, 2000, bot_here=True) == 2
    assert concurrency(2, 8000, bot_here=True) == 0
    assert concurrency(11, 15000, bot_here=False) == 10
    assert concurrency(11, 600, bot_here=False) == 2
    assert concurrency(4, 300, bot_here=True) == 1
    assert concurrency(1, 100, bot_here=False) == 1

def test_time_control_is_scaled_to_box_speed():
    assert scaled_tc(2_850_000, 2_850_000) == "8.00+0.080"
    assert scaled_tc(2_850_000, 5_000_000) == "4.56+0.046"

def test_fastchess_arguments():
    a = Engine("candidate", Path("/b/new/vektr"), {"Hash": "16", "Move Overhead": "10"})
    b = Engine("champion", Path("/b/old/vektr"), {"Hash": "16", "Move Overhead": "10"})
    args = fastchess_args(Path("/t/fastchess"), a, b, tc="8.00+0.080", book=Path("/t/8moves_v3.pgn"),
                          pairs=50, jobs=2, seed=7, pgn_out=Path("/w/out.pgn"))
    joined = " ".join(args)
    assert args[0] == "/t/fastchess"
    assert "cmd=/b/new/vektr name=candidate option.Hash=16" in joined
    assert "option.Move Overhead=10" in args                    # one argument, space kept
    assert "-rounds 50 -games 2 -repeat" in joined
    assert "-concurrency 2" in joined and "-srand 7" in joined
    assert "file=/t/8moves_v3.pgn format=pgn order=random" in joined
    assert "tc=8.00+0.080" in joined and "-recover" in args
    assert "file=/w/out.pgn" in joined
```

- [ ] **Step 2: Run** `uv run --project ops pytest ops/tests/test_match.py -q` — Expected: FAIL, module missing.
- [ ] **Step 3: Implement.** Check every flag against `tools/fastchess --help` for 1.8.3 and correct the spelling there and in the test if one differs. Adjudication: `-draw movenumber=40 movecount=8 score=10 -resign movecount=3 score=600`. `run_batch` starts the process with `os.nice(19)` when `low_priority`, and on Ctrl-C or SIGTERM stops fastchess and still returns what the PGN holds.
- [ ] **Step 4: Run** — Expected: 3 passed.
- [ ] **Step 5: Prove the pairing assumption against the real tool.** Run a 4-pair batch of the HEAD build against itself (`tc=2+0.02`) and check the PGN: each `Round` value appears exactly twice, with colours swapped, and `read_pgn` reports `games == 8`. If fastchess numbers rounds differently, change the grouping rule in `results.py` and its test helper to match, and note it in the commit.
- [ ] **Step 6: Commit** `feat(ops): size and run one match batch`.

---

### Task 7: The gate

**Files:**
- Create: `ops/src/vektr_ops/gate.py`
- Modify: `ops/src/vektr_ops/cli.py`
- Test: `ops/tests/test_gate.py`

**Interfaces:**
- Consumes: Tasks 1–6.
- Produces:
  - `GAME_CAP = 20_000`, `ELO0 = 0.0`, `ELO1 = 5.0`
  - `test_id(champion_id: str, candidate_id: str) -> str` — `"<champion>__<candidate>"`.
  - `batch_name(host: str, now: datetime, rand: str) -> str` — `"<host>-<YYYYMMDDTHHMMSSZ>-<rand>"`.
  - `upload_batch(store, test: str, name: str, result: BatchResult, pgn: str, meta: dict) -> None` — writes `data:gate/<test>/<name>.json` and `.pgn`; an invalid batch is written under `data:gate/<test>/discarded/` instead.
  - `pooled(store, test: str) -> tuple[Penta, int]` — summed counts and games over every `gate/<test>/*.json` directly in the folder.
  - `verdict(penta: Penta, games: int) -> dict` — `{"decision", "llr", "penta", "games", "elo", "margin", "elo0", "elo1", "reason"}`; decision is `accept`, `reject` or `continue`; at `games >= GAME_CAP` a `continue` becomes `reject` with `reason: "cap"`.
  - `step(store, root, candidate: Version, runner=run_batch) -> dict` — one batch for one candidate, upload, pool, and when decided write the result to the manifest (`promote` on accept, `reject` otherwise); returns the verdict.
  - `work(store, root, *, once: bool = False, idle_s: int = 300) -> None` — the worker loop: each round, `step` every candidate in the manifest, then anchor work if any is due (Task 8); sleep `idle_s` when there was nothing to do.
- CLI:
  - `vektr-ops gate <git-ref>` — resolve the ref to a commit, refuse (exit 2) if the commit is not on `origin` (`git branch -r --contains`), add it as a candidate, then `step` until decided, printing one line per batch: `games=400 llr=0.84 elo=+6.1 ±14.0`.
  - `vektr-ops gate --work` — run `work`.
  - `vektr-ops promote <id> --bootstrap`, `vektr-ops rollback`.

- [ ] **Step 1: Write the failing tests**

```python
def result(penta, bad=0):
    games = 2 * sum(penta); return BatchResult(penta, 0, games, 0, games, bad)

def test_pooling_sums_valid_batches_from_all_machines(tmp_path):
    s = LocalStore(tmp_path)
    upload_batch(s, "a__b", "box-1", result((1, 4, 10, 6, 2)), "pgn", {})
    upload_batch(s, "a__b", "laptop-1", result((1, 4, 10, 6, 2)), "pgn", {})
    assert pooled(s, "a__b") == ((2, 8, 20, 12, 4), 92)

def test_invalid_batch_is_kept_but_not_counted(tmp_path):
    s = LocalStore(tmp_path)
    upload_batch(s, "a__b", "box-1", result((0, 0, 50, 0, 0), bad=5), "pgn", {})
    assert pooled(s, "a__b") == ((0, 0, 0, 0, 0), 0)
    assert s.list("data", "gate/a__b/discarded/") != []

def test_same_batch_name_cannot_be_uploaded_twice(tmp_path):
    s = LocalStore(tmp_path)
    upload_batch(s, "a__b", "box-1", result((0, 0, 1, 0, 0)), "pgn", {})
    with pytest.raises(FileExistsError):
        upload_batch(s, "a__b", "box-1", result((0, 0, 1, 0, 0)), "pgn", {})

def test_verdict_accepts_rejects_and_continues():
    assert verdict((100, 400, 1000, 480, 120), 4200)["decision"] == "accept"
    assert verdict((200, 600, 1000, 400, 100), 4600)["decision"] == "reject"
    assert verdict((5, 20, 50, 20, 5), 200)["decision"] == "continue"

def test_game_cap_rejects():
    v = verdict((5, 20, 50, 20, 5), 20_000)
    assert v["decision"] == "reject" and v["reason"] == "cap"

def test_step_promotes_on_accept(tmp_path):
    s, m = store_with_champion_and_candidate(tmp_path)        # helper: champion A, candidate B
    runner = lambda *a, **k: (result((100, 400, 1000, 480, 120)), "pgn")
    v = step(s, tmp_path, m.get(B[:12]), runner=runner)
    after = load(s)
    assert v["decision"] == "accept" and after.champion == B[:12]
    assert after.get(B[:12]).gate["decision"] == "accept"

def test_step_rejects_and_keeps_the_champion(tmp_path):
    s, m = store_with_champion_and_candidate(tmp_path)
    step(s, tmp_path, m.get(B[:12]), runner=lambda *a, **k: (result((200, 600, 1000, 400, 100)), "pgn"))
    after = load(s)
    assert after.champion == A[:12] and after.get(B[:12]).status == "rejected"

def test_step_on_an_already_decided_candidate_plays_nothing(tmp_path):
    s, m = store_with_champion_and_candidate(tmp_path)
    step(s, tmp_path, m.get(B[:12]), runner=lambda *a, **k: (result((100, 400, 1000, 480, 120)), "pgn"))
    def boom(*a, **k): raise AssertionError("must not play")
    assert step(s, tmp_path, load(s).get(B[:12]), runner=boom)["decision"] == "accept"

def test_work_with_no_candidates_does_nothing(tmp_path):
    s, _ = store_with_champion_only(tmp_path)
    work(s, tmp_path, once=True, idle_s=0)                      # returns, plays nothing
```

In the `step` tests the helper patches `ensure_build`, `bench_nps`, `tools.ensure` and `bot_here` so nothing is built or downloaded.

- [ ] **Step 2: Run** `uv run --project ops pytest ops/tests/test_gate.py -q` — Expected: FAIL, module missing.
- [ ] **Step 3: Implement.** In `step`: reload the manifest first and return the recorded gate if the candidate is no longer `candidate`; time control is `scaled_tc(champion.box_nps, bench_nps(champion build here))`; engines run with `Hash=16`, `Threads=1`, `Move Overhead=10`; the seed is derived from the batch name; when `concurrency(...)` is 0, sleep and return `continue` without playing. The decision is taken from the pool re-read after the upload, so two machines finishing together reach the same verdict, and `promote` on an already promoted version changes nothing.
- [ ] **Step 4: Run** — Expected: 9 passed; then the whole ops suite passes.
- [ ] **Step 5: Commit** `feat(ops): pooled sequential gate, worker loop, promote and rollback commands`.

---

### Task 8: Anchor ladder

**Files:**
- Create: `ops/src/vektr_ops/anchors.py`
- Modify: `ops/src/vektr_ops/cli.py`, `ops/src/vektr_ops/gate.py` (the `work` loop calls `anchor_step`)
- Test: `ops/tests/test_anchors.py`

**Interfaces:**
- Consumes: Tasks 1–6.
- Produces:
  - `ANCHORS = (1500, 1800, 2100, 2400)`, `GAMES_PER_ANCHOR = 200`
  - `rating(results: list[tuple[int, int, int, int]]) -> float | None` — each item is `(anchor_elo, wins, draws, losses)`; the rating `R` at which expected points equal actual points, found by bisection on `[0, 4000]`; `None` with no games; clamped to the ends when every game was won or lost.
  - `margin(results) -> float` — 95% half-width, from the summed per-game variance.
  - `anchor_step(store, root, version: Version, runner=run_batch) -> dict | None` — plays one batch against the anchor with the fewest games so far, uploads to `data:anchor/<version id>/<anchor>-<batch name>.json`, and when every anchor has `GAMES_PER_ANCHOR` games writes `{"elo", "margin", "games"}` to the version's `anchor` field. Returns `None` when the version is already measured.
- CLI: `vektr-ops anchors [<id>]` — default is the champion; loops `anchor_step` until measured, printing the running estimate.

- [ ] **Step 1: Write the failing tests**

```python
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

def test_margin_shrinks_with_more_games():
    assert margin([(2000, 200, 0, 200)]) < margin([(2000, 20, 0, 20)])

def test_anchor_step_records_the_rating_when_complete(tmp_path):
    s, m = store_with_champion_only(tmp_path)
    runner = lambda *a, **k: (BatchResult((0, 0, 50, 0, 0), 50, 0, 50, 100, 0), "pgn")
    while anchor_step(s, tmp_path, load(s).champion_version(), runner=runner) is not None:
        pass
    a = load(s).champion_version().anchor
    assert a["games"] == 800 and a["elo"] == approx(1950, abs=1)
```

- [ ] **Step 2: Run** `uv run --project ops pytest ops/tests/test_anchors.py -q` — Expected: FAIL, module missing.
- [ ] **Step 3: Implement.** The Stockfish engine runs with `Threads=1`, `Hash=16`, `UCI_LimitStrength=true`, `UCI_Elo=<anchor>`, `Move Overhead=10`. Same time control scaling, concurrency and discard rule as the gate.
- [ ] **Step 4: Run** — Expected: 8 passed; whole ops suite passes.
- [ ] **Step 5: Commit** `feat(ops): anchor ladder and absolute rating estimate`.

---

### Task 9: The box follows the manifest

**Files:**
- Create: `ops/src/vektr_ops/sync.py`, `deploy/vektr64-sync.service`, `deploy/vektr64-sync.timer`, `deploy/vektr64-worker.service`
- Modify: `ops/src/vektr_ops/cli.py` (`sync` command; `play` engine directory), `deploy/install-box.sh`
- Test: `ops/tests/test_sync.py`, `ops/tests/test_cli.py`

**Interfaces:**
- Consumes: Tasks 3–5.
- Produces:
  - `SPEED_MARGIN = 0.10`
  - `current_id(root: Path) -> str | None` — the directory name `root/builds/current` points at.
  - `switch(root: Path, id: str) -> None` — atomic: make a new symlink beside it and rename over `current`.
  - `sync(store, root, *, build=ensure_build, bench=bench_nps) -> str` — returns `"unchanged"`, `"switched to <id>"` or raises; on any failure writes the message to `root/state/sync-error` and re-raises; on success removes that file.
  - `upload_games(store, root) -> int` — every `bot/game_records/*.pgn` not yet in `data:games/lichess/`; returns how many were uploaded.
  - `engine_dir(root: Path) -> Path` — `root/builds/current` when it exists, else `root/engine/target/release`.
- `vektr-ops play` uses `engine_dir(root)` and builds the working tree only when it falls back.

- [ ] **Step 1: Write the failing tests**

```python
def fake_build(root, commit):
    p = root / "builds" / commit[:12] / "vektr"; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b"x"); return p

def test_first_sync_switches_to_the_champion(tmp_path):
    s = store_with_champion(tmp_path / "store", A)
    assert sync(s, tmp_path, build=fake_build, bench=lambda p: 2_850_000) == f"switched to {A[:12]}"
    assert current_id(tmp_path) == A[:12]
    assert load(s).get(A[:12]).box_nps == 2_850_000

def test_sync_is_idempotent(tmp_path):
    s = store_with_champion(tmp_path / "store", A)
    sync(s, tmp_path, build=fake_build, bench=lambda p: 2_850_000)
    assert sync(s, tmp_path, build=fake_build, bench=lambda p: 2_850_000) == "unchanged"

def test_slow_build_is_refused(tmp_path):
    s = store_with_champion(tmp_path / "store", A)
    sync(s, tmp_path, build=fake_build, bench=lambda p: 2_850_000)
    promote_accepted(s, B)                                       # helper
    speeds = {A[:12]: 2_850_000, B[:12]: 2_500_000}
    with pytest.raises(SyncRefused, match="slower"):
        sync(s, tmp_path, build=fake_build, bench=lambda p: speeds[p.parent.name])
    assert current_id(tmp_path) == A[:12]
    assert "slower" in (tmp_path / "state" / "sync-error").read_text()

def test_build_failure_leaves_the_current_engine(tmp_path):
    s = store_with_champion(tmp_path / "store", A)
    sync(s, tmp_path, build=fake_build, bench=lambda p: 2_850_000)
    promote_accepted(s, B)
    def broken(root, commit): raise BuildFailed("cargo said no")
    with pytest.raises(BuildFailed):
        sync(s, tmp_path, build=broken, bench=lambda p: 1)
    assert current_id(tmp_path) == A[:12] and (tmp_path / "state" / "sync-error").exists()

def test_unreachable_store_leaves_the_current_engine(tmp_path):
    s = store_with_champion(tmp_path / "store", A)
    sync(s, tmp_path, build=fake_build, bench=lambda p: 2_850_000)
    class Down:
        def read_bytes(self, *a): raise StoreUnavailable("offline")
    with pytest.raises(StoreUnavailable):
        sync(Down(), tmp_path, build=fake_build, bench=lambda p: 1)
    assert current_id(tmp_path) == A[:12]

def test_rollback_switches_back_without_rebuilding(tmp_path):
    s = store_with_champion(tmp_path / "store", A)
    sync(s, tmp_path, build=fake_build, bench=lambda p: 2_850_000)
    promote_accepted(s, B); sync(s, tmp_path, build=fake_build, bench=lambda p: 2_850_000)
    m = load(s); rollback(m); save(s, m)
    assert sync(s, tmp_path, build=fake_build, bench=lambda p: 2_850_000) == f"switched to {A[:12]}"

def test_success_clears_an_old_error(tmp_path):
    s = store_with_champion(tmp_path / "store", A)
    (tmp_path / "state").mkdir(); (tmp_path / "state" / "sync-error").write_text("old")
    sync(s, tmp_path, build=fake_build, bench=lambda p: 2_850_000)
    assert not (tmp_path / "state" / "sync-error").exists()

def test_games_are_uploaded_once(tmp_path):
    s = LocalStore(tmp_path / "store")
    rec = tmp_path / "bot" / "game_records"; rec.mkdir(parents=True); (rec / "a.pgn").write_text("1. e4")
    assert upload_games(s, tmp_path) == 1 and upload_games(s, tmp_path) == 0

def test_engine_dir_falls_back_to_the_working_tree(tmp_path):
    assert engine_dir(tmp_path) == tmp_path / "engine" / "target" / "release"
```

- [ ] **Step 2: Run** `uv run --project ops pytest ops/tests/test_sync.py -q` — Expected: FAIL, module missing.
- [ ] **Step 3: Implement.** A new build is checked before the switch: it must answer `uci` with `uciok` within 5 seconds, and its bench speed must be at least `(1 - SPEED_MARGIN)` of the current build's, both measured now, one after the other. `box_nps` is written to the manifest only when it is still null. Nothing in `builds/` is ever deleted by `sync`.
- [ ] **Step 4: Write the units.**
  - `vektr64-sync.service`: `Type=oneshot`, `User=irom`, `WorkingDirectory=/home/irom/Vektr64`, `EnvironmentFile=/home/irom/Vektr64/.env`, `ExecStart=/home/irom/.local/bin/uv run --project ops vektr-ops sync`, `Nice=10`.
  - `vektr64-sync.timer`: `OnBootSec=2min`, `OnUnitActiveSec=10min`, `WantedBy=timers.target`.
  - `vektr64-worker.service`: as the sync service but `Type=simple`, `ExecStart=... vektr-ops gate --work`, `Nice=19`, `MemoryMax=700M`, `Restart=always`, `RestartSec=60`.
  - `install-box.sh` copies all units and runs `daemon-reload`; it enables none of them.
- [ ] **Step 5: Run** — Expected: 9 passed; whole ops suite passes; `uv run --project ops vektr-ops play --dry-run` on the laptop still prints a config whose engine directory is `engine/target/release`.
- [ ] **Step 6: Commit** `feat(ops): box follows the manifest champion with a build check`.

---

### Task 10: Status and the page

**Files:**
- Create: `ops/src/vektr_ops/page.py`, `deploy/vektr64-page.service`
- Modify: `ops/src/vektr_ops/cli.py` (`page` command, extended `status`), `ops/src/vektr_ops/sync.py` (the `sync` command regenerates the page last, and a page failure does not fail the sync), `docs/runbook.md`, `README.md`
- Test: `ops/tests/test_page.py`

**Interfaces:**
- Consumes: `Manifest` (Task 4), `current_id` (Task 9).
- Produces:
  - `@dataclass PageData: generated: str; blitz_rating: int | None; blitz_rd: int | None; blitz_games: int; history: list[tuple[str, int]]; versions: list[Version]; champion: str | None; playing_id: str | None; sync_error: str | None`
  - `render(data: PageData) -> str` — one self-contained HTML document: no scripts, no external requests, charts as inline SVG.
  - `line_chart(points: list[tuple[str, float]], *, width=720, height=240) -> str` — an `<svg>` element; `"<p>No data yet.</p>"` for fewer than two points.
  - `collect(store, root, fetch_json) -> PageData` — rating history from `https://lichess.org/api/user/Vektr64/rating-history` (the `Blitz` entry; months there are zero-based), the rest from `https://lichess.org/api/user/Vektr64`.
- CLI: `vektr-ops page --out site` writes `site/index.html`. `vektr-ops status` adds the lines `champion:`, `playing:`, `anchor rating:`, `last gate:` and `sync error:` (only when present), and still works with no store configured, printing `store: not configured` instead of those lines.

- [ ] **Step 1: Write the failing tests**

```python
def data(**over):
    base = dict(generated="2026-11-28 10:00 UTC", blitz_rating=1850, blitz_rd=62, blitz_games=73,
                history=[("2026-11-27", 3000), ("2026-11-28", 1850)], versions=[], champion=None,
                playing_id=None, sync_error=None)
    return PageData(**{**base, **over})

def test_page_is_self_contained():
    html = render(data())
    assert "<script" not in html and "http://" not in html.replace("http://www.w3.org", "")
    assert "<svg" in html and "1850" in html

def test_eligibility_shows_both_thresholds():
    html = render(data(blitz_games=12, blitz_rd=140))
    assert "12 / 50" in html and "140 / 75" in html and "not yet eligible" in html.lower()
    assert "eligible" in render(data()).lower() and "not yet eligible" not in render(data()).lower()

def test_gate_history_lists_every_decision():
    vs = [Version("a" * 12, "a" * 40, None, None, "champion", "t0", None, {"elo": 1950.0, "margin": 30.0, "games": 800}, 2850000),
          Version("b" * 12, "b" * 40, None, None, "rejected", "t1", {"decision": "reject", "llr": -2.97, "games": 5200, "elo": -1.2, "margin": 6.0}, None, None)]
    html = render(data(versions=vs, champion="a" * 12, playing_id="a" * 12))
    assert "bbbbbbbbbbbb" in html and "reject" in html and "5200" in html and "1950" in html

def test_mismatch_between_champion_and_playing_is_called_out():
    html = render(data(champion="a" * 12, playing_id="b" * 12))
    assert "not the champion" in html

def test_sync_error_is_shown_and_escaped():
    assert "&lt;b&gt;boom" in render(data(sync_error="<b>boom"))

def test_empty_account_renders():
    html = render(data(blitz_rating=None, blitz_rd=None, blitz_games=0, history=[]))
    assert "No data yet." in html

def test_line_chart_needs_two_points():
    assert line_chart([("a", 1.0)]) == "<p>No data yet.</p>"
    assert line_chart([("a", 1.0), ("b", 2.0)]).startswith("<svg")
```

- [ ] **Step 2: Run** `uv run --project ops pytest ops/tests/test_page.py -q` — Expected: FAIL, module missing.
- [ ] **Step 3: Implement.** Sections in this order: headline (rating, games, eligibility), Lichess blitz rating over time, anchor rating by version (one point per measured version, in promotion order), gate history table (newest first: id, decision, games, Elo ± margin, date), footer with the generation time. Every value from outside goes through `html.escape`. Readable in light and dark (`color-scheme: light dark`, `currentColor` for chart strokes).
- [ ] **Step 4: Write `vektr64-page.service`:** `User=irom`, `ExecStart=/usr/bin/python3 -m http.server 8064 --directory /home/irom/Vektr64/site`, `Restart=always`, `MemoryMax=100M`. Add it to `install-box.sh`.
- [ ] **Step 5: Run** — Expected: 7 passed; whole ops suite passes. Then `VEKTR_STORE=/tmp/x uv run --project ops vektr-ops page --out site` and open `site/index.html`: the rating chart shows the account's real history.
- [ ] **Step 6: Update the docs.** `docs/runbook.md` gains: how to gate a change, read the page, roll back, and what `sync error` means. `README.md` gains the new commands.
- [ ] **Step 7: Commit** `feat(ops): status page and extended status command`.

---

### Task 11: Go live

**Files:**
- Modify: `docs/bench.md` (gate time control and first measured numbers), `docs/runbook.md`

- [ ] **Step 1: Local proof of the whole gate, no accounts needed.** With `VEKTR_STORE=$PWD/state/local-store`: `vektr-ops promote $(git rev-parse --short=12 origin/main) --bootstrap`, then gate that same commit against itself for exactly 4 batches (a temporary `--max-batches 4` option is acceptable and stays). Expected: 400 games, zero discarded batches, `llr` between −2.94 and 2.94, Elo estimate within its own margin of 0.
- [ ] **Step 2: OWNER — Hugging Face token.** Ask the owner to create a Hugging Face account if needed and a **write** token, and to put `HF_TOKEN=` in `.env` on the laptop. Copy that line to the box's `.env` (mode 600) without printing it.
- [ ] **Step 3: Register the first version.** `vektr-ops promote <main commit id> --bootstrap` against the real store. Expected: two private repos exist, `manifest.json` names the champion. Confirm both repos are private.
- [ ] **Step 4: Box.** Pull `stage2-measuring-tools` on the box, run `./deploy/install-box.sh`, then `vektr-ops sync` once by hand. Expected: `switched to <id>`, `builds/current` exists, the manifest has `box_nps`. Restart `vektr64` between games (`in a game now: no`) so the bridge starts from `builds/current`; confirm the next rated game completes normally.
- [ ] **Step 5: Enable the units.** `sudo systemctl enable --now vektr64-sync.timer vektr64-page.service vektr64-worker.service`. Expected: the page loads at `http://100.114.82.19:8064/`; the worker starts anchor matches for the champion.
- [ ] **Step 6: Watch the first hour for harm to rated play.** After the worker has run through at least 3 rated games: no time forfeit in those games (`journalctl -u vektr64`), the engine's reported speed in game logs is not more than 15% below its stage 1 figure (2.7M nps), no discarded anchor batches. If any of these fails, stop the worker, record it in the runbook, and change the box rule to pause during games.
- [ ] **Step 7: Let the anchor ladder finish** (800 games; the laptop can contribute with `vektr-ops anchors`). Expected: the champion has an anchor rating with a margin, and the page shows the first point on the curve. Record it in `docs/bench.md`.
- [ ] **Step 8: Commit, push, open a pull request** to `main`. Do not merge.
