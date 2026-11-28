from typer.testing import CliRunner

from vektr_ops.cli import app


def test_play_without_token_fails_clearly(monkeypatch, tmp_path):
    monkeypatch.delenv("LICHESS_TOKEN", raising=False)
    monkeypatch.setattr("vektr_ops.cli.repo_root", lambda: tmp_path)
    r = CliRunner().invoke(app, ["play", "--dry-run"])
    assert r.exit_code == 2 and "LICHESS_TOKEN is not set" in r.output


def test_dry_run_prints_config_without_token(monkeypatch):
    monkeypatch.setenv("LICHESS_TOKEN", "lip_secret_value")
    monkeypatch.setattr("vektr_ops.cli.build_engine", lambda root: None)
    r = CliRunner().invoke(app, ["play", "--dry-run"])
    assert r.exit_code == 0 and "Hash" in r.output and "lip_secret_value" not in r.output


def test_bootstrap_promote_then_rollback_explains(monkeypatch, tmp_path):
    from vektr_ops.manifest import load
    from vektr_ops.store import LocalStore

    monkeypatch.setenv("VEKTR_STORE", str(tmp_path))
    monkeypatch.setattr("vektr_ops.cli.resolve_commit", lambda root, ref: "a" * 40)
    r = CliRunner().invoke(app, ["promote", "main", "--bootstrap"])
    assert r.exit_code == 0 and "champion: aaaaaaaaaaaa" in r.output
    assert load(LocalStore(tmp_path)).champion == "a" * 12
    r = CliRunner().invoke(app, ["rollback"])
    assert r.exit_code == 2 and "no previous champion" in r.output


def test_promote_without_a_passed_gate_is_refused(monkeypatch, tmp_path):
    from helpers import B, store_with_champion_and_candidate

    store_with_champion_and_candidate(tmp_path)
    monkeypatch.setenv("VEKTR_STORE", str(tmp_path / "store"))
    r = CliRunner().invoke(app, ["promote", B[:12]])
    assert r.exit_code == 2 and "has not passed the gate" in r.output


def test_gate_refuses_a_commit_that_is_not_pushed(monkeypatch, tmp_path):
    from helpers import store_with_champion_only

    store_with_champion_only(tmp_path)
    monkeypatch.setenv("VEKTR_STORE", str(tmp_path / "store"))
    monkeypatch.setattr("vektr_ops.cli.resolve_commit", lambda root, ref: "b" * 40)
    monkeypatch.setattr("vektr_ops.cli.is_pushed", lambda root, commit: False)
    r = CliRunner().invoke(app, ["gate", "HEAD"])
    assert r.exit_code == 2 and "push it first" in r.output


def test_gate_without_a_champion_explains(monkeypatch, tmp_path):
    monkeypatch.setenv("VEKTR_STORE", str(tmp_path))
    monkeypatch.setattr("vektr_ops.cli.resolve_commit", lambda root, ref: "b" * 40)
    monkeypatch.setattr("vektr_ops.cli.is_pushed", lambda root, commit: True)
    r = CliRunner().invoke(app, ["gate", "HEAD"])
    assert r.exit_code == 2 and "promote" in r.output and "--bootstrap" in r.output


def test_store_commands_explain_a_missing_store(monkeypatch, tmp_path):
    monkeypatch.delenv("VEKTR_STORE", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr("vektr_ops.cli.repo_root", lambda: tmp_path)
    r = CliRunner().invoke(app, ["rollback"])
    assert r.exit_code == 2 and "HF_TOKEN or VEKTR_STORE" in r.output


def test_play_uses_the_followed_champion_when_there_is_one(monkeypatch, tmp_path):
    import shutil

    from vektr_ops.cli import repo_root

    real = repo_root()
    (tmp_path / "bot").mkdir()
    shutil.copy(real / "bot" / "config.template.yml", tmp_path / "bot" / "config.template.yml")
    build = tmp_path / "builds" / "abc"
    build.mkdir(parents=True)
    (build / "vektr").write_bytes(b"x")
    (tmp_path / "builds" / "current").symlink_to("abc")
    monkeypatch.setenv("LICHESS_TOKEN", "lip_secret_value")
    monkeypatch.setattr("vektr_ops.cli.repo_root", lambda: tmp_path)
    monkeypatch.setattr("vektr_ops.cli.build_engine", lambda root: 1 / 0)  # must not rebuild
    r = CliRunner().invoke(app, ["play", "--dry-run"])
    assert r.exit_code == 0 and str(tmp_path / "builds" / "current") in r.output


def test_sync_command_reports_a_refusal(monkeypatch, tmp_path):
    offline(monkeypatch)
    monkeypatch.setenv("VEKTR_STORE", str(tmp_path / "store"))
    monkeypatch.setattr("vektr_ops.cli.repo_root", lambda: tmp_path)
    r = CliRunner().invoke(app, ["sync"])
    assert r.exit_code == 1 and "no champion" in r.output
    assert (tmp_path / "state" / "sync-error").exists()


def offline(monkeypatch):
    def down(*a, **k):
        raise OSError("offline")

    monkeypatch.setattr("vektr_ops.cli.urllib.request.urlopen", down)
    monkeypatch.setattr("vektr_ops.page.urllib.request.urlopen", down)


def test_status_without_a_store_still_works(monkeypatch, tmp_path):
    offline(monkeypatch)
    monkeypatch.delenv("VEKTR_STORE", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr("vektr_ops.cli.repo_root", lambda: tmp_path)
    r = CliRunner().invoke(app, ["status"])
    assert r.exit_code == 0 and "store: not configured" in r.output
    assert "lichess: unreachable" in r.output


def test_status_shows_champion_gate_and_sync_error(monkeypatch, tmp_path):
    from helpers import B, store_with_champion_and_candidate

    from vektr_ops.manifest import load, reject, save

    offline(monkeypatch)
    s, m = store_with_champion_and_candidate(tmp_path)
    m.champion_version().anchor = {"elo": 1950.0, "margin": 30.0, "games": 800}
    reject(m, B[:12], {"decision": "reject", "games": 5200, "elo": -1.2, "margin": 6.0})
    save(s, m)
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "sync-error").write_text("SyncRefused: boom")
    monkeypatch.setenv("VEKTR_STORE", str(tmp_path / "store"))
    monkeypatch.setattr("vektr_ops.cli.repo_root", lambda: tmp_path)
    out = CliRunner().invoke(app, ["status"]).output
    assert "champion: aaaaaaaaaaaa" in out and "playing: working-tree build" in out
    assert "anchor rating: 1950 ±30" in out
    assert "last gate: bbbbbbbbbbbb reject" in out and "sync error: SyncRefused: boom" in out


def test_status_shows_selfplay_progress(monkeypatch, tmp_path):
    from helpers import store_with_champion_only

    offline(monkeypatch)
    s, _ = store_with_champion_only(tmp_path)
    for i in range(2):
        s.write_bytes("data", f"selfplay/20261128T00000{i}Z-none-box-0000.vkd.gz", b"x")
    s.write_bytes("data", "experience/game.fens", b"fen\n")
    monkeypatch.setenv("VEKTR_STORE", str(tmp_path / "store"))
    monkeypatch.setattr("vektr_ops.cli.repo_root", lambda: tmp_path)
    out = CliRunner().invoke(app, ["status"]).output
    assert "self-play: 2 of 120 shards for network none, lost games: 1" in out


def test_page_command_writes_the_file(monkeypatch, tmp_path):
    offline(monkeypatch)
    monkeypatch.setenv("VEKTR_STORE", str(tmp_path / "store"))
    monkeypatch.setattr("vektr_ops.cli.repo_root", lambda: tmp_path)
    r = CliRunner().invoke(app, ["page", "--out", str(tmp_path / "site")])
    assert r.exit_code == 0
    assert "<title>Vektr64 status</title>" in (tmp_path / "site" / "index.html").read_text()


def test_reject_command_withdraws_a_candidate(monkeypatch, tmp_path):
    from helpers import B, store_with_champion_and_candidate

    from vektr_ops.manifest import load

    s, _ = store_with_champion_and_candidate(tmp_path)
    monkeypatch.setenv("VEKTR_STORE", str(tmp_path / "store"))
    r = CliRunner().invoke(app, ["reject", B[:12]])
    after = load(s).get(B[:12])
    assert r.exit_code == 0 and "rejected" in r.output
    assert after.status == "rejected" and after.gate["reason"] == "withdrawn"
    r = CliRunner().invoke(app, ["reject", "nosuchid"])
    assert r.exit_code == 2 and "no version" in r.output


def test_gate_on_a_withdrawn_version_says_so(monkeypatch, tmp_path):
    from helpers import B, store_with_champion_and_candidate

    store_with_champion_and_candidate(tmp_path)
    monkeypatch.setenv("VEKTR_STORE", str(tmp_path / "store"))
    monkeypatch.setattr("vektr_ops.cli.resolve_commit", lambda root, ref: B)
    monkeypatch.setattr("vektr_ops.cli.is_pushed", lambda root, commit: True)
    CliRunner().invoke(app, ["reject", B[:12]])
    r = CliRunner().invoke(app, ["gate", "some-ref"])
    assert r.exit_code == 0 and "already rejected (withdrawn)" in r.output


def cli_setup(monkeypatch, tmp_path):
    from vektr_ops.store import LocalStore

    monkeypatch.setattr("vektr_ops.cli.repo_root", lambda: tmp_path)
    monkeypatch.setenv("VEKTR_STORE", str(tmp_path / "store"))
    nets = tmp_path / "engine" / "nets"
    nets.mkdir(parents=True)
    (nets / "default.vknn").write_bytes(b"old net")
    (nets / "default.check").write_bytes(b"old check")
    store = LocalStore(tmp_path / "store")
    store.write_bytes("nets", "runs/r1/net.vknn", b"network")
    store.write_bytes("nets", "runs/r1/net.check", b"check")
    return nets


def test_install_net_of_an_unknown_run_exits_2(monkeypatch, tmp_path):
    cli_setup(monkeypatch, tmp_path)
    r = CliRunner().invoke(app, ["install-net", "nope"])
    assert r.exit_code == 2 and "no network" in r.output


def test_install_net_with_a_failing_netcheck_exits_1_and_leaves_the_engine_alone(monkeypatch, tmp_path):
    nets = cli_setup(monkeypatch, tmp_path)
    monkeypatch.setattr("vektr_ops.cli.engine_check", lambda root: ["mismatch: score 12 != 13"])
    r = CliRunner().invoke(app, ["install-net", "r1"])
    assert r.exit_code == 1 and "mismatch: score 12 != 13" in r.output
    assert (nets / "default.vknn").read_bytes() == b"old net"
    assert (nets / "default.check").read_bytes() == b"old check"


def test_install_net_installs_when_the_check_passes(monkeypatch, tmp_path):
    nets = cli_setup(monkeypatch, tmp_path)
    monkeypatch.setattr("vektr_ops.cli.engine_check", lambda root: [])
    r = CliRunner().invoke(app, ["install-net", "r1"])
    assert r.exit_code == 0 and (nets / "default.vknn").read_bytes() == b"network"


def test_train_reports_a_failed_run_and_exits_1(monkeypatch, tmp_path):
    cli_setup(monkeypatch, tmp_path)

    def boom(*a, **k):
        raise RuntimeError("trainer exited with status 1")

    monkeypatch.setattr("vektr_ops.training.run_training", boom)
    r = CliRunner().invoke(app, ["train", "--run", "r1", "--minutes", "1"])
    assert r.exit_code == 1 and "trainer exited with status 1" in r.output


from vektr_ops.selfplay import PER_NET


def test_train_passes_selfplay(monkeypatch, tmp_path):
    cli_setup(monkeypatch, tmp_path)
    seen = {}

    def fake(*a, **k):
        seen.update(k)
        return {"net": "n", "val_loss": 1.0}

    monkeypatch.setattr("vektr_ops.training.run_training", fake)
    r = CliRunner().invoke(app, ["train", "--run", "r1", "--minutes", "1", "--selfplay", "7"])
    assert r.exit_code == 0 and seen["selfplay"] == 7
    CliRunner().invoke(app, ["train", "--run", "r1", "--minutes", "1"])
    assert seen["selfplay"] == PER_NET


def gate_setup(monkeypatch, tmp_path, net, passes):
    from helpers import store_with_champion_only

    store_with_champion_only(tmp_path)
    monkeypatch.setenv("VEKTR_STORE", str(tmp_path / "store"))
    monkeypatch.setattr("vektr_ops.cli.resolve_commit", lambda root, ref: "b" * 40)
    monkeypatch.setattr("vektr_ops.cli.is_pushed", lambda root, commit: True)
    monkeypatch.setattr("vektr_ops.builds.net_hash", lambda root, commit: net)
    monkeypatch.setattr("vektr_ops.builds.ensure_build", lambda root, commit: tmp_path / "vektr")
    monkeypatch.setattr("vektr_ops.builds.netcheck", lambda binary: passes)
    monkeypatch.setattr("vektr_ops.gate.step", lambda *a, **k: {"decision": "continue", "games": 0, "llr": 0.0, "elo": 0.0, "margin": 0.0})


def test_gate_refuses_a_candidate_that_fails_netcheck(monkeypatch, tmp_path):
    from vektr_ops.manifest import load
    from vektr_ops.store import LocalStore

    gate_setup(monkeypatch, tmp_path, "0123456789abcdef", False)
    r = CliRunner().invoke(app, ["gate", "HEAD"])
    assert r.exit_code == 2 and "engine and trainer disagree on the network" in r.output
    assert [v.id for v in load(LocalStore(tmp_path / "store")).versions] == ["a" * 12]


def test_gate_registers_the_network_hash(monkeypatch, tmp_path):
    from vektr_ops.manifest import load
    from vektr_ops.store import LocalStore

    gate_setup(monkeypatch, tmp_path, "0123456789abcdef", True)
    r = CliRunner().invoke(app, ["gate", "HEAD", "--max-batches", "1"])
    assert r.exit_code == 0, r.output
    assert load(LocalStore(tmp_path / "store")).get("b" * 12).net == "0123456789abcdef"


def test_sync_reports_lost_games(monkeypatch, tmp_path):
    import os

    from test_experience import LOSS
    from vektr_ops import sync as sync_mod

    offline(monkeypatch)
    monkeypatch.setenv("VEKTR_STORE", str(tmp_path / "store"))
    monkeypatch.setattr("vektr_ops.cli.repo_root", lambda: tmp_path)
    monkeypatch.setattr(sync_mod, "sync", lambda *a, **k: "unchanged")
    records = tmp_path / "bot" / "game_records"
    records.mkdir(parents=True)
    game = records / "x vs Vektr64 - abc.pgn"
    game.write_text(LOSS)
    os.utime(game, (1, 1))
    r = CliRunner().invoke(app, ["sync"])
    assert r.exit_code == 0 and "lost games added: 1" in r.output


def test_an_unavailable_store_for_lost_games_does_not_fail_the_sync(monkeypatch, tmp_path):
    from vektr_ops import experience, sync as sync_mod
    from vektr_ops.store import StoreUnavailable

    def down(*a, **k):
        raise StoreUnavailable("hub down")

    offline(monkeypatch)
    monkeypatch.setenv("VEKTR_STORE", str(tmp_path / "store"))
    monkeypatch.setattr("vektr_ops.cli.repo_root", lambda: tmp_path)
    monkeypatch.setattr(sync_mod, "sync", lambda *a, **k: "unchanged")
    monkeypatch.setattr(experience, "upload_experience", down)
    r = CliRunner().invoke(app, ["sync"])
    assert r.exit_code == 0 and "lost games not added: StoreUnavailable: hub down" in r.output


def test_a_vanishing_record_does_not_fail_the_sync(monkeypatch, tmp_path):
    from vektr_ops import experience, sync as sync_mod

    def gone(*a, **k):
        raise OSError("record vanished")

    offline(monkeypatch)
    monkeypatch.setenv("VEKTR_STORE", str(tmp_path / "store"))
    monkeypatch.setattr("vektr_ops.cli.repo_root", lambda: tmp_path)
    monkeypatch.setattr(sync_mod, "sync", lambda *a, **k: "unchanged")
    monkeypatch.setattr(experience, "upload_experience", gone)
    r = CliRunner().invoke(app, ["sync"])
    assert r.exit_code == 0 and "lost games not added: OSError: record vanished" in r.output


def test_the_page_is_refreshed_even_when_the_game_upload_fails_badly(monkeypatch, tmp_path):
    from vektr_ops import sync as sync_mod

    def boom(*a, **k):
        raise RuntimeError("boom")

    refreshed = []
    offline(monkeypatch)
    monkeypatch.setenv("VEKTR_STORE", str(tmp_path / "store"))
    monkeypatch.setattr("vektr_ops.cli.repo_root", lambda: tmp_path)
    monkeypatch.setattr("vektr_ops.cli.refresh_page", lambda root: refreshed.append(root))
    monkeypatch.setattr(sync_mod, "sync", lambda *a, **k: "unchanged")
    monkeypatch.setattr(sync_mod, "upload_games", boom)
    r = CliRunner().invoke(app, ["sync"])
    assert r.exit_code != 0 and refreshed == [tmp_path]


def test_datagen_refuses_when_this_machine_must_not_play(monkeypatch, tmp_path):
    monkeypatch.setenv("VEKTR_STORE", str(tmp_path))
    monkeypatch.setattr("vektr_ops.match.bot_here", lambda: True)
    monkeypatch.setattr("vektr_ops.cli.gate_mod.machine", lambda: (2, 4000))
    r = CliRunner().invoke(app, ["datagen"])
    assert r.exit_code == 1 and "must not play" in r.output
