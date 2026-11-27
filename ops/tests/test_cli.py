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


def test_page_command_writes_the_file(monkeypatch, tmp_path):
    offline(monkeypatch)
    monkeypatch.setenv("VEKTR_STORE", str(tmp_path / "store"))
    monkeypatch.setattr("vektr_ops.cli.repo_root", lambda: tmp_path)
    r = CliRunner().invoke(app, ["page", "--out", str(tmp_path / "site")])
    assert r.exit_code == 0
    assert "<title>Vektr64 status</title>" in (tmp_path / "site" / "index.html").read_text()
