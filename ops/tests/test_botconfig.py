import copy
from pathlib import Path

import yaml

from vektr_ops.botconfig import load_env, render

ROOT = Path(__file__).resolve().parents[2]


def tpl():
    return yaml.safe_load((ROOT / "bot/config.template.yml").read_text())


def test_render_sets_engine_and_hash():
    t = tpl()
    before = copy.deepcopy(t)
    c = render(t, engine_dir="/x/engine/target/release", engine_name="vektr", hash_mb=64)
    assert c["engine"]["dir"] == "/x/engine/target/release" and c["engine"]["name"] == "vektr"
    assert c["engine"]["uci_options"]["Hash"] == 64 and c["engine"]["uci_options"]["Threads"] == 1
    assert t == before


def test_template_pins_the_spec():
    c = tpl()
    assert c["challenge"]["concurrency"] == 1
    assert c["challenge"]["variants"] == ["standard"]
    assert c["challenge"]["time_controls"] == ["blitz"]
    assert c["matchmaking"]["allow_matchmaking"] is True
    assert c["matchmaking"]["challenge_mode"] == "rated"
    assert c["matchmaking"]["challenge_initial_time"] == [180, 300]
    assert c["matchmaking"]["challenge_increment"] == [2, 3]


def test_no_token_in_template_or_render():
    text = (ROOT / "bot/config.template.yml").read_text()
    assert "lip_" not in text


def test_env_overrides_dotenv(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("LICHESS_TOKEN=from_file\nOTHER=1\n")
    monkeypatch.setenv("LICHESS_TOKEN", "from_env")
    e = load_env(tmp_path)
    assert e["LICHESS_TOKEN"] == "from_env" and e["OTHER"] == "1"
