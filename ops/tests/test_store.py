import pytest

from vektr_ops.store import LocalStore, StoreUnavailable, open_store, read_json, write_json


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


def test_json_helpers(tmp_path):
    s = LocalStore(tmp_path)
    assert read_json(s, "nets", "m.json") is None
    write_json(s, "nets", "m.json", {"a": 1})
    assert read_json(s, "nets", "m.json") == {"a": 1}


def test_open_store_prefers_the_local_directory(tmp_path):
    assert isinstance(open_store({"VEKTR_STORE": str(tmp_path), "HF_TOKEN": "x"}), LocalStore)


def test_open_store_without_settings_explains():
    with pytest.raises(StoreUnavailable, match="HF_TOKEN or VEKTR_STORE"):
        open_store({})
