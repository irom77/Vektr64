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


def test_versioned_write_detects_a_change_since_the_read(tmp_path):
    from vektr_ops.store import StoreConflict

    s = LocalStore(tmp_path)
    data, seen = s.read_versioned("nets", "manifest.json")
    assert data is None
    s.write_bytes("nets", "manifest.json", b"one", expect=seen)
    data, seen = s.read_versioned("nets", "manifest.json")
    assert data == b"one"
    s.write_bytes("nets", "manifest.json", b"two", overwrite=True)
    with pytest.raises(StoreConflict):
        s.write_bytes("nets", "manifest.json", b"three", expect=seen)
    assert s.read_bytes("nets", "manifest.json") == b"two"
    assert s.list("nets", "") == ["manifest.json"]


class FakeApi:
    def __init__(self, sha="c0ffee", fail=None):
        self.sha, self.fail, self.uploads = sha, fail, []

    def repo_info(self, repo_id, repo_type):
        return type("Info", (), {"sha": self.sha})()

    def file_exists(self, *a, **k):
        return False

    def upload_file(self, **kwargs):
        self.uploads.append(kwargs)
        if self.fail:
            raise self.fail


def hub(api=None):
    from vektr_ops.store import HubStore

    s = HubStore("hf_test")
    s._api, s._user, s._ready = api or FakeApi(), "someone", {"nets", "data"}
    return s


def http_error(cls, status):
    import httpx2

    response = httpx2.Response(status, request=httpx2.Request("GET", "https://hub.invalid/x"))
    return cls("refused", response=response)


def test_hub_read_treats_only_a_remote_miss_as_missing(monkeypatch):
    from huggingface_hub.errors import LocalEntryNotFoundError, RemoteEntryNotFoundError

    def missing(*a, **k):
        raise http_error(RemoteEntryNotFoundError, 404)

    def unreachable(*a, **k):
        raise LocalEntryNotFoundError("cannot reach the Hub and the file is not cached")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", missing)
    assert hub().read_bytes("data", "gate/t/box-1.json") is None
    monkeypatch.setattr("huggingface_hub.hf_hub_download", unreachable)
    with pytest.raises(StoreUnavailable):
        hub().read_bytes("data", "gate/t/box-1.json")


def test_hub_versioned_read_pins_the_revision_it_reports(monkeypatch, tmp_path):
    seen = {}

    def download(repo_id, path, **kwargs):
        seen.update(kwargs)
        (tmp_path / "f").write_bytes(b"body")
        return str(tmp_path / "f")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    assert hub().read_versioned("nets", "manifest.json") == (b"body", "c0ffee")
    assert seen["revision"] == "c0ffee"


def test_hub_versioned_write_names_its_parent_and_reports_a_conflict():
    from huggingface_hub.errors import HfHubHTTPError

    from vektr_ops.store import StoreConflict

    api = FakeApi()
    hub(api).write_bytes("nets", "manifest.json", b"x", overwrite=True, expect="c0ffee")
    assert api.uploads[0]["parent_commit"] == "c0ffee"
    for status in (409, 412):
        with pytest.raises(StoreConflict):
            hub(FakeApi(fail=http_error(HfHubHTTPError, status))).write_bytes(
                "nets", "manifest.json", b"x", overwrite=True, expect="c0ffee"
            )
    with pytest.raises(StoreUnavailable):
        hub(FakeApi(fail=http_error(HfHubHTTPError, 500))).write_bytes(
            "nets", "manifest.json", b"x", overwrite=True, expect="c0ffee"
        )
