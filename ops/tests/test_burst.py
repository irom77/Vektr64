import http.client
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from vektr_ops import runpod
from vektr_ops.burst import (
    GPUS, BurstFailed, hard_limit, kill_all, launch, pod_body, start_cmd, watch,
)
from vektr_ops.cli import app
from vektr_ops.store import LocalStore


class FakeApi:
    """Stands in for the `runpod` module; records every call."""

    RunpodError = runpod.RunpodError

    def __init__(self, creates=(), pods=(), states=(), delete_fails=()):
        self.delete_fails = set(delete_fails)
        self.creates = list(creates)  # each: a pod dict or a RunpodError to raise
        self.pods = list(pods)
        self.states = list(states)  # answers of get_pod, in order; the last repeats
        self.calls = []
        self.bodies = []

    def create_pod(self, key, body):
        self.calls.append(("create", body["gpu"]["id"]))
        self.bodies.append(body)
        answer = self.creates.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    def get_pod(self, key, id):
        self.calls.append(("get", id))
        answer = self.states.pop(0) if len(self.states) > 1 else self.states[0]
        if isinstance(answer, BaseException):
            raise answer
        return answer

    def delete_pod(self, key, id):
        self.calls.append(("delete", id))
        if id in self.delete_fails:
            raise runpod.RunpodError(500, "cannot delete")

    def list_pods(self, key):
        return self.pods

    def deleted(self):
        return [c[1] for c in self.calls if c[0] == "delete"]


def bodies(n=2):
    return [pod_body("r1", "abc", 45, "hf_SECRET", g) for g in GPUS[:n]]


def test_the_watchdog_and_trap_come_before_anything_else():
    script = start_cmd("abc123", 3900)[2]
    assert script.index("bye() {") < script.index("sleep 3900") < script.index("trap") < script.index("curl -fsSL")
    assert "(sleep 3900; bye) &" in script and "raw.githubusercontent.com/irom77/Vektr64/abc123/deploy/burst.sh" in script


def test_the_hard_limit_is_setup_plus_training():
    assert hard_limit(45) == 3900


def run_pod_script(tmp_path, uploader=None, timeout_s=1, runpodctl=None, extra_env=None):
    """Execute the generated pod script with stub runpodctl and curl (and optionally vektr-ops)."""
    import signal

    stubs = tmp_path / "bin"
    stubs.mkdir()
    record = tmp_path / "record"
    (stubs / "runpodctl").write_text(
        (runpodctl or '#!/bin/sh\necho "$@" >> RECORD\n').replace("RECORD", str(record))
    )
    # curl: the script download breaks; a DELETE call is recorded (without its headers) and works
    (stubs / "curl").write_text(
        '#!/bin/sh\ncase "$*" in *DELETE*) for a in "$@"; do last=$a; done; '
        f'echo "curl DELETE $last" >> {record}; exit 0;; esac\n'
        "echo 'curl: the download broke' >&2\nexit 22\n"
    )
    if uploader is not None:
        (stubs / "vektr-ops").write_text(uploader.replace("RECORD", str(record)))
    for f in stubs.iterdir():
        f.chmod(0o755)
    env = {
        "PATH": f"{stubs}:/usr/bin:/bin", "RUNPOD_POD_ID": "pod42", "VEKTR_RUN": "r1",
        "VEKTR_UPLOAD_TIMEOUT": str(timeout_s), "VEKTR_RETRY_S": "0", **(extra_env or {}),
    }
    with open(tmp_path / "out", "wb") as out:
        proc = subprocess.Popen(start_cmd("abc", 30), env=env, stdout=out, stderr=out, start_new_session=True)
        try:
            code = proc.wait(timeout=20)
        finally:
            os.killpg(proc.pid, signal.SIGKILL)  # the watchdog's sleep must not outlive the test
    return code, record.read_text().strip().splitlines()


def test_a_failing_start_deletes_the_pod(tmp_path):
    code, record = run_pod_script(tmp_path)
    assert code != 0
    assert record == ["pod delete pod42"]


OLD_RUNPODCTL = '#!/bin/sh\necho "$@" >> RECORD\n[ "$1" = remove ]\n'  # knows only `remove pod`
NO_RUNPODCTL = '#!/bin/sh\necho "$@" >> RECORD\nexit 1\n'


def test_an_older_runpodctl_still_deletes_the_pod(tmp_path):
    code, record = run_pod_script(tmp_path, runpodctl=OLD_RUNPODCTL)
    assert code != 0
    assert record == ["pod delete pod42", "remove pod pod42"]


def test_the_rest_api_deletes_the_pod_when_runpodctl_cannot(tmp_path):
    code, record = run_pod_script(tmp_path, runpodctl=NO_RUNPODCTL, extra_env={"RUNPOD_API_KEY": "rpa_INPOD"})
    assert record == ["pod delete pod42", "remove pod pod42", "curl DELETE https://api.runpod.io/v2/pods/pod42"]
    assert "rpa_INPOD" not in (tmp_path / "out").read_text()


def test_a_delete_that_keeps_failing_is_tried_three_times_and_ends(tmp_path):
    code, record = run_pod_script(tmp_path, runpodctl=NO_RUNPODCTL)
    assert code != 0
    assert record == ["pod delete pod42", "remove pod pod42"] * 3


UPLOADERS = {
    "works": '#!/bin/sh\necho "$@" > RECORD.args\ncp "$4" RECORD.log\n',
    "fails": "#!/bin/sh\nexit 3\n",
    "hangs": "#!/bin/sh\nexec sleep 30\n",
    "absent": None,
}


@pytest.mark.parametrize("kind", UPLOADERS)
def test_the_pod_is_deleted_once_whatever_the_log_upload_does(tmp_path, kind):
    code, record = run_pod_script(tmp_path, UPLOADERS[kind])
    assert code != 0
    assert record == ["pod delete pod42"]


def test_a_failed_start_uploads_the_output_tail_before_deleting(tmp_path):
    run_pod_script(tmp_path, UPLOADERS["works"])
    assert (tmp_path / "record.args").read_text().startswith("burst-log --run r1 ")
    assert "curl: the download broke" in (tmp_path / "record.log").read_text()


def test_the_log_is_not_uploaded_when_the_script_succeeded(tmp_path):
    import signal

    stubs = tmp_path / "bin"
    stubs.mkdir()
    (stubs / "runpodctl").write_text("#!/bin/sh\ntrue\n")
    (stubs / "curl").write_text("#!/bin/sh\nexit 0\n")
    (stubs / "bash").write_text("#!/bin/sh\nexit 0\n")  # stands in for the downloaded script
    (stubs / "vektr-ops").write_text(f"#!/bin/sh\ntouch {tmp_path}/uploaded\n")
    for f in stubs.iterdir():
        f.chmod(0o755)
    env = {"PATH": f"{stubs}:/usr/bin:/bin", "RUNPOD_POD_ID": "p", "VEKTR_RUN": "r1"}
    script = start_cmd("abc", 30)[2]
    proc = subprocess.Popen(["/bin/bash", "-c", script], env=env, start_new_session=True)
    try:
        assert proc.wait(timeout=20) == 0
    finally:
        os.killpg(proc.pid, signal.SIGKILL)
    assert not (tmp_path / "uploaded").exists()


def test_the_token_is_only_in_env():
    body = pod_body("r1", "abc", 45, "hf_SECRET", GPUS[0])
    assert body["env"]["HF_TOKEN"] == "hf_SECRET" and "hf_SECRET" not in " ".join(body["cmd"]) + body["name"]
    assert body["cloud"] == "SECURE" and body["gpu"] == {"id": GPUS[0], "count": 1}


def test_the_default_cloud_is_secure():
    assert pod_body("r1", "abc", 45, "t", GPUS[0])["cloud"] == "SECURE"
    assert pod_body("r1", "abc", 45, "t", GPUS[0], cloud="COMMUNITY")["cloud"] == "COMMUNITY"


@pytest.mark.parametrize("failure", [TimeoutError("timed out"), ValueError("bad json")])
def test_a_network_error_during_create_stops_and_says_to_kill(failure):
    api = FakeApi(creates=[failure, {"id": "p2", "cost": 0.2}])
    with pytest.raises(BurstFailed, match="burst --kill"):
        launch("k", bodies(), api=api)
    assert [c for c in api.calls if c[0] == "create"] == [("create", GPUS[0])]


@pytest.mark.parametrize(
    "failure",
    [runpod.RunpodError(500, "oops"), runpod.RunpodError(502, "bad gateway"), runpod.RunpodError(0, "?"),
     http.client.IncompleteRead(b"x"), http.client.BadStatusLine("x")],
)
def test_an_ambiguous_create_failure_removes_a_pod_named_for_the_run(failure):
    api = FakeApi(
        creates=[failure, {"id": "p2", "cost": 0.2}],
        pods=[{"id": "x", "name": "vektr64-r1"}, {"id": "y", "name": "vektr64-other"}],
    )
    with pytest.raises(BurstFailed, match="burst --kill"):
        launch("k", bodies(), api=api)
    assert api.deleted() == ["x"]
    assert [c for c in api.calls if c[0] == "create"] == [("create", GPUS[0])]


def test_a_failing_cleanup_still_ends_in_burst_failed_with_guidance():
    class Broken(FakeApi):
        def list_pods(self, key):
            raise OSError("down")

    api = Broken(creates=[runpod.RunpodError(503, "x")])
    with pytest.raises(BurstFailed, match="burst --kill"):
        launch("k", bodies(), api=api)


@pytest.mark.parametrize("status", [401, 402, 404, 422])
def test_a_clean_refusal_does_not_touch_other_pods(status):
    api = FakeApi(creates=[runpod.RunpodError(status, "no")], pods=[{"id": "x", "name": "vektr64-r1"}])
    with pytest.raises(BurstFailed):
        launch("k", bodies(), api=api)
    assert api.deleted() == []


def test_a_reply_without_an_id_stops_and_says_to_kill():
    api = FakeApi(creates=[{"cost": 0.2}, {"id": "p2", "cost": 0.2}])
    with pytest.raises(BurstFailed, match="burst --kill"):
        launch("k", bodies(), api=api)
    assert [c for c in api.calls if c[0] == "create"] == [("create", GPUS[0])]


def test_the_next_gpu_is_tried_when_one_has_no_capacity():
    api = FakeApi(creates=[runpod.RunpodError(400, "no capacity"), {"id": "p2", "cost": 0.2}])
    pod = launch("k", bodies(), api=api)
    assert pod["id"] == "p2" and api.calls == [("create", GPUS[0]), ("create", GPUS[1])]


def test_no_capacity_anywhere_fails_without_a_pod():
    api = FakeApi(creates=[runpod.RunpodError(400, "none"), runpod.RunpodError(403, "no access")])
    with pytest.raises(BurstFailed):
        launch("k", bodies(), api=api)
    assert api.deleted() == []


def test_a_rate_above_the_cap_is_deleted_and_not_used():
    api = FakeApi(creates=[{"id": "p1", "cost": 0.90}])
    with pytest.raises(BurstFailed):
        launch("k", bodies(1), api=api)
    assert api.deleted() == ["p1"]


def test_a_pod_without_a_known_rate_is_deleted_too():
    api = FakeApi(creates=[{"id": "p1"}, {"id": "p2", "cost": 0.3}])
    assert launch("k", bodies(), api=api)["id"] == "p2" and api.deleted() == ["p1"]


def test_insufficient_balance_stops_at_once():
    api = FakeApi(creates=[runpod.RunpodError(402, "balance")])
    with pytest.raises(BurstFailed):
        launch("k", bodies(), api=api)
    assert [c for c in api.calls if c[0] == "create"] == [("create", GPUS[0])]


@pytest.mark.parametrize(
    "garbage", [ValueError("bad json"), http.client.IncompleteRead(b"x"), "not a dict", ["x"], KeyError("k")]
)
def test_watch_treats_any_garbled_poll_as_unknown(garbage):
    api = FakeApi(states=[{"status": "RUNNING"}, garbage, {"status": "RUNNING"}, None])
    assert watch("k", "p1", 100, api=api, sleep=lambda s: None) == "gone"
    assert api.deleted() == []


def test_a_404_before_the_pod_was_ever_seen_is_not_gone():
    api = FakeApi(states=[None, None, {"status": "RUNNING"}, None])
    assert watch("k", "p1", 100, api=api, sleep=lambda s: None) == "gone"
    assert [c for c in api.calls if c[0] == "get"] == [("get", "p1")] * 4


def test_a_pod_never_seen_is_still_deleted_at_the_deadline():
    now = [0.0]
    api = FakeApi(states=[None])

    def sleep(s):
        now[0] += s

    result = watch("k", "p1", 100, api=api, sleep=sleep, clock=lambda: now[0], poll_s=30)
    assert result.startswith("deleted") and api.deleted() == ["p1"]


def test_watch_returns_when_the_pod_is_gone():
    api = FakeApi(states=[{"status": "RUNNING"}, None])
    assert watch("k", "p1", 100, api=api, sleep=lambda s: None) == "gone"
    assert api.deleted() == []


@pytest.mark.parametrize("status", ["EXITED", "ERROR"])
def test_watch_deletes_a_pod_that_exited_or_errored(status):
    api = FakeApi(states=[{"status": "RUNNING"}, {"status": status}])
    assert watch("k", "p1", 100, api=api, sleep=lambda s: None).startswith("deleted")
    assert api.deleted() == ["p1"]


def test_watch_deletes_a_pod_that_outlives_the_hard_limit():
    now = [0.0]
    api = FakeApi(states=[{"status": "RUNNING"}])

    def sleep(s):
        now[0] += s

    result = watch("k", "p1", 100, api=api, sleep=sleep, clock=lambda: now[0], poll_s=30)
    assert result.startswith("deleted") and api.deleted() == ["p1"] and now[0] >= 400


def test_watch_deletes_the_pod_once_the_run_is_in_the_store_on_two_polls():
    api = FakeApi(states=[{"status": "RUNNING"}])
    answers = iter([None, "the network is in the store", None, "the network is in the store", "the network is in the store"])
    result = watch("k", "p1", 100, api=api, sleep=lambda s: None, finished=lambda: next(answers))
    assert result == "deleted: the network is in the store"
    assert api.deleted() == ["p1"] and api.calls.count(("get", "p1")) == 4


def test_watch_ignores_a_store_that_cannot_be_read():
    api = FakeApi(states=[{"status": "RUNNING"}, None])

    def broken():
        raise RuntimeError("store down")

    assert watch("k", "p1", 100, api=api, sleep=lambda s: None, finished=broken) == "gone"


def test_watch_survives_a_failed_poll():
    api = FakeApi(states=[{"status": "RUNNING"}, runpod.RunpodError(500, "oops"), None])
    assert watch("k", "p1", 100, api=api, sleep=lambda s: None) == "gone"


def test_kill_all_touches_only_our_pods():
    api = FakeApi(pods=[{"id": "a", "name": "vektr64-r1"}, {"id": "b", "name": "other"}])
    assert kill_all("k", api=api) == (["a"], []) and api.deleted() == ["a"]


def test_kill_all_continues_past_a_failed_delete_and_reports_it():
    api = FakeApi(
        pods=[{"id": "a", "name": "vektr64-r1"}, {"id": "b", "name": "vektr64-r2"}], delete_fails=["a"]
    )
    killed, failed = kill_all("k", api=api)
    assert killed == ["b"] and api.deleted() == ["a", "b"]
    assert len(failed) == 1 and failed[0].startswith("a ")


# --- the command ---------------------------------------------------------------------------

@pytest.fixture
def isolated(monkeypatch, tmp_path):
    """No real .env, no real keys, a local store, no git: nothing here can reach the network."""
    for name in ("RUNPOD_API_KEY", "HF_TOKEN", "VEKTR_STORE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("vektr_ops.cli.repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        "vektr_ops.cli.runpod_api", lambda: pytest.fail("a test reached for the real RunPod client")
    )
    monkeypatch.setattr("vektr_ops.burst.time.sleep", lambda s: None)
    monkeypatch.setattr("vektr_ops.cli.resolve_commit", lambda root, ref: "c" * 40)
    monkeypatch.setattr("vektr_ops.cli.is_pushed", lambda root, commit: True)
    return tmp_path


def configured(monkeypatch, tmp_path, api):
    monkeypatch.setenv("RUNPOD_API_KEY", "rpa_SECRET")
    monkeypatch.setenv("HF_TOKEN", "hf_SECRET")
    monkeypatch.setenv("VEKTR_STORE", str(tmp_path / "store"))
    monkeypatch.setattr("vektr_ops.cli.runpod_api", lambda: api)


GOOD = {"id": "p1", "cost": 0.2, "status": "RUNNING"}


def test_burst_command_never_prints_a_token(isolated, monkeypatch):
    api = FakeApi(creates=[GOOD], states=[{"status": "RUNNING"}, None])
    configured(monkeypatch, isolated, api)
    r = CliRunner().invoke(app, ["burst", "--minutes", "45", "--run", "r1", "--yes"])
    assert "hf_SECRET" not in r.output and "rpa_SECRET" not in r.output
    assert "worst case $" in r.output and "$0.20/h" in r.output
    assert api.deleted() == []  # the pod removed itself
    assert r.exit_code == 1 and "no network in the store" in r.output  # gone, but nothing was trained


def test_burst_succeeds_when_the_network_is_in_the_store(isolated, monkeypatch):
    api = FakeApi(creates=[GOOD], states=[{"status": "RUNNING"}, None])
    configured(monkeypatch, isolated, api)
    LocalStore(isolated / "store").write_bytes("nets", "runs/r1/net.vknn", b"n")
    r = CliRunner().invoke(app, ["burst", "--minutes", "45", "--run", "r1", "--yes"])
    assert r.exit_code == 0 and "install-net r1" in r.output


def test_the_default_cloud_is_secure_and_is_shown(isolated, monkeypatch):
    api = FakeApi(creates=[GOOD], states=[{"status": "RUNNING"}, None])
    configured(monkeypatch, isolated, api)
    r = CliRunner().invoke(app, ["burst", "--minutes", "45", "--run", "r1", "--yes"])
    assert api.bodies[0]["cloud"] == "SECURE" and "SECURE" in r.output
    assert "HF_BURST_TOKEN" in r.output  # the recommendation, since only HF_TOKEN is set


def test_community_cloud_is_opt_in_and_case_insensitive(isolated, monkeypatch):
    api = FakeApi(creates=[GOOD], states=[{"status": "RUNNING"}, None])
    configured(monkeypatch, isolated, api)
    r = CliRunner().invoke(app, ["burst", "--minutes", "45", "--run", "r1", "--yes", "--cloud", "Community"])
    assert api.bodies[0]["cloud"] == "COMMUNITY" and "COMMUNITY" in r.output


def test_an_unknown_cloud_exits_2(isolated, monkeypatch):
    api = FakeApi(creates=[GOOD])
    configured(monkeypatch, isolated, api)
    r = CliRunner().invoke(app, ["burst", "--minutes", "45", "--cloud", "moon"])
    assert r.exit_code == 2 and api.calls == []


def test_the_burst_token_is_preferred_and_nothing_is_printed(isolated, monkeypatch):
    api = FakeApi(creates=[GOOD], states=[{"status": "RUNNING"}, None])
    configured(monkeypatch, isolated, api)
    monkeypatch.setenv("HF_BURST_TOKEN", "hf_BURSTSECRET")
    r = CliRunner().invoke(app, ["burst", "--minutes", "45", "--run", "r1", "--yes"])
    assert api.bodies[0]["env"]["HF_TOKEN"] == "hf_BURSTSECRET"
    for secret in ("hf_BURSTSECRET", "hf_SECRET", "rpa_SECRET"):
        assert secret not in r.output
    assert "HF_BURST_TOKEN" not in r.output  # no recommendation when it is set


def test_declining_deletes_the_pod(isolated, monkeypatch):
    api = FakeApi(creates=[GOOD], states=[{"status": "RUNNING"}])
    configured(monkeypatch, isolated, api)
    r = CliRunner().invoke(app, ["burst", "--minutes", "45", "--run", "r1"], input="n\n")
    assert r.exit_code == 1 and api.deleted() == ["p1"]


def test_an_interrupt_after_the_pod_exists_deletes_it(isolated, monkeypatch):
    api = FakeApi(creates=[GOOD], states=[KeyboardInterrupt()])
    configured(monkeypatch, isolated, api)
    CliRunner().invoke(app, ["burst", "--minutes", "45", "--run", "r1", "--yes"])
    assert api.deleted() == ["p1"]


def test_an_error_outside_the_poll_after_the_pod_exists_deletes_it(isolated, monkeypatch):
    api = FakeApi(creates=[GOOD], states=[{"status": "RUNNING"}])
    configured(monkeypatch, isolated, api)

    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr("vektr_ops.burst.watch", boom)
    r = CliRunner().invoke(app, ["burst", "--minutes", "45", "--run", "r1", "--yes"])
    assert r.exit_code != 0 and api.deleted() == ["p1"]


def test_a_garbled_poll_does_not_delete_a_healthy_pod_in_the_command(isolated, monkeypatch):
    api = FakeApi(creates=[GOOD], states=[{"status": "RUNNING"}, ValueError("boom"), None])
    configured(monkeypatch, isolated, api)
    CliRunner().invoke(app, ["burst", "--minutes", "45", "--run", "r1", "--yes"])
    assert api.deleted() == []


def test_a_pod_above_the_cap_is_never_kept_by_the_command(isolated, monkeypatch):
    api = FakeApi(creates=[{"id": f"p{i}", "cost": 0.9} for i in range(4)])
    configured(monkeypatch, isolated, api)
    r = CliRunner().invoke(app, ["burst", "--minutes", "45", "--yes"])
    assert r.exit_code == 1 and api.deleted() == ["p0", "p1", "p2", "p3"]


def test_an_exited_pod_is_deleted_by_the_launcher(isolated, monkeypatch):
    api = FakeApi(creates=[GOOD], states=[{"status": "EXITED"}])
    configured(monkeypatch, isolated, api)
    r = CliRunner().invoke(app, ["burst", "--minutes", "45", "--run", "r1", "--yes"])
    assert api.deleted() == ["p1"] and "pod deleted" in r.output


def test_burst_without_a_runpod_key_explains_what_to_set(isolated):
    r = CliRunner().invoke(app, ["burst", "--minutes", "45"])
    assert r.exit_code == 2 and "RUNPOD_API_KEY" in r.output and ".env" in r.output


def test_burst_without_a_hugging_face_token_is_refused(isolated, monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "rpa_SECRET")
    monkeypatch.setattr("vektr_ops.cli.runpod_api", lambda: FakeApi())
    r = CliRunner().invoke(app, ["burst", "--minutes", "45"])
    assert r.exit_code == 2 and "HF_TOKEN" in r.output


def test_burst_refuses_an_unpushed_commit(isolated, monkeypatch):
    api = FakeApi()
    configured(monkeypatch, isolated, api)
    monkeypatch.setattr("vektr_ops.cli.is_pushed", lambda root, commit: False)
    r = CliRunner().invoke(app, ["burst", "--minutes", "45", "--yes"])
    assert r.exit_code == 2 and "push it first" in r.output and api.calls == []


def test_list_and_kill_only_show_our_pods(isolated, monkeypatch):
    api = FakeApi(pods=[{"id": "a", "name": "vektr64-r1", "status": "RUNNING", "cost": 0.2}, {"id": "b", "name": "other"}])
    configured(monkeypatch, isolated, api)
    listed = CliRunner().invoke(app, ["burst", "--list"])
    assert "vektr64-r1" in listed.output and "other" not in listed.output
    killed = CliRunner().invoke(app, ["burst", "--kill"])
    assert "a" in killed.output and api.deleted() == ["a"]


def test_kill_exits_non_zero_and_names_the_pods_it_could_not_delete(isolated, monkeypatch):
    api = FakeApi(
        pods=[{"id": "a", "name": "vektr64-r1"}, {"id": "b", "name": "vektr64-r2"}], delete_fails=["a"]
    )
    configured(monkeypatch, isolated, api)
    r = CliRunner().invoke(app, ["burst", "--kill"])
    assert r.exit_code == 1 and api.deleted() == ["a", "b"]
    assert "deleted: b" in r.output and "could not delete: a" in r.output


def test_a_failed_burst_shows_the_log_the_pod_left_in_the_store(isolated, monkeypatch):
    api = FakeApi(creates=[GOOD], states=[{"status": "RUNNING"}, None])
    configured(monkeypatch, isolated, api)
    LocalStore(isolated / "store").write_bytes("nets", "runs/r1/burst.log", b"pip exploded\n")
    r = CliRunner().invoke(app, ["burst", "--minutes", "45", "--run", "r1", "--yes"])
    assert r.exit_code == 1 and "pip exploded" in r.output
    assert r.output.index("pip exploded") < r.output.index("start again")


def test_burst_log_uploads_with_every_secret_redacted(isolated, monkeypatch, tmp_path):
    secrets = {"HF_TOKEN": "hf_AAA", "HF_BURST_TOKEN": "hf_BBB", "RUNPOD_API_KEY": "rpa_CCC", "LICHESS_TOKEN": "lip_DDD"}
    for name, value in secrets.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("VEKTR_STORE", str(tmp_path / "store"))
    log = tmp_path / "tail.log"
    log.write_text("boom hf_AAA and hf_BBB then rpa_CCC / lip_DDD twice lip_DDD\n")
    r = CliRunner().invoke(app, ["burst-log", "--run", "r9", str(log)])
    assert r.exit_code == 0, r.output
    stored = LocalStore(tmp_path / "store").read_bytes("nets", "runs/r9/burst.log").decode()
    assert stored == "boom *** and *** then *** / *** twice ***\n"
    again = CliRunner().invoke(app, ["burst-log", "--run", "r9", str(log)])  # overwrite is allowed
    assert again.exit_code == 0


def test_the_app_never_shows_locals_in_tracebacks():
    assert app.pretty_exceptions_show_locals is False


@pytest.mark.slow
def test_local_dry_run_trains_and_uploads(tmp_path):
    root = Path(__file__).resolve().parents[2]
    store = tmp_path / "store"
    shard = tmp_path / "shard.npz"
    make = (
        "import itertools, sys, numpy as np\n"
        "from pathlib import Path\n"
        "from vektr_trainer import vknn\n"
        "from vektr_trainer.features import EMPTY, fen_features\n"
        "from vektr_trainer.prep import NO_RESULT, write_shard\n"
        "n = 8192\n"
        "f = np.full((n, 32), EMPTY, dtype=np.uint16); s = np.zeros(n, dtype=np.uint8); c = np.zeros(n, dtype=np.int16)\n"
        "for i, fen in zip(range(n), itertools.cycle(vknn.check_fens())):\n"
        "    idx, side = fen_features(fen); f[i, :len(idx)] = idx; s[i] = side\n"
        "    c[i] = 100 * (fen.split()[0].count('P') - fen.split()[0].count('p'))\n"
        "write_shard(Path(sys.argv[1]), f, s, c, np.full(n, NO_RESULT, dtype=np.uint8))\n"
    )
    subprocess.run(["uv", "run", "--project", str(root / "trainer"), "python", "-c", make, str(shard)], check=True)
    LocalStore(store).write_bytes("data", "bootstrap/dry/shard-00000.npz", shard.read_bytes())
    env = {k: v for k, v in os.environ.items() if k not in ("RUNPOD_API_KEY", "HF_TOKEN")}
    env.update(VEKTR_STORE=str(store))
    env.pop("VEKTR_TRAINER_PYTHON", None)
    try:
        done = subprocess.run(
            ["uv", "run", "--project", str(root / "ops"), "vektr-ops", "burst", "--local", "--minutes", "0.1", "--run", "dry"],
            cwd=root, env=env, capture_output=True, text=True, timeout=900,
        )
        assert done.returncode == 0, done.stdout + done.stderr
        assert (store / "nets" / "runs" / "dry" / "net.vknn").is_file()
        assert (store / "nets" / "runs" / "dry" / "run.json").is_file()
    finally:
        shutil.rmtree(root / "state" / "train" / "dry", ignore_errors=True)


def test_the_launcher_deletes_the_pod_when_the_run_record_appears(isolated, monkeypatch):
    from vektr_ops.store import LocalStore

    api = FakeApi(creates=[GOOD], states=[{"status": "RUNNING"}])  # the pod never leaves by itself
    configured(monkeypatch, isolated, api)
    store = LocalStore(isolated / "store")
    store.write_bytes("nets", "runs/r1/run.json", b"an older run")
    polls = []

    def get_pod(key, id):
        polls.append(id)
        if len(polls) == 2:  # the pod finishes: the new record and the network land in the store
            store.write_bytes("nets", "runs/r1/run.json", b"the new run", overwrite=True)
            store.write_bytes("nets", "runs/r1/net.vknn", b"net", overwrite=True)
        assert len(polls) < 50, "the launcher never deleted the pod"
        return {"status": "RUNNING"}

    monkeypatch.setattr(api, "get_pod", get_pod)
    r = CliRunner().invoke(app, ["burst", "--minutes", "45", "--run", "r1", "--yes"])
    assert api.deleted() == ["p1"] and len(polls) == 3
    assert "pod deleted: the network is in the store" in r.output and "network ready" in r.output
