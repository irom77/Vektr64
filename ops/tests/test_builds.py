import pytest

from vektr_ops.builds import ensure_build, parse_bench


def test_parse_bench():
    assert parse_bench("38791222 nodes 2850000 nps\n") == (38791222, 2850000)


def test_parse_bench_rejects_other_output():
    with pytest.raises(ValueError):
        parse_bench("error: no such command")


def test_ensure_build_reuses_an_existing_binary(tmp_path):
    binary = tmp_path / "builds" / ("a" * 12) / "vektr"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"x")
    assert ensure_build(tmp_path, "a" * 40) == binary  # no git, no cargo needed


class Done:
    def __init__(self, returncode=0, stdout=b"", stderr=b""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


def fake_tools(monkeypatch, cargo):
    """Replace git, tar and cargo; `cargo(cwd)` decides how the build ends."""
    calls = []

    def run(args, **kwargs):
        calls.append(args[0])
        if args[0] == "cargo":
            return cargo(kwargs["cwd"])
        return Done()

    monkeypatch.setattr("vektr_ops.builds.subprocess.run", run)
    return calls


def test_failed_build_leaves_a_binary_another_process_finished(tmp_path, monkeypatch):
    from vektr_ops.builds import BuildFailed

    binary = tmp_path / "builds" / ("a" * 12) / "vektr"

    def cargo(cwd):  # the other process finishes first, then this build fails
        binary.parent.mkdir(parents=True, exist_ok=True)
        binary.write_bytes(b"theirs")
        return Done(1, "", "error: could not compile")

    fake_tools(monkeypatch, cargo)
    with pytest.raises(BuildFailed):
        ensure_build(tmp_path, "a" * 40)
    assert binary.read_bytes() == b"theirs"


def test_successful_build_installs_the_binary_and_leaves_no_scratch(tmp_path, monkeypatch):
    def cargo(cwd):
        out = cwd / "target" / "release"
        out.mkdir(parents=True)
        (out / "vektr").write_bytes(b"built")
        return Done(0, "", "")

    fake_tools(monkeypatch, cargo)
    binary = ensure_build(tmp_path, "a" * 40)
    assert binary == tmp_path / "builds" / ("a" * 12) / "vektr" and binary.read_bytes() == b"built"
    left = sorted(p.name for p in (tmp_path / "builds").iterdir())
    assert left == ["a" * 12, "a" * 12 + ".lock"]


def test_failed_build_is_not_retried_at_once(tmp_path, monkeypatch):
    from vektr_ops.builds import BuildFailed

    calls = fake_tools(monkeypatch, lambda cwd: Done(1, "", "error: could not compile"))
    with pytest.raises(BuildFailed, match="could not compile"):
        ensure_build(tmp_path, "a" * 40)
    with pytest.raises(BuildFailed, match="could not compile"):
        ensure_build(tmp_path, "a" * 40)
    assert calls.count("cargo") == 1


def test_failed_build_is_retried_after_the_wait(tmp_path, monkeypatch):
    import os

    from vektr_ops import builds

    calls = fake_tools(monkeypatch, lambda cwd: Done(1, "", "error: could not compile"))
    with pytest.raises(builds.BuildFailed):
        ensure_build(tmp_path, "a" * 40)
    note = tmp_path / "builds" / ("a" * 12 + ".failed")
    old = note.stat().st_mtime - builds.RETRY_AFTER_S - 1
    os.utime(note, (old, old))
    with pytest.raises(builds.BuildFailed):
        ensure_build(tmp_path, "a" * 40)
    assert calls.count("cargo") == 2


def script(tmp_path, body):
    path = tmp_path / f"engine{len(list(tmp_path.iterdir()))}"
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(0o755)
    return path


def test_netcheck_is_true_only_for_the_ok_line(tmp_path):
    from vektr_ops.builds import netcheck

    assert netcheck(script(tmp_path, 'echo "netcheck ok 256 positions"')) is True
    assert netcheck(script(tmp_path, 'echo "mismatch x"; exit 1')) is False
    assert netcheck(script(tmp_path, 'echo "netcheck ok 1 positions"; exit 1')) is False
    assert netcheck(script(tmp_path, "cat >/dev/null")) is False  # an engine without the command
    assert netcheck(tmp_path / "missing") is False


def test_net_hash_reads_the_file_at_that_commit(tmp_path):
    import hashlib
    import subprocess

    from vektr_ops.builds import net_hash

    def git(*args):
        return subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
            cwd=tmp_path, capture_output=True, text=True, check=True,
        ).stdout.strip()

    git("init", "-q")
    (tmp_path / "a.txt").write_text("x")
    git("add", ".")
    git("commit", "-q", "-m", "one")
    before = git("rev-parse", "HEAD")
    net = tmp_path / "engine" / "nets" / "default.vknn"
    net.parent.mkdir(parents=True)
    net.write_bytes(b"network bytes")
    git("add", ".")
    git("commit", "-q", "-m", "two")
    after = git("rev-parse", "HEAD")
    assert net_hash(tmp_path, before) is None
    assert net_hash(tmp_path, after) == hashlib.sha256(b"network bytes").hexdigest()[:16]
