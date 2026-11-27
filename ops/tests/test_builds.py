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
