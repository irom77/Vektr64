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
