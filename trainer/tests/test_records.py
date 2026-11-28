import numpy as np
import pytest

from vektr_trainer.records import RECORD_BYTES, read_records, write_records

REFERENCE = "0c00" "4401" "fc02" + "0003" * 29 + "dbff" "01" "00"


def rows(n, seed=0):
    rng = np.random.default_rng(seed)
    return (
        rng.integers(0, 769, (n, 32)).astype(np.uint16),
        rng.integers(0, 2, n).astype(np.uint8),
        rng.integers(-3000, 3001, n).astype(np.int16),
        rng.integers(0, 3, n).astype(np.uint8),
    )


def test_reference_record(tmp_path):
    (tmp_path / "a.vkd").write_bytes(bytes.fromhex(REFERENCE))
    got = read_records(tmp_path / "a.vkd")
    assert got["features"].shape == (1, 32) and got["features"].dtype == np.uint16
    assert got["features"][0][:3].tolist() == [12, 324, 764]
    assert (got["features"][0][3:] == 768).all()
    assert got["score"][0] == -37 and got["stm"][0] == 1 and got["result"][0] == 0


@pytest.mark.parametrize("name", ["a.vkd", "a.vkd.gz"])
def test_round_trip_plain_and_gzip(tmp_path, name):
    data = rows(1000)
    write_records(tmp_path / name, *data)
    assert ((tmp_path / name).read_bytes()[:2] == b"\x1f\x8b") == name.endswith(".gz")
    got = read_records(tmp_path / name)
    for key, expected in zip(("features", "stm", "score", "result"), data):
        assert np.array_equal(got[key], expected)
    if not name.endswith(".gz"):
        assert (tmp_path / name).stat().st_size == 1000 * RECORD_BYTES


def test_truncated_file_names_the_file(tmp_path):
    (tmp_path / "t.vkd").write_bytes(bytes(100))
    with pytest.raises(ValueError, match="t.vkd"):
        read_records(tmp_path / "t.vkd")


def test_cut_gzip_names_the_file(tmp_path):
    write_records(tmp_path / "c.vkd.gz", *rows(1000))
    raw = (tmp_path / "c.vkd.gz").read_bytes()
    (tmp_path / "c.vkd.gz").write_bytes(raw[: len(raw) // 2])
    with pytest.raises(ValueError, match="c.vkd.gz"):
        read_records(tmp_path / "c.vkd.gz")


def test_bad_values_are_refused(tmp_path):
    good = bytearray(bytes.fromhex(REFERENCE))
    cases = {"feature": (2, 769), "stm": (66, 2), "result": (67, 3)}
    for name, (offset, value) in cases.items():
        bad = bytearray(good)
        if name == "feature":
            bad[offset : offset + 2] = value.to_bytes(2, "little")
        else:
            bad[offset] = value
        path = tmp_path / f"{name}.vkd"
        path.write_bytes(bytes(bad))
        with pytest.raises(ValueError, match=name):
            read_records(path)
