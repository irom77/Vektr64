import pytest

from vektr_ops.sizing import hash_mb


@pytest.mark.parametrize(
    "avail,want", [(2048, 128), (1024, 64), (600, 32), (300, 16), (0, 16), (64000, 128)]
)
def test_hash_mb(avail, want):
    assert hash_mb(avail) == want
