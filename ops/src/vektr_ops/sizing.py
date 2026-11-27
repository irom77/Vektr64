"""Size the engine from the machine it is running on."""

MIN_HASH_MB = 16
MAX_HASH_MB = 128


def hash_mb(available_mb: int) -> int:
    """Hash table size: a sixteenth of available memory, as a power of two, 16-128 MiB."""
    target = max(available_mb // 16, 1)
    power = 1 << (target.bit_length() - 1)
    return min(max(power, MIN_HASH_MB), MAX_HASH_MB)
