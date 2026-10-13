"""Backend-neutral integer helpers (no Triton import)."""


def next_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (int(n) - 1).bit_length()


def cdiv(a: int, b: int) -> int:
    return -(-int(a) // int(b))


def compute_tile_n(block_size: int, max_tile: int = 128) -> int:
    """Pick a tile size: one tile if the padded block fits, otherwise ``max_tile``."""
    padded = max(16, next_power_of_2(block_size))
    return padded if padded <= max_tile else max_tile
