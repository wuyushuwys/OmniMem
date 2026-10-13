"""Stateless generator-update schedules indexed by completed optimizer steps."""

import hashlib


VALID_GENERATOR_SCHEDULES = frozenset({"fixed", "random_phase"})


def generator_update_due(
    step: int,
    ratio: int,
    *,
    schedule: str = "fixed",
    seed: int = 42,
) -> bool:
    """Pick one generator update per ``ratio`` steps; the hash must stay stable across resumes."""
    if type(step) is not int or step < 0:
        raise ValueError("step must be a nonnegative integer")
    if type(ratio) is not int or ratio <= 0:
        raise ValueError("ratio must be a positive integer")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if not isinstance(schedule, str) or schedule not in VALID_GENERATOR_SCHEDULES:
        raise ValueError(f"Unknown generator schedule: {schedule!r}")

    if schedule == "fixed":
        return step % ratio == 0

    group = step // ratio
    payload = f"omnimem-generator-phase-v1:{seed}:{group}".encode("ascii")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    phase = int.from_bytes(digest, "little") % ratio
    return step % ratio == phase
