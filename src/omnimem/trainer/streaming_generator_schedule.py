"""Stateless generator-update schedules indexed by completed optimizer steps."""

VALID_GENERATOR_SCHEDULES = frozenset({"fixed"})


def generator_update_due(
    step: int,
    ratio: int,
    *,
    schedule: str = "fixed",
) -> bool:
    """Pick one generator update per ``ratio`` steps."""
    if type(step) is not int or step < 0:
        raise ValueError("step must be a nonnegative integer")
    if type(ratio) is not int or ratio <= 0:
        raise ValueError("ratio must be a positive integer")
    if not isinstance(schedule, str) or schedule not in VALID_GENERATOR_SCHEDULES:
        raise ValueError(f"Unknown generator schedule: {schedule!r}")

    return step % ratio == 0
