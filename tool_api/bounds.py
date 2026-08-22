"""Hard bounds. The language model does not get a vote here."""

BOUNDS = {
    "hotend": (180.0, 260.0),
    "bed": (40.0, 90.0),
    "feedrate": (50.0, 120.0),
}

ALLOWED_COMMANDS = frozenset({"none", "pause", "set_parameter"})
ALLOWED_PARAMS = frozenset(BOUNDS)


def clip_parameter(name: str, value: float) -> tuple[float, bool]:
    if name not in BOUNDS:
        raise ValueError(f"parameter {name!r} is not on the whitelist")
    lo, hi = BOUNDS[name]
    clipped = min(hi, max(lo, float(value)))
    return clipped, clipped != float(value)
