"""Hardware peak-throughput lookup tables."""
import os

from src.hardware.m4_max import M4_MAX
from src.hardware.rtx_5080 import RTX_5080
from src.hardware.rtx_pro_6000 import RTX_PRO_6000

TARGETS = {
    "RTX_PRO_6000": RTX_PRO_6000,
    "RTX_5080": RTX_5080,
    "M4_MAX": M4_MAX,
}

ENV_VAR = "KERNELBENCH_HARDWARE"


def get(name: str):
    if name not in TARGETS:
        raise ValueError(f"Unknown hardware {name!r}; available: {list(TARGETS)}")
    return TARGETS[name]


def select(meta: dict):
    """Resolve the hardware target for a problem run.

    Order of precedence:
      1. ``KERNELBENCH_HARDWARE`` env var, if set. Must appear in
         ``meta["hardware"]`` (so a problem can declare which targets it
         supports — running on an unsupported target is an explicit error).
      2. ``meta["hardware"][0]`` — first entry from problem.yaml.

    The env var override is the integrity-preserving way to run on a second
    SM120 SKU (e.g. RTX 5080) without auto-detection magic.
    """
    declared = list(meta.get("hardware", []))
    if not declared:
        raise ValueError("problem.yaml is missing a non-empty `hardware:` list")

    override = os.environ.get(ENV_VAR)
    if override:
        if override not in TARGETS:
            raise ValueError(
                f"{ENV_VAR}={override!r} is not a known hardware target; "
                f"available: {list(TARGETS)}"
            )
        if override not in declared:
            raise ValueError(
                f"{ENV_VAR}={override!r} is not in this problem's supported "
                f"hardware list {declared}. Either add it to problem.yaml or "
                f"unset {ENV_VAR} to use the default ({declared[0]})."
            )
        return TARGETS[override]

    return get(declared[0])
