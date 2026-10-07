"""Runtime state and module-level tuning knobs shared by the whole package: the quiet flag, the gradient-acceptance gate, the calibration constant, the version string and the log() helper."""
from __future__ import annotations

import os

EPS = 1e-12

# Set LOGO_TRACE_REFINE_DEBUG=1 to print every adaptive-refinement candidate and its decision
# (diagnostic only; it never changes the result).
_REFINE_DEBUG = os.environ.get("LOGO_TRACE_REFINE_DEBUG", "") == "1"

# The defaults are calibrated for this canvas size; smaller images scale the pixel-based parameters down proportionally (see main)
AUTOSCALE_REF = 1254.0

_QUIET = False


def log(msg: str = "") -> None:
    """Progress output; silent under --quiet. The final summary does not go through here and is always printed."""
    if not _QUIET:
        print(msg)

__version__ = "1.0.0"


# Accept/reject threshold for "flat color vs linear gradient": a fitted gradient must remove at
# least this fraction of the squared error. Raised/lowered at runtime from --grad-min-gain, or
# automatically by --auto-gradient on smooth (gradient-heavy) images. Module-level so every call
# site shares one value without threading an extra argument through.
GRAD_MIN_GAIN = 0.12


def set_quiet(value: bool) -> None:
    """Set the package-wide quiet flag used by log()."""
    global _QUIET
    _QUIET = bool(value)
