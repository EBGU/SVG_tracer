"""``python -m svg_tracer`` support: the same entry point as ``SVG_tracer.py`` / ``svg-tracer``."""
from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
