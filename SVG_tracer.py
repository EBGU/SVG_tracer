#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SVG_tracer - bitmap tracing / vectorization into SVG.

This is the program entry point:

    python SVG_tracer.py --in openai.png --preset logo

The implementation lives in the :mod:`svg_tracer` package; this module only sets up the import
path for a plain source checkout and hands control to :func:`svg_tracer.cli.main`. Installing the
project also provides the equivalent ``svg-tracer`` console script.
"""
from __future__ import annotations

import os
import sys

# A plain source checkout is run as ``python SVG_tracer.py`` from the repository root, where the
# package is already importable; keeping the directory on the path makes the script work from
# anywhere (and from a symlink).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from svg_tracer.cli import main  # noqa: E402


if __name__ == "__main__":
    sys.exit(main())
