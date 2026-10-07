#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility shim: the implementation now lives in the :mod:`svg_tracer` package.

``python logo_trace.py ...`` and every ``import logo_trace`` keep working exactly as before.
The two knobs that the runtime rebinds (``GRAD_MIN_GAIN``, ``_QUIET``) are forwarded to
:mod:`svg_tracer.state`, so reads and writes stay live across the split.
"""
from __future__ import annotations

import sys
import types

from svg_tracer import *          # noqa: F401,F403
from svg_tracer import main       # noqa: F401


class _ShimModule(types.ModuleType):
    """Forward access to the knobs that the runtime rebinds."""

    _FORWARD = ("GRAD_MIN_GAIN", "_QUIET")

    def __getattr__(self, name):
        if name in _ShimModule._FORWARD:
            from svg_tracer import state
            return getattr(state, name)
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    def __setattr__(self, name, value):
        if getattr(self, "__dict__", None) and name in _ShimModule._FORWARD:
            from svg_tracer import state
            setattr(state, name, value)
            return
        super().__setattr__(name, value)


sys.modules[__name__].__class__ = _ShimModule


if __name__ == "__main__":
    sys.exit(main())
