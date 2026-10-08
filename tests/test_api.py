"""The public API surface of the :mod:`svg_tracer` package.

The implementation is split across the pipeline stages, but the package re-exports one flat
namespace, and a few module-level knobs are rebound at runtime (``GRAD_MIN_GAIN`` from
``--grad-min-gain``, ``_QUIET`` from ``--quiet``) -- those must stay live rather than becoming
stale copies in every module that imported them.
"""
from __future__ import annotations

import os

import pytest

# The names documented as the package's public surface.
PUBLIC_NAMES = [
    "main", "parse_args", "resolve_io", "ensure_dir", "EXAMPLES", "PRESETS", "log", "EPS",
    "AUTOSCALE_REF", "__version__", "structure_tensor", "classify_regions", "texture_map",
    "segment_hybrid", "segment_colors", "segment_edges", "fit_region_gradient",
    "predict_region", "merge_gradient_regions", "refine_regions", "refine_contour",
    "mask_to_paths", "region_path_d", "make_streamlines", "ribbon_d", "skeleton_paths",
    "build_svg", "build_shade_stack", "aa_band_regions", "batch_bookkeeping", "to_hex", "fnum",
    "polyline_to_bezier_d", "fit_bezier_d", "_REFINE_STATS", "_AA_STATS", "_PAR_CTX",
]


@pytest.fixture
def mods():
    import svg_tracer
    from svg_tracer import state
    return svg_tracer, state


def test_package_exposes_the_public_api(mods):
    svg_tracer, _ = mods
    assert svg_tracer.__version__ == "1.0.0"
    assert svg_tracer.main is svg_tracer.cli.main
    assert svg_tracer.resolve_io is svg_tracer.cli.resolve_io
    for name in PUBLIC_NAMES:
        assert hasattr(svg_tracer, name), f"svg_tracer.{name} is missing"


def test_runtime_knobs_stay_live(mods):
    """A rebind must be visible through the package namespace, not only through state."""
    svg_tracer, state = mods
    keep_gain, keep_quiet = state.GRAD_MIN_GAIN, state._QUIET
    try:
        svg_tracer.GRAD_MIN_GAIN = 0.42
        assert state.GRAD_MIN_GAIN == 0.42 == svg_tracer.GRAD_MIN_GAIN
        svg_tracer._QUIET = True
        assert state._QUIET is True and svg_tracer._QUIET is True
        state.set_quiet(False)
        assert svg_tracer._QUIET is False
    finally:
        state.GRAD_MIN_GAIN = keep_gain
        state.set_quiet(keep_quiet)


def test_log_is_gated_by_the_shared_quiet_flag(mods, capsys):
    svg_tracer, state = mods
    keep = state._QUIET
    try:
        state.set_quiet(False)
        svg_tracer.log("visible")
        assert "visible" in capsys.readouterr().out
        state.set_quiet(True)
        svg_tracer.log("hidden")
        assert "hidden" not in capsys.readouterr().out
    finally:
        state.set_quiet(keep)


def test_shared_counters_are_the_same_objects(mods):
    svg_tracer, _ = mods
    assert svg_tracer._REFINE_STATS is svg_tracer.contours._REFINE_STATS
    assert svg_tracer._AA_STATS is svg_tracer.svg_out._AA_STATS
    assert svg_tracer._PAR_CTX is svg_tracer.cli._PAR_CTX


def test_thread_cap_is_installed_before_numpy():
    import svg_tracer  # importing the package installs the cap
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        assert os.environ.get(var), f"{var} was not set by svg_tracer/__init__.py"
    assert svg_tracer._THREADS == os.environ.get("SVG_TRACER_THREADS", "4")


def test_parse_args_defaults_and_preset(mods):
    svg_tracer, _ = mods
    ns = svg_tracer.parse_args([])
    assert ns.preset is None and ns.scale == 4.0
    ns2 = svg_tracer.parse_args(["--preset", "logo"])
    assert ns2.no_strokes is True and ns2.kmeans_k == 20      # the logo preset is applied
    ns3 = svg_tracer.parse_args(["--preset", "painting"])
    assert ns3.kmeans_k == 48 and ns3.no_strokes is False


def test_version_flag(mods, capsys):
    svg_tracer, _ = mods
    with pytest.raises(SystemExit) as exc:
        svg_tracer.parse_args(["--version"])
    assert exc.value.code == 0
    assert "SVG_tracer 1.0.0" in capsys.readouterr().out
