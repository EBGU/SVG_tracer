"""The compatibility surface of the package split.

``logo_trace.py`` is a thin shim in front of ``svg_tracer``: the documented
``python logo_trace.py ...`` invocation, ``import logo_trace`` and the ``svg-tracer`` console
script must keep working, and the module-level knobs that the runtime rebinds must stay live
across the split instead of becoming stale copies.
"""
from __future__ import annotations

import os

import pytest

# Names that used to live in the flat logo_trace module and are still reachable through it.
LEGACY_NAMES = [
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
    import logo_trace
    import svg_tracer
    from svg_tracer import state
    return logo_trace, svg_tracer, state


def test_shim_exposes_the_historical_api(mods):
    logo_trace, svg_tracer, _ = mods
    assert os.path.basename(logo_trace.__file__) == "logo_trace.py"
    assert svg_tracer.__version__ == "1.0.0" == logo_trace.__version__
    assert logo_trace.main is svg_tracer.cli.main
    assert logo_trace.resolve_io is svg_tracer.cli.resolve_io
    for name in LEGACY_NAMES:
        assert hasattr(logo_trace, name), f"logo_trace.{name} disappeared in the split"


def test_runtime_knobs_stay_live(mods):
    """Writes through the shim must reach svg_tracer.state, not a stale copy."""
    logo_trace, _, state = mods
    keep_gain, keep_quiet = state.GRAD_MIN_GAIN, state._QUIET
    try:
        logo_trace.GRAD_MIN_GAIN = 0.42
        assert state.GRAD_MIN_GAIN == 0.42 == logo_trace.GRAD_MIN_GAIN
        logo_trace._QUIET = True
        assert state._QUIET is True and logo_trace._QUIET is True
        state.set_quiet(False)
        assert logo_trace._QUIET is False
    finally:
        state.GRAD_MIN_GAIN = keep_gain
        state.set_quiet(keep_quiet)


def test_log_is_gated_by_the_shared_quiet_flag(mods, capsys):
    logo_trace, _, state = mods
    keep = state._QUIET
    try:
        state.set_quiet(False)
        logo_trace.log("visible")
        assert "visible" in capsys.readouterr().out
        state.set_quiet(True)
        logo_trace.log("hidden")
        assert "hidden" not in capsys.readouterr().out
    finally:
        state.set_quiet(keep)


def test_shared_counters_are_the_same_objects(mods):
    _, svg_tracer, _ = mods
    assert svg_tracer._REFINE_STATS is svg_tracer.contours._REFINE_STATS
    assert svg_tracer._AA_STATS is svg_tracer.svg_out._AA_STATS
    assert svg_tracer._PAR_CTX is svg_tracer.cli._PAR_CTX


def test_thread_cap_is_installed_before_numpy():
    import svg_tracer  # noqa: F401
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        assert os.environ.get(var), f"{var} was not set by svg_tracer/__init__.py"


def test_parse_args_defaults_and_preset(mods):
    logo_trace, _, _ = mods
    ns = logo_trace.parse_args([])
    assert ns.preset is None and ns.scale == 4.0
    ns2 = logo_trace.parse_args(["--preset", "logo"])
    assert ns2.no_strokes is True and ns2.kmeans_k == 20      # the logo preset is applied
    ns3 = logo_trace.parse_args(["--preset", "painting"])
    assert ns3.kmeans_k == 48 and ns3.no_strokes is False


def test_version_flag(mods, capsys):
    logo_trace, _, _ = mods
    with pytest.raises(SystemExit) as exc:
        logo_trace.parse_args(["--version"])
    assert exc.value.code == 0
    assert "logotrace 1.0.0" in capsys.readouterr().out


def test_thread_cap_variable(mods):
    """The thread cap reads LOGO_TRACE_THREADS and defaults to 4, exactly as before the split."""
    import svg_tracer
    assert svg_tracer._THREADS == os.environ.get("LOGO_TRACE_THREADS", "4")
