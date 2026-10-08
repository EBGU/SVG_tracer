"""SVG_tracer - bitmap tracing / vectorization (SVG) based on the "multi-scale structure tensor"

Core idea
---------
Structure tensor  J = G_rho * (∇I ∇I^T)
Multi-channel images use the Di Zenzo form: J = Σ_c G_rho * (∇I_c ∇I_c^T)

Eigendecomposition gives:
    λ1 ≥ λ2                      : energy (edge strength)
    coherence = (λ1-λ2)/(λ1+λ2)  : coherence (1=linear structure/edge, 0=corner/flat)
    principal eigenvector        : gradient direction ∇I  → the direction of the "gradient"
    minor eigenvector            : isophote direction → the "stroke" direction (flowing along the shape)

Two scales:
    fine   (σd=1.0, σi=2.5) : edges / golden seam lines / contours
    coarse (σd=2.0, σi=12 ) : smooth stroke flow field, still stable in weak-energy areas

The three-layer structure of the vector output
----------------------------------------------
1. regions : color quantization + RAG merging yields the color regions; contour → Douglas-Peucker
             simplification → Catmull-Rom converted to cubic Beziers (with overshoot clamping);
             the color field is binned along the "structure tensor gradient axis" → SVG linear
             gradient (gradient vectorization)
2. strokes : evenly spaced streamlines obtained by integrating the coarse-scale isophotes,
             filled as variable-width ribbons; width/opacity modulated by coherence; color
             sampled along the line → strokes
3. edges   : fine-scale "high energy + high coherence" ridges → skeletonization → centerline
             vector strokes (golden seam lines etc.)

Usage
-----
    python SVG_tracer.py --in inputs/openai.png --out out/openai_traced.svg \
        --preview out/openai_traced_preview.png --debug out/openai_debug.png
"""

from __future__ import annotations

import os
import sys
import types

# ---- thread cap (must come before "import numpy") ----------------------
# This script's numeric work is memory-bandwidth bound, while OpenBLAS/OpenMP on large
# shared nodes threads by core count by default, and thread oversubscription slows down
# rgb2lab / matrix operations by more than tenfold. Override with SVG_TRACER_THREADS.
_THREADS = os.environ.get("SVG_TRACER_THREADS", "4")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, _THREADS)

# Fail early with a readable message when scikit-image is missing, before any
# submodule import can raise a bare ImportError.
from . import _deps  # noqa: F401

from .cli import (  # noqa: F401
    EXAMPLES,
    PRESETS,
    _Fmt,
    _PAR_CTX,
    _n_jobs,
    _par_act_one,
    _par_detail_one,
    _par_region_one,
    _par_stroke_one,
    batch_bookkeeping,
    ensure_dir,
    main,
    parse_args,
    resolve_io,
)
from .contours import (  # noqa: F401
    _REFINE_STATS,
    bilin_arr,
    mask_to_paths,
    refine_contour,
    region_path_d,
)
from .edges import (  # noqa: F401
    _NB8,
    skeleton_paths,
)
from .geometry import (  # noqa: F401
    _chord_param,
    _clamp_ctrl,
    _corner_indices,
    _cross2,
    _eval_cubic,
    _fit_cubic_ls,
    _fit_segment,
    _unit,
    bilin,
    fit_bezier_d,
    fit_bezier_segments,
    fnum,
    poly_area,
    polyline_to_bezier_d,
    to_hex,
)
from .gradient import (  # noqa: F401
    GRAD_RADIAL_MARGIN,
    MERGE_SAMPLES,
    _fit_core,
    _grad_pred,
    _radial_centers,
    _radial_fit,
    _ramp_fit,
    _sample_mse,
    fit_region_gradient,
    merge_gradient_regions,
    predict_region,
)
from .refine import (  # noqa: F401
    _fill_stats,
    _region_interior,
    fit_subregion,
    refine_regions,
    region_fill_error,
    regroup_regions,
    split_region_blobs,
    split_region_mask,
)
from .segment import (  # noqa: F401
    _adjacent_pairs,
    _kmeans,
    classify_regions,
    edge_map,
    merge_small_regions,
    rag_merge,
    segment_colors,
    segment_edges,
    segment_flat,
    segment_hybrid,
    texture_map,
)
from .shade import (  # noqa: F401
    _shade_alpha,
    _shade_apply,
    _shade_basis,
    _shade_candidates,
    _shade_emit,
    _shade_fit_layer,
    _shade_smooth_weight,
    _shade_win,
    build_shade_stack,
)
from .state import (  # noqa: F401
    AUTOSCALE_REF,
    EPS,
    _REFINE_DEBUG,
    __version__,
    log,
)
from .strokes import (  # noqa: F401
    make_streamlines,
    ribbon_d,
    sample_color,
    trace_streamline,
)
from .svg_out import (  # noqa: F401
    _AA_STATS,
    aa_band_regions,
    aa_ramp_levels,
    build_svg,
)
from .tensor import (  # noqa: F401
    structure_tensor,
)

from . import state  # noqa: F401

__all__ = [
    "AUTOSCALE_REF",
    "EPS",
    "EXAMPLES",
    "GRAD_RADIAL_MARGIN",
    "MERGE_SAMPLES",
    "PRESETS",
    "_AA_STATS",
    "_Fmt",
    "_NB8",
    "_PAR_CTX",
    "_REFINE_DEBUG",
    "_REFINE_STATS",
    "__version__",
    "_adjacent_pairs",
    "_chord_param",
    "_clamp_ctrl",
    "_corner_indices",
    "_cross2",
    "_eval_cubic",
    "_fill_stats",
    "_fit_core",
    "_fit_cubic_ls",
    "_fit_segment",
    "_grad_pred",
    "_kmeans",
    "_n_jobs",
    "_par_act_one",
    "_par_detail_one",
    "_par_region_one",
    "_par_stroke_one",
    "_radial_centers",
    "_radial_fit",
    "_ramp_fit",
    "_region_interior",
    "_sample_mse",
    "_shade_alpha",
    "_shade_apply",
    "_shade_basis",
    "_shade_candidates",
    "_shade_emit",
    "_shade_fit_layer",
    "_shade_smooth_weight",
    "_shade_win",
    "_unit",
    "aa_band_regions",
    "aa_ramp_levels",
    "batch_bookkeeping",
    "bilin",
    "bilin_arr",
    "build_shade_stack",
    "build_svg",
    "classify_regions",
    "edge_map",
    "ensure_dir",
    "fit_bezier_d",
    "fit_bezier_segments",
    "fit_region_gradient",
    "fit_subregion",
    "fnum",
    "log",
    "main",
    "make_streamlines",
    "mask_to_paths",
    "merge_gradient_regions",
    "merge_small_regions",
    "parse_args",
    "poly_area",
    "polyline_to_bezier_d",
    "predict_region",
    "rag_merge",
    "refine_contour",
    "refine_regions",
    "region_fill_error",
    "region_path_d",
    "regroup_regions",
    "resolve_io",
    "ribbon_d",
    "sample_color",
    "segment_colors",
    "segment_edges",
    "segment_flat",
    "segment_hybrid",
    "skeleton_paths",
    "split_region_blobs",
    "split_region_mask",
    "structure_tensor",
    "texture_map",
    "to_hex",
    "trace_streamline",
]


# ----------------------------------------------------------------------
# Live view of the runtime knobs
# ----------------------------------------------------------------------
# GRAD_MIN_GAIN and _QUIET (see state.py) are rebound at runtime -- by --grad-min-gain /
# --auto-gradient and by --quiet -- while several modules import them once at import time. The
# package therefore does not expose copies of those two names: reads and writes are routed to
# state, so ``svg_tracer.GRAD_MIN_GAIN = x`` and ``state.GRAD_MIN_GAIN = x`` are the same object
# seen from two directions.
_KNOWNS = ("GRAD_MIN_GAIN", "_QUIET")


class _LiveKnobsModule(types.ModuleType):
    """Module type that forwards the two runtime knobs to :mod:`svg_tracer.state`."""

    def __getattr__(self, name):
        if name in _KNOWNS:
            return getattr(state, name)
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    def __setattr__(self, name, value):
        # `state` is always bound by the time the module class is swapped in (end of this file).
        if name in _KNOWNS:
            setattr(state, name, value)
            return
        super().__setattr__(name, value)


sys.modules[__name__].__class__ = _LiveKnobsModule
