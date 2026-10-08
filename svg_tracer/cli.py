"""Command line interface, presets, the parallel per-region stages and the end-to-end tracing pipeline."""
from __future__ import annotations

import argparse
import gzip
import math
import os
import sys
import time
import warnings

import numpy as np
from PIL import Image
from PIL import ImageDraw
from scipy import ndimage as ndi
from skimage import filters
from skimage import measure
from skimage.color import rgb2lab
from skimage.restoration import denoise_bilateral
from skimage.segmentation import find_boundaries
from svg_slim import slim_svg
from . import gpu

from .contours import _REFINE_STATS, region_path_d
from .edges import skeleton_paths
from .geometry import polyline_to_bezier_d, to_hex
from .gradient import GRAD_RADIAL_MARGIN, fit_region_gradient, merge_gradient_regions, predict_region
from .refine import refine_regions, regroup_regions
from .segment import classify_regions, merge_small_regions, segment_colors, segment_edges, segment_flat, segment_hybrid
from .shade import build_shade_stack
from .state import AUTOSCALE_REF, EPS, __version__, log
from .strokes import make_streamlines, ribbon_d, sample_color
from .svg_out import _AA_STATS, aa_band_regions, build_svg
from .tensor import structure_tensor
from . import state

EXAMPLES = """\
Examples
--------
  # Hard-edged flat artwork (logo / icon / illustration): 2x grid + subpixel localization + conformal Bezier, done in one line
  python SVG_tracer.py --in openai.png --preset logo

  # Freehand painting / photo: detail preset (many small color regions, dense fine strokes)
  python SVG_tracer.py --in water_lilies.jpg --preset painting

  # Only the geometric layer (no strokes), plus a .svgz on the side
  python SVG_tracer.py --in apple.png --preset logo --no-strokes --gzip

  # Smooth/glossy artwork: turn the gradient gate down automatically (expect banding)
  python SVG_tracer.py --in apple.png --preset logo --auto-gradient

  # Chasing high fidelity: 4x grid on a large canvas (about 20 minutes)
  python SVG_tracer.py --in openai.png --preset logo --scale 4 --compress slim --gzip

  # Self-check: run the whole pipeline on a small synthetic image and assert the quality
  python selfcheck.py

Input defaults to inputs/<name>, output defaults to out/; a bare filename is looked up under inputs/.
"""

# ======================================================================
# main pipeline
# ======================================================================
# Presets: parameter combinations for different art styles (explicit command-line arguments win over presets)
PRESETS = {
    # Hard-edged flat artwork (logo / icon / illustration): trace on an enlarged grid, subpixel boundary localization, conformal Beziers in geometric regions
    # These are exactly the defaults; the preset's extra job is to also set --scale 2.0, so that a one-line call works
    "logo": dict(
        seg="full",
        no_strokes=True,   # logo mode emits only the geometric layer by default
        kmeans_k=20, min_area=600, contour_tol=0.75, contour_smooth=1.0,
        snap_mode="refine", snap_sub="half", fit="auto", fit_tol=0.10,
        aa_levels=0, grad_stops=10, detail_chroma=12.0,
    ),
    # Freehand painting / photographic: many small regions, contours hugging the detail, dense thin strokes
    "painting": dict(
        seg="full",
        kmeans_k=48, thresh=8.0, merge_thresh2=6.0, merge_passes=4,
        min_area=180, min_width=0, contour_tol=0.5, contour_smooth=1.3,
        contour_min_area=12.0, grad_stops=12, grad_min_range=0.003,
        detail_chroma=0.0, spacing=7.0, stroke_width=4.0, stroke_alpha=0.5,
        stroke_nodes=10, stroke_min=14.0, stroke_max=90.0, coh_min=0.10,
        stroke_activity_ref=0.02, stroke_alpha_min=0.02, aa_levels=0,
        compress="tight",   # watercolor-like images barely lose quality (measured -0.01~-0.11 dB) while saving about 1/3 of the volume
    ),
}


class _Fmt(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """Show the defaults while keeping the example line breaks from the epilog."""


# ---- parallelism for the per-region stage: fork + copy-on-write (large arrays are neither copied nor pickled) ----
_PAR_CTX = {}


def _n_jobs(args, n_items, min_items=24):
    """How many processes to use for the per-region stage.

    0/negative = auto: min(cores, 16); with too few candidate regions (default <24) it is fixed at 1
    -- so that the fork and pool startup cost does not eat the gain, and so that small images keep
    behaving exactly as they always have. Without fork (non-POSIX) it uses 1.
    """
    import multiprocessing
    if "fork" not in multiprocessing.get_all_start_methods():
        return 1
    want = int(getattr(args, "jobs", 0) or 0)
    if want <= 0:
        want = min(os.cpu_count() or 1, 16)
    if int(n_items) < int(min_items):
        return 1
    return max(1, min(want, int(n_items)))


def batch_bookkeeping(labels, rgb_s, use_gpu=True):
    """Compute the two rule-dense sub-steps of the per-region stage over the whole image in one batch; the result is bit-identical to the per-region serial version.

    1) inner: the original does binary_erosion(labels[win]==li, iterations=3) per region. The default
       structuring element is a 3x3 cross, so three iterations = a diamond (L1 radius 3)
       neighbourhood; hence the whole-image equivalent is "all labels in this pixel's diamond
       neighbourhood are identical", obtained with min/max over a radius-3 diamond footprint. The
       border is treated as 0 (erosion's border_value): using cval=-1 / cval=nl makes border pixels
       fail both the min and the max test, which is exactly equivalent.
    2) cols: per-region median color. When pixel values lie on the 1/255 grid the median is a rank
       statistic: odd counts take the middle rank, even counts the average of the two middle ranks;
       with a 256-bucket histogram + cumsum it can be computed in O(1) per label, averaging in
       numpy's operation order ((b_lo/255 + b_hi/255)/2), hence bit-identical. When the image has
       been resampled (scale != 1) the pixels are not on the 1/255 grid and np.median cannot be
       reproduced exactly from a histogram -- fall back.

    Returns (inner_all, cols, backend); when unavailable (None, None, "").
    """
    q = rgb_s * 255.0
    if not np.array_equal(q, np.rint(q)):
        return None, None, ""
    nl = int(labels.max()) + 1
    # With too few labels, whole-image filtering costs more than "each region computing just its own patch", so skip batching entirely.
    if nl > 60000 or nl < 256:
        return None, None, ""
    yy, xx = np.mgrid[-3:4, -3:4]
    dia = (np.abs(yy) + np.abs(xx)) <= 3          # 3 iterations of cross erosion == this diamond
    qq = np.rint(q).astype(np.int32)

    # GPU batching only pays off when "very large + many labels": measured at 1580 labels/1.1Mpx the CPU batch takes 0.4s,
    # while the cupy path takes 3.4s (its first run also pays about 18s for kernel compilation, cached on disk afterwards). So the threshold is set very high.
    if use_gpu and nl >= 4000 and labels.size >= 8000000:
        try:
            import cupy as cp
            from cupyx.scipy import ndimage as gndi
            gl = cp.asarray(labels)
            gdia = cp.asarray(dia)
            gmn = gndi.minimum_filter(gl, footprint=gdia, mode="constant", cval=-1)
            gmx = gndi.maximum_filter(gl, footprint=gdia, mode="constant", cval=nl)
            ginner = (gl == gmn) & (gl == gmx)
            gq = cp.asarray(qq)
            glf = gl.ravel().astype(cp.int64)
            gcin = cp.bincount(gl[ginner].ravel(), minlength=nl)
            gtot = cp.bincount(gl.ravel(), minlength=nl)
            gcols = cp.zeros((nl, 3), dtype=cp.float64)
            for c in range(3):
                qc = gq[..., c].ravel().astype(cp.int64)
                ha = cp.bincount(glf * 256 + qc, minlength=nl * 256).reshape(nl, 256)
                hb = cp.bincount(glf[ginner.ravel()] * 256 + qc[ginner.ravel()],
                                 minlength=nl * 256).reshape(nl, 256)
                use = gcin >= 200
                h = cp.where(use[:, None], hb, ha)
                tot = cp.where(use, gcin, gtot).astype(cp.int64)
                cs = cp.cumsum(h, axis=1)
                bl = (cs <= ((tot - 1) // 2)[:, None]).sum(1).astype(cp.float64)
                bh = (cs <= (tot // 2)[:, None]).sum(1).astype(cp.float64)
                gcols[:, c] = (bl / 255.0 + bh / 255.0) / 2.0
            out = (cp.asnumpy(ginner), cp.asnumpy(gcols), "GPU")
            cp._default_memory_pool.free_all_blocks()
            return out
        except Exception:
            pass

    mn = ndi.minimum_filter(labels, footprint=dia, mode="constant", cval=-1)
    mx = ndi.maximum_filter(labels, footprint=dia, mode="constant", cval=nl)
    inner_all = (labels == mn) & (labels == mx)
    lf = labels.ravel()
    lfi = lf.astype(np.int64)
    cin = np.bincount(labels[inner_all].ravel(), minlength=nl)
    tot_all = np.bincount(lf, minlength=nl)
    keep = inner_all.ravel()
    cols = np.zeros((nl, 3))
    for c in range(3):
        qc = qq[..., c].ravel()
        ha = np.bincount(lfi * 256 + qc, minlength=nl * 256).reshape(nl, 256)
        hb = np.bincount(lfi[keep] * 256 + qc[keep], minlength=nl * 256).reshape(nl, 256)
        use = cin >= 200
        h = np.where(use[:, None], hb, ha)
        tot = np.where(use, cin, tot_all).astype(np.int64)
        cs = np.cumsum(h, axis=1)
        bl = (cs <= ((tot - 1) // 2)[:, None]).sum(1).astype(np.float64)
        bh = (cs <= (tot // 2)[:, None]).sum(1).astype(np.float64)
        cols[:, c] = (bl / 255.0 + bh / 255.0) / 2.0
    return inner_all, cols, "CPU"


def _par_region_one(li):
    """Parallel unit: process one region (bit-identical to the serial version).

    Region masks do not overlap and no rng is used anywhere, so the units share no state; the parent
    writes regions back in the original order, so the output bytes match the serial version. Child
    processes inherit the large arrays in `_PAR_CTX` directly through fork.
    """
    P = _PAR_CTX
    args, labels, areas = P["args"], P["labels"], P["areas"]
    rgb_s, tc, tf = P["rgb_s"], P["tc"], P["tf"]
    thr_ref, bg_label, geo_r = P["thr_ref"], P["bg_label"], P["geo_r"]
    H, W, _MG, objs = P["H"], P["W"], P["MG"], P["objs"]
    _REFINE_STATS["pts"] = 0
    _REFINE_STATS["moved"] = 0
    _REFINE_STATS["sum"] = 0.0
    sl = objs[li] if 0 <= li < len(objs) else None
    if sl is None:
        win, y0, x0 = (slice(0, 0), slice(0, 0)), 0, 0
    else:
        y0 = max(0, sl[0].start - _MG)
        y1 = min(H, sl[0].stop + _MG)
        x0 = max(0, sl[1].start - _MG)
        x1 = min(W, sl[1].stop + _MG)
        win = (slice(y0, y1), slice(x0, x1))
    mask = labels[win] == li
    if int(areas[li]) < args.min_area:      # equals mask.sum(), but in O(1)
        return None, "", dict(_REFINE_STATS)
    _inb = P.get("inner_all")
    if _inb is not None:
        inner = _inb[win] & mask          # the whole-image batch result must be restricted back to this region
        if int(inner.sum()) < 200:
            inner = mask
        col = P["cols"][li]
    else:
        inner = ndi.binary_erosion(mask, iterations=3)
        if inner.sum() < 200:
            inner = mask
        col = np.median(rgb_s[win][inner], axis=0)
    if li == bg_label:
        return ({"idx": li, "mask": mask, "area": int(areas[li]), "d": "",
                 "fill": to_hex(col), "grad": None, "bg": True,
                 "geo": float(geo_r[li]), "win": win, "off": (y0, x0), "col": col},
                "", dict(_REFINE_STATS))
    m = ndi.binary_dilation(mask, iterations=1)
    geo_frac = float(geo_r[li])
    use_fit = args.fit
    if use_fit == "auto":
        use_fit = "bezier" if geo_frac >= args.auto_geo_frac else "cr"
    rf = None
    if args.snap_mode == "refine" and geo_frac >= args.auto_geo_frac:
        rf = (tf, thr_ref, args.snap_shift, args.snap_step, args.snap_sub)
    d = region_path_d(m, args.contour_tol, args.contour_smooth, args.contour_min_area,
                      fit=use_fit, fit_tol=args.fit_tol, corner_deg=args.corner_deg,
                      refine=rf, offset=(y0, x0))
    if not d:
        return None, "", dict(_REFINE_STATS)
    grad = fit_region_gradient(rgb_s, inner, tc, args.grad_stops,
                               args.grad_min_range, off=(y0, x0),
                               allow_radial=args.grad_radial,
                               radial_margin=args.grad_radial_margin)
    msg = ""
    if args.verbose:
        q = f"{grad['quality']:.2f}({grad['axis']})" if grad else "flat"
        msg = (f"      · region #{li:<2d} area={int(areas[li]):>7d} {to_hex(col)} "
               f"gradient quality={q} fit={use_fit}(geometricity {geo_frac:.2f})")
    return ({"idx": li, "mask": mask, "area": int(areas[li]), "d": d,
             "fill": to_hex(col), "grad": grad, "bg": False,
             "geo": geo_frac, "fit": use_fit, "win": win, "off": (y0, x0), "col": col},
            msg, dict(_REFINE_STATS))


def _par_act_one(i):
    """Parallel unit: stroke-activity residual for region i.

    Instead of writing into a shared array it hands (rows, cols, vals) back to the parent to be
    written in the original order -- regions do not overlap, so the write-back order does not affect
    the result, making it bit-identical to the serial version.
    """
    P = _PAR_CTX
    r = P["regions"][i]
    if r["bg"] or r.get("aa"):
        return None
    rgb_s = P["rgb_s"]
    inner = ndi.binary_erosion(r["mask"], iterations=3)
    if inner.sum() < 50:
        return None
    rows, cols_, pred = predict_region(P["shape"], inner, r["grad"], r.get("off", (0, 0)))
    obs = rgb_s[rows, cols_]
    if pred is None:
        pred = np.median(obs, axis=0)[None, :]
    return rows, cols_, np.abs(obs - pred).mean(1)


def _par_detail_one(k):
    """Parallel unit: one colored thin seam line (no rng, bit-identical to the serial version).

    Computed only inside the bounding-box window given by ``find_objects(lb + 1)``. It used to start
    with ``mk = P["lb"] == k``, building an (H,W) boolean array over the **whole image**, after which
    erosion / dilation / color sampling each scanned the whole image again; microbenchmark (wave's
    real size 3859x2594) 364.7ms per component versus 1.053ms for the windowed version -- **346x**.
    """
    P = _PAR_CTX
    args, rgb_s = P["args"], P["rgb_s"]
    tc, tf, thr_ref = P["tc"], P["tf"], P["thr_ref"]
    H, W, MG, objs = P["H"], P["W"], P["MG"], P["objs"]
    sl = objs[k] if 0 <= k < len(objs) else None
    if sl is None:
        win, y0, x0 = (slice(0, 0), slice(0, 0)), 0, 0
    else:
        y0 = max(0, sl[0].start - MG)
        y1 = min(H, sl[0].stop + MG)
        x0 = max(0, sl[1].start - MG)
        x1 = min(W, sl[1].stop + MG)
        win = (slice(y0, y1), slice(x0, x1))
    mk = P["lb"][win] == k
    inner = ndi.binary_erosion(mk, iterations=2)
    if inner.sum() < 20:
        inner = mk
    col = np.median(rgb_s[win][inner], axis=0)
    _t_rp = time.time()
    d = region_path_d(ndi.binary_dilation(mk, iterations=1),
                      args.contour_tol * 0.5,
                      args.contour_smooth * 0.6, args.contour_min_area,
                      fit=("bezier" if args.fit == "auto" else args.fit),
                      fit_tol=args.fit_tol,
                      corner_deg=args.corner_deg,
                      refine=((tf, thr_ref, args.snap_shift, args.snap_step)
                              if args.snap_mode == "refine" else None),
                      offset=(y0, x0))
    if not d:
        return None, ""
    grad = fit_region_gradient(rgb_s, inner, tc, args.grad_stops,
                               args.grad_min_range, off=(y0, x0),
                               allow_radial=args.grad_radial,
                               radial_margin=args.grad_radial_margin)
    # The long tail of the detail layer is "the contour DP of a few very large components": there are very few components, so a line is always reported here
    # (including the component area and the region_path_d time), so the next round can pinpoint which component and how many seconds.
    _dt = time.time() - _t_rp
    q = f"{grad['quality']:.2f}" if grad else "flat"
    msg = (f"      · seam #{k} area={int(mk.sum()):>7d} window={mk.shape[0]}x{mk.shape[1]} "
           f"contour={_dt:.1f}s {to_hex(col)} gradient={q}")
    if args.verbose:
        msg += f" (candidate component #{k})"
    return ({"idx": f"d{k}", "mask": mk, "area": int(mk.sum()),
             "d": d, "fill": to_hex(col), "grad": grad,
             "bg": False, "detail": True, "win": win, "off": (y0, x0)}, msg)


def _par_stroke_one(i):
    """Parallel unit: strokes for one region (streamline tracing + color sampling).

    In parallel mode each region uses a "deterministic derived seed", independent of scheduling
    order and process count → reproducible for the same arguments and process count. The cost: the
    random jitter grid differs from the serial version (shared rng), so --jobs 1 is the setting that
    is byte-identical to out/.
    """
    P = _PAR_CTX
    args, tc, fg = P["args"], P["tc"], P["fg"]
    act, rgb_s = P["act"], P["rgb_s"]
    H, W = P["H"], P["W"]
    r = P["sregs"][i]
    rng = np.random.default_rng((int(P["seed"]) * 1000003 + int(i) + 1) & 0x7FFFFFFF)
    win = r.get("win")
    if win is None:
        sub = r["mask"] & fg
    else:
        sub_w = r["mask"] & fg[win]
        if sub_w.sum() < 400:
            return None
        sub = np.zeros((H, W), bool)
        sub[win] = sub_w
    if sub.sum() < 400:
        return None
    segs = make_streamlines(tc, sub, args, rng)
    if not segs:
        return None
    items = []
    for pts, wid, _c in segs:
        iy = np.clip(pts[:, 1].astype(int), 0, H - 1)
        ix = np.clip(pts[:, 0].astype(int), 0, W - 1)
        a = float(act[iy, ix].mean())
        alpha = args.stroke_alpha * min(1.0, a / (args.stroke_activity_ref + EPS))
        if alpha < args.stroke_alpha_min:
            continue  # the base fill is accurate enough -> no stroke (avoids creating artifacts in flat areas)
        col = sample_color(rgb_s, pts, args.stroke_jitter, rng)
        items.append((ribbon_d(pts, wid), col, min(alpha, 0.95)))
    return {"idx": r["idx"], "d": r["d"], "area": r["area"], "strokes": items}


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="SVG_tracer.py",
        description="Bitmap tracing / vectorization tool based on the multi-scale structure tensor"
                    " (structure tensor → partition routing → SVG)",
        epilog=EXAMPLES, formatter_class=_Fmt)
    p.add_argument("--version", action="version", version=f"SVG_tracer {__version__}")

    g = p.add_argument_group("input / output")
    g.add_argument("--in", dest="src", default="inputs/openai.png",
                   help="input bitmap; a bare filename is looked up under inputs/")
    g.add_argument("--out", default=None,
                   help="output SVG; defaults to out/<input name>_traced.svg")
    g.add_argument("--preview", nargs="?", const="auto", default=None,
                   help="preview PNG; without a value = auto-named under out/")
    g.add_argument("--no-preview", dest="preview", action="store_const", const="",
                   help="do not render a preview")
    g.add_argument("--debug", nargs="?", const="auto", default=None,
                   help="structure tensor debug image; off by default, without a value = auto-named under out/")
    g.add_argument("--gzip", action="store_true",
                   help="also write .svgz (gzip -9; usable directly by browsers / <img>)")
    g.add_argument("--compress", choices=["off", "slim", "tight"], default="slim",
                   help="SVG slimming: off=as is; slim=precision trimming (lossless per digit, can still be gzipped); "
                        "tight=additionally drop duplicate clipPath (smallest, but strokes spill outside their color region: "
                        "measured -1.9 dB at logo 2x / -5.7 dB at 4x, watercolor -0.01 dB; "
                        "**--preset painting defaults to tight**). "
                        "To go even smaller on hard-edged artwork use svgzip.py --prec 0 (nearly lossless, another ~1/3 off)")

    g = p.add_argument_group("run")
    g.add_argument("--scale", type=float, default=4.0,
                   help="Lanczos upscale factor before tracing (hard-edged artwork gains clearly from 2.0 up)")
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--no-autoscale", dest="no_autoscale", action="store_true",
                   help=f"turn off parameter auto-scaling for small images (on by default: for a canvas < {AUTOSCALE_REF:.0f}px it scales "
                        f"absolute pixel parameters such as min_area/spacing by k=max(W,H)/{AUTOSCALE_REF:.0f})")
    g.add_argument("--verbose", dest="verbose", action="store_true", default=True,
                   help=argparse.SUPPRESS)
    g.add_argument("--quiet", dest="verbose", action="store_false",
                   help="print only the final summary")

    g = p.add_argument_group("presets")
    g.add_argument("--preset", choices=sorted(PRESETS), default=None,
                   help="parameter preset (explicit command-line arguments win): "
                        "logo=hard-edged flat artwork (2x grid + subpixel localization + conformal Bezier); "
                        "painting=freehand/photo (many small color regions, dense fine strokes)")

    g = p.add_argument_group("structure tensor")
    g.add_argument("--sigma-d", type=float, default=1.0, help="fine scale differentiation scale σd")
    g.add_argument("--sigma-i", type=float, default=2.5, help="fine scale integration scale σi")
    g.add_argument("--cs-sigma-d", type=float, default=2.0, help="coarse scale σd (stroke flow field)")
    g.add_argument("--cs-sigma-i", type=float, default=12.0, help="coarse scale σi (stroke flow field)")

    g = p.add_argument_group("segmentation")
    g.add_argument("--method", choices=["colors", "edges", "hybrid", "watershed"],
                   default="hybrid",
                   help="hybrid=partition routing (default): color segmentation fixes the region identity, solid-color geometric regions "
                        "snap their boundary subpixel onto edges and use conformal Beziers, stroke/gradient complex regions keep the original approach; "
                        "colors=pure color segmentation (Catmull-Rom); edges=pure edge-detection watershed; "
                        "watershed=energy watershed")
    g.add_argument("--labels-cache", default="",
                   help="segmentation cache file (.npy): load it when present, otherwise save after computing; empty=off")
    g.add_argument("--gpu", choices=["off", "auto", "on"], default="auto",
                   help="whether the structure tensor + k-means go on the GPU (default auto). auto=use it when available (cupy or the in-house "
                        "CUDA backend cuda_backend.py), otherwise silently fall back to CPU; off=CPU only; "
                        "on=GPU required. Measured: tensor stage 1.7~2.6x, end to end the float32 "
                        "tensor costs 37.54 dB (CPU 37.57), and the gain is larger on big images")
    g.add_argument("--jobs", type=int, default=0,
                   help="number of parallel processes for the per-region stage (fork + copy-on-write, large arrays are neither copied nor pickled)."
                        "0=auto: min(cores,16), 1 when there are fewer than 24 candidate regions; 1=parallelism off."
                        "The stages without rng (region geometry/activity/detail layer) produce byte-identical output when parallel; "
                        "the stroke layer switches to a per-region deterministic seed, so --jobs 1 is the setting that matches "
                        "out/ byte for byte")
    g.add_argument("--batch-book", choices=("auto", "off"), default="auto",
                   help="turn the rule-dense sub-steps of the per-region stage (mask erosion / per-region median color) into one whole-image batch: "
                        "auto=compute it when possible (cupy preferred on the GPU), off=keep the per-region loop. The batch version is "
                        "byte-identical to the per-region version (the median has a closed form on the 1/255 grid); when the image was resampled "
                        "(scale != 1) the pixels are not on that grid and it falls back to per-region automatically.")
    g.add_argument("--seg", choices=["full", "auto", "flat"], default="full",
                   help="segmentation route. full=the complete route (default, k-means+RAG); auto/flat=flat-color fast path "
                        "(histogram dominant colors + LUT + connected components). Measured: a hard-edged flat-color icon 27.57→27.37 dB while segmentation "
                        "drops 9.3→4.1s; a logo with gradients 37.57→34.7 dB -- so it is off by default and only recommended"
                        "when explicitly requested for purely hard-edged flat artwork")
    g.add_argument("--flat-cov", type=float, default=0.90,
                   help="fast-path criterion: pixel share the dominant-color buckets must cover, below which the full route is taken")
    g.add_argument("--flat-nb", type=int, default=5,
                   help="fast-path histogram bits per channel (5 → 32 levels/channel)")
    g.add_argument("--flat-share", type=float, default=2e-4,
                   help="fast path: minimum pixel share for a bucket to count as a dominant color")
    g.add_argument("--flat-merge", type=float, default=0.0,
                   help="fast path: color distance threshold for palette merging (0-1)")
    g.add_argument("--edge-nms-hi", type=float, default=0.97,
                   help="edges method: hysteresis high threshold after non-maximum suppression (energy quantile)")
    g.add_argument("--edge-nms-lo", type=float, default=0.92,
                   help="edges method: hysteresis low threshold (energy quantile)")
    g.add_argument("--edge-core", type=float, default=3.0,
                   help="edges method: distance to the edge required of a flat-region seed (px)")
    g.add_argument("--edge-close", type=int, default=0,
                   help="edges method: edge closing iterations (fills 1~2px gaps)")
    g.add_argument("--edge-dsmooth", type=float, default=0.6,
                   help="edges method: distance-field smoothing σ")
    g.add_argument("--kmeans-k", type=int, default=20)
    g.add_argument("--kmeans-iters", type=int, default=16)
    g.add_argument("--pre-smooth", type=float, default=1.6, help="Gaussian smoothing σ before quantization")
    g.add_argument("--thresh", type=float, default=12.0, help="RAG merging threshold (0-255 color distance)")
    g.add_argument("--merge-thresh2", type=float, default=10.0,
                   help="threshold for another RAG merging pass at the connected-component level (0=off)")
    g.add_argument("--detail-chroma", type=float, default=12.0,
                   help="Lab chroma threshold of the colored thin seam-line detail layer (0=off)")
    g.add_argument("--detail-scale", type=float, default=1.0,
                   help="chroma map Gaussian smoothing σ")
    g.add_argument("--detail-min-area", type=int, default=40)
    g.add_argument("--protect-sat", type=float, default=0.0,
                   help="colored regions whose average saturation (rgb range) exceeds this value take no part in merging/absorption "
                        "(0=off; golden seam lines are already handled separately by the detail layer)")
    g.add_argument("--merge-passes", type=int, default=4)
    g.add_argument("--min-area", type=int, default=600)
    g.add_argument("--min-width", type=int, default=0,
                   help="label opening kernel width (px), 0=off; it shrinks genuinely thin long features, use with care")
    g.add_argument("--denoise", type=float, default=0.0,
                   help="edge-preserving smoothing sigma_color (0=off; 0.03~0.05 is recommended for noisy input)")
    g.add_argument("--contour-tol", type=float, default=0.75)
    g.add_argument("--contour-smooth", type=float, default=1.0)
    g.add_argument("--fit", choices=["auto", "cr", "bezier"], default="auto",
                   help="curve fitting: auto=choose from the region's geometricity (bezier for geometric regions / cr for complex ones); "
                        "cr=Douglas-Peucker+Catmull-Rom; "
                        "bezier=conformal optimal Bezier (emits l for straight lines, preserves sharp corners)")
    g.add_argument("--fit-tol", type=float, default=0.10,
                   help="maximum allowed error of the bezier fit (px)")
    g.add_argument("--corner-deg", type=float, default=62.0,
                   help="corner detection angle of the bezier fit")
    g.add_argument("--auto-geo-frac", type=float, default=0.5,
                   help="geometricity threshold of --fit auto (share of geometric pixels in the region)")
    g.add_argument("--tex-sigma", type=float, default=2.5,
                   help="texture map high-pass scale σ")
    g.add_argument("--tex-smooth", type=float, default=6.0,
                   help="texture map energy smoothing σ")
    g.add_argument("--tex-norm", type=float, default=99.5,
                   help="texture normalization quantile")
    g.add_argument("--tex-thr", type=float, default=3.0,
                   help="texture ceiling for classifying a region as geometric (after normalization)")
    g.add_argument("--range-thr", type=float, default=0.02,
                   help="interior color spread ceiling for classifying a region as geometric (0.02≈5/255)")
    g.add_argument("--tex-close", type=int, default=0)
    g.add_argument("--tex-open", type=int, default=0)
    g.add_argument("--snap-w", type=float, default=3.0,
                   help="hybrid method: bonus weight of edge lines in the elevation (the larger, the closer to the edge)")
    g.add_argument("--snap-sigma", type=float, default=1.0,
                   help="hybrid method: smoothing σ applied before the edge-line bonus")
    g.add_argument("--snap-erode", type=int, default=2,
                   help="hybrid method: erosion radius of the color-region seeds (leaves room for the boundary to snap)")
    g.add_argument("--snap-band", type=float, default=3.0,
                   help="hybrid method: how many pixels a boundary may move at most (0=unlimited)")
    g.add_argument("--snap-mode", choices=["refine", "watershed", "off"],
                   default="refine",
                   help="snapping mode: refine=snap contour points subpixel along the normal onto the edge energy ridge "
                        "(local, topology-preserving, default); watershed=marker-controlled watershed re-cuts the boundary; off=no snapping")
    g.add_argument("--snap-shift", type=float, default=1.5,
                   help="refine mode: maximum snapping displacement per point (px)")
    g.add_argument("--snap-sub", choices=["half", "peak"], default="half",
                   help="subpixel localization: half=50%% crossing of the luma profile (more accurate), peak=energy peak")
    g.add_argument("--snap-step", type=float, default=0.15,
                   help="refine mode: sampling step of the normal profile (px)")
    g.add_argument("--aa-levels", type=int, default=0,
                   help="number of levels for anti-aliasing transition-band reconstruction (how many per side; 0=off). The source boundary has a 2~3px"
                        "anti-aliasing transition and nine tenths of the squared error sits in that band; but measured, once the boundary is placed accurately, "
                        "approximating the ramp with same-color steps slightly lowers PSNR (see README), hence off by default")
    g.add_argument("--aa-width", type=float, default=2.5,
                   help="total width of the anti-aliasing transition band (px), should be ≈ the source boundary transition width")
    g.add_argument("--aa-smooth", type=float, default=0.8,
                   help="distance-field smoothing σ (smaller hugs the original boundary)")
    g.add_argument("--aa-tol", type=float, default=0.15,
                   help="Bezier fitting tolerance of the transition-band contour (px)")
    g.add_argument("--aa-geo-thr", type=float, default=-1.0,
                   help="geometricity threshold that admits a region to the transition band (<0 uses --auto-geo-frac)")
    g.add_argument("--aa-min-radius", type=float, default=1.4,
                   help="no transition band when the region's maximum inscribed radius is below this value (avoids the two offsets crossing)")
    g.add_argument("--aa-profile", choices=["shape", "uniform"], default="shape",
                   help="transition-band grading: shape=optimal quantization from the measured transition profile, uniform=equal width")
    g.add_argument("--aa-min-area", type=int, default=8,
                   help="minimum pixel count of one transition-band segment")
    g.add_argument("--aa-synthetic", action=argparse.BooleanOptionalAction, default=False,
                   help="also build anti-aliasing transition bands on **synthetic boundaries**. A synthetic boundary = the divide between two child"
                        "regions that adaptive refinement cut out of one and the same original region (the colors are continuous by construction, so the band is essentially wasted bytes); "
                        "background boundaries / other original-region boundaries / real image edges are unaffected and always keep their band. "
                        "Off by default (=saves the synthetic-boundary bands); --aa-synthetic restores the old behaviour (byte for byte)")
    g.add_argument("--aa-edge-dedup", action=argparse.BooleanOptionalAction, default=True,
                   help="carry the origin of **one and the same original region** all the way down every level of the refinement tree and dedup with it: after the refinement recursion, "
                        "every divide between descendants of the same original region (even when they are not direct parent/child but uncle/nephew or cousins) counts as a synthetic boundary, "
                        "and only one band is kept. When off it only knows one level of parent/child (old behaviour, byte for byte). "
                        "Bands on real image edges are unaffected (apple@4: 1223→1043 KB, PSNR/MAE/JUMP unchanged)")
    g.add_argument("--aa-edge-canon", action=argparse.BooleanOptionalAction, default=False,
                   help="[experimental, off by default] more aggressive dedup: within one group of descendants of the same original region, only"
                        "the largest descendant may build transition bands on a **real image edge**, the other descendants keep only the bands on synthetic boundaries. "
                        "Measured to change the bands on real edges (each descendant is responsible for its own stretch of the edge), so it is not recommended")
    g.add_argument("--contour-min-area", type=float, default=30.0)

    g = p.add_argument_group("gradients")
    g.add_argument("--grad-stops", type=int, default=10)
    g.add_argument("--grad-min-gain", type=float, default=0.12,
                   help="gradient acceptance gate: a linear gradient must remove at least this share of the squared error, otherwise the region falls back to flat color. "
                        "Smooth gradient images (e.g. soft logos / renders) can go down to about 0.02, which markedly reduces color steps")
    g.add_argument("--auto-gradient", action="store_true",
                   help="relax the gradient gate automatically from the image smoothness (smooth images get a lower gate), no manual tuning needed")
    g.add_argument("--grad-min-range", type=float, default=0.004,
                   help="lower bound on the color spread that enables a linear gradient (0.004≈1/255; this image's gradients are extremely shallow)")
    g.add_argument("--grad-radial", action=argparse.BooleanOptionalAction, default=True,
                   help="allow radial-gradient (center+radius) candidates: they use the same error-gain criterion as linear, and are adopted only when the radial residual"
                        "is smaller than the best linear one by --grad-radial-margin (turn off with --no-grad-radial)")
    g.add_argument("--grad-radial-margin", type=float, default=GRAD_RADIAL_MARGIN,
                   help="residual-advantage threshold for adopting radial: the radial residual must be at least this ratio smaller than the linear one")
    g.add_argument("--merge-grad", action=argparse.BooleanOptionalAction, default=True,
                   help="gradient-aware region merging: when the union of adjacent color regions can still be explained by a single gradient (linear/radial), merge it into"
                        "a larger, smoother region and remove the color steps (turn off with --no-merge-grad)")
    g.add_argument("--merge-grad-tol", type=float, default=0.05,
                   help="merge criterion: single-gradient residual of the union <= the sum of the two independent fit residuals ×(1+this value)")
    g.add_argument("--merge-grad-passes", type=int, default=4,
                   help="number of iterations of gradient-aware merging (fixed scan order per pass, deterministic result)")

    g = p.add_argument_group("adaptive refinement (error-driven)")
    g.add_argument("--adaptive-refine", action=argparse.BooleanOptionalAction, default=True,
                   help="error-driven adaptive refinement: measure the residual of every region's current fill (flat/linear/radial), and when the residual exceeds the threshold"
                        "bisect along the direction of largest error and re-fit recursively, spending the fixed SVG element budget where it is needed most, "
                        "which removes the visible flat patches on smooth/glossy images (--no-adaptive-refine restores the old behaviour; combined with"
                        " --no-grad-radial --no-merge-grad it reproduces the legacy output byte for byte)")
    g.add_argument("--refine-err", type=float, default=0.02,
                   help="refinement residual threshold (full-scale share, 0.02≈5/255): a bisection is only considered when the number of pixels in the region interior (eroded core) whose max-channel"
                        "error exceeds this value reaches max(16, max(32, refine-min-area/4)/2) or more (i.e. a truly visible patch of error exists); the local criterion is essential, because a small patch of error on a large region barely"
                        "raises the global RMS")
    g.add_argument("--refine-min-area", type=int, default=400,
                   help="minimum region area that takes part in refinement; smaller regions are left as they are (prevents chasing noise/thin seams)")
    g.add_argument("--refine-max-depth", type=int, default=6,
                   help="maximum recursion depth of refinement (0=one round, no recursion)")
    g.add_argument("--refine-gain", type=float, default=0.20,
                   help="minimum residual drop required to accept a bisection (area-weighted RMS of the children <= (1-this value)×the parent); "
                        "this is a natural protection against noise/texture and also the valve on how fast it grows")
    g.add_argument("--refine-budget", type=int, default=0,
                   help="cap on the number of regions refinement may add; 0=auto (max(64, min(256, region count)))")
    g.add_argument("--refine-order", choices=("merge-first", "refine-first"), default="merge-first",
                   help="order of refinement and gradient-aware merging: merge-first=merge then refine (default); "
                        "refine-first=refine the original partition first, then gradient-merge the refined partition")
    g.add_argument("--dump-refined-labels", default="",
                   help="write the refined label map to a .npy (diagnostics / reproducing the refine-first order); empty=do not write")

    g = p.add_argument_group("strokes (structure-tensor streamlines)")
    g.add_argument("--no-strokes", action="store_true",
                   help="emit only the geometric layer; --preset logo enables this by default")
    g.add_argument("--spacing", type=float, default=10.0, help="stroke spacing")
    g.add_argument("--occ-ratio", type=float, default=0.45, help="occupancy radius / spacing")
    g.add_argument("--stroke-step", type=float, default=3.0)
    g.add_argument("--stroke-max", type=float, default=120.0)
    g.add_argument("--stroke-min", type=float, default=24.0)
    g.add_argument("--stroke-nodes", type=int, default=12)
    g.add_argument("--stroke-width", type=float, default=7.5)
    g.add_argument("--stroke-grid-units", action="store_true",
                   help="interpret stroke length parameters as tracing-grid px (old behaviour); by default they are native pixels and are multiplied by --scale")
    g.add_argument("--stroke-alpha", type=float, default=0.6)
    g.add_argument("--stroke-jitter", type=float, default=0.012)
    g.add_argument("--stroke-activity-ref", type=float, default=0.012,
                   help="stroke opacity=stroke-alpha×min(1,residual/this value); the residual comes from 'image - base fill'")
    g.add_argument("--stroke-alpha-min", type=float, default=0.03,
                   help="strokes below this opacity are dropped")
    g.add_argument("--coh-min", type=float, default=0.12)
    g.add_argument("--max-turn", type=float, default=70.0, help="maximum turning angle per step (degrees)")

    g = p.add_argument_group("edges / seam lines (this image's golden seam lines are already rebuilt by the color-region layer, hence off by default)")
    g.add_argument("--edge-mode", choices=["off", "chroma", "energy", "both"], default="off",
                   help="off=do not draw; chroma=Lab chroma thin lines; energy=structure-tensor energy ridges")
    g.add_argument("--edge-chroma", type=float, default=9.0, help="Lab chroma threshold (chroma mode)")
    g.add_argument("--edge-hi", type=float, default=0.975, help="high energy quantile (energy mode)")
    g.add_argument("--edge-lo", type=float, default=0.90, help="hysteresis low threshold quantile (energy mode)")
    g.add_argument("--edge-coh", type=float, default=0.45, help="coherence lower bound (energy mode)")
    g.add_argument("--edge-width", type=float, default=2.0)
    g.add_argument("--edge-alpha", type=float, default=0.9)
    g.add_argument("--edge-min-area", type=int, default=24)
    g.add_argument("--edge-min-len", type=int, default=8)
    g.add_argument("--edge-tol", type=float, default=0.6, help="skeleton path simplification tolerance")

    g = p.add_argument_group("shade layers (stacked translucent radial gradients / gradient boosting; experimental, off by default)")
    g.add_argument("--shade-blobs", action=argparse.BooleanOptionalAction, default=False,
                   help="overlay stacked translucent radial gradients to cancel the 1..6/255 steps "
                        "between neighbouring region fills (the flat polygonal patches / Mach banding "
                        "that survive even when the pixel error is small). Every layer is a smooth "
                        "radial colour curve inside a soft disk that may only live in a large "
                        "low-gradient area; a layer is kept only while it strictly reduces the "
                        "smooth-weighted residual error of its disk, so the stack is a boosting "
                        "sequence. Plain SVG 1.1 (radialGradient + stop-opacity), so cairosvg and "
                        "mainstream browsers paint it identically. Costs one extra cairosvg render "
                        "plus about 1 s per layer per megapixel (measured: 50 s for 48 layers at "
                        "1000x1248).")
    g.add_argument("--shade-layers", type=int, default=48,
                   help="maximum number of stacked layers (the fit stops early once no candidate can "
                        "improve the smooth residual any more)")
    g.add_argument("--shade-radius", type=float, default=0.0,
                   help="largest disk radius in native px; 0 = auto = 0.19*max(width,height). The five "
                        "radii tried are 0.17/0.29/0.46/0.67/1.0 of it, and --scale is applied")
    g.add_argument("--shade-min-area", type=int, default=0,
                   help="only smooth connected areas of at least this many tracing-grid px^2 are "
                        "corrected; 0 = auto = (0.05*max(width,height)*--scale)^2")
    g.add_argument("--shade-eps", default="0.2,0.35,0.5",
                   help="candidate centre alphas of a disk (stronger alpha attenuates the boundary "
                        "steps more, but leans harder on the radial colour model)")
    g.add_argument("--shade-stops", type=int, default=4,
                   help="colour stops per layer (the radial colour curve is piecewise linear on them)")
    g.add_argument("--shade-tedge", type=float, default=0.55,
                   help="radius fraction that still has full alpha; the ramp to 0 runs from there to r")
    g.add_argument("--shade-smooth-gtol", type=float, default=2.5,
                   help="gradient magnitude (0..255 per px) at which the smooth weight reaches 0")
    g.add_argument("--shade-smooth-ttol", type=float, default=2.0,
                   help="detail energy at which the texture weight reaches 0 (keeps layers off noisy "
                        "areas)")
    g.add_argument("--shade-non-smooth", type=float, default=0.35,
                   help="largest share of non-smooth pixels a disk interior may contain")
    g.add_argument("--shade-halo", type=float, default=6.0,
                   help="largest RMS colour move (0..255) a layer may apply to non-smooth pixels "
                        "inside its disk (halo guard)")
    g.add_argument("--shade-gain", type=float, default=0.0,
                   help="minimum relative reduction of a disk's weighted residual error required to "
                        "keep the layer")
    g.add_argument("--shade-cand", type=int, default=8,
                   help="candidate disk centres evaluated per layer")
    g.add_argument("--shade-margin", type=float, default=28.0,
                   help="non-maximum-suppression radius between candidate centres, in grid px")
    pre, _ = p.parse_known_args(argv)
    if pre.preset:
        p.set_defaults(**PRESETS[pre.preset])
    return p.parse_args(argv)


# ======================================================================
# output post-processing: slimming / paths / directories
# ======================================================================


def ensure_dir(path: str) -> None:
    """Make sure the parent directory of the output file exists."""
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)


def resolve_io(args) -> None:
    """Complete the input path (a bare filename automatically looks in inputs/) and auto-name the three outputs (default out/)."""
    if not os.path.isfile(args.src):
        alt = os.path.join("inputs", args.src)
        if os.path.isfile(alt):
            args.src = alt
    stem = os.path.splitext(os.path.basename(args.src))[0] or "out"

    def expand(val, suffix):
        if val is None or val == "auto":
            return os.path.join("out", stem + suffix)
        return val

    if not args.out:
        args.out = os.path.join("out", stem + "_traced.svg")
    args.preview = expand(args.preview, "_preview.png")
    args.debug = "" if args.debug is None else expand(args.debug, "_debug.png")
    for pth in (args.out, args.preview, args.debug):
        if pth:
            ensure_dir(pth)


def main(argv=None):
    args = parse_args(argv)
    # Gradient gate: explicit value, or auto-tuned from image smoothness (--auto-gradient).
    _argv = list(sys.argv[1:] if argv is None else argv)
    # Was --aa-levels given by the user? (a preset value does not count as explicit)
    _aa_explicit = any(_a == "--aa-levels" or _a.startswith("--aa-levels=") for _a in _argv)
    _re_explicit = any(_a == "--refine-err" or _a.startswith("--refine-err=") for _a in _argv)
    if args.auto_gradient:
        from PIL import Image as _Im
        with _Im.open(args.src) as _im:
            _a = np.asarray(_im.convert("RGB").resize((64, 64))).astype(np.float64) / 255.0
        _g = float(np.abs(np.diff(_a, axis=0)).mean() + np.abs(np.diff(_a, axis=1)).mean())
        state.GRAD_MIN_GAIN = 0.02 if _g > 0.008 else args.grad_min_gain
        if _g > 0.008:
            # Smooth/glossy images: be aggressive about merging patches into large gradient-filled
            # regions -- this is what removes the visible colour steps the user sees. Measured on
            # apple.png @scale4: >=40px plateaus 35.5% -> 26.6% (source 24.8%), JUMP% back to the
            # source level, for about 0.5 dB of PSNR. Texture-rich photos should use --no-merge-grad,
            # where merging costs ~0.5 dB and radial-only is strictly better.
            args.merge_grad_tol = max(args.merge_grad_tol, 0.3)
            args.merge_grad_passes = max(args.merge_grad_passes, 8)
        print(f"      · automatic gradient routing: smoothness {_g:.4f} → gradient gate {state.GRAD_MIN_GAIN}",
              file=sys.stderr)
        # Smooth images: their remaining hard steps are the k-means patch boundaries, so also turn on
        # the boundary anti-aliasing staircase (3 levels) unless the user pinned --aa-levels. Tied to
        # the smooth-image feature stack so that --no-grad-radial --no-merge-grad still reproduces the
        # legacy output byte for byte for every other flag combination.
        if (_g > 0.008 and not _aa_explicit and int(args.aa_levels) <= 0
                and (args.grad_radial or args.merge_grad)):
            args.aa_levels = 3
            print("      · smooth image: boundary anti-aliasing transition bands enabled automatically (--aa-levels 3, turn off with --aa-levels 0)",
                  file=sys.stderr)
        # Smooth images: the residual that is left after merging is a *graded* residual (the fill is
        # 1-2/255 off over broad shallow ramps), so the refinement error gate can be tightened
        # without chasing sensor noise. Measured on apple.png @scale4: --refine-err 0.008 + the auto
        # budget takes >=40px plateaus from 26.6% to 23.9% (source 24.8%) and JUMP% from 11.16 to
        # 11.09 (source 11.20), with PSNR 33.32 -> 33.50 dB. Textured photos keep --refine-err.
        if _g > 0.008 and not _re_explicit and args.adaptive_refine:
            _re0 = float(args.refine_err)
            args.refine_err = min(_re0, 0.008)
            if args.refine_err < _re0:
                print(f"      · smooth image: adaptive-refinement error threshold tightened {_re0} → {args.refine_err} "
                      f"(override with --refine-err)", file=sys.stderr)
    else:
        state.GRAD_MIN_GAIN = args.grad_min_gain
    state.set_quiet(not args.verbose)
    resolve_io(args)
    if args.scale != 1.0 and not args.stroke_grid_units:
        # Length-like stroke parameters are interpreted in "native pixels": multiply by --scale automatically.
        # Otherwise the spacing would be in grid units and the stroke density would grow with scale² (59->9567 strokes at 4x, volume x4).
        for _nm in ("spacing", "stroke_min", "stroke_max", "stroke_width"):
            setattr(args, _nm, getattr(args, _nm) * args.scale)
    t0 = time.time()
    rng = np.random.default_rng(args.seed)

    def el():
        return f"{time.time()-t0:6.1f}s"

    # ---------------- read the image ----------------
    if not os.path.isfile(args.src):
        sys.exit(f"[error] input file not found: {args.src}")
    im0 = Image.open(args.src).convert("RGB")
    im = im0
    W0, H0 = im.size
    # ---------------- auto-scaling of parameters for small images ----------------
    # The defaults are calibrated for a ~1254px canvas (min_area=600px², spacing=10px ...). When the image is markedly smaller,
    # these absolute pixel quantities become relatively too large: regions are over-merged and strokes become thick bands smeared over the picture.
    # Here k = max(W,H)/AUTOSCALE_REF scales length-like parameters by k and area-like ones by k²; for k >= 1 nothing changes,
    # so large canvases (logo 1254², watercolor 1511x1600) keep exactly the historical parameters and results.
    if not getattr(args, "no_autoscale", False):
        k = max(W0, H0) / AUTOSCALE_REF
        if k < 1.0:
            for _nm in ("spacing", "stroke_min", "stroke_max", "stroke_width", "stroke_step",
                        "contour_tol", "snap_band", "snap_sigma", "snap_shift", "edge_width",
                        "edge_tol", "tex_sigma", "tex_smooth", "pre_smooth", "detail_scale",
                        "aa_width", "aa_smooth", "aa_min_radius"):
                if hasattr(args, _nm):
                    setattr(args, _nm, getattr(args, _nm) * k)
            for _nm in ("min_area", "detail_min_area", "edge_min_area", "edge_min_len",
                        "contour_min_area", "aa_min_area"):
                if hasattr(args, _nm):
                    setattr(args, _nm, max(1, int(round(getattr(args, _nm) * k * k))))
            log(f"      · parameter auto-scaling for small image k={k:.3f} (reference canvas {AUTOSCALE_REF:.0f}px): "
                f"min_area={args.min_area}  spacing={args.spacing:.1f}px")
    if args.scale != 1.0:
        im = im.resize((max(32, int(W0 * args.scale)), max(32, int(H0 * args.scale))),
                       Image.LANCZOS)
    W, H = im.size
    rgb = np.asarray(im).astype(np.float64) / 255.0
    log(f"[{el()}] read in {args.src}  {W}x{H}"
          + (f" (source {W0}x{H0})" if args.scale != 1.0 else ""))

    # ---------------- edge-preserving smoothing ----------------
    rgb_s = rgb
    if args.denoise > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rgb_s = denoise_bilateral(rgb, sigma_color=args.denoise, sigma_spatial=2,
                                      channel_axis=-1)
    rgb255 = np.clip(rgb_s, 0, 1) * 255.0

    # ---------------- structure tensor ----------------
    tf = tc = None
    if gpu.enabled(args):
        try:
            tf = gpu.structure_tensor(rgb_s, args.sigma_d, args.sigma_i, eps=EPS)
            tc = gpu.structure_tensor(rgb_s, args.cs_sigma_d, args.cs_sigma_i, eps=EPS)
            log(f"[{el()}] structure tensor: GPU {gpu.device_name()} "
                f"(in device memory, accumulating in double like the CPU)")
        except Exception as _e:
            print(f"      ! GPU structure tensor failed ({_e}), falling back to CPU", file=sys.stderr)
            tf = tc = None
    if tf is None:
        tf = structure_tensor(rgb_s, args.sigma_d, args.sigma_i)
        tc = structure_tensor(rgb_s, args.cs_sigma_d, args.cs_sigma_i)
    log(f"[{el()}] structure tensor: fine scale (σd={args.sigma_d},σi={args.sigma_i}) "
          f"mean coherence={tf['coh'].mean():.3f} | coarse scale (σd={args.cs_sigma_d},"
          f"σi={args.cs_sigma_i}) mean coherence={tc['coh'].mean():.3f}")

    # ---------------- segmentation ----------------
    # --labels-cache: cache the segmentation result. When tuning stroke/gradient/fitting parameters there is no need to rerun segmentation,
    # and it is off by default so default results are unaffected.
    geo_px = None
    if args.labels_cache and os.path.exists(args.labels_cache):
        labels = np.load(args.labels_cache).astype(np.int32)
        log(f"[{el()}] loaded segmentation cache {args.labels_cache} "
              f"({int(labels.max()) + 1} color regions)")
    else:
        if args.method == "colors":
            labels = segment_colors(rgb_s, rgb255, args, rng)
        elif args.method == "edges":
            labels = segment_edges(rgb_s, tf, args, rng)
        elif args.method == "hybrid":
            if args.seg in ("auto", "flat"):
                labels = segment_flat(rgb_s, rgb255, args, rng)
                if labels is None and args.seg == "flat":
                    print("      ! flat-color fast path not applicable, falling back to the full route", file=sys.stderr)
            if args.seg == "flat" and labels is None:
                labels, geo_px = segment_hybrid(rgb_s, rgb255, tf, args, rng)
            elif args.seg == "full":
                labels, geo_px = segment_hybrid(rgb_s, rgb255, tf, args, rng)
            elif labels is None:
                labels, geo_px = segment_hybrid(rgb_s, rgb255, tf, args, rng)
        else:
            from skimage.feature import peak_local_max
            from skimage.segmentation import watershed
            elev = ndi.gaussian_filter(tf["energy"], 1.2)
            elev /= elev.max() + EPS
            coords = peak_local_max(-elev, min_distance=9, exclude_border=False,
                                    labels=np.ones(elev.shape, bool))
            markers = np.zeros(elev.shape, np.int32)
            markers[tuple(coords.T)] = np.arange(1, coords.shape[0] + 1)
            labels = watershed(elev, markers, compactness=0.0005).astype(np.int32)
        if args.labels_cache:
            np.save(args.labels_cache, labels)
    labels = merge_small_regions(labels, rgb_s, args.min_area,
                                 core_radius=max(0, (args.min_width - 1) // 2),
                                 protect_sat=args.protect_sat)
    # gradient-aware spatial merging: adjacent patches whose union is still one gradient become a
    # single (larger, smoother) region. Off -> labels untouched, so every downstream byte is unchanged.
    if getattr(args, "merge_grad", False):
        _t_mg = time.time()
        labels, _n_mg0, _n_mg1 = merge_gradient_regions(rgb_s, labels, tc, args)
        log(f"[{el()}] gradient-aware merging: {_n_mg0} → {_n_mg1} color regions "
            f"(-{_n_mg0 - _n_mg1}, {time.time() - _t_mg:.1f}s)")
    n_lab = int(labels.max()) + 1
    areas = np.bincount(labels.ravel(), minlength=n_lab)
    log(f"[{el()}] segmentation: {n_lab} color regions, area {int(areas.min())}~{int(areas.max())} px")
    # shared partition-routing criteria (used by curve fitting / boundary snapping / anti-aliasing bands)
    geo_r, tex_r_all, rng_r_all, _t_norm = classify_regions(
        rgb_s, labels, areas, args.tex_sigma, args.tex_smooth, args.tex_norm,
        args.tex_thr, args.range_thr, args.min_area)
    if args.verbose:
        _keep = [int(li) for li in np.nonzero(areas >= args.min_area)[0]]
        _ngeo = sum(1 for li in _keep if geo_r[li])
        _head = ", ".join("#%d:%s(texture%.1e/spread%.3f)"
                          % (li, "geo" if geo_r[li] else "complex",
                             tex_r_all[li], rng_r_all[li]) for li in _keep[:8])
        print("      · partition routing (unified criterion): %d/%d color regions classified as geometric%s"
              % (_ngeo, len(_keep), ("; e.g. " + _head) if _head else ""))
    # energy threshold for subpixel edge snapping (the same threshold as the edge map)
    thr_ref = float(np.quantile(tf["energy"], args.edge_nms_hi))
    _REFINE_STATS.update(pts=0, moved=0, sum=0.0)

    # ---------------- background region ----------------
    border = np.zeros((H, W), bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    bl = np.bincount(labels[border].ravel(), minlength=n_lab)
    bg_label = int(np.argmax(bl))
    bg_mask = labels == bg_label
    bg_col = np.median(rgb_s[bg_mask], axis=0) if bg_mask.any() else np.ones(3)

    # ---------------- assembling regions ----------------
    # Each region is computed only inside a "bounding box + safety margin" window. Outside the window nothing affects the local operators here
    # (erosion/dilation/median/contour/pixel indexing all depend only on the region's surroundings), so the result is bit-identical to the full-image version,
    # while the complexity drops from O(H*W) per region to O(region area) per region.
    _objs = ndi.find_objects(labels + 1)   # index k corresponds to label k -- including label 0!
    # The margin must cover the adaptive cropping inside mask_to_paths: dilation 1 + pad 3 + mar(3*smooth+3);
    # if it is too small, the Gaussian smoothing sees the "window border" instead of the real all-zero background and the contour coordinates are no longer bit-identical.
    _MG = 8 + int(3 * args.contour_smooth) + 4

    def _win_of(li):
        sl = _objs[li] if 0 <= li < len(_objs) else None
        if sl is None:
            return (slice(0, 0), slice(0, 0)), 0, 0
        y0 = max(0, sl[0].start - _MG)
        y1 = min(H, sl[0].stop + _MG)
        x0 = max(0, sl[1].start - _MG)
        x1 = min(W, sl[1].stop + _MG)
        return (slice(y0, y1), slice(x0, x1)), y0, x0

    def _recon(r):
        """Window mask → full-image-coordinate mask (only called where a full-image tensor is genuinely needed)."""
        win = r.get("win")
        if win is None:
            return r["mask"]
        full = np.zeros((H, W), bool)
        full[win] = r["mask"]
        return full

    # The rule-dense sub-steps (mask erosion / per-region median color) are batch-computed once over the whole image, preferring the GPU.
    # Byte-identical to the per-region version (see the derivation in batch_bookkeeping), and it falls back automatically when the image was resampled.
    _inner_all = _cols_bk = None
    _bk_backend = ""
    if getattr(args, "batch_book", "auto") != "off":
        _t_bk = time.time()
        _inner_all, _cols_bk, _bk_backend = batch_bookkeeping(labels, rgb_s,
                                                             use_gpu=gpu.enabled(args))
        if _inner_all is None:
            log(f"[{el()}] batch bookkeeping: unavailable (pixels are not on the 1/255 grid / too many labels), falling back to per-region")
        else:
            log(f"[{el()}] batch bookkeeping: mask + median color of {len(areas)} labels computed in one pass "
                f"({time.time() - _t_bk:.1f}s, {_bk_backend})")

    regions = []
    _order = [int(x) for x in np.argsort(-areas)]
    _jobs = _n_jobs(args, len(_order))
    if _jobs > 1:
        # fork: the large arrays labels/rgb_s/tc/tf are shared by copy-on-write, neither copied nor pickled;
        # imap returns in the original order, so the order of regions is exactly that of the serial version.
        import multiprocessing
        _PAR_CTX.clear()
        _PAR_CTX.update(dict(args=args, labels=labels, areas=areas, rgb_s=rgb_s, tc=tc,
                             tf=tf, thr_ref=thr_ref, bg_label=bg_label, geo_r=geo_r,
                             H=H, W=W, MG=_MG, objs=_objs,
                             inner_all=_inner_all, cols=_cols_bk))
        _t_par = time.time()
        log(f"[{el()}] per-region stage: {len(_order)} candidate regions, {_jobs} processes in parallel")
        with multiprocessing.get_context("fork").Pool(_jobs) as _pool:
            for _r, _msg, _st in _pool.imap(_par_region_one, _order, chunksize=4):
                if _r is not None:
                    regions.append(_r)
                if _msg:
                    log(_msg)
                _REFINE_STATS["pts"] += _st["pts"]
                _REFINE_STATS["moved"] += _st["moved"]
                _REFINE_STATS["sum"] += _st["sum"]
        log(f"      · parallel assembly of {len(regions)} regions took {time.time() - _t_par:.1f}s")
    else:
        _t_ser = time.time()
        for li in np.argsort(-areas):
            li = int(li)
            win, y0, x0 = _win_of(li)
            mask = labels[win] == li
            if int(areas[li]) < args.min_area:      # equals mask.sum(), but in O(1)
                continue
            if _inner_all is not None:
                inner = _inner_all[win] & mask    # same as above: restrict back to this region
                if int(inner.sum()) < 200:
                    inner = mask
                col = _cols_bk[li]
            else:
                inner = ndi.binary_erosion(mask, iterations=3)
                if inner.sum() < 200:
                    inner = mask
                col = np.median(rgb_s[win][inner], axis=0)
            if li == bg_label:
                regions.append({"idx": li, "mask": mask, "area": int(areas[li]), "d": "",
                                "fill": to_hex(col), "grad": None, "bg": True,
                                "geo": float(geo_r[li]), "win": win, "off": (y0, x0), "col": col})
                continue
            m = ndi.binary_dilation(mask, iterations=1)
            geo_frac = float(geo_r[li])
            use_fit = args.fit
            if use_fit == "auto":
                use_fit = "bezier" if geo_frac >= args.auto_geo_frac else "cr"
            rf = None
            if args.snap_mode == "refine" and geo_frac >= args.auto_geo_frac:
                rf = (tf, thr_ref, args.snap_shift, args.snap_step, args.snap_sub)
            d = region_path_d(m, args.contour_tol, args.contour_smooth, args.contour_min_area,
                              fit=use_fit, fit_tol=args.fit_tol, corner_deg=args.corner_deg,
                              refine=rf, offset=(y0, x0))
            if not d:
                continue
            # rgb_s/tc are passed as the full image and off only converts pixel indices to absolute coordinates -> values are bit-identical to the full-image version
            grad = fit_region_gradient(rgb_s, inner, tc, args.grad_stops,
                                       args.grad_min_range, off=(y0, x0),
                                       allow_radial=args.grad_radial,
                                       radial_margin=args.grad_radial_margin)
            regions.append({"idx": li, "mask": mask, "area": int(areas[li]), "d": d,
                            "fill": to_hex(col), "grad": grad, "bg": False,
                            "geo": geo_frac, "fit": use_fit, "win": win, "off": (y0, x0),
                            "col": col})
            if args.verbose:
                q = f"{grad['quality']:.2f}({grad['axis']})" if grad else "flat"
                log(f"      · region #{li:<2d} area={int(areas[li]):>7d} {to_hex(col)} "
                      f"gradient quality={q} fit={use_fit}(geometricity {geo_frac:.2f})")

        log(f"      · serial assembly of {len(regions)} regions took {time.time() - _t_ser:.1f}s")

    # ---- error-driven adaptive refinement (see refine_regions) ----
    # Gated on the gradient feature stack as well, so that the legacy flat-colour route
    # (--no-grad-radial --no-merge-grad) still reproduces the old output byte for byte.
    if args.adaptive_refine and (args.grad_radial or args.merge_grad):
        _t_rf = time.time()
        _lab_ref = labels.copy()
        regions, _rst = refine_regions(rgb_s, tc, regions, args, tf, thr_ref, labels_out=_lab_ref)
        _extra = f", {len(regions) - _rst['n0']:+d}" if _rst["n1"] != _rst["n0"] else ""
        log(f"[{el()}] adaptive refinement: {_rst['n0']} → {_rst['n1']} regions{_extra} "
            f"(bisections {_rst['split']}, background {_rst['bg_split']}, max depth {_rst['depth']}, "
            f"evaluated {_rst['eval']}, rejected {_rst['rejected']}, budget {_rst['budget']}, "
            f"took {time.time() - _t_rf:.1f}s)")
        if args.merge_grad and args.refine_order == "refine-first":
            _t_rm = time.time()
            _mg, _m0, _m1 = merge_gradient_regions(rgb_s, _lab_ref, tc, args)
            if _m1 < _m0:
                regions = regroup_regions(rgb_s, regions, _lab_ref, _mg, args, tc, tf, thr_ref)
                _lab_ref = _mg
            log(f"[{el()}] post-refinement gradient merging (refine-first): {_m0} → {_m1} color regions, "
                f"regrouped into {len(regions)} regions ({time.time() - _t_rm:.1f}s)")
        labels = _lab_ref                      # keep AA bands / debug consistent with the refined ids
        if args.dump_refined_labels:
            np.save(args.dump_refined_labels, labels)

    # ---- anti-aliasing transition bands: hard edges -> the source image's 2~3px transition (aa_band_regions) ----
    if args.aa_levels > 0 and args.aa_width > 0:
        t_aa = time.time()
        aa_segs = aa_band_regions(rgb_s, regions, labels, args, tf, thr_ref)
        regions.extend(aa_segs)
        if aa_segs:
            print("[%s] anti-aliasing transition bands: %d segments (%d levels per side, total width %.1fpx), covering %d px, "
                  "skipped %d synthetic-boundary segments, skipped %d duplicate real-edge segments | %.1fs"
                  % (el(), len(aa_segs), args.aa_levels, args.aa_width,
                     sum(a["area"] for a in aa_segs), _AA_STATS["skipped_syn"],
                     _AA_STATS["skipped_canon"], time.time() - t_aa))

    # ---- detail layer: colored thin seam lines (golden strokes) ----
    # k-means is very unfriendly to "thin and colored" narrow bands (they often get merged into the neighbouring large region), yet such seam lines
    # are exactly the key elements of a logo, so they are extracted separately as one vector layer using Lab chroma (drawn on top).
    if _REFINE_STATS["moved"] > 0 and args.verbose:
        _m = _REFINE_STATS["moved"]
        log(f"      · subpixel edge snapping: {_m}/{_REFINE_STATS['pts']} contour points snapped onto edges, "
              f"mean displacement {_REFINE_STATS['sum'] / _m:.3f}px (limit {args.snap_shift}px)")
    if args.detail_chroma > 0:
        _t_det = time.time()
        lab = rgb2lab(np.clip(rgb_s, 0, 1))
        chroma = ndi.gaussian_filter(np.hypot(lab[..., 1], lab[..., 2]),
                                     args.detail_scale)
        dm = ndi.binary_closing(chroma > args.detail_chroma, np.ones((3, 3), bool))
        lb = measure.label(dm, connectivity=2)
        szs = np.bincount(lb.ravel(), minlength=1)
        _ks = [int(k) for k in np.nonzero(szs >= args.detail_min_area)[0] if int(k) != 0]
        _PAR_CTX.clear()
        _PAR_CTX.update(args=args, rgb_s=rgb_s, tc=tc, tf=tf, thr_ref=thr_ref, lb=lb,
                        H=H, W=W, MG=_MG, objs=ndi.find_objects(lb + 1))
        # Every component of the detail layer is heavy enough (whole-image filtering + contours), so the threshold is lowered to 2: the component count is often far below 24,
        # and the default threshold would degrade it to serial -- measured on the full image, those two stages were literally "only 1 of 16 workers busy".
        _jobs_det = _n_jobs(args, len(_ks), min_items=2)
        _det = []
        if _jobs_det > 1:
            import multiprocessing
            with multiprocessing.get_context("fork").Pool(_jobs_det) as _pool:
                for _r, _msg in _pool.imap(_par_detail_one, _ks, chunksize=4):
                    if _r is not None:
                        _det.append(_r)
                    if _msg:
                        log(_msg)
        else:
            for _k in _ks:
                _r, _msg = _par_detail_one(_k)
                if _r is not None:
                    _det.append(_r)
                if _msg:
                    log(_msg)
        regions.extend(_det)
        n_det = len(_det)
        log(f"[{el()}] detail layer (Lab chroma>{args.detail_chroma}): {n_det} colored seam lines "
              f"(candidate components {len(_ks)}, {_jobs_det} processes, took {time.time() - _t_det:.1f}s)")

    n_aa = sum(1 for r in regions if r.get("aa"))
    n_grad = sum(1 for r in regions if r["grad"] is not None)
    n_rad = sum(1 for r in regions if r["grad"] is not None and r["grad"].get("kind") == "radial")
    log(f"[{el()}] region geometry: {len(regions) - n_aa} paths (plus {n_aa} anti-aliasing band segments), "
          f"of which {n_grad} carry a gradient (linear {n_grad - n_rad} / radial {n_rad})")

    # ---- stroke activity: residual between the image and the per-region base fill (gradient/flat color), deciding where strokes are needed ----
    act = np.zeros((H, W), np.float32)
    _PAR_CTX.clear()
    _PAR_CTX.update(args=args, regions=regions, rgb_s=rgb_s, shape=rgb.shape)
    _jobs_act = _n_jobs(args, len(regions))
    if _jobs_act > 1:
        import multiprocessing
        with multiprocessing.get_context("fork").Pool(_jobs_act) as _pool:
            _it = _pool.imap(_par_act_one, range(len(regions)), chunksize=8)
            for _res in _it:
                if _res is not None:
                    _rows, _cols, _vals = _res
                    act[_rows, _cols] = _vals
    else:
        for _i in range(len(regions)):
            _res = _par_act_one(_i)
            if _res is not None:
                _rows, _cols, _vals = _res
                act[_rows, _cols] = _vals
    act = ndi.gaussian_filter(act, 2.5)
    if args.verbose and act.max() > 0:
        print(f"      activity residual: mean={act[act>0].mean()*255:.2f}/255 "
              f"p99={np.percentile(act[act>0],99)*255:.2f}/255")

    # ---------------- strokes ----------------
    stroke_groups = []
    if not args.no_strokes:
        bg_mask2 = _recon(regions[0]) if regions and regions[0]["bg"] else bg_mask
        fg = ~bg_mask2
        fg = ndi.binary_erosion(fg, iterations=1)
        _sregs = sorted([r for r in regions if not r["bg"] and not r.get("aa")],
                        key=lambda z: -z["area"])
        _jobs_str = _n_jobs(args, len(_sregs))
        if _jobs_str > 1:
            # parallel: each child builds its own rng from (seed, index), so the result is scheduling-independent and reproducible;
            # but the jitter pattern differs from the serial version (shared rng) -- use --jobs 1 to reproduce out/ byte for byte.
            import multiprocessing
            _PAR_CTX.clear()
            _PAR_CTX.update(args=args, tc=tc, fg=fg, act=act, rgb_s=rgb_s, H=H, W=W,
                            sregs=_sregs, seed=int(args.seed or 0))
            with multiprocessing.get_context("fork").Pool(_jobs_str) as _pool:
                for _g in _pool.imap(_par_stroke_one, range(len(_sregs)), chunksize=2):
                    if _g is None:
                        continue
                    stroke_groups.append(_g)
                    if args.verbose:
                        log(f"      · region #{_g['idx']} (area={_g['area']}) "
                              f"→ {len(_g['strokes'])} strokes")
        else:
            for r in sorted([r for r in regions if not r["bg"] and not r.get("aa")],
                            key=lambda z: -z["area"]):
                # The sub.sum()<400 test is computed inside the window first (exactly equivalent to the full image, and O(window));
                # only regions that really need streamlines pay the one-off O(H*W) restoration -- make_streamlines needs the whole-image
                # tensor field anyway, so the per-region O(H*W) cost is inherent to it.
                win = r.get("win")
                if win is None:
                    sub = r["mask"] & fg
                else:
                    sub_w = r["mask"] & fg[win]
                    if sub_w.sum() < 400:
                        continue
                    sub = np.zeros((H, W), bool)
                    sub[win] = sub_w
                if sub.sum() < 400:
                    continue
                segs = make_streamlines(tc, sub, args, rng)
                if not segs:
                    continue
                items = []
                for pts, wid, _c in segs:
                    iy = np.clip(pts[:, 1].astype(int), 0, H - 1)
                    ix = np.clip(pts[:, 0].astype(int), 0, W - 1)
                    a = float(act[iy, ix].mean())
                    alpha = args.stroke_alpha * min(1.0, a / (args.stroke_activity_ref + EPS))
                    if alpha < args.stroke_alpha_min:
                        continue  # the base fill is accurate enough -> no stroke (avoids creating artifacts in flat areas)
                    col = sample_color(rgb_s, pts, args.stroke_jitter, rng)
                    items.append((ribbon_d(pts, wid), col, min(alpha, 0.95)))
                stroke_groups.append({"idx": r["idx"], "d": r["d"], "strokes": items})
                if args.verbose:
                    log(f"      · region #{r['idx']} (area={r['area']}) → {len(items)} strokes")
    n_strokes = sum(len(g["strokes"]) for g in stroke_groups)
    log(f"[{el()}] strokes: {n_strokes} (spacing {args.spacing}px)")

    # ---------------- edges / seam lines ----------------
    edges = []
    if args.edge_mode != "off":
        fgm = ~bg_mask
        em = np.zeros((H, W), bool)
        chroma = None
        if args.edge_mode in ("chroma", "both"):
            lab = rgb2lab(np.clip(rgb_s, 0, 1))
            chroma = np.hypot(lab[..., 1], lab[..., 2])
            em |= (ndi.gaussian_filter(chroma, 0.8) > args.edge_chroma) & fgm
        if args.edge_mode in ("energy", "both"):
            e = tf["energy"]
            vals = e[fgm]
            hi = float(np.quantile(vals, args.edge_hi)) if vals.size else 0.0
            lo = float(np.quantile(vals, args.edge_lo)) if vals.size else 0.0
            em |= (filters.apply_hysteresis_threshold(e, lo, hi) & fgm
                   & (tf["coh"] > args.edge_coh))
        lb = measure.label(em, connectivity=2)
        sizes = np.bincount(lb.ravel())
        keep = np.nonzero(sizes >= args.edge_min_area)[0]
        keep = keep[keep != 0]
        em = np.isin(lb, keep) if keep.size else np.zeros_like(em)
        from skimage import morphology   # lazy import: needed in this one place only, and importing at startup costs 2.6s
        skel = morphology.skeletonize(em)
        for p in skeleton_paths(skel, min_len=args.edge_min_len, tol=args.edge_tol):
            col = sample_color(rgb_s, p, 0.0, rng)
            iy = np.clip(p[:, 1].astype(int), 0, H - 1)
            ix = np.clip(p[:, 0].astype(int), 0, W - 1)
            if chroma is not None:
                ch = float(np.median(chroma[iy, ix]))
                wq = args.edge_width * (0.75 + 0.5 * min(1.0, max(0.0, ch / args.edge_chroma - 1.0)))
            else:
                wq = args.edge_width * (0.8 + 0.8 * min(1.0, float(np.median(tf["coh"][iy, ix]))))
            edges.append((polyline_to_bezier_d(p, closed=False), col, wq, args.edge_alpha))
        log(f"[{el()}] edges/seam lines: {len(edges)} lines ({args.edge_mode} mode)")
    else:
        log(f"[{el()}] edges/seam lines: off (the color-region layer already contains the golden seam lines)")

    # ---------------- emit SVG ----------------
    meta = {"bg_fill": to_hex(bg_col), "src": os.path.basename(args.src),
            "desc": (f"structure-tensor vector tracing | fine(σd={args.sigma_d},σi={args.sigma_i}) "
                     f"coarse(σd={args.cs_sigma_d},σi={args.cs_sigma_i}) | regions={len(regions)} "
                     f"gradients={n_grad} strokes={n_strokes} edges={len(edges)}")}
    svg = build_svg(W0, H0, regions, stroke_groups, edges, meta, view_box=(W, H))
    sh_stats = None
    if args.shade_blobs:
        # Defaults that depend on the picture: a "smooth blob" is a fraction of the canvas, not a
        # fixed number of pixels, so the radius and the minimum area follow the native size.
        if args.shade_radius <= 0:
            args.shade_radius = max(48.0, 0.19 * max(W0, H0) * args.scale)
        if args.shade_min_area <= 0:
            args.shade_min_area = max(900, int(round((0.05 * max(W0, H0) * args.scale) ** 2)))
        _sh = build_shade_stack(rgb255, svg, W, H, args, log)
        if _sh is not None:
            svg = build_svg(W0, H0, regions, stroke_groups, edges, meta, view_box=(W, H),
                            shade=_sh[:2])
            sh_stats = _sh[2]
            log(f"[{el()}] shade layers: {sh_stats['n']} stacked radial gradients "
                f"(smooth area {sh_stats['area']}px, disk area/smooth area {sh_stats['cov']:.1f}x, "
                f"base {sh_stats['psnr0']:.2f} → {sh_stats['psnr']:.2f} dB after stacking, "
                f"took {sh_stats['secs']:.1f}s)")
    raw_kb = len(svg.encode()) / 1024
    if args.compress != "off":
        svg = slim_svg(svg, prec_contour=1,
                       prec_stroke=(0 if args.compress == "tight" else 1),
                       drop_clip=(args.compress == "tight"))
    ensure_dir(args.out)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(svg)
    out_kb = os.path.getsize(args.out) / 1024
    extra = "" if args.compress == "off" else f", {args.compress} slimming {raw_kb:.0f}→{out_kb:.0f} KB"
    log(f"[{el()}] wrote {args.out} ({out_kb:.1f} KB{extra})")

    gz_path = ""
    if args.gzip:
        gz_path = (args.out[:-4] if args.out.lower().endswith(".svg") else args.out) + ".svgz"
        # mtime=0: makes the .svgz reproducible (otherwise the gzip header carries a build timestamp)
        with open(gz_path, "wb") as fh, gzip.GzipFile(
                fileobj=fh, mode="wb", compresslevel=9, mtime=0) as f:
            f.write(svg.encode())
        log(f"      · gzip {gz_path} ({os.path.getsize(gz_path)/1024:.1f} KB)")

    # ---------------- structure tensor debug image ----------------
    if args.debug:
        base = 0.42 * rgb_s + 0.58
        img = Image.fromarray((np.clip(base, 0, 1) * 255).astype(np.uint8))
        dr = ImageDraw.Draw(img)
        step = 14
        ys, xs = np.mgrid[step // 2:H:step, step // 2:W:step]
        ys, xs = ys.ravel(), xs.ravel()
        cohv = tc["coh"][ys, xs]
        k = cohv > 0.15
        ys, xs, cohv = ys[k], xs[k], cohv[k]
        L = 5.0
        for y, x, c in zip(ys.tolist(), xs.tolist(), cohv.tolist()):
            ang = math.atan2(tc["ty"][y, x], tc["tx"][y, x])
            dx, dy = math.cos(ang) * L, math.sin(ang) * L
            dr.line([x - dx, y - dy, x + dx, y + dy],
                    fill=(int(255 * c), int(90 * (1 - c)), int(255 * (1 - c))), width=1)
        by, bx = np.nonzero(find_boundaries(labels, mode="outer"))
        for y, x in zip(by[::2].tolist(), bx[::2].tolist()):
            img.putpixel((x, y), (0, 160, 0))
        img.save(args.debug)
        log(f"      · debug image {args.debug}: segment direction=isophote, hue=coherence, green=color-region boundary")

    # ---------------- preview + fidelity ----------------
    psnr = None
    if args.preview:
        try:
            import cairosvg
            # Render back at **native size** before comparing: that way the self-reported PSNR and comparisons across --scale use the same basis
            cairosvg.svg2png(url=args.out, write_to=args.preview,
                             output_width=W0, output_height=H0)
            prev = np.asarray(Image.open(args.preview).convert("RGB")).astype(np.float64)
            ref = np.asarray(im0).astype(np.float64)
            if prev.shape == ref.shape:
                mae = float(np.abs(prev - ref).mean())
                mse = float(((prev - ref) ** 2).mean())
                psnr = 99.0 if mse <= 0 else 10 * math.log10(255.0 ** 2 / mse)
                log(f"      · preview {args.preview}: rendered back at native {W0}x{H0}, MAE={mae:.2f}/255 PSNR={psnr:.2f} dB")
        except ImportError:
            print("      · cairosvg is not installed, skipping preview rendering")
        except Exception as exc:
            log(f"      · preview rendering failed: {exc}")

    # ---------------- final summary ----------------
    log(f"[{el()}] done")
    if W != W0:
        grid = f"  →  grid {W}x{H} (--scale {args.scale:g})"
    else:
        grid = ""
    print("  " + "-" * 62)
    print(f"  input    {args.src}  {W0}x{H0}{grid}")
    print(f"  regions  {len(regions)} areas · {n_grad} gradients · {n_strokes} strokes · {len(edges)} seams")
    print(f"  vector   {args.out}  {out_kb:.0f} KB" + (f"   ({psnr:.2f} dB)" if psnr else ""))
    if sh_stats:
        print(f"  shade    {sh_stats['n']} radial-gradient layers (smooth area {sh_stats['area']}px, coverage "
              f"{sh_stats['cov']:.1f}x)")
    if gz_path:
        print(f"  gzip     {gz_path}  {os.path.getsize(gz_path)/1024:.0f} KB")
    if args.preview and os.path.exists(args.preview):
        print(f"  preview  {args.preview}")
    if args.debug:
        print(f"  debug    {args.debug}")
    print("  " + "-" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
