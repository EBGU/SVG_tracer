#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
logo_trace.py —— bitmap tracing / vectorization (SVG) based on the "multi-scale structure tensor"

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
    python logo_trace.py --in logo.png --out logo_traced.svg \
        --preview logo_traced_preview.png --debug logo_tensor_debug.png
"""

from __future__ import annotations

import argparse
import gzip
import heapq
import math
import os
import re
import sys
import time
import warnings

# ---- thread cap (must come before "import numpy") ----------------------
# This script's numeric work is memory-bandwidth bound, while OpenBLAS/OpenMP on large
# shared nodes threads by core count by default, and thread oversubscription slows down
# rgb2lab / matrix operations by more than tenfold. Override with LOGO_TRACE_THREADS.
_THREADS = os.environ.get("LOGO_TRACE_THREADS", "4")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, _THREADS)

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

try:
    from skimage import filters, measure
    from skimage.color import rgb2lab
    from skimage.restoration import denoise_bilateral
    from skimage.segmentation import find_boundaries
except Exception as exc:  # pragma: no cover
    sys.exit(f"[错误] 需要 scipy 与 scikit-image: {exc}")

from svg_slim import slim_path, slim_svg  # slimming kernel, shared with svgzip.py (standard library only)
import gpu_backend as gpu

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

EXAMPLES = """\
示例
----
  # 硬边平面图（logo / 图标 / 插画）：2 倍网格 + 亚像素定位 + 保角贝塞尔，一行搞定
  python logo_trace.py --in logo.png --preset logo

  # 写意绘画 / 照片：细节预设（色块小而多、笔触密而细）
  python logo_trace.py --in water-lilies-29.jpg --preset painting

  # 只要几何层（不要笔触）并顺带出 .svgz
  python logo_trace.py --in logo.png --preset logo --no-strokes --gzip

  # 追高保真：4 倍网格（5016²，约 20 分钟）
  python logo_trace.py --in logo.png --preset logo --scale 4 --compress slim --gzip

  # 自检：合成小图跑通全流程并断言质量
  python selfcheck.py

输入默认为 inputs/<名>，输出默认写到 out/；裸文件名会自动到 inputs/ 下找。
"""


# ======================================================================
# generic helpers
# ======================================================================
def to_hex(c) -> str:
    r, g, b = (np.clip(np.asarray(c, float), 0.0, 1.0) * 255).round().astype(int)
    return f"#{r:02x}{g:02x}{b:02x}"


def fnum(v, nd=1) -> str:
    s = f"{float(v):.{nd}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s if s not in ("", "-0") else "0"


def bilin(a: np.ndarray, x: float, y: float) -> float:
    """Bilinear sampling (scalar)."""
    h, w = a.shape
    x = min(max(x, 0.0), w - 1.001)
    y = min(max(y, 0.0), h - 1.001)
    x0 = int(x)
    y0 = int(y)
    fx = x - x0
    fy = y - y0
    return float(
        a[y0, x0] * (1 - fx) * (1 - fy)
        + a[y0, x0 + 1] * fx * (1 - fy)
        + a[y0 + 1, x0] * (1 - fx) * fy
        + a[y0 + 1, x0 + 1] * fx * fy
    )


def _clamp_ctrl(p1, c, p2, max_ratio=0.55):
    """Clamp control-point offsets so that Catmull-Rom does not overshoot at sharp corners and create 'spikes'."""
    v = c - p1
    seg = float(np.hypot(*(p2 - p1)))
    n = float(np.hypot(*v))
    if seg > 1e-9 and n > max_ratio * seg:
        v = v * (max_ratio * seg / n)
    return p1 + v


def polyline_to_bezier_d(pts, closed=True, nd=1) -> str:
    """Polyline → cubic Bezier path (Catmull-Rom tangents, relative-coordinate output: compact and smooth)."""
    pts = np.asarray(pts, float)
    if closed and len(pts) > 2 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    n = len(pts)
    if n < 2:
        return ""
    if n == 2:
        d = (f"M{fnum(pts[0,0],nd)} {fnum(pts[0,1],nd)}"
             f"L{fnum(pts[1,0],nd)} {fnum(pts[1,1],nd)}")
        return d + ("z" if closed else "")
    out = [f"M{fnum(pts[0,0],nd)} {fnum(pts[0,1],nd)}"]
    last = n if closed else n - 1
    for i in range(last):
        p0 = pts[(i - 1) % n] if closed else pts[max(i - 1, 0)]
        p1 = pts[i % n]
        p2 = pts[(i + 1) % n]
        p3 = pts[(i + 2) % n] if closed else pts[min(i + 2, n - 1)]
        c1 = _clamp_ctrl(p1, p1 + (p2 - p0) / 6.0, p2)
        c2 = _clamp_ctrl(p2, p2 - (p3 - p1) / 6.0, p1)
        out.append(
            "c"
            + " ".join(
                fnum(v, nd)
                for v in (
                    c1[0] - p1[0], c1[1] - p1[1],
                    c2[0] - p2[0], c2[1] - p2[1],
                    p2[0] - p1[0], p2[1] - p1[1],
                )
            )
        )
    if closed:
        out.append("z")
    return "".join(out)


def poly_area(pts) -> float:
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _cross2(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


# ----------------------------------------------------------------------
# Conformal Bezier fitting (Schneider, Graphics Gems)
# Endpoints are fixed, the tangent follows the line through the neighbouring points, and the two
# control-point lengths a1/a2 are solved by least squares; if the error exceeds the limit, split
# recursively at the maximum-error point, sharing the tangent there between both sides -> G1-continuous joints; nearly straight segments are emitted as L, so straight edges are not baked into curves and round corners stay round.
# ----------------------------------------------------------------------
def _chord_param(pts: np.ndarray) -> np.ndarray:
    seg = np.hypot(*np.diff(pts, axis=0).T)
    d = np.r_[0.0, np.cumsum(seg)]
    if d[-1] < 1e-12:
        return np.linspace(0.0, 1.0, len(pts))
    return d / d[-1]


def _eval_cubic(p0, p1, p2, p3, u):
    u = np.asarray(u, float)[:, None]
    return (((1 - u) ** 3) * p0 + (3 * (1 - u) ** 2 * u) * p1
            + (3 * (1 - u) * u ** 2) * p2 + (u ** 3) * p3)


def _unit(v):
    n = float(np.hypot(*v))
    return v / n if n > 1e-12 else np.zeros(2)


def _fit_cubic_ls(pts: np.ndarray, t1, t2):
    """t1: unit tangent at p0 pointing into the curve; t2: unit tangent at p3 pointing into the curve."""
    u = _chord_param(pts)
    p0, p3 = pts[0], pts[-1]
    b0 = (((1 - u) ** 3) + 3 * (1 - u) ** 2 * u)[:, None] * p0 \
        + (3 * (1 - u) * u ** 2 + u ** 3)[:, None] * p3
    a1 = (3 * (1 - u) ** 2 * u)[:, None] * t1
    a2 = (3 * (1 - u) * u ** 2)[:, None] * t2
    r = pts - b0
    c11 = float((a1 * a1).sum())
    c12 = float((a1 * a2).sum())
    c22 = float((a2 * a2).sum())
    x1 = float((a1 * r).sum())
    x2 = float((a2 * r).sum())
    det = c11 * c22 - c12 * c12
    chord = float(np.hypot(*(p3 - p0)))
    if abs(det) < 1e-12:
        al1 = al2 = chord / 3.0
    else:
        al1 = (x1 * c22 - x2 * c12) / det
        al2 = (x2 * c11 - x1 * c12) / det
    lim = chord * 1.5 + 1e-9
    al1 = float(np.clip(al1, 0.0, lim))
    al2 = float(np.clip(al2, 0.0, lim))
    c1 = p0 + al1 * t1
    c2 = p3 + al2 * t2
    err = float(np.hypot(*(_eval_cubic(p0, c1, c2, p3, u) - pts).T).max())
    return (p0, c1, c2, p3), err


def _fit_segment(pts: np.ndarray, t1, t2, tol: float, depth: int = 0):
    if len(pts) < 2:
        return []
    if len(pts) == 2:
        return [(pts[0], pts[0], pts[1], pts[1])]
    bez, err = _fit_cubic_ls(pts, t1, t2)
    if err <= tol or depth >= 8 or len(pts) <= 3:
        return [bez]
    u = _chord_param(pts)
    p0, c1, c2, p3 = bez
    i = int(np.argmax(np.hypot(*(_eval_cubic(p0, c1, c2, p3, u) - pts).T)))
    i = min(max(i, 1), len(pts) - 2)
    tc = _unit(pts[i - 1] - pts[i + 1])          # tangent at i pointing into the curve
    return (_fit_segment(pts[:i + 1], t1, tc, tol, depth + 1)
            + _fit_segment(pts[i:], -tc, t2, tol, depth + 1))


def _corner_indices(pts: np.ndarray, k: int, ang_tol_deg: float):
    n = len(pts)
    if n < 2 * k + 2:
        return []
    v1 = pts - np.roll(pts, k, axis=0)
    v2 = np.roll(pts, -k, axis=0) - pts
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    dot = (v1 * v2).sum(1)
    ang = np.degrees(np.abs(np.arctan2(cross, dot)))
    return [int(i) for i in np.nonzero(ang > ang_tol_deg)[0]]


def fit_bezier_segments(pts: np.ndarray, closed: bool, tol: float = 0.25,
                        corner_deg: float = 62.0, corner_win: int = 4):
    """Subpixel contour → list of Bezier segments [(p0, c1, c2, p3), ...] (in contour order)."""
    pts = np.asarray(pts, float)
    if closed and len(pts) > 2 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    n = len(pts)
    if n < 2:
        return []
    cor = sorted(set(_corner_indices(pts, corner_win, corner_deg)))
    if closed:
        if len(cor) < 2:
            cor = [0, n // 2]           # fully smooth closed contour: two cuts are enough
        bounds = cor + [cor[0] + n]
        subs = [pts[np.arange(a, b + 1) % n] for a, b in zip(bounds[:-1], bounds[1:])]
    else:
        cor = sorted(set([0] + cor + [n - 1]))
        subs = [pts[a:b + 1] for a, b in zip(cor[:-1], cor[1:])]
    segs = []
    for sub in subs:
        if len(sub) < 2:
            continue
        t1 = _unit(sub[1] - sub[0])
        t2 = _unit(sub[-2] - sub[-1])
        segs.extend(_fit_segment(sub, t1, t2, tol))
    return segs


def fit_bezier_d(pts: np.ndarray, closed: bool, tol: float = 0.25,
                 corner_deg: float = 62.0, corner_win: int = 4,
                 nd: int = 1, straight_tol: float = 0.12) -> str:
    """Subpixel contour polyline → conformal Bezier SVG path string."""
    segs = fit_bezier_segments(pts, closed, tol, corner_deg, corner_win)
    if not segs:
        return ""
    out = [f"M{fnum(segs[0][0][0], nd)} {fnum(segs[0][0][1], nd)}"]
    cur = segs[0][0]
    for p0, c1, c2, p3 in segs:
        chord = p3 - p0
        L = float(np.hypot(*chord))
        if L < 1e-9:
            continue
        d1 = abs(_cross2(chord, c1 - p0)) / L
        d2 = abs(_cross2(chord, c2 - p0)) / L
        if max(d1, d2) <= straight_tol:
            out.append(f"l{fnum(p3[0] - cur[0], nd)} {fnum(p3[1] - cur[1], nd)}")
        else:
            out.append("c" + " ".join(fnum(v, nd) for v in (
                c1[0] - cur[0], c1[1] - cur[1],
                c2[0] - cur[0], c2[1] - cur[1],
                p3[0] - cur[0], p3[1] - cur[1])))
        cur = p3
    return "".join(out) + ("z" if closed else "")


# ======================================================================
# 1. structure tensor
# ======================================================================
def structure_tensor(rgb: np.ndarray, sigma_d: float, sigma_i: float,
                     color: bool = True) -> dict:
    """Multi-channel (Di Zenzo) structure tensor and its eigendecomposition.

    Returns l1, l2, coh, energy(=l1), gx/gy (principal eigenvector = gradient direction),
        tx/ty (minor eigenvector = stroke/isophote direction)
    """
    chans = [rgb[..., i] for i in range(3)] if color else [rgb.mean(2)]
    jxx = np.zeros(rgb.shape[:2], np.float64)
    jyy = np.zeros_like(jxx)
    jxy = np.zeros_like(jxx)
    for c in chans:
        gx = ndi.gaussian_filter(c, sigma_d, order=(0, 1), mode="nearest")
        gy = ndi.gaussian_filter(c, sigma_d, order=(1, 0), mode="nearest")
        jxx += ndi.gaussian_filter(gx * gx, sigma_i, mode="nearest")
        jyy += ndi.gaussian_filter(gy * gy, sigma_i, mode="nearest")
        jxy += ndi.gaussian_filter(gx * gy, sigma_i, mode="nearest")
    n_ch = len(chans)
    jxx /= n_ch
    jyy /= n_ch
    jxy /= n_ch

    tr = jxx + jyy
    dif = jxx - jyy
    tmp = np.sqrt(dif * dif + 4.0 * jxy * jxy)
    l1 = 0.5 * (tr + tmp)
    l2 = 0.5 * (tr - tmp)

    # Principal eigenvector: of the two candidates take the one with the larger magnitude, to avoid degeneracy
    ax, ay = jxy, l1 - jxx
    bx, by = l1 - jyy, jxy
    use_b = np.hypot(bx, by) > np.hypot(ax, ay)
    vx = np.where(use_b, bx, ax)
    vy = np.where(use_b, by, ay)
    nrm = np.hypot(vx, vy)
    nrm[nrm < 1e-12] = 1.0
    vx /= nrm
    vy /= nrm

    luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    coh = (l1 - l2) / (l1 + l2 + EPS)
    return {"l1": l1, "l2": l2, "coh": coh, "energy": l1, "luma": luma,
            "gx": vx, "gy": vy, "tx": -vy, "ty": vx}


# ======================================================================
# 2. region segmentation
# ======================================================================
def _kmeans(feats: np.ndarray, k: int, iters: int, rng,
            fit_sample: int = 150000, chunk: int = 200000, use_gpu: bool = False):
    """k-means++ initialization + Lloyd iterations.

    The centroids are fitted on a subsample (accurate enough and fast), then the full image is
    assigned once; the distance computation is blocked to bound peak memory.
    """
    n = len(feats)
    if n > fit_sample:
        sample = feats[np.sort(rng.choice(n, fit_sample, replace=False))]
    else:
        sample = feats

    idx = [int(rng.integers(len(sample)))]
    d2 = ((sample - sample[idx[0]]) ** 2).sum(1)
    for _ in range(k - 1):
        p = d2 / (d2.sum() + EPS)
        idx.append(int(rng.choice(len(sample), p=p)))
        d2 = np.minimum(d2, ((sample - sample[idx[-1]]) ** 2).sum(1))
    cen = sample[idx].copy()

    if use_gpu:
        try:
            return gpu.lloyd(sample, feats, cen, k, iters)
        except Exception as _e:
            print(f"      ! GPU k-means 失败({_e}), 退回 CPU", file=sys.stderr)

    def assign(f):
        """Expand (x-c)^2 into gemm form: an order of magnitude faster than broadcast subtraction and lighter on memory."""
        out = np.empty(len(f), np.int32)
        cn = (cen * cen).sum(1)
        for i in range(0, len(f), chunk):
            blk = f[i:i + chunk]
            d = cn[None, :] - 2.0 * (blk @ cen.T)
            out[i:i + chunk] = d.argmin(1)
        return out

    lab_sub = assign(sample)
    for _ in range(iters):
        cnt = np.bincount(lab_sub, minlength=k).astype(np.float64)
        new = np.zeros_like(cen)
        for c in range(cen.shape[1]):
            new[:, c] = np.bincount(lab_sub, weights=sample[:, c], minlength=k) / np.maximum(cnt, 1)
        cen = new
        lab_sub = assign(sample)
    return cen, assign(feats)


def _adjacent_pairs(labels: np.ndarray) -> np.ndarray:
    """All spatially adjacent label pairs (encoded as 1D integers and deduplicated with unique, tens of times faster than unique(axis=0))."""
    a = np.concatenate([labels[:, :-1].ravel(), labels[:-1, :].ravel()])
    b = np.concatenate([labels[:, 1:].ravel(), labels[1:, :].ravel()])
    m = a != b
    a, b = a[m].astype(np.int64), b[m].astype(np.int64)
    if a.size == 0:
        return np.zeros((0, 2), np.int64)
    n = int(labels.max()) + 1
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    codes = lo * n + hi
    # Bucket counting (O(pixels)) is more than ten times faster than sort-and-dedupe (O(N log N)); fall back to sorting when the label count is too large,
    # so that an n^2 histogram does not blow up memory.
    if n * n <= 20_000_000:
        hits = np.nonzero(np.bincount(codes))[0]
    else:
        hits = np.unique(codes)
    return np.stack([hits // n, hits % n], 1)


def rag_merge(labels: np.ndarray, rgb255: np.ndarray, thresh: float,
              passes: int = 4, protect_sat: float = 0.0) -> np.ndarray:
    """Greedily merge regions whose "average color distance between neighbours < thresh" (union-find), over several rounds.

    When protect_sat > 0, regions whose average saturation (rgb range) exceeds that value are
    protected and excluded from merging -- this keeps golden seam lines from being absorbed into
    grey-white regions, while thin anti-aliasing rings between neutral colors are still merged away.
    """
    labels = labels.astype(np.int32)
    rgb255 = np.asarray(rgb255, dtype=np.float64)
    for _ in range(passes):
        n = int(labels.max()) + 1
        flat = labels.ravel()
        cnt = np.bincount(flat, minlength=n).astype(np.float64)
        means = np.zeros((n, 3))
        for c in range(3):
            means[:, c] = np.bincount(flat, weights=rgb255[..., c].ravel(),
                                      minlength=n) / np.maximum(cnt, 1)
        prot = np.zeros(n, bool)
        if protect_sat > 0:
            sat = (rgb255.max(2) - rgb255.min(2)).ravel()
            prot = (np.bincount(flat, weights=sat, minlength=n)
                    / np.maximum(cnt, 1)) > protect_sat
        pairs = _adjacent_pairs(labels)
        if pairs.size == 0:
            break
        d = np.linalg.norm(means[pairs[:, 0]] - means[pairs[:, 1]], axis=1)
        order = np.argsort(d)
        parent = np.arange(n)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        merged = False
        for i in order:
            if d[i] > thresh:
                break
            ra, rb = find(int(pairs[i, 0])), find(int(pairs[i, 1]))
            if ra == rb:
                continue
            if protect_sat > 0 and (prot[ra] or prot[rb]):
                continue  # colored region: keep it as is
            parent[rb] = ra
            merged = True
        if not merged:
            break
        labels = parent[labels]
        _, inv = np.unique(labels, return_inverse=True)
        labels = inv.reshape(labels.shape).astype(np.int32)
    return labels


def segment_colors(rgb: np.ndarray, rgb255: np.ndarray, args, rng) -> np.ndarray:
    """Color quantization (Lab k-means) + RAG color merging → smooth region boundaries."""
    t = time.time()
    sm = ndi.gaussian_filter(rgb, (args.pre_smooth, args.pre_smooth, 0.0))
    lab = rgb2lab(np.clip(sm, 0, 1))
    feats = lab.reshape(-1, 3).astype(np.float64)
    cen, lab_img = _kmeans(feats, args.kmeans_k, args.kmeans_iters, rng,
                           use_gpu=gpu.enabled(args) and gpu.has_kmeans())
    labels = lab_img.reshape(rgb.shape[:2]).astype(np.int32)
    t1 = time.time()
    labels = rag_merge(labels, rgb255.astype(np.float64), args.thresh,
                       args.merge_passes, args.protect_sat)
    t2 = time.time()
    # the +1 avoids skimage.measure.label treating the 0 value in the input as background and discarding it
    labels = measure.label(labels + 1, connectivity=2).astype(np.int32) - 1
    t3 = time.time()
    # merge once more at the connected-component level: components of the same cluster that were split apart are merged back if their colors are nearly identical.
    # Key effect: anti-aliased transition bands are often quantized into "thin rings around shapes" (differing from the neighbouring region by only a few grey levels);
    # without merging them back, a bright line that the source image does not have gets drawn along the edge.
    if args.merge_thresh2 > 0:
        labels = rag_merge(labels, rgb255.astype(np.float64),
                           args.merge_thresh2, 3, args.protect_sat)
    t4 = time.time()
    if args.verbose:
        log(f"      · k-means(k={args.kmeans_k}) {t1-t:.1f}s → RAG 合并 "
              f"{t2-t1:.1f}s → 连通域拆分 {t3-t2:.1f}s → 连通域再合并 "
              f"{t4-t3:.1f}s, {labels.max()+1} 个色块")
    return labels


def segment_flat(rgb: np.ndarray, rgb255: np.ndarray, args, rng):
    """Fast path for flat-color images: coarse histogram dominant colors + LUT nearest color + connected components + RAG re-merging.

    Aimed at logos / icons / vector-style illustrations -- images with only a few solid colors
    plus one anti-aliased edge, where iterating k-means+RAG to approximate them is a waste: here
    we run only a few O(N) histogram and lookup passes, about 0.5~1s on 1254² (versus about 28s
    for the full route). Whether the fast path is worth taking is decided by the dominant-color
    coverage cov; if it falls below --flat-cov it returns None and hands back to the full route,
    so photos/oil paintings are not harmed. Fill colors and gradients are both resampled from the
    source image, so the bucket centers are used only for "grouping" and do not affect color
    accuracy.
    """
    t = time.time()
    sm = ndi.gaussian_filter(rgb, (args.pre_smooth, args.pre_smooth, 0.0))
    nb = int(np.clip(args.flat_nb, 3, 7))
    shift = 8 - nb
    q = np.clip(sm * 255.0, 0, 255).astype(np.uint8) >> shift
    key = ((q[..., 0].astype(np.int32) << (2 * nb))
           | (q[..., 1].astype(np.int32) << nb) | q[..., 2].astype(np.int32))
    nbin = 1 << (3 * nb)
    hist = np.bincount(key.ravel(), minlength=nbin)
    peaks = np.nonzero(hist >= max(64, int(args.flat_share * key.size)))[0]
    cov = float(hist[peaks].sum() / key.size) if len(peaks) else 0.0
    if len(peaks) < 2 or len(peaks) > 512 or cov < args.flat_cov:
        if args.verbose:
            log(f"      · 平色快路径不适用: 主色 {len(peaks)} 个, 覆盖率 {cov*100:.1f}% "
                  f"(需 ≥{args.flat_cov*100:.0f}%), 走完整路线")
        return None

    def bucket_rgb(b):
        return ((np.stack([(b >> (2 * nb)) & ((1 << nb) - 1),
                           (b >> nb) & ((1 << nb) - 1),
                           b & ((1 << nb) - 1)], 1).astype(np.float64) + 0.5)
                * (1 << shift) / 255.0)

    # Key: first merge "buckets with similar colors" into a palette -- do the merging at the class level in O(bucket count),
    # rather than on an O(pixel-pair) region adjacency graph (rag_merge takes 9~18s on 1254²,
    # measured to cost more than all the other fast-path steps combined). Anti-aliased transition colors are therefore absorbed nearby,
    # and the remaining connected-component splitting plus merge_small_regions is enough to finish up.
    cen0 = bucket_rgb(peaks)
    cen_list = []
    for i in np.argsort(-hist[peaks]):          # most populous buckets become centers first
        c = cen0[i]
        if not cen_list or np.sqrt(((np.asarray(cen_list) - c) ** 2).sum(1)).min() > args.flat_merge:
            cen_list.append(c)
    cen = np.asarray(cen_list)
    lut = np.empty(nbin, np.int32)
    allb = np.arange(nbin)
    for i in range(0, nbin, 8192):
        blk = bucket_rgb(allb[i:i + 8192])
        lut[i:i + 8192] = ((blk[:, None, :] - cen[None, :, :]) ** 2).sum(2).argmin(1)
    labels = lut[key]
    t1 = time.time()
    labels = measure.label(labels + 1, connectivity=2).astype(np.int32) - 1
    t2 = time.time()
    # Anti-aliased transition bands get quantized into "thin rings around shapes" and must be merged back into their neighbours, or the boundary gains
    # bright lines that the source image does not have. rag_merge now uses bincount + connected components, so at this scale it only takes ~1s.
    if args.merge_thresh2 > 0:
        labels = rag_merge(labels, rgb255.astype(np.float64),
                           args.merge_thresh2, 3, args.protect_sat)
    t3 = time.time()
    if args.verbose:
        log(f"      · 平色快路径: 主色 {len(peaks)} 个(覆盖 {cov*100:.1f}%) → "
              f"{len(cen)} 种调色板 → LUT {t1-t:.2f}s → 连通域 {t2-t1:.2f}s → "
              f"RAG 合并 {t3-t2:.2f}s → {labels.max()+1} 个色块")
    return labels


def edge_map(tf: dict, args):
    """Fine edges: non-maximum suppression along the gradient direction + hysteresis thresholding. Returns (edge boolean map, energy field)."""
    mag = np.hypot(tf["gx"], tf["gy"])
    ux, uy = tf["gx"] / (mag + EPS), tf["gy"] / (mag + EPS)
    e = tf["energy"]                      # λ1 = Di Zenzo gradient energy (already smoothed by σi)
    yy, xx = np.mgrid[0:e.shape[0], 0:e.shape[1]]
    a = ndi.map_coordinates(e, [yy + uy, xx + ux], order=1, mode="nearest")
    b = ndi.map_coordinates(e, [yy - uy, xx - ux], order=1, mode="nearest")
    thin = np.where((e >= a) & (e >= b), e, 0.0)   # non-maximum suppression along the gradient direction
    # The threshold is a quantile of the "energy field" (not of the suppressed thin: many pixels in thin are exactly 0,
    # so the quantile lands on 0, making either the whole image an edge or none of it an edge)
    hi = float(np.quantile(e, args.edge_nms_hi))
    lo = float(np.quantile(e, args.edge_nms_lo))
    edge = filters.apply_hysteresis_threshold(thin, lo, hi)
    frac = float(edge.mean())
    if frac <= 0.0 or frac > 0.4:
        print(f"      ! 警告: 边缘像素占比 {frac*100:.1f}% (阈值 hi={hi:.3g} lo={lo:.3g}) "
              f"不合理, 请调 --edge-nms-hi/--edge-nms-lo", file=sys.stderr)
        if frac <= 0.0:
            edge = thin > hi          # fallback: plain threshold
    return edge, e


def segment_edges(rgb: np.ndarray, tf: dict, args, rng) -> np.ndarray:
    """Pure edge route: edges act as barriers and a distance-field watershed cuts out flat regions (cleanest geometry, coarsest regions).

    1) derive fine edge lines from edge_map;
    2) use the "distance to edge" field d as terrain; inside flat regions (large d) take connected
       components as seeds, and run the watershed with -d as elevation: two seeds meet at the
       minimum of d, which is exactly at the edge.
    """
    t = time.time()
    from skimage.segmentation import watershed
    edge, _ = edge_map(tf, args)
    if args.edge_close > 0:
        edge = ndi.binary_closing(edge, np.ones((3, 3), bool),
                                  iterations=int(args.edge_close))
    d = ndi.distance_transform_edt(~edge)
    if args.edge_dsmooth > 0:
        d = ndi.gaussian_filter(d, args.edge_dsmooth)
    core = d > args.edge_core
    markers, n_mark = measure.label(core, connectivity=2, return_num=True)
    if n_mark < 2:
        print(f"      ! 警告: 只找到 {n_mark} 个平坦区种子, 分割会退化成整图一块; "
              f"请减小 --edge-core 或调 --edge-nms-lo", file=sys.stderr)
    labels = watershed(-d, markers).astype(np.int32) - 1
    if args.verbose:
        log(f"      · 边缘(非极大值抑制+滞后): 边缘像素 {edge.mean()*100:.2f}% → "
              f"种子 {n_mark} 个 → 分水岭 {time.time()-t:.1f}s")
    return labels


def texture_map(rgb: np.ndarray, sigma: float = 2.5, smooth: float = 6.0):
    """Local high-frequency texture strength: local energy of the high-pass residual (large where there are many strokes/details)."""
    lo = ndi.gaussian_filter(rgb, (sigma, sigma, 0.0))
    res = rgb - lo
    return ndi.gaussian_filter((res * res).sum(-1), smooth)


def segment_hybrid(rgb: np.ndarray, rgb255: np.ndarray, tf: dict, args, rng):
    """Partition routing: color segmentation decides "region identity", edge detection decides "boundary position".

    The method is marker-controlled watershed: seeds = the eroded cores of the color regions
    (region identity/count unchanged), elevation = normalized gradient energy + edge-line bonus.
    A flood can only take the lowest-energy path, so where two floods meet is a gradient ridge
    = edge → the boundary snaps onto the real edge and closes automatically, with none of the
    1~2px offset caused by regions competing. Where there are strokes/gradients the gradient is
    diffuse, the flood finds no clear ridge and can only stop near the color boundary, which is
    equivalent to the original behavior.

    Returns (labels, geo_px): geo_px marks "solid-color + geometric" pixels (both texture and
    color spread small), used to choose the curve-fitting method per region (geometric regions use
    conformal Beziers).
    """
    t = time.time()
    from skimage.segmentation import watershed
    labels_c = segment_colors(rgb, rgb255, args, rng)
    labels_c = merge_small_regions(labels_c, rgb, args.min_area,
                                   core_radius=max(0, (args.min_width - 1) // 2),
                                   protect_sat=args.protect_sat)
    n = int(labels_c.max()) + 1
    areas = np.bincount(labels_c.ravel(), minlength=n)
    # ---- first judge "geometricity" per color region: high-frequency texture and interior color spread both small -> solid-color/geometric region ----
    geo_r, tex_r, rng_r, _tn = classify_regions(
        rgb, labels_c, areas, args.tex_sigma, args.tex_smooth, args.tex_norm,
        args.tex_thr, args.range_thr, args.min_area)
    geo_px = geo_r[labels_c]
    if args.tex_open > 0:
        geo_px = ndi.binary_opening(geo_px, np.ones((3, 3), bool),
                                    iterations=args.tex_open)
    if args.tex_close > 0:
        geo_px = ndi.binary_closing(geo_px, np.ones((3, 3), bool),
                                    iterations=args.tex_close)
    # ---- boundary snapping: only in "solid-color/geometric" regions; stroke/gradient regions keep their original color boundary ----
    edge, e = edge_map(tf, args)
    n_moved = 0
    if args.snap_mode == "watershed":
        elev = e / (float(e.max()) + EPS)
        if args.snap_w > 0:
            elev = elev + args.snap_w * ndi.gaussian_filter(edge.astype(np.float64),
                                                            args.snap_sigma)
        markers = np.zeros(labels_c.shape, np.int32)
        r = int(args.snap_erode)
        for li in range(n):
            m = labels_c == li
            if not m.any():
                continue
            core = ndi.binary_erosion(m, iterations=r) if r > 0 else m
            if not core.any():
                core = m
            markers[core] = li + 1
        labels_ws = watershed(elev, markers, watershed_line=False).astype(np.int32) - 1
        if args.snap_band > 0:
            # The boundary may move only within ±snap_band pixels of the color boundary: close enough to snap to the real edge,
            # yet not enough for "another nearby edge" to drag it away (region topology stays unchanged)
            cb = find_boundaries(labels_c, mode="inner")
            move = ndi.distance_transform_edt(~cb) <= args.snap_band
        else:
            move = np.ones(labels_c.shape, bool)
        labels = np.where(move & geo_px, labels_ws, labels_c).astype(np.int32)
        n_moved = int((labels != labels_c).sum())
    else:
        labels = labels_c
    if args.verbose:
        keep = list(np.nonzero(areas >= args.min_area)[0])
        info = ", ".join(f"#{li}:{'几何' if geo_r[li] else '复杂'}"
                         f"(纹理{tex_r[li]:.1e}/色跨{rng_r[li]:.3f})" for li in keep)
        log(f"      · 分区路由: 边缘 {edge.mean()*100:.2f}% | 吸附方式 {args.snap_mode}"
              f" | 几何区像素 {geo_px.mean()*100:.1f}%"
              + (f", 边界移动 {n_moved} 像素" if args.snap_mode == "watershed" else ""))
        print(f"        [{info}] (几何判据: 纹理<{args.tex_thr} 且 色跨<{args.range_thr})"
              f" | {time.time()-t:.1f}s")
    return labels, geo_px


def merge_small_regions(labels: np.ndarray, rgb: np.ndarray, min_area: int,
                        core_radius: int = 0, max_pass: int = 8,
                        protect_sat: float = 0.0) -> np.ndarray:
    """Merge regions with area < min_area into the nearest large region (a single EDT, O(N)).

    With core_radius > 0 it first performs a "label opening": use min/max filtering to find each
    region's core (pixels whose neighbourhood holds a single label), then relabel by nearest core
    -- this eats the thin rings/thin tails that quantization produces in anti-aliased transition
    bands. Note that it also shrinks genuinely thin, long features (such as golden seam lines), so
    it is off by default; in this codebase thin rings are instead removed by "RAG color merging at
    the connected-component level".
    """
    labels = labels.astype(np.int32)
    if core_radius > 0:
        size = 2 * core_radius + 1
        n = int(labels.max()) + 1
        areas = np.bincount(labels.ravel(), minlength=n)
        uni = (ndi.minimum_filter(labels, size=size)
               == ndi.maximum_filter(labels, size=size))
        has_core = np.zeros(n, bool)
        has_core[labels[uni]] = True
        big = areas >= min_area
        core = (uni & big[labels]) | (big & ~has_core)[labels]
        if core.any() and not core.all():
            ind = ndi.distance_transform_edt(~core, return_distances=False,
                                             return_indices=True)
            labels = labels[ind[0], ind[1]]
    # Colorful regions with high average saturation (golden seam lines) are kept even when small
    rgb255 = np.asarray(rgb, dtype=np.float64) * 255.0
    for _ in range(max_pass):
        n = int(labels.max()) + 1
        flat = labels.ravel()
        areas = np.bincount(flat, minlength=n)
        keep = areas >= min_area
        if protect_sat > 0:
            sat = (rgb255.max(2) - rgb255.min(2)).ravel()
            cnt = np.maximum(areas, 1).astype(np.float64)
            keep |= (np.bincount(flat, weights=sat, minlength=n) / cnt) > protect_sat
        if not keep.any():
            break
        big_px = keep[labels]
        if big_px.all():
            break
        ind = ndi.distance_transform_edt(~big_px, return_distances=False,
                                         return_indices=True)
        new = labels[ind[0], ind[1]]
        if np.array_equal(new, labels):
            break
        labels = new
    _, inv = np.unique(labels, return_inverse=True)
    return inv.reshape(labels.shape).astype(np.int32)


# ======================================================================
# 3. per-region gradient fitting (the structure tensor supplies the gradient axis)
# ======================================================================
def _ramp_fit(xs, ys, cols, u, n_stops, nb=8):
    """Bin along the unit axis u and take the median color → gradient stops; also reports the flat-color error and the along-axis residual.

    The criterion is "error gain" = 1 - along-axis residual / flat-color residual, i.e. how much of
    the color variance this 1D gradient removes; it is more robust than explained variance (it maps
    directly to the final fill error).
    """
    cx, cy = float(xs.mean()), float(ys.mean())
    t = (xs - cx) * u[0] + (ys - cy) * u[1]
    lo, hi = np.percentile(t, 1.0), np.percentile(t, 99.0)
    if hi - lo < 4.0:
        return None
    tn = np.clip((t - lo) / (hi - lo), 0.0, 1.0)
    stops = []
    for i in range(n_stops):
        a = i / n_stops
        sel = (tn >= a) & (tn < (i + 1) / n_stops) if i < n_stops - 1 else (tn >= a)
        if sel.sum() < 15:
            continue
        stops.append(((a + (i + 1) / n_stops) * 0.5, np.median(cols[sel], axis=0)))
    if len(stops) < 2:
        return None
    arr = np.array([s[1] for s in stops])
    rng_col = float(np.abs(arr.max(0) - arr.min(0)).max())

    err_flat = float(((cols - cols.mean(0)) ** 2).sum())
    err_axis = 0.0
    for i in range(nb):
        sel = (tn >= i / nb) & (tn < (i + 1) / nb)
        if sel.sum() > 15:
            seg = cols[sel]
            err_axis += float(((seg - np.median(seg, axis=0)) ** 2).sum())
    gain = 1.0 - err_axis / (err_flat + EPS)
    return {"stops": stops, "range": rng_col, "gain": gain,
            "lo": float(lo), "hi": float(hi), "cx": cx, "cy": cy}


# Accept/reject threshold for "flat color vs linear gradient": a fitted gradient must remove at
# least this fraction of the squared error. Raised/lowered at runtime from --grad-min-gain, or
# automatically by --auto-gradient on smooth (gradient-heavy) images. Module-level so every call
# site shares one value without threading an extra argument through.
GRAD_MIN_GAIN = 0.12

# A radial candidate must remove at least this much MORE of the squared error than the best linear
# candidate before it is preferred (relative margin on the residual 1-gain). A few percent keeps
# linear gradients for genuinely linear ramps and switches to radial only for real centre/radius
# shading (vignettes, glossy highlights). Overridable per call from --grad-radial-margin.
GRAD_RADIAL_MARGIN = 0.05


def _radial_fit(xs, ys, cols, c, n_stops, nb=8):
    """Fit a radial (center + radius) gradient: bin the pixels by distance to the center c and take the median color per bin.

    The criterion is the SAME "error gain" as _ramp_fit (1 - along-radius residual / flat-color
    residual), so the radial and linear candidates are directly comparable. The returned stop
    offsets are expressed as fractions of the radius r (= the 99th percentile of |p-c|), so that
    predict_region() and the SVG writer agree exactly on where each stop sits.
    """
    t = np.hypot(xs - float(c[0]), ys - float(c[1]))
    lo, hi = np.percentile(t, 1.0), np.percentile(t, 99.0)
    if hi - lo < 2.0:                       # (almost) constant radius: no radial ramp to speak of
        return None
    tn = np.clip((t - lo) / (hi - lo), 0.0, 1.0)
    stops = []
    for i in range(n_stops):
        a = i / n_stops
        sel = (tn >= a) & (tn < (i + 1) / n_stops) if i < n_stops - 1 else (tn >= a)
        if sel.sum() < 15:
            continue
        tc = (a + (i + 1) / n_stops) * 0.5  # bin centre in normalized radius
        stops.append(((lo + tc * (hi - lo)) / hi, np.median(cols[sel], axis=0)))
    if len(stops) < 2:
        return None
    arr = np.array([s[1] for s in stops])
    rng_col = float(np.abs(arr.max(0) - arr.min(0)).max())

    err_flat = float(((cols - cols.mean(0)) ** 2).sum())
    err_axis = 0.0
    for i in range(nb):
        sel = (tn >= i / nb) & (tn < (i + 1) / nb)
        if sel.sum() > 15:
            seg = cols[sel]
            err_axis += float(((seg - np.median(seg, axis=0)) ** 2).sum())
    gain = 1.0 - err_axis / (err_flat + EPS)
    return {"stops": stops, "range": rng_col, "gain": gain,
            "lo": float(lo), "hi": float(hi), "r": float(hi),
            "cx": float(c[0]), "cy": float(c[1])}


def _radial_centers(xs, ys, w, gx, gy, bbox):
    """Candidate radial centers for a region (deterministic, no rng).

      (1) region centroid,
      (2) bounding-box center,
      (3) least-squares intersection of the local gradient lines -- for a truly radial colour field
          every local gradient direction points at/away from the center, so the center minimizes the
          weighted squared distance to those lines. The normal matrix of a linear field is (nearly)
          rank one, so a badly conditioned system is rejected instead of producing a far-away center.

    xs/ys are absolute image coordinates and w/gx/gy are the gathered structure-tensor energy and
    gradient components at those pixels (so this works both for a windowed mask and for the sampled
    pixels used by the merge pass).
    The caller fits all candidates and keeps the best gain, so extra candidates can only improve the
    fit; the order is fixed, and ties keep the earlier (geometric) candidate.
    """
    cx, cy = float(xs.mean()), float(ys.mean())
    y0, y1, x0, x1 = bbox
    ext = float(np.hypot(x1 - x0, y1 - y0)) + 1.0     # region extent, for a sanity bound
    out = [(cx, cy),
           (0.5 * (float(x0) + float(x1)), 0.5 * (float(y0) + float(y1)))]
    sel = w > 1e-9
    if int(sel.sum()) >= 100:
        gxx, gyy = gx[sel], gy[sel]
        px = xs[sel].astype(np.float64)
        py = ys[sel].astype(np.float64)
        ww = w[sel]
        nrm = np.hypot(gxx, gyy) + EPS
        ax, ay = gyy / nrm, -gxx / nrm                  # normal of the gradient line
        d = -(px * ax + py * ay)
        a11 = float((ww * ax * ax).sum())
        a12 = float((ww * ax * ay).sum())
        a22 = float((ww * ay * ay).sum())
        tr = a11 + a22
        det = a11 * a22 - a12 * a12
        # rank-1 (linear field) => det ~ 0; require a strongly conditioned 2x2 system
        if tr > EPS and det > 1e-3 * tr * tr:
            b1 = -float((ww * ax * d).sum())
            b2 = -float((ww * ay * d).sum())
            qx = (b1 * a22 - b2 * a12) / det
            qy = (a11 * b2 - a12 * b1) / det
            if (np.isfinite(qx) and np.isfinite(qy)
                    and np.hypot(qx - cx, qy - cy) <= 3.0 * ext + 8.0):
                out.append((float(qx), float(qy)))
    return out


def _fit_core(xs, ys, cols, w, gx, gy, n_stops, min_range,
              allow_radial=False, radial_margin=None):
    """Linear/radial gradient selection over already gathered per-pixel samples.

    xs/ys: absolute image coordinates (float or int), cols: (n,3) colours in 0..1,
    w/gx/gy: structure-tensor energy and gradient components at those pixels.
    Shared by fit_region_gradient (windowed region mask) and merge_gradient_regions (deterministic
    subsample per region), so both take exactly the same decisions for the same samples.
    """
    n = len(xs)
    if n < 200:
        return None
    axes = []
    # (1) color-field PCA
    P = np.stack([xs - xs.mean(), ys - ys.mean()], 1).astype(np.float64)
    C = cols - cols.mean(0)
    M = (P.T @ C) / len(P)
    _, V = np.linalg.eigh(M @ M.T)
    axes.append(("pca", V[:, -1]))
    # (2) structure-tensor mean gradient direction
    g = np.array([float((gx * w).mean()), float((gy * w).mean())])
    if np.hypot(*g) > 1e-9:
        axes.append(("tensor", g / np.hypot(*g)))

    best = None
    for name, u in axes:
        u = u / (np.hypot(*u) + EPS)
        r = _ramp_fit(xs, ys, cols, u, n_stops)
        if r is None:
            continue
        if best is None or r["gain"] > best[1]["gain"]:
            best = (name, r, u)
    if best is None:
        return None
    name, r, u = best
    # ---- radial candidate (center + radius, two endpoint colours at minimum): it reuses the SAME
    # error/gain criterion as the linear fit and is preferred only when its residual is better than
    # the best linear residual by a relative margin (and it passes the same acceptance gate);
    # otherwise the linear fit is kept. ----
    if allow_radial:
        rad = None
        for c in _radial_centers(xs, ys, w, gx, gy,
                                 (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max()))):
            rr = _radial_fit(xs, ys, cols, c, n_stops)
            if rr is not None and (rad is None or rr["gain"] > rad["gain"]):
                rad = rr
        if rad is not None:
            _mg = GRAD_RADIAL_MARGIN if radial_margin is None else float(radial_margin)
            better = (1.0 - rad["gain"]) < (1.0 - r["gain"]) * (1.0 - _mg)
            if better and rad["range"] >= min_range and rad["gain"] >= GRAD_MIN_GAIN:
                return {"kind": "radial", "cx": rad["cx"], "cy": rad["cy"], "r": rad["r"],
                        "stops": rad["stops"], "quality": rad["gain"], "axis": "radial",
                        "range": rad["range"]}
    # flat color vs linear gradient: the gradient must remove at least 12% of the squared error and span a wide enough color range, otherwise flat color is cheaper
    if r["range"] < min_range or r["gain"] < GRAD_MIN_GAIN:
        return None
    return {"p0": (r["cx"] + u[0] * r["lo"], r["cy"] + u[1] * r["lo"]),
            "p1": (r["cx"] + u[0] * r["hi"], r["cy"] + u[1] * r["hi"]),
            "stops": r["stops"], "quality": r["gain"], "axis": name,
            "u": (float(u[0]), float(u[1])), "lo": r["lo"], "hi": r["hi"],
            "cx": r["cx"], "cy": r["cy"]}


def fit_region_gradient(rgb: np.ndarray, mask: np.ndarray, tensor: dict,
                        n_stops: int, min_range: float, off=(0, 0),
                        allow_radial: bool = False, radial_margin=None):
    """Fit a gradient to a region: a linear one by default, plus a radial candidate when allow_radial.

    There are two candidate gradient axes; take the one with the better explained variance:
      (1) color-field PCA / least squares                        —— globally optimal linear direction, never degenerates
      (2) structure-tensor energy-weighted mean gradient ∇I      —— the physical direction given by local edges
    When allow_radial is set, the best radial (center + radius) candidate is scored with the same
    error/gain criterion and returned as {"kind": "radial", ...} when its residual beats the linear
    residual by radial_margin (default GRAD_RADIAL_MARGIN); otherwise the linear result is kept.
    Returning None means the region is handled as flat color.

    off=(dy, dx): mask is a window sub-image. Here pixel indices are converted into **absolute
    image coordinates**, and the full image's rgb / tensor are still used for indexing (an
    O(region pixel count) gather, not an O(H*W) mask), so the order and values of the pixels
    obtained -- and every subsequent floating-point operation -- are bit-identical to the
    full-image version.
    """
    ys, xs = np.nonzero(mask)
    n = ys.size
    if n < 200:
        return None
    ys = ys + int(off[0])
    xs = xs + int(off[1])
    rng = np.random.default_rng(0)
    if n > 60000:
        sel = rng.choice(n, 60000, replace=False)
        ys, xs = ys[sel], xs[sel]
    cols = rgb[ys, xs].astype(np.float64)
    w = tensor["energy"][ys, xs]
    gx, gy = tensor["gx"][ys, xs], tensor["gy"][ys, xs]
    return _fit_core(xs, ys, cols, w, gx, gy, n_stops, min_range,
                     allow_radial=allow_radial, radial_margin=radial_margin)


def predict_region(rgb_shape, mask, grad, off=(0, 0)):
    """Predict pixel colors from the per-region base fill (linear/radial gradient or flat color); returns (rows, cols, pred).

    off=(dy, dx): mask is a window sub-image and the returned rows/cols are **absolute image
    coordinates** (the caller uses them to index act/rgb), so grad must also be in absolute image
    coordinates.
    """
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return rows, cols, np.zeros((0, 3))
    rows = rows + int(off[0])
    cols = cols + int(off[1])
    if grad is None:
        return rows, cols, None  # handled as flat color by the caller
    if grad.get("kind") == "radial":
        # Radial fill: |p-c| / r, with the stop offsets already expressed as radius fractions
        # (see _radial_fit), so this matches the emitted <radialGradient> exactly.
        t = np.hypot(cols - grad["cx"], rows - grad["cy"])
        tn = np.clip(t / (grad["r"] + EPS), 0.0, 1.0)
        off = np.array([o for o, _ in grad["stops"]])
        cols_arr = np.array([c for _, c in grad["stops"]])
        pred = np.stack([np.interp(tn, off, cols_arr[:, k]) for k in range(3)], 1)
        return rows, cols, pred
    ux, uy = grad["u"]
    t = (cols - grad["cx"]) * ux + (rows - grad["cy"]) * uy
    tn = np.clip((t - grad["lo"]) / (grad["hi"] - grad["lo"] + EPS), 0.0, 1.0)
    off = np.array([o for o, _ in grad["stops"]])
    cols_arr = np.array([c for _, c in grad["stops"]])
    pred = np.stack([np.interp(tn, off, cols_arr[:, k]) for k in range(3)], 1)
    return rows, cols, pred


# ======================================================================
# 3.5 gradient-aware region merging (spatial merging of colour patches)
# ======================================================================
# Pixels sampled per region by the merge pass. The merge decision only compares "one gradient over
# the union" against "the two separate fills", and both sides are scored on exactly the same sample
# pixels, so a bounded sample is enough (and makes the pass O(#pairs) instead of O(area)).
MERGE_SAMPLES = 2000


def _grad_pred(grad, xs, ys):
    """Predicted colours of a fitted fill at the sample points; None means flat color."""
    if grad is None:
        return None
    if grad.get("kind") == "radial":
        t = np.hypot(xs - grad["cx"], ys - grad["cy"])
        tn = np.clip(t / (grad["r"] + EPS), 0.0, 1.0)
    else:
        ux, uy = grad["u"]
        t = (xs - grad["cx"]) * ux + (ys - grad["cy"]) * uy
        tn = np.clip((t - grad["lo"]) / (grad["hi"] - grad["lo"] + EPS), 0.0, 1.0)
    offs = np.array([o for o, _ in grad["stops"]])
    cst = np.array([c for _, c in grad["stops"]])
    return np.stack([np.interp(tn, offs, cst[:, k]) for k in range(3)], 1)


def _sample_mse(cols, grad, xs, ys) -> float:
    """Mean squared error of one region's fill over its sample (flat median when there is no fit)."""
    pred = _grad_pred(grad, xs, ys)
    if pred is None:
        pred = np.median(cols, axis=0)[None, :]
    return float(((cols - pred) ** 2).mean())


def merge_gradient_regions(rgb: np.ndarray, labels: np.ndarray, tensor: dict, args):
    """Merge adjacent regions whose union is still well explained by ONE gradient (linear or radial).

    k-means in colour space + connected components make region boundaries hard steps by construction:
    on smooth/glossy images the patches stay visible even when every patch is individually filled
    with its own gradient. This pass re-merges such patches in SPACE. For every pair of 4-connected
    regions it fits a single gradient over the union of the two samples (same gate and same
    linear/radial choice as the normal per-region fit) and accepts the merge when

      * the union fit is accepted by the gradient gate (GRAD_MIN_GAIN / --grad-min-range), and
      * the union's mean squared error is at most (1 + --merge-grad-tol) times the area-weighted mean
        of the two regions' own squared errors -- one gradient explains the union nearly as well as
        the two fills did.

    The second test is what protects hard-edged artwork: two flat regions have a (near) zero separate
    error, so a gradient that merely bridges their step is rejected even though it "fits" the union.
    Scan order is fixed (pairs sorted by (min,max) label), each label is merged at most once per pass,
    and the only rng use is the per-label sample draw, seeded by the label index -- so the result is
    deterministic for a given input and parameter set.

    Returns (labels, n_before, n_after) with the relabelled, gap-free label image.
    """
    labels = labels.astype(np.int32)
    H, W = labels.shape
    passes = max(1, int(getattr(args, "merge_grad_passes", 4)))
    tol = float(getattr(args, "merge_grad_tol", 0.05))
    allow_radial = bool(getattr(args, "grad_radial", False))
    margin = float(getattr(args, "grad_radial_margin", GRAD_RADIAL_MARGIN))
    n_stops = int(args.grad_stops)
    min_range = float(args.grad_min_range)
    # The background label becomes one flat fill later; never absorb it into a gradient region.
    border = np.zeros((H, W), bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    n_before = int(labels.max()) + 1
    bg = int(np.argmax(np.bincount(labels[border].ravel())))
    # Label ids stay STABLE across passes (merged-away ids simply disappear, leaving gaps that are
    # only renumbered once at the end). That makes both caches below sound: a label whose pixel set
    # did not change has the same sample and the same individual fit, so a pair rejected in an
    # earlier pass cannot have become acceptable unless one of its two labels changed.
    cache = {}                                  # label id -> sample tuple (persists across passes)
    rejected = set()                            # pair -> already refused; persists while both are unchanged
    changed = None                              # ids whose pixel set changed in the previous pass
    n = n_before
    for _p in range(passes):
        pairs = _adjacent_pairs(labels)
        if pairs.size == 0:
            break
        if changed is not None:
            rejected = {k for k in rejected if k[0] not in changed and k[1] not in changed}
            for li in changed:
                cache.pop(li, None)
        areas = np.bincount(labels.ravel(), minlength=n).astype(np.float64)
        objs = ndi.find_objects(labels + 1)     # index k corresponds to label k; vanished ids give None

        def info(li):
            """(xs, ys, cols, en, gx, gy, grad, mse) from the label's deterministic sample; cached."""
            e = cache.get(li)
            if e is not None:
                return e
            sl = objs[li] if 0 <= li < len(objs) else None
            if sl is None:
                e = (None, None, None, None, None, None, None, 0.0)
            else:
                y0 = max(0, int(sl[0].start) - 1)
                y1 = min(H, int(sl[0].stop) + 1)
                x0 = max(0, int(sl[1].start) - 1)
                x1 = min(W, int(sl[1].stop) + 1)
                yy, xx = np.nonzero(labels[y0:y1, x0:x1] == li)
                yy = yy + y0
                xx = xx + x0
                if yy.size > MERGE_SAMPLES:
                    sel = np.random.default_rng(li).choice(yy.size, MERGE_SAMPLES, replace=False)
                    yy, xx = yy[sel], xx[sel]
                cc = rgb[yy, xx].astype(np.float64)
                gx = tensor["gx"][yy, xx]
                gy = tensor["gy"][yy, xx]
                en = tensor["energy"][yy, xx]
                g = _fit_core(xx, yy, cc, en, gx, gy, n_stops, min_range,
                              allow_radial=allow_radial, radial_margin=margin)
                e = (xx, yy, cc, en, gx, gy, g, _sample_mse(cc, g, xx, yy))
            cache[li] = e
            return e

        done, acc = set(), []
        for a, b in pairs.tolist():
            if a == bg or b == bg or a in done or b in done:
                continue
            if (a, b) in rejected:               # unchanged pair: same decision as before
                continue
            xa, ya, ca, ena, gxa, gya, _ga, ea = info(a)
            xb, yb, cb, enb, gxb, gyb, _gb, eb = info(b)
            if xa is None or xb is None:
                continue
            # Expected per-pixel error if the two regions keep their own fills, area-weighted.
            base = (areas[a] * ea + areas[b] * eb) / (areas[a] + areas[b] + EPS)
            xs_u = np.concatenate([xa, xb])
            ys_u = np.concatenate([ya, yb])
            cu = np.concatenate([ca, cb])
            gu = None
            if base > 1e-12:                     # both fills already perfect -> nothing to gain
                gu = _fit_core(xs_u, ys_u, cu,
                               np.concatenate([ena, enb]), np.concatenate([gxa, gxb]),
                               np.concatenate([gya, gyb]),
                               n_stops, min_range, allow_radial=allow_radial, radial_margin=margin)
            if gu is None:
                rejected.add((a, b))             # not gradient-explainable as one region -> keep both
                continue
            eu = _sample_mse(cu, gu, xs_u, ys_u)
            if eu <= base * (1.0 + tol) + 1e-12:
                acc.append((a, b))
                done.add(a)
                done.add(b)
            else:
                rejected.add((a, b))
        if args.verbose and (len(pairs) or acc):
            log(f"        · 合并第{_p + 1}轮: {len(pairs)} 个邻接对 → {len(acc)} 次合并")
        if not acc:
            break
        parent = np.arange(n)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for a, b in acc:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
        labels = parent[labels]
        changed = {x for pair in acc for x in pair}
    _, inv = np.unique(labels, return_inverse=True)
    labels = inv.reshape(labels.shape).astype(np.int32)
    return labels, n_before, int(labels.max()) + 1


# ======================================================================
# 3.6 error-driven adaptive refinement
# ======================================================================
# "Segment, then fit each region once" cannot represent a smooth 2-D colour field: whatever the
# thresholds, the pieces that are still too coarse keep a visible error (a flat patch, or a step at
# a boundary), which is exactly the banding the user sees on glossy/rendered images. This stage does
# not tune the segmentation; it measures the reconstruction residual of every region's CHOSEN fill
# and splits the region where that residual is large, re-fits both halves with the normal
# flat/linear/radial machinery and recurses. A fixed budget of SVG elements therefore buys the best
# approximation instead of an arbitrary colour-quantisation partition.
#
# Residual definition: RMS of the per-pixel MAX-CHANNEL |observed - predicted| over the region's
# eroded interior (erosion 3), in 0..1 full-scale units. The anti-aliased rim is excluded on
# purpose: on hard-edged artwork the rim error is the source's own AA and must not be chased.
#
# Determinism: the region order, the split axis, the cut position, the child order and every fit are
# fixed functions of the pixel data (no rng anywhere), so the same input and flags give the same
# bytes. A split is accepted only when it measurably lowers the fill residual, which keeps the
# growth bounded and stops the stage from chasing sensor noise / texture.


def region_fill_error(rgb_s: np.ndarray, mask: np.ndarray, region: dict, off):
    """Per-pixel max-channel fill error over `mask`, in np.nonzero(mask) order."""
    rows, cols, pred = predict_region(rgb_s.shape, mask, region["grad"], off)
    if rows.size == 0:
        return np.zeros(0)
    obs = rgb_s[rows, cols]
    if pred is None:
        pred = np.asarray(region["col"], float)[None, :]
    return np.abs(obs - pred).max(1)


def _region_interior(mask: np.ndarray) -> np.ndarray:
    """Eroded core used for fitting / residual statistics; the mask itself when it is tiny."""
    inner = ndi.binary_erosion(mask, iterations=3)
    if int(inner.sum()) < 200:
        return mask
    return inner


def split_region_mask(mask: np.ndarray, err: np.ndarray, min_child: int):
    """Binary split of `mask` aimed at the error field; returns (a, b) boolean sub-masks or None.

    The cut is a straight line perpendicular to the error-weighted principal axis of the region,
    placed at the error-weighted median of the projection but clamped into the inter-quartile range
    of the unweighted projection, so neither child can degenerate into a sliver. `err` is aligned
    with np.nonzero(mask); the rim is expected to be zeroed by the caller.
    """
    ys, xs = np.nonzero(mask)
    n = int(ys.size)
    if n < 2 * min_child:
        return None
    mx, my = float(xs.mean()), float(ys.mean())
    dx = xs.astype(np.float64) - mx
    dy = ys.astype(np.float64) - my
    w = err.astype(np.float64) ** 2
    sxx = float((w * dx * dx).sum())
    syy = float((w * dy * dy).sum())
    sxy = float((w * dx * dy).sum())
    tr = sxx + syy
    det = sxx * syy - sxy * sxy
    disc = math.sqrt(max(0.0, 0.25 * tr * tr - det))
    l1 = 0.5 * tr + disc
    if tr <= 1e-18 or l1 <= 1e-18:
        # no usable error anisotropy: fall back to the geometric principal axis
        ux, uy = (1.0, 0.0) if float((dx * dx).sum()) >= float((dy * dy).sum()) else (0.0, 1.0)
    else:
        vx, vy = sxy, l1 - sxx
        if abs(vx) + abs(vy) < 1e-12:
            vx, vy = l1 - syy, sxy
        nrm = math.hypot(vx, vy)
        ux, uy = (vx / nrm, vy / nrm) if nrm > 0.0 else (1.0, 0.0)
    proj = dx * ux + dy * uy
    q25, q75 = (float(v) for v in np.percentile(proj, [25.0, 75.0]))
    if q75 - q25 < 1e-9:
        return None
    order = np.argsort(proj, kind="stable")
    cw = np.cumsum(w[order])
    if cw[-1] <= 0.0:
        t = 0.5 * (q25 + q75)                    # error is flat: split at the geometric centre
    else:
        k = int(np.searchsorted(cw, 0.5 * cw[-1], side="left"))
        t = float(proj[order[min(k, n - 1)]])
    t = min(max(t, q25), q75)                    # anti-sliver clamp
    sel = proj <= t
    a = np.zeros_like(mask)
    b = np.zeros_like(mask)
    a[ys[sel], xs[sel]] = True
    b[ys[~sel], xs[~sel]] = True
    if int(a.sum()) < min_child or int(b.sum()) < min_child:
        return None
    return a, b


def fit_subregion(rgb_s: np.ndarray, mask: np.ndarray, tensor: dict, args, off, win,
                  geo_frac: float, tf, thr_ref):
    """Fit + path one split child with exactly the machinery used for a top-level region.

    `mask` lives in the parent's window `win` / origin `off`, so all coordinates stay absolute and
    the result integrates with strokes / activity / AA exactly like a normal region.
    """
    inner = _region_interior(mask)
    col = np.median(rgb_s[win][inner], axis=0)
    grad = fit_region_gradient(rgb_s, inner, tensor, args.grad_stops, args.grad_min_range,
                               off=off, allow_radial=args.grad_radial,
                               radial_margin=args.grad_radial_margin)
    use_fit = args.fit
    if use_fit == "auto":
        use_fit = "bezier" if geo_frac >= args.auto_geo_frac else "cr"
    rf = None
    if args.snap_mode == "refine" and geo_frac >= args.auto_geo_frac:
        rf = (tf, thr_ref, args.snap_shift, args.snap_step, args.snap_sub)
    d = region_path_d(ndi.binary_dilation(mask, iterations=1), args.contour_tol,
                      args.contour_smooth, args.contour_min_area, fit=use_fit,
                      fit_tol=args.fit_tol, corner_deg=args.corner_deg, refine=rf, offset=off)
    if not d:
        return None
    return {"idx": -1, "mask": mask, "area": int(mask.sum()), "d": d, "fill": to_hex(col),
            "grad": grad, "bg": False, "geo": float(geo_frac), "fit": use_fit,
            "win": win, "off": off, "col": col}


def split_region_blobs(mask: np.ndarray, err: np.ndarray, err_thr: float, min_child: int):
    """Split off the largest connected blob of above-threshold error as one child.

    This is the split that targets a *localised* error (a misplaced flat patch, a glow band along a
    boundary) which a straight-line split cannot isolate: the blob plus a 2px context margin becomes
    one child, everything else the other. Thin error lines are rejected so the recursion cannot
    shave a region into slivers.
    """
    ys, xs = np.nonzero(mask)
    n = int(ys.size)
    sel = err > err_thr
    if int(sel.sum()) < min_child:
        return None
    hot = np.zeros_like(mask)
    hot[ys[sel], xs[sel]] = True
    hot = ndi.binary_dilation(hot, iterations=2) & mask
    lab, nl = ndi.label(hot, structure=np.ones((3, 3), bool))
    if nl < 1:
        return None
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    a = lab == int(sizes.argmax())
    na = int(a.sum())
    if na < min_child or na > 0.9 * n:
        return None
    # sliver guard: a usable blob must keep a 2px-thick core after erosion
    if int(ndi.binary_erosion(a, iterations=2).sum()) < max(8, int(0.02 * na)):
        return None
    b = mask & ~a
    if int(b.sum()) < min_child:
        return None
    return a, b


def _fill_stats(rgb_s: np.ndarray, region: dict, err_thr: float):
    """(rms, excess energy E, n_bad, err, interior selector) of one region's chosen fill.

    `rms` is the RMS max-channel error over the eroded interior; `E` is the mean squared error
    *above* the threshold (the part that is actually visible): a split is judged on E, because a
    localised visible patch is diluted to nothing by the global rms of a large region, whereas E
    isolates exactly what the user sees as banding.
    """
    mask = region["mask"]
    err = region_fill_error(rgb_s, mask, region, region["off"])
    if err.size == 0:
        return 0.0, 0.0, 0, err, None
    inner = _region_interior(mask)
    ys, xs = np.nonzero(mask)
    inside = inner[ys, xs]
    if not inside.any():
        inside = np.ones(err.size, bool)
    e = err[inside]
    rms = float(np.sqrt((e ** 2).mean()))
    ex = np.maximum(e - err_thr, 0.0)
    E = float((ex ** 2).mean())
    nbad = int((e > err_thr).sum())
    return rms, E, nbad, err, inside


def refine_regions(rgb_s: np.ndarray, tensor: dict, regions: list, args, tf, thr_ref,
                   labels_out: np.ndarray = None):
    """Recursively split regions whose chosen fill still leaves visible error.

    Returns (new_regions, stats). Regions are visited in the caller's order (largest area first)
    and each split emits its leaves in a fixed order, so the result is deterministic. `labels_out`,
    when given, is a copy of the label image that receives the refined partition (children get
    fresh ids); the caller uses it to keep the AA-band / debug passes consistent and to run the
    refine-then-merge order.

    The background region is refinable too: it is emitted as a flat base rect, so on a smooth image
    it is usually the largest single source of flat patches. Its children are ordinary (non-bg)
    regions painted on top of that rect.
    """
    stats = {"n0": len(regions), "n1": len(regions), "split": 0, "depth": 0, "rejected": 0,
             "budget": 0, "eval": 0,
             "bg_split": 0}
    if not bool(getattr(args, "adaptive_refine", False)):
        return regions, stats
    err_thr = float(args.refine_err)
    min_area = max(1, int(args.refine_min_area))
    max_depth = max(0, int(args.refine_max_depth))
    gain_thr = min(0.95, max(0.0, float(args.refine_gain)))
    budget = int(getattr(args, "refine_budget", 0))
    if budget <= 0:
        # Default cap on accepted splits. It is deliberately sub-linear in the region count: the
        # effort per candidate grows with the region's area, so a texture-rich photo with thousands
        # of regions must not be allowed to double its geometry count (wave.jpg @scale1 has 3197
        # regions and would otherwise spend tens of minutes fitting splits). 64 keeps a small image
        # well covered (+80% on apple.png's 80 regions) while 256 bounds a large textured one.
        budget = max(64, min(256, len(regions)))
    min_child = max(32, min_area // 4)
    min_bad = max(16, min_child // 2)
    next_id = max((int(r["idx"]) for r in regions if isinstance(r["idx"], (int, np.integer))),
                  default=-1) + 1
    left = [budget]
    stats["budget"] = budget
    # Origin ("root") of every region: children created by a split inherit the root of the region
    # they came from, so the AA pass can recognise a boundary between two descendants of the same
    # original region as a *synthetic* split (colour-continuous by construction) instead of a real
    # image edge. Regions that were never split are their own root.
    #
    # Two ways to carry the origin down a *recursive* split tree:
    #   --aa-edge-dedup off (default): one level, exactly as before -- a child that is split again
    #     becomes the root of its own grandchildren, so a grandchild and its "uncle" are treated as
    #     unrelated and both keep their AA band on the (synthetic) boundary between them.
    #   --aa-edge-dedup on: the *original* pre-refinement region, carried down the whole tree, so
    #     every boundary inside one former region -- however deep the split -- is recognised and the
    #     duplicated AA treatment between its descendants is dropped.
    _full_root = bool(getattr(args, "aa_edge_dedup", False))
    roots = {int(r["idx"]): int(r["idx"]) for r in regions
             if isinstance(r["idx"], (int, np.integer))}

    def try_candidate(sp, r):
        """Fit both children of a candidate split; returns (E_new, rms_new, kids) or None."""
        if sp is None:
            return None
        kids = []
        for cm in sp:
            c = fit_subregion(rgb_s, cm, tensor, args, r["off"], r["win"], r["geo"], tf, thr_ref)
            if c is None:
                return None
            kids.append(c)
        tot = sum(k["area"] for k in kids)
        accE = 0.0
        accR = 0.0
        for k in kids:
            kr, kE, _, _, _ = _fill_stats(rgb_s, k, err_thr)
            accE += k["area"] * kE
            accR += k["area"] * (kr ** 2)
        if tot <= 0:
            return None
        return accE / tot, math.sqrt(accR / tot), kids

    def visit(r, depth):
        nonlocal next_id
        if (r.get("aa") or r.get("detail") or int(r["area"]) < min_area
                or depth >= max_depth or left[0] <= 0):
            return [r]
        mask = r["mask"]
        rms, E, nbad, err, inside = _fill_stats(rgb_s, r, err_thr)
        # Trigger: enough interior pixels exceed --refine-err (a *visible* patch). The local test
        # matters: on a large region a localised patch of visible error barely moves the global RMS
        # (so an RMS-only gate ignores the background entirely), yet it is exactly what the user sees.
        # A region whose *global* RMS exceeds the threshold necessarily has such pixels too.
        if err.size == 0 or E <= 0.0 or nbad < min_bad:
            return [r]
        if _REFINE_DEBUG:
            print(f"        · 细化候选#{r['idx']}{'[bg]' if r.get('bg') else ''} area={r['area']} "
                  f"rms={rms:.4f} E={E:.2e} nbad={nbad} depth={depth}")
        # zero the rim so the splits follow the *interior* error only
        err_w = np.where(inside, err, 0.0) if inside is not None else err
        cands = []
        line = try_candidate(split_region_mask(mask, err_w, min_child), r)
        if line is not None:
            cands.append(("line", line))
        if int(mask.sum()) >= 2 * min_area and int((err_w > err_thr).sum()) >= min_bad:
            blobs = try_candidate(split_region_blobs(mask, err_w, err_thr, min_child), r)
            if blobs is not None:
                cands.append(("blob", blobs))
        if not cands:
            stats["rejected"] += 1
            if _REFINE_DEBUG:
                print("          → 放弃 (无法二分 / 会切出碎片)")
            return [r]
        name, (newE, new_rms, kids) = min(cands, key=lambda z: (z[1][0], z[1][1]))
        okE = newE <= (1.0 - gain_thr) * E
        okR = new_rms <= rms * 1.05
        if not (okE and okR):
            stats["rejected"] += 1
            if _REFINE_DEBUG:
                print(f"          → 放弃 ({name}: E {E:.2e} → {newE:.2e}, 需 ≤ "
                      f"{(1.0 - gain_thr) * E:.2e}; rms {rms:.4f} → {new_rms:.4f})")
            return [r]
        if _REFINE_DEBUG:
            print(f"          → 二分[{name}] (E {E:.2e} → {newE:.2e}, rms {rms:.4f} → "
                  f"{new_rms:.4f}, 子区域 {kids[0]['area']}/{kids[1]['area']} px, 渐变 "
                  f"{int(kids[0]['grad'] is not None)}+{int(kids[1]['grad'] is not None)})")
        left[0] -= (len(kids) - 1)
        stats["split"] += 1
        if r.get("bg"):
            stats["bg_split"] += 1
        stats["depth"] = max(stats["depth"], depth + 1)
        return kids

    # Best-first over the largest regions. A *depth-first* walk would evaluate every small texture
    # region even after the split budget is gone; ordering by area means the budget also caps the
    # total fitting effort, because the expensive candidates are always the ones we look at first
    # and we stop the moment the budget is spent. The final list stays in descending-area order,
    # which is the order the region assembler already produces.
    heap = [(-int(r["area"]), i, 0, r) for i, r in enumerate(regions)]
    heapq.heapify(heap)
    _seq = len(heap)
    out = []

    def push(k, depth):
        nonlocal _seq
        _seq += 1
        heapq.heappush(heap, (-int(k["area"]), _seq, depth, k))

    while heap:
        if left[0] <= 0 or stats["eval"] >= budget + 512:
            # budget spent (or evaluation safety cap reached): keep the rest untouched
            while heap:
                out.append(heapq.heappop(heap)[3])
            break
        _neg, _s, depth, r = heapq.heappop(heap)
        stats["eval"] += 1
        my_root = int(r.get("root", r["idx"])) if _full_root else roots.get(int(r["idx"]),
                                                                             int(r["idx"]))
        kids = visit(r, depth)
        if len(kids) == 1 and kids[0] is r:
            out.append(r)
            continue
        for k in kids:
            k["idx"] = next_id
            k["root"] = my_root
            next_id += 1
            if labels_out is not None:
                y0, x0 = k["off"]
                sub = labels_out[y0:y0 + k["mask"].shape[0], x0:x0 + k["mask"].shape[1]]
                sub[k["mask"]] = k["idx"]
            push(k, depth + 1)
    stats["n1"] = len(out)
    return out, stats


def regroup_regions(rgb_s: np.ndarray, regions: list, pre: np.ndarray, merged: np.ndarray,
                    args, tensor: dict, tf, thr_ref):
    """Re-fit the partition produced by merging the *refined* label image (refine-then-merge order).

    Only the groups whose pixel set actually changed are re-fitted; an untouched region keeps its
    already-computed fill (and therefore its exact bytes). Group order follows the input region
    order, so the largest regions stay first and the paint order is stable.
    """
    n_lab = int(merged.max()) + 1
    H, W = merged.shape
    objs = ndi.find_objects(merged + 1)
    border = np.zeros((H, W), bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    bgl = int(np.argmax(np.bincount(merged[border].ravel(), minlength=n_lab)))
    bgr = next((r for r in regions if r.get("bg")), None)
    by_id = {int(z["idx"]): z for z in regions}
    group_geo = {}
    for z in regions:
        zids = np.unique(merged[z["win"]][z["mask"]])
        if zids.size:
            group_geo.setdefault(int(zids[0]), float(z["geo"]))
    out, emitted = [], set()
    for r in regions:
        win, m = r["win"], r["mask"]
        ids = np.unique(merged[win][m])
        if ids.size == 0:
            continue
        li = int(ids[0])
        if li in emitted:
            continue
        emitted.add(li)
        sl = objs[li] if 0 <= li < len(objs) else None
        if sl is None:
            continue
        y0, y1 = int(sl[0].start), int(sl[0].stop)
        x0, x1 = int(sl[1].start), int(sl[1].stop)
        win2 = (slice(y0, y1), slice(x0, x1))
        mask2 = merged[win2] == li
        if li == bgl:
            col = np.median(rgb_s[win2][mask2], axis=0)
            out.append({"idx": li, "mask": mask2, "area": int(mask2.sum()), "d": "",
                        "fill": to_hex(col), "grad": None, "bg": True,
                        "geo": float(bgr["geo"]) if bgr is not None else 1.0,
                        "win": win2, "off": (y0, x0), "col": col})
            continue
        mem = np.unique(pre[win2][mask2])
        mem_roots = {int(by_id[int(z)].get("root", int(z))) for z in mem if int(z) in by_id}
        if mem.size == 1:
            src = by_id.get(int(mem[0]))
            if src is not None and src["mask"] is not None and int(src["area"]) == int(mask2.sum()):
                src["idx"] = li                      # keep the computed fill, refresh the id
                out.append(src)
                continue
        geo = group_geo.get(li, 0.5)
        c = fit_subregion(rgb_s, mask2, tensor, args, (y0, x0), win2, geo, tf, thr_ref)
        if c is None:
            continue
        c["idx"] = li
        # A merged group that descends from a single original region is still a synthetic split of
        # that region, so keep the root; a group spanning several origins is a boundary in its own
        # right and becomes its own root (no AA suppression around it).
        c["root"] = mem_roots.pop() if len(mem_roots) == 1 else li
        out.append(c)
    return out


# ======================================================================
# 3.7 mesh-gradient fill candidate (SVG 2 meshgradient; opt-in)
# ======================================================================
# A single linear or radial gradient is a rank-1 model: it cannot represent a smooth 2-D colour
# field such as a glossy highlight that curves in two directions. A Coons-patch mesh fits a coarse
# lattice of colour stops and can, so it is offered as an extra fill candidate for large smooth
# regions.
#
# COMPATIBILITY (important): SVG 2 mesh gradients are NOT implemented by mainstream browsers
# (Chrome / Firefox / Safari) and NOT by cairosvg. A bare mesh fill would therefore leave the region
# unpainted (cairosvg paints an unresolvable paint server black). The mesh is consequently OFF by
# default and every mesh fill is written as an SVG 2 *paint fallback list* --
# `fill="url(#meshN) url(#gN)"` -- so the document always names a valid non-mesh paint server for
# the region. Renderers that do not implement paint fallback lists (cairosvg, mainstream browsers)
# ignore the second entry, so a mesh-ON document is a structural/experimental artifact there, not a
# faithful preview; only a mesh-capable SVG 2 renderer shows the mesh itself.

MESH_SAMPLES = 60000          # cap on the pixels used for the least-squares mesh fit


def _mesh_hat(t, n):
    """Piecewise-linear hat basis coordinates: (lower node, upper node, upper weight)."""
    z = np.clip(t, 0.0, 1.0) * float(n)
    i0 = np.minimum(np.floor(z).astype(np.int64), n - 1)
    return i0, z - i0


def _mesh_design(u, v, n_patch):
    """Design matrix of the bilinear Coons mesh (tensor product of hat functions)."""
    m = n_patch + 1
    iu, wu = _mesh_hat(u, n_patch)
    iv, wv = _mesh_hat(v, n_patch)
    D = np.zeros((u.size, m * m), np.float64)
    rows = np.arange(u.size)
    for dj, wj in ((0, 1.0 - wv), (1, wv)):
        for di, wi in ((0, 1.0 - wu), (1, wu)):
            D[rows, (iv + dj) * m + (iu + di)] += wi * wj
    return D


def fit_region_mesh(rgb_s: np.ndarray, region: dict, n_patch: int):
    """Least-squares Coons-patch mesh over a region's bounding box.

    The basis is the tensor product of piecewise-linear hat functions on a uniform (n_patch+1)
    lattice; a straight-edged Coons patch mesh interpolates its stops in exactly that basis, so the
    fitted nodal colours are precisely the <stop> colours to emit. The residual is evaluated over
    the whole region (not just the fit sample) so it is directly comparable with the fallback fill.
    """
    mask = region["mask"]
    ys, xs = np.nonzero(mask)
    n = int(ys.size)
    if n < 400:
        return None
    o = region["off"]
    ax = xs.astype(np.float64) + int(o[1])
    ay = ys.astype(np.float64) + int(o[0])
    x0, x1 = float(ax.min()), float(ax.max())
    y0, y1 = float(ay.min()), float(ay.max())
    if x1 - x0 < 2.0 or y1 - y0 < 2.0:
        return None
    ua = (ax - x0) / (x1 - x0)
    va = (ay - y0) / (y1 - y0)
    obs = rgb_s[ay.astype(np.int64), ax.astype(np.int64)].astype(np.float64)
    if n > MESH_SAMPLES:                       # deterministic stride, no rng
        sel = np.arange(0, n, int(np.ceil(n / MESH_SAMPLES)))
    else:
        sel = slice(None)
    D = _mesh_design(ua[sel], va[sel], n_patch)
    obs_s = obs[sel]
    # Ridge-regularised normal equations: the bbox corners of a non-rectangular region can have no
    # nearby pixels, and a plain least-squares solve extrapolates them to impossible colours (black /
    # saturated cyan at the outer nodes). Pulling the solution toward the region mean keeps every
    # node inside the observed colour range; the clips below enforce that as a hard guarantee.
    mu = obs_s.mean(0)
    lam = 1e-3 * float(max(1, D.shape[0]))
    try:
        sol = np.linalg.solve(D.T @ D + lam * np.eye(D.shape[1]), D.T @ obs_s + lam * mu[None, :])
    except np.linalg.LinAlgError:
        sol, *_ = np.linalg.lstsq(D, obs_s, rcond=None)
    lo = np.percentile(obs_s, 1.0, axis=0) - 0.02
    hi = np.percentile(obs_s, 99.0, axis=0) + 0.02
    sol = np.clip(sol, np.maximum(lo, 0.0)[None, :], np.minimum(hi, 1.0)[None, :])
    pred = _mesh_design(ua, va, n_patch) @ sol
    err = np.abs(obs - pred).max(1)
    m = n_patch + 1
    return {"nodes": np.clip(sol.reshape(m, m, 3), 0.0, 1.0),
            "bbox": (x0, y0, x1, y1), "n_patch": int(n_patch),
            "rms": float(np.sqrt((err ** 2).mean())), "max": float(err.max()),
            "n_px": n}


def apply_mesh_fills(rgb_s: np.ndarray, regions: list, args) -> dict:
    """Give large smooth regions a mesh candidate when it clearly beats the linear/radial fill.

    The current fill is kept (it becomes the SVG paint fallback); the mesh is adopted only when its
    full-region RMS max-channel error beats the current fill's by --grad-mesh-margin.
    """
    margin = float(args.grad_mesh_margin)
    min_area = int(args.grad_mesh_min_area)
    n_patch = max(1, int(args.grad_mesh_patches))
    thr = float(args.refine_err)
    stats = {"tried": 0, "n": 0, "gain_sum": 0.0, "gain_max": 0.0, "rms0": 0.0, "rms1": 0.0}
    for r in regions:
        if r.get("bg") or r.get("aa") or r.get("detail") or int(r["area"]) < min_area:
            continue
        err0 = region_fill_error(rgb_s, r["mask"], r, r["off"])
        if err0.size == 0:
            continue
        rms0 = float(np.sqrt((err0 ** 2).mean()))
        if rms0 <= thr:
            continue                              # one gradient is already good enough
        stats["tried"] += 1
        mesh = fit_region_mesh(rgb_s, r, n_patch)
        if mesh is None:
            continue
        if mesh["rms"] <= (1.0 - margin) * rms0 and mesh["rms"] < rms0:
            r["mesh"] = mesh
            stats["n"] += 1
            stats["gain_sum"] += 1.0 - mesh["rms"] / max(rms0, EPS)
            stats["gain_max"] = max(stats["gain_max"], 1.0 - mesh["rms"] / max(rms0, EPS))
            stats["rms0"] += rms0
            stats["rms1"] += mesh["rms"]
    return stats


def _mesh_def(mid: str, mesh: dict) -> str:
    """<meshgradient gradientUnits="userSpaceOnUse"> for a fitted lattice (SVG 2 mesh).

    Stops are the four corners of each patch in the order TR, BR, BL, TL, each written as a
    relative path from the previous stop's absolute position, so the absolute corner positions --
    and therefore the mesh geometry -- are unambiguous.
    """
    n_patch = int(mesh["n_patch"])
    m = n_patch + 1
    x0, y0, x1, y1 = mesh["bbox"]
    w = (x1 - x0) / n_patch
    h = (y1 - y0) / n_patch
    nodes = mesh["nodes"]
    cx_, cy_ = x0, y0
    out = [f'<meshgradient id="{mid}" gradientUnits="userSpaceOnUse" '
           f'x="{fnum(x0)}" y="{fnum(y0)}">']
    for j in range(n_patch):
        out.append("<meshrow>")
        for i in range(n_patch):
            corners = ((x0 + (i + 1) * w, y0 + j * h, nodes[j, i + 1]),
                       (x0 + (i + 1) * w, y0 + (j + 1) * h, nodes[j + 1, i + 1]),
                       (x0 + i * w, y0 + (j + 1) * h, nodes[j + 1, i]),
                       (x0 + i * w, y0 + j * h, nodes[j, i]))
            out.append("<meshpatch>")
            for px, py, col in corners:
                out.append(f'<stop path="l {fnum(px - cx_)},{fnum(py - cy_)}" '
                           f'stop-color="{to_hex(col)}"/>')
                cx_, cy_ = px, py
            out.append("</meshpatch>")
        out.append("</meshrow>")
    out.append("</meshgradient>")
    return "".join(out)


# ======================================================================
# 4. contour -> SVG path
# ======================================================================
_REFINE_STATS = {"pts": 0, "moved": 0, "sum": 0.0}


def bilin_arr(a: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear sampling (array coordinates, vectorized)."""
    h, w = a.shape
    x = np.clip(x, 0.0, w - 1.001)
    y = np.clip(y, 0.0, h - 1.001)
    x0 = x.astype(np.int32)
    y0 = y.astype(np.int32)
    fx = x - x0
    fy = y - y0
    return (a[y0, x0] * (1 - fx) * (1 - fy) + a[y0, x0 + 1] * fx * (1 - fy)
            + a[y0 + 1, x0] * (1 - fx) * fy + a[y0 + 1, x0 + 1] * fx * fy)


def refine_contour(pts: np.ndarray, tf: dict, thr: float, max_shift: float = 1.5,
                   step: float = 0.15, sub: str = "half") -> np.ndarray:
    """Snap contour points along the normal to the subpixel edge position.

    The principal eigenvector of the fine-scale structure tensor is the color-gradient direction
    = contour normal. Sample the profile along that direction; two centering modes:
      half (default) — take the 50% intensity crossing of the luma profile. The source image's
        boundary is an anti-aliased ramp, and the 50% point of the ramp is exactly the geometric
        boundary; this is insensitive to the ramp shape and is an unbiased estimate. Synthetic
        straight-edge experiment: error against truth RMS 0.045px / max 0.067px.
      peak — take the gradient-energy peak + parabolic interpolation (old method): RMS 0.12px / max 0.19px.
    Only points with strong enough energy (> thr, the same threshold as the edge map) are moved,
    and the displacement never exceeds max_shift, so the topology is unchanged and the boundary is
    not dragged onto another structure. This step also removes the 1px jitter caused by
    quantization/anti-aliasing.
    """
    if len(pts) == 0:
        return pts
    e = tf["energy"]
    mag = np.hypot(tf["gx"], tf["gy"])
    ux = tf["gx"] / (mag + EPS)
    uy = tf["gy"] / (mag + EPS)
    x, y = pts[:, 0], pts[:, 1]
    nx = bilin_arr(ux, x, y)
    ny = bilin_arr(uy, x, y)
    if sub == "half":
        luma = tf["luma"]
        offs = np.arange(-max_shift * 1.2, max_shift * 1.2 + 1e-9, step)
        prof = np.stack([bilin_arr(luma, x + t * nx, y + t * ny) for t in offs])
        if len(offs) >= 3:                      # 3-point moving average, robust against brightness noise
            prof = np.vstack([prof[:1],
                              (prof[:-2] + prof[1:-1] + prof[2:]) / 3.0,
                              prof[-1:]])
        # At the ends take the median of the first/last 3 samples for noise robustness; too wide a window bleeds onto a neighbouring structure
        lo = np.median(prof[:3], axis=0)
        hi = np.median(prof[-3:], axis=0)
        mid = 0.5 * (lo + hi)
        idx = np.arange(len(pts))

        def _cross(pr):
            """50% crossing offset when pr is monotonically increasing (in the offs coordinate system)."""
            k = np.argmax(pr >= mid[None, :], axis=0)
            kc = np.clip(k, 1, len(offs) - 1)
            a0 = pr[kc - 1, idx]
            a1 = pr[kc, idx]
            f = np.where(np.abs(a1 - a0) > 1e-9,
                         (mid - a0) / np.where(a1 == a0, 1.0, a1 - a0), 0.0)
            return offs[kc - 1] + np.clip(f, 0.0, 1.0) * (offs[kc] - offs[kc - 1])

        # The principal eigenvector points towards the "brighter" side; when the region is darker than its neighbour the profile decreases, so solve in reverse
        sh_up = _cross(prof)
        sh_dn = -_cross(prof[::-1])
        shift = np.where(hi >= lo, sh_up, sh_dn)
        shift = np.clip(shift, -max_shift, max_shift)
        # The threshold must use **energy** (the same threshold as the edge map): a luma threshold is dimensionally wrong and would make every point move at random
        gs = np.arange(-max_shift, max_shift + 1e-9, max_shift / 2.0)
        emax = np.max(np.stack([bilin_arr(e, x + t * nx, y + t * ny) for t in gs]), 0)
        good = (np.abs(hi - lo) > 0.06) & (emax > thr)
        shift = np.where(good, shift, 0.0)
        _REFINE_STATS["pts"] += int(len(pts))
        _REFINE_STATS["moved"] += int(good.sum())
        _REFINE_STATS["sum"] += float(np.abs(shift).sum())
        return pts + shift[:, None] * np.stack([nx, ny], 1)
    offs = np.arange(-max_shift, max_shift + 1e-9, step)
    prof = np.stack([bilin_arr(e, x + t * nx, y + t * ny) for t in offs])
    k = np.argmax(prof, axis=0)
    kc = np.clip(k, 1, len(offs) - 2)
    idx = np.arange(len(pts))
    a0 = prof[kc - 1, idx]
    a1 = prof[kc, idx]
    a2 = prof[kc + 1, idx]
    den = a0 - 2.0 * a1 + a2
    d = np.where(np.abs(den) > 1e-12, 0.5 * (a0 - a2) / np.where(den == 0, 1.0, den), 0.0)
    d = np.clip(d, -1.0, 1.0)
    shift = np.clip(offs[kc] + d * step, -max_shift, max_shift)
    good = a1 > thr
    shift = np.where(good, shift, 0.0)
    _REFINE_STATS["pts"] += int(len(pts))
    _REFINE_STATS["moved"] += int(good.sum())
    _REFINE_STATS["sum"] += float(np.abs(shift).sum())
    return pts + shift[:, None] * np.stack([nx, ny], 1)


def mask_to_paths(mask: np.ndarray, tol: float, smooth: float,
                  min_area: float = 25.0, pad: int = 3, simplify: bool = True,
                  refine=None, offset=(0, 0)):
    """offset=(dy, dx): mask comes from some bounding-box window; this is that window's origin in the full image.

    The output coordinates are always **absolute image coordinates** (downstream uses them to index
    labels/rgb/tensor fields). Implementation note: offset is added to the window's slice origin
    using **integer** arithmetic only, then added to the contour coordinates, so the floating-point
    addition sequence is exactly that of computing directly on the full image -- bit-identical, not
    merely "approximately equal".
    """
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return []
    mar = int(3 * smooth) + 3
    y0 = max(0, int(ys.min()) - pad - mar)
    y1 = min(mask.shape[0], int(ys.max()) + 1 + pad + mar)
    x0 = max(0, int(xs.min()) - pad - mar)
    x1 = min(mask.shape[1], int(xs.max()) + 1 + pad + mar)
    m = np.pad(mask[y0:y1, x0:x1].astype(np.float32), pad)
    if smooth > 0:
        m = ndi.gaussian_filter(m, smooth)
    y0 += int(offset[0])
    x0 += int(offset[1])
    paths = []
    for c in measure.find_contours(m, 0.5):
        c = c - pad
        pts = np.stack([c[:, 1] + x0, c[:, 0] + y0], 1)
        if refine is not None:
            pts = refine_contour(pts, *refine)
        if simplify:
            pts = measure.approximate_polygon(pts, tolerance=tol)
            if len(pts) < 4:
                continue
        if poly_area(pts) < min_area:
            continue
        paths.append(pts)
    return paths


def region_path_d(mask: np.ndarray, tol: float, smooth: float,
                  min_area: float = 25.0, nd: int = 1, fit: str = "cr",
                  fit_tol: float = 0.25, corner_deg: float = 62.0,
                  refine=None, offset=(0, 0)) -> str:
    """Region mask → SVG path. fit='bezier' uses conformal optimal fitting, 'cr' uses simplification + Catmull-Rom.

    offset=(dy, dx): the window origin of mask in the full image; the path is emitted in absolute image coordinates.
    """
    if fit == "bezier":
        return "".join(fit_bezier_d(p, True, tol=fit_tol, corner_deg=corner_deg, nd=nd)
                       for p in mask_to_paths(mask, tol, smooth, min_area,
                                              simplify=False, refine=refine,
                                              offset=offset))
    return "".join(polyline_to_bezier_d(p, closed=True, nd=nd)
                   for p in mask_to_paths(mask, tol, smooth, min_area,
                                          refine=refine, offset=offset))


# ======================================================================
# 5. strokes: isophote streamlines (evenly spaced streamlines)
# ======================================================================
def trace_streamline(x, y, sgn, h, max_steps, coh, tx, ty, mask, occ,
                     coh_min, cos_max_turn):
    h_, w_ = mask.shape
    pts = [(x, y)]
    prev = None
    for _ in range(max_steps):
        if bilin(coh, x, y) < coh_min:
            break
        vx, vy = bilin(tx, x, y), bilin(ty, x, y)
        n = math.hypot(vx, vy)
        if n < 1e-9:
            break
        vx, vy = vx / n * sgn, vy / n * sgn
        if prev is not None and (vx * prev[0] + vy * prev[1]) < cos_max_turn:
            break
        mx, my = x + 0.5 * h * vx, y + 0.5 * h * vy
        if bilin(coh, mx, my) < coh_min:
            break
        ux, uy = bilin(tx, mx, my), bilin(ty, mx, my)
        n2 = math.hypot(ux, uy)
        if n2 < 1e-9:
            break
        ux, uy = ux / n2 * sgn, uy / n2 * sgn
        nx, ny = x + h * ux, y + h * uy
        if not (0.0 <= nx <= w_ - 1.001 and 0.0 <= ny <= h_ - 1.001):
            break
        ix, iy = int(round(nx)), int(round(ny))
        if not mask[iy, ix] or occ[iy, ix]:
            break
        x, y = nx, ny
        pts.append((x, y))
        prev = (ux, uy)
    return pts


def make_streamlines(tens, mask, args, rng):
    """Returns [(pts(N,2), widths(N,), coh(N,)), ...]"""
    h_, w_ = mask.shape
    coh, tx, ty = tens["coh"], tens["tx"], tens["ty"]
    occ = np.zeros((h_, w_), bool)
    rad = max(2, int(round(args.spacing * args.occ_ratio)))
    yy, xx = np.mgrid[0:2 * rad + 1, 0:2 * rad + 1]
    disc = (yy - rad) ** 2 + (xx - rad) ** 2 <= rad * rad
    dy, dx = np.nonzero(disc)
    dy = dy - rad
    dx = dx - rad
    cos_max_turn = math.cos(math.radians(args.max_turn))

    sp = args.spacing
    gy, gx = np.mgrid[sp: h_ - sp: sp, sp: w_ - sp: sp]
    sy = gy.ravel().astype(float) + rng.uniform(-0.45 * sp, 0.45 * sp, gy.size)
    sx = gx.ravel().astype(float) + rng.uniform(-0.45 * sp, 0.45 * sp, gx.size)
    np.clip(sy, 1, h_ - 2, out=sy)
    np.clip(sx, 1, w_ - 2, out=sx)
    order = np.argsort(-coh[sy.astype(int), sx.astype(int)])

    max_steps = int(args.stroke_max / args.stroke_step) + 1
    out = []
    for k in order:
        x0, y0 = float(sx[k]), float(sy[k])
        ix, iy = int(round(x0)), int(round(y0))
        if occ[iy, ix] or not mask[iy, ix] or coh[iy, ix] < args.coh_min:
            continue
        fwd = trace_streamline(x0, y0, +1.0, args.stroke_step, max_steps,
                               coh, tx, ty, mask, occ, args.coh_min, cos_max_turn)
        bwd = trace_streamline(x0, y0, -1.0, args.stroke_step, max_steps,
                               coh, tx, ty, mask, occ, args.coh_min, cos_max_turn)
        pts = np.array(bwd[::-1] + fwd, float)
        if len(pts) < 3:
            continue
        arc = float(np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1])).sum())
        if arc < args.stroke_min:
            continue
        for px, py in pts[::2]:
            iy2 = np.clip((py + dy).astype(int), 0, h_ - 1)
            ix2 = np.clip((px + dx).astype(int), 0, w_ - 1)
            occ[iy2, ix2] = True
        n = len(pts)
        cnt = int(min(n, max(4, args.stroke_nodes)))
        idx = np.unique(np.linspace(0, n - 1, cnt).round().astype(int))
        p = pts[idx]
        c = coh[p[:, 1].astype(int), p[:, 0].astype(int)]
        wid = args.stroke_width * (0.5 + 0.5 * c)
        t = np.linspace(0.0, 1.0, len(p))
        wid = wid * np.clip(np.minimum(t, 1 - t) / 0.16, 0.25, 1.0)
        out.append((p, wid, c))
    return out


def ribbon_d(pts, widths, nd=1) -> str:
    """Variable-width stroke → closed outline polygon path."""
    p = np.asarray(pts, float)
    w = np.maximum(np.asarray(widths, float), 0.4)
    n = len(p)
    if n < 2:
        return ""
    d = np.gradient(p, axis=0)
    ln = np.hypot(d[:, 0], d[:, 1])
    ln[ln == 0] = 1.0
    nx, ny = -d[:, 1] / ln, d[:, 0] / ln
    left = np.stack([p[:, 0] + nx * w / 2, p[:, 1] + ny * w / 2], 1)
    right = np.stack([p[:, 0] - nx * w / 2, p[:, 1] - ny * w / 2], 1)[::-1]
    return polyline_to_bezier_d(np.vstack([left, right]), closed=True, nd=nd)


def sample_color(rgb, pts, jitter, rng) -> np.ndarray:
    xs = np.clip(pts[:, 0].round().astype(int), 0, rgb.shape[1] - 1)
    ys = np.clip(pts[:, 1].round().astype(int), 0, rgb.shape[0] - 1)
    col = np.median(rgb[ys, xs], axis=0)
    if jitter:
        # Additive jitter (multiplicative jitter saturates in near-white areas and creates glaring bright streaks)
        col = col + rng.normal(0.0, jitter, 3)
    return np.clip(col, 0.0, 1.0)


# ======================================================================
# 6. edge ridges -> vector strokes
# ======================================================================
_NB8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def skeleton_paths(skel: np.ndarray, min_len: int = 6, tol: float = 0.8):
    ys, xs = np.nonzero(skel)
    nodes = set(zip(ys.tolist(), xs.tolist()))
    if not nodes:
        return []
    deg = {}
    for (y, x) in nodes:
        d = 0
        for dy, dx in _NB8:
            if (y + dy, x + dx) in nodes:
                d += 1
        deg[(y, x)] = d
    junctions = {p for p, d in deg.items() if d >= 3}
    used = set()
    paths = []

    def walk(a, b):
        path = [a, b]
        used.add((min(a, b), max(a, b)))
        prev, cur = a, b
        while True:
            if cur in junctions or deg[cur] == 1:
                break
            nxt = None
            for dy, dx in _NB8:
                q = (cur[0] + dy, cur[1] + dx)
                if q in nodes and q != prev and (min(cur, q), max(cur, q)) not in used:
                    nxt = q
                    break
            if nxt is None:
                break
            used.add((min(cur, nxt), max(cur, nxt)))
            path.append(nxt)
            prev, cur = cur, nxt
        return path

    for s in [p for p, d in deg.items() if d == 1] + list(junctions):
        for dy, dx in _NB8:
            q = (s[0] + dy, s[1] + dx)
            if q in nodes and (min(s, q), max(s, q)) not in used:
                paths.append(walk(s, q))
    for a in nodes:  # remaining closed loops
        for dy, dx in _NB8:
            b = (a[0] + dy, a[1] + dx)
            if b in nodes and (min(a, b), max(a, b)) not in used:
                paths.append(walk(a, b))

    out = []
    for p in paths:
        if len(p) < min_len:
            continue
        arr = np.array([(x, y) for (y, x) in p], float)
        arr = measure.approximate_polygon(arr, tolerance=tol)
        if len(arr) >= 2:
            out.append(arr)
    return out


# ======================================================================
# 7. SVG assembly
# ======================================================================
def classify_regions(rgb: np.ndarray, labels: np.ndarray, areas: np.ndarray,
                     tex_sigma: float = 2.5, tex_smooth: float = 6.0,
                     tex_norm: float = 99.5, tex_thr: float = 3.0,
                     range_thr: float = 0.02, min_area: int = 600):
    """Decide per region whether it is "solid color + geometric" or "stroke/gradient complex".

    All criteria are computed on the eroded interior, avoiding the anti-aliased transition band:
    texture = high-frequency residual energy (large where there are strokes/details), color spread
    = difference between the 5% and 95% quantiles (large where there is a gradient).
    """
    tex = texture_map(rgb, tex_sigma, tex_smooth)
    n = int(len(areas))
    tex_r = np.zeros(n)
    rng_r = np.zeros(n)
    # Each region is eroded / sampled only inside a window of its bounding box + 3px margin: the radius of 3 erosions is exactly 3,
    # and outside the window is an all-zero background (consistent with the full-image convention), so the windowed and full-image versions are bit-identical.
    # find_objects(labels + 1): list index k corresponds directly to label k, so label 0 is not missed.
    _objs = ndi.find_objects(labels + 1)
    for li in np.nonzero(areas >= min_area)[0]:
        sl = _objs[li] if 0 <= li < len(_objs) else None
        if sl is None:
            win = (slice(0, 0), slice(0, 0))
        else:
            win = (slice(max(0, sl[0].start - 3), sl[0].stop + 3),
                   slice(max(0, sl[1].start - 3), sl[1].stop + 3))
        mk = labels[win] == li
        m = ndi.binary_erosion(mk, iterations=3)
        if m.sum() < 30:
            m = mk
        px = rgb[win][m]
        rng_r[li] = float((np.percentile(px, 95, axis=0)
                           - np.percentile(px, 5, axis=0)).max())
        tex_r[li] = float(tex[win][m].mean())
    t_norm = float(np.percentile(tex, tex_norm)) + EPS
    geo_r = (tex_r / t_norm < tex_thr) & (rng_r < range_thr)
    return geo_r, tex_r, rng_r, t_norm


def neighbor_map(labels: np.ndarray, win: int = 3, band=None) -> np.ndarray:
    """For each pixel, the "nearest different label" (first occurrence within the window), used to split transition bands by adjacent region.

    Edges are filled in with edge to avoid the false adjacencies that np.roll produces by wrapping
    around from the opposite side.
    """
    H, W = labels.shape
    pad = np.pad(labels, win, mode="edge")
    out = np.full(labels.shape, -1, labels.dtype)
    for dy in range(-win, win + 1):
        for dx in range(-win, win + 1):
            if dy == 0 and dx == 0:
                continue
            sh = pad[win + dy:win + dy + H, win + dx:win + dx + W]
            new = (sh != labels) & (out < 0)
            if band is not None:
                new &= band
            out[new] = sh[new]
    return out


def aa_ramp_levels(rgb: np.ndarray, tf: dict, thr: float, W2: float, N: int):
    """Measure the average anti-aliasing transition profile along the source image's boundary normals, then run 1D Lloyd quantization along depth.

    Returns (bnds, samps): bnds are the N+1 visible boundary depths (from the boundary into the
    region interior, in pixels), samps are the N color-sampling depths. Equal-width grading
    concentrates the error where the transition is steepest; grading by profile shape (equivalent
    to equal increments of intensity) minimizes the "within-segment variance", i.e. it is the
    optimal quantization approximating a continuous gradient with an N-step staircase. When the
    profile is insufficient it returns (None, None) and the caller falls back to equal width.
    """
    if tf is None:
        return None, None
    e = tf["energy"]
    ys, xs = np.nonzero(e > thr)
    if len(ys) < 200:
        return None, None
    rng = np.random.default_rng(0)
    sel = rng.choice(len(ys), size=min(6000, len(ys)), replace=False)
    ys = ys[sel].astype(np.float64)
    xs = xs[sel].astype(np.float64)
    luma = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    mag = np.hypot(tf["gx"], tf["gy"]) + EPS
    nx = bilin_arr(tf["gx"] / mag, xs, ys)
    ny = bilin_arr(tf["gy"] / mag, xs, ys)
    ts = np.arange(-W2, W2 + 1e-9, 0.25)
    prof = np.stack([bilin_arr(luma, xs + s * nx, ys + s * ny) for s in ts], 1)
    lo, hi = prof[:, 0], prof[:, -1]
    ok = np.abs(hi - lo) > 0.08
    if int(ok.sum()) < 100:
        return None, None
    s = (prof[ok] - lo[ok, None]) / (hi - lo)[ok, None]
    shape = np.median(s, axis=0)
    half = shape[len(ts) // 2:]                  # boundary (0.5) -> interior (1.0)
    dg = np.linspace(0.0, W2, len(half))
    bnds = np.linspace(0.0, W2, N + 1)
    for _ in range(24):
        edges = np.searchsorted(dg, bnds[1:-1])
        parts = [p for p in np.split(np.arange(len(dg)), edges) if len(p)]
        if len(parts) != N:
            break
        means = np.array([half[p].mean() for p in parts])
        nb = [0.0]
        for k in range(N - 1):
            nb.append(float(np.interp(0.5 * (means[k] + means[k + 1]), half, dg)))
        nb.append(float(W2))
        nb = np.maximum.accumulate(np.asarray(nb))
        done = np.allclose(nb, bnds, atol=0.01)
        bnds = nb
        if done:
            break
    edges = np.searchsorted(dg, bnds[1:-1])
    parts = [p for p in np.split(np.arange(len(dg)), edges) if len(p)]
    if len(parts) != N:
        return None, None
    samps = np.array([float(np.interp(half[p].mean(), half, dg)) for p in parts])
    if not np.all(np.diff(bnds) > 1e-3):
        return None, None
    return bnds, samps


# Diagnostics for one anti-aliasing pass: how many bands were emitted and how many were dropped
# because the boundary is a *synthetic* split between two descendants of the same original region.
# Setting LOGO_AA_DUMP=<path> additionally writes one row per candidate boundary segment, which is
# how the byte cost of a given boundary class is audited.
_AA_STATS = {"skipped_syn": 0}


def aa_band_regions(rgb: np.ndarray, entries: list, labels: np.ndarray, args,
                    tf=None, thr: float = 0.0) -> list:
    """Restore the boundary of "solid color + geometric" regions as an n-level anti-aliasing staircase.

    Each region's fill contour is already subpixel (and edge-snapped where needed); here we take
    the same contour and offset it inward along its normal by step, giving n parallel lines;
    adjacent lines plus end caps enclose a closed "band", and the band is filled with the source
    image's color at that place (bilinear sampling along the band's centerline, then the median).
    Adjacent regions each contribute their own half, the two staircases meet at the boundary, and
    together they restore the source image's 2~3px wide anti-aliasing transition.

    Why not distance-field isolines: the EDT is a quantized field, and smoothing it further with a
    Gaussian on a narrow band is pulled down by the zero background, so high-level isolines shatter
    into hundreds of fragments; "contour normal offsetting" is naturally subpixel and shares its
    edge with the fill.
    """
    W2 = args.aa_width * 0.5
    N = max(1, int(args.aa_levels))
    H, W = labels.shape
    bnds = samps = None
    if args.aa_profile == "shape":
        bnds, samps = aa_ramp_levels(rgb, tf, thr, W2, N)
        if args.verbose and bnds is not None:
            print("      · 过渡带分级(实测剖面): 边界深度 "
                  + "/".join("%.2f" % b for b in bnds[1:])
                  + " | 取色深度 " + "/".join("%.2f" % s for s in samps))
    if bnds is None:
        bnds = np.linspace(0.0, W2, N + 1)
        samps = (bnds[:-1] + bnds[1:]) / 2.0
    out = []
    saved = dict(_REFINE_STATS)
    # Origin of every region in this pass: a refined child carries the root (the pre-split region)
    # it descends from, everything else is its own root. Two neighbours with the same root are two
    # halves of one *synthetic* split, so the boundary between them is colour-continuous by
    # construction and does not need an AA staircase; a neighbour with a different root (another
    # original region, the background, the image border) keeps the full AA treatment. With
    # --aa-synthetic the old behaviour is restored exactly.
    keep_syn = bool(getattr(args, "aa_synthetic", True))
    _AA_STATS["skipped_syn"] = 0
    _AA_STATS["skipped_canon"] = 0
    _aa_rows = [] if os.environ.get("LOGO_AA_DUMP") else None
    root_of = {}
    for r in entries:
        if isinstance(r.get("idx"), (int, np.integer)):
            root_of[int(r["idx"])] = int(r.get("root", r["idx"]))
    # Aggressive variant (--aa-edge-canon, default off, measured to change real edges): for one
    # origin group only the *largest* descendant may emit AA bands on a real edge; the smaller
    # descendants keep only their synthetic (internal) boundaries. Real edges are shared among the
    # descendants, so this deletes the AA of every stretch that a small descendant owns.
    keep_canon = bool(getattr(args, "aa_edge_canon", False))
    _thr0 = (args.auto_geo_frac if args.aa_geo_thr < 0 else args.aa_geo_thr)
    canon = {}
    if keep_canon:
        for r in entries:
            if isinstance(r.get("idx"), (int, np.integer)) and r.get("geo") is not None \
                    and float(r["geo"]) >= _thr0:
                k = root_of.get(int(r["idx"]), int(r["idx"]))
                if k not in canon or int(r.get("area", 0)) > int(canon[k][1]):
                    canon[k] = (int(r["idx"]), int(r.get("area", 0)))
        canon = {k: v[0] for k, v in canon.items()}
    for r in entries:
        if r.get("aa"):
            continue
        geo = r.get("geo")
        _thr = (args.auto_geo_frac if args.aa_geo_thr < 0 else args.aa_geo_thr)
        if geo is None or geo < _thr:
            continue                      # complex region keeps the original approach: no anti-aliasing reconstruction
        mask = r["mask"]
        ys, xs = np.nonzero(mask)
        if len(ys) == 0:
            continue
        sl = (slice(max(0, ys.min() - 2), ys.max() + 3),
              slice(max(0, xs.min() - 2), xs.max() + 3))
        rr = float(ndi.distance_transform_edt(mask[sl]).max())   # maximum inscribed radius
        if rr < args.aa_min_radius:
            continue                      # the region is too thin overall: no band
        rf = None
        if tf is not None and args.snap_mode == "refine":
            rf = (tf, thr, args.snap_shift, args.snap_step, args.snap_sub)
        _w = r.get("win")
        _o = (_w[0].start, _w[1].start) if _w is not None else (0, 0)
        pls = mask_to_paths(ndi.binary_dilation(mask, iterations=1),
                            args.contour_tol, args.contour_smooth,
                            args.contour_min_area, simplify=False, refine=rf,
                            offset=_o)
        for pts in pls:
            npt = len(pts)
            if npt < 12:
                continue
            t = np.roll(pts, -1, 0) - np.roll(pts, 1, 0)
            t = t / (np.hypot(t[:, 0], t[:, 1])[:, None] + EPS)
            nv = np.stack([-t[:, 1], t[:, 0]], 1)
            ok_or = False
            for s in (1.0, -1.0):                     # make the normal point into the region
                q = pts + nv * s
                yy = np.clip(np.round(q[:, 1]).astype(int), 0, H - 1)
                xx = np.clip(np.round(q[:, 0]).astype(int), 0, W - 1)
                if (labels[yy, xx] == r["idx"]).mean() > 0.6:
                    if s < 0:
                        nv = -nv
                    ok_or = True
                    break
            if not ok_or:
                continue
            # Per-point local half width: sample along the inward normal until leaving the region. Where the region is locally thin/narrow (thin protrusions,
            # sharp corners) the band must narrow, otherwise the two offsets cross and the band self-intersects into one big blob.
            exit_d = np.full(npt, 4.0)
            for d in np.arange(0.5, 4.0, 0.5):
                q = pts + nv * d
                yy = np.clip(np.round(q[:, 1]).astype(int), 0, H - 1)
                xx = np.clip(np.round(q[:, 0]).astype(int), 0, W - 1)
                out_ = (labels[yy, xx] != r["idx"]) & (exit_d > d - 1e-9)
                exit_d[out_] = d
            halfw = np.clip(0.75 * (exit_d - 0.5), 0.15, W2)
            if npt >= 5:                  # smooth along the contour to avoid abrupt width changes
                kk = np.ones(5) / 5.0
                halfw = np.convolve(np.r_[halfw[-2:], halfw, halfw[:2]], kk, "valid")
            wr = (halfw / W2)[:, None]
            q = pts - nv * 1.0                        # step 1px outward to see who is opposite
            yy = np.clip(np.round(q[:, 1]).astype(int), 0, H - 1)
            xx = np.clip(np.round(q[:, 0]).astype(int), 0, W - 1)
            B = labels[yy, xx].astype(np.int64)
            B = np.where(B == r["idx"], -1, B)
            for i in range(1, npt):                   # -1 fills with the previous value
                if B[i] < 0:
                    B[i] = B[i - 1]
            if B[0] < 0:
                B[0] = B[-1] if B[-1] >= 0 else -1
            for _ in range(2):                        # merge over-short segments
                i = 0
                while i < npt and npt > 24:
                    j = i
                    while j + 1 < npt and B[j + 1] == B[i]:
                        j += 1
                    if j - i + 1 < 5 and i > 0:
                        B[i:j + 1] = B[i - 1]
                    i = j + 1
            if B[0] < 0:
                continue
            brk = np.nonzero(B != np.roll(B, 1))[0]
            if len(brk) == 0:
                segs = [(0, npt - 1, int(B[0]))]
            else:
                segs = [(int(s0), int((brk[(k + 1) % len(brk)] - 1) % npt), int(B[s0]))
                        for k, s0 in enumerate(brk)]
            for (st, en, bl) in segs:
                if bl < 0:
                    continue
                m = (en - st) % npt + 1
                # A boundary segment is *synthetic* when the region on the other side descends from
                # the same pre-refinement region as this one.
                _syn = root_of.get(int(bl), int(bl)) == root_of.get(int(r["idx"]), int(r["idx"]))
                if _aa_rows is not None:
                    _aa_rows.append((int(r["idx"]), int(bl), int(m), int(npt),
                                     1 if _syn else 0))
                if m < 6 or m > npt - 3 or m < args.aa_min_area:
                    continue
                if _syn and not keep_syn:
                    _AA_STATS["skipped_syn"] += 1
                    continue          # synthetic split boundary: skip the AA staircase
                if (keep_canon and not _syn
                        and canon.get(root_of.get(int(r["idx"]), int(r["idx"]))) not in
                        (None, int(r["idx"]))):
                    _AA_STATS["skipped_canon"] += 1
                    continue          # aggressive variant: real edge kept for one descendant only
                ii = np.arange(st, st + m) % npt
                o = pts[ii]
                u = nv[ii]
                for j in range(N):
                    w_ = wr[ii]
                    poly = np.concatenate([o + u * (bnds[j] * w_),
                                           (o + u * (bnds[j + 1] * w_))[::-1]])
                    poly = measure.approximate_polygon(poly, tolerance=0.06)
                    d = fit_bezier_d(poly, True, tol=args.aa_tol,
                                     corner_deg=args.corner_deg)
                    if not d:
                        continue
                    mid = o + u * (samps[j] * wr[ii])
                    col = np.array([np.median(bilin_arr(rgb[:, :, c], mid[:, 0],
                                                        mid[:, 1])) for c in range(3)])
                    out.append({"idx": "aa%d-%d-%d" % (r["idx"], bl, j),
                                "mask": None,
                                "area": int(m * (bnds[j + 1] - bnds[j])
                                            * float(wr[ii].mean())), "d": d,
                                "fill": to_hex(np.clip(col, 0, 1)), "grad": None,
                                "bg": False, "aa": True})
    _REFINE_STATS.update(saved)
    if _aa_rows is not None:
        # region, neighbour, segment length, contour point count, is_synthetic
        with open(os.environ["LOGO_AA_DUMP"], "w") as fh:
            fh.write("idx\tbl\tm\tnpt\tsyn\n")
            for row in _aa_rows:
                fh.write("%d\t%d\t%d\t%d\t%d\n" % row)
    return out


def build_svg(width, height, regions, stroke_groups, edges, meta, view_box=None,
              mesh_underlay=True) -> str:
    """width/height = canvas (default display size), view_box = user unit range (the tracing grid).

    The two are separate because of --scale: the geometry runs on the enlarged grid, but the default
    display size should still be the native size (otherwise a 4x-grid SVG opens 5016px wide in a
    browser).
    """
    vw, vh = view_box or (width, height)
    defs, body = [], []
    body.append(f'<rect x="0" y="0" width="{vw}" height="{vh}" fill="{meta["bg_fill"]}"/>')

    body.append('<g id="regions">')
    for i, r in enumerate(regions):
        if r["bg"]:
            continue
        if r["grad"] is not None or r.get("mesh") is not None:
            gid = f"g{i}"
            g = r["grad"]
            mesh = r.get("mesh")
            if g is None:
                # Mesh with a flat fallback: a two-stop constant linear gradient keeps the fallback
                # a real paint server (its id is present in <defs>) as an SVG 2 paint list requires.
                c = to_hex(r["col"])
                defs.append(
                    f'<linearGradient id="{gid}" gradientUnits="userSpaceOnUse" '
                    f'x1="0" y1="0" x2="1" y2="0">'
                    f'<stop offset="0" stop-color="{c}"/>'
                    f'<stop offset="1" stop-color="{c}"/></linearGradient>')
            elif g.get("kind") == "radial":
                # Radial fill: center + radius in user units; the stop offsets are already radius
                # fractions (see _radial_fit), so anything past the last stop is clamped by SVG.
                defs.append(
                    f'<radialGradient id="{gid}" gradientUnits="userSpaceOnUse" '
                    f'cx="{fnum(g["cx"])}" cy="{fnum(g["cy"])}" r="{fnum(g["r"])}">'
                    + "".join(f'<stop offset="{fnum(o,3)}" stop-color="{to_hex(c)}"/>'
                              for o, c in g["stops"])
                    + "</radialGradient>")
            else:
                defs.append(
                    f'<linearGradient id="{gid}" gradientUnits="userSpaceOnUse" '
                    f'x1="{fnum(g["p0"][0])}" y1="{fnum(g["p0"][1])}" '
                    f'x2="{fnum(g["p1"][0])}" y2="{fnum(g["p1"][1])}">'
                    + "".join(f'<stop offset="{fnum(o,3)}" stop-color="{to_hex(c)}"/>'
                              for o, c in g["stops"])
                    + "</linearGradient>")
            if mesh is not None:
                mid = f"mesh{i}"
                defs.append(_mesh_def(mid, mesh))
                r["fill"] = f"url(#{mid}) url(#{gid})"
                r["fallback_fill"] = f"url(#{gid})"
            else:
                r["fill"] = f"url(#{gid})"
        if mesh_underlay and r.get("mesh") is not None:
            # cairosvg (measured) and mainstream browsers do not implement SVG 2 paint *fallback
            # lists*: they resolve the first paint server, fail, and paint nothing at all -- so the
            # region would vanish. Painting the fallback fill underneath makes the region correct
            # everywhere, while a mesh-capable renderer still draws the mesh on top of it.
            body.append(f'<path d="{r["d"]}" fill="{r["fallback_fill"]}" fill-rule="evenodd"/>')
        body.append(f'<path d="{r["d"]}" fill="{r["fill"]}" fill-rule="evenodd"/>')
    body.append("</g>")

    if stroke_groups:
        body.append('<g id="strokes">')
        for grp in stroke_groups:
            cid = f'clip{grp["idx"]}'
            defs.append(f'<clipPath id="{cid}"><path d="{grp["d"]}"/></clipPath>')
            body.append(f'<g clip-path="url(#{cid})">')
            for d, col, alpha in grp["strokes"]:
                body.append(f'<path d="{d}" fill="{to_hex(col)}" '
                            f'fill-opacity="{fnum(alpha,3)}"/>')
            body.append("</g>")
        body.append("</g>")

    if edges:
        body.append('<g id="edges" fill="none" stroke-linecap="round" stroke-linejoin="round">')
        for d, col, w, alpha in edges:
            body.append(f'<path d="{d}" stroke="{to_hex(col)}" stroke-width="{fnum(w,2)}" '
                        f'stroke-opacity="{fnum(alpha,3)}"/>')
        body.append("</g>")

    head = ('<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'width="{width}" height="{height}" viewBox="0 0 {vw} {vh}">\n'
            f"<title>{meta.get('src', 'input')} · 结构张量描摹 (structure-tensor tracing)</title>\n"
            f'<desc>{meta["desc"]}</desc>\n')
    return head + "<defs>\n" + "\n".join(defs) + "\n</defs>\n" + "\n".join(body) + "\n</svg>\n"


# ======================================================================
# 8. main pipeline
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
        q = f"{grad['quality']:.2f}({grad['axis']})" if grad else "平色"
        msg = (f"      · 区域#{li:<2d} area={int(areas[li]):>7d} {to_hex(col)} "
               f"渐变质量={q} 拟合={use_fit}(几何度{geo_frac:.2f})")
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
    q = f"{grad['quality']:.2f}" if grad else "平色"
    msg = (f"      · 缝线#{k} area={int(mk.sum()):>7d} 窗口={mk.shape[0]}x{mk.shape[1]} "
           f"轮廓={_dt:.1f}s {to_hex(col)} 渐变={q}")
    if args.verbose:
        msg += f" (候选分量中第 {k} 个)"
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
        prog="logo_trace.py",
        description="基于多尺度结构张量的位图描摹 / 矢量化工具"
                    "（结构张量 → 分区路由 → SVG）",
        epilog=EXAMPLES, formatter_class=_Fmt)
    p.add_argument("--version", action="version", version=f"logotrace {__version__}")

    g = p.add_argument_group("输入输出")
    g.add_argument("--in", dest="src", default="inputs/logo.png",
                   help="输入位图; 给裸文件名会自动到 inputs/ 下找")
    g.add_argument("--out", default=None,
                   help="输出 SVG; 默认 out/<输入名>_traced.svg")
    g.add_argument("--preview", nargs="?", const="auto", default=None,
                   help="预览 PNG; 不带值=自动命名到 out/")
    g.add_argument("--no-preview", dest="preview", action="store_const", const="",
                   help="不渲染预览")
    g.add_argument("--debug", nargs="?", const="auto", default=None,
                   help="结构张量调试图; 默认不输出, 不带值=自动命名到 out/")
    g.add_argument("--gzip", action="store_true",
                   help="同时输出 .svgz (gzip -9; 浏览器 / <img> 可直接用)")
    g.add_argument("--compress", choices=["off", "slim", "tight"], default="slim",
                   help="SVG 瘦身: off=原样; slim=数字精度裁剪(逐位无损, 可再 gzip); "
                        "tight=再多删掉重复 clipPath(体积最小, 但笔触会越出所属色块: "
                        "logo 2 倍实测 -1.9 dB / 4 倍 -5.7 dB, 水彩 -0.01 dB; "
                        "**--preset painting 默认就是 tight**). "
                        "硬边图形还想更小, 用 svgzip.py 的 --prec 0(几乎无损, 体积再降约 1/3)")

    g = p.add_argument_group("运行")
    g.add_argument("--scale", type=float, default=4.0,
                   help="描摹前 Lanczos 放大倍数 (硬边图形 2.0 起收益明显)")
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--no-autoscale", dest="no_autoscale", action="store_true",
                   help=f"关掉小图自动缩参 (默认开启: 盘面 < {AUTOSCALE_REF:.0f}px 时按 "
                        f"k=max(W,H)/{AUTOSCALE_REF:.0f} 缩小 min_area/spacing 等绝对像素参数)")
    g.add_argument("--verbose", dest="verbose", action="store_true", default=True,
                   help=argparse.SUPPRESS)
    g.add_argument("--quiet", dest="verbose", action="store_false",
                   help="只打印结束摘要")

    g = p.add_argument_group("预设")
    g.add_argument("--preset", choices=sorted(PRESETS), default=None,
                   help="参数预设 (命令行显式参数优先): "
                        "logo=硬边平面图(2 倍网格 + 亚像素定位 + 保角贝塞尔); "
                        "painting=写意/照片(色块小而多、笔触密而细)")

    g = p.add_argument_group("结构张量")
    g.add_argument("--sigma-d", type=float, default=1.0, help="细尺度 微分尺度 σd")
    g.add_argument("--sigma-i", type=float, default=2.5, help="细尺度 积分尺度 σi")
    g.add_argument("--cs-sigma-d", type=float, default=2.0, help="粗尺度 σd (笔触流场)")
    g.add_argument("--cs-sigma-i", type=float, default=12.0, help="粗尺度 σi (笔触流场)")

    g = p.add_argument_group("分割")
    g.add_argument("--method", choices=["colors", "edges", "hybrid", "watershed"],
                   default="hybrid",
                   help="hybrid=分区路由(默认): 颜色分割定区域身份, 纯色几何区把边界亚像素"
                        "吸附到边缘并走保角贝塞尔, 笔触/渐变复杂区走原来的做法; "
                        "colors=纯颜色分割(Catmull-Rom); edges=纯边缘检测分水岭; "
                        "watershed=能量分水岭")
    g.add_argument("--labels-cache", default="",
                   help="分割结果缓存文件 (.npy): 存在则载入, 否则算完保存; 空=不用")
    g.add_argument("--gpu", choices=["off", "auto", "on"], default="auto",
                   help="结构张量+k-means 是否上 GPU(默认 auto)。auto=可用就用(cupy 或自研 "
                        "CUDA 后端 cuda_backend.py), 没有就静默退回 CPU; off=只用 CPU; "
                        "auto=可用就用(自研 CUDA 后端 cuda_backend.py, 或 cupy), 否则静默"
                        "退回 CPU; on=必须有 GPU。实测: 张量阶段 1.7~2.6×, 端到端因 float32 "
                        "张量变成 37.54 dB (CPU 37.57), 且大图上收益更明显")
    g.add_argument("--jobs", type=int, default=0,
                   help="逐区域阶段的并行进程数 (fork + 写时复制, 大数组不拷贝、不 pickle)。"
                        "0=自动: min(核数,16), 候选区域少于 24 个时用 1; 1=关闭并行。"
                        "无 rng 的环节(区域矢量/活动度/细节层)并行后输出逐字节不变; "
                        "笔触层并行时改用每区域确定性种子, 所以 --jobs 1 才是与 out/ "
                        "逐字节相同的口径")
    g.add_argument("--batch-book", choices=("auto", "off"), default="auto",
                   help="逐区域阶段的规则密集子步骤(掩码腐蚀 / 逐区域中位色)改成整图批量: "
                        "auto=能算就算(GPU 上优先 cupy), off=沿用逐区域循环。批量版与"
                        "逐区域版逐字节相同(1/255 网格上中位数有闭式解); 图经过重采样"
                        "(scale != 1)时像素不在该网格上, 自动回退到逐区域。")
    g.add_argument("--seg", choices=["full", "auto", "flat"], default="full",
                   help="分割路线。full=完整路线(默认, k-means+RAG); auto/flat=平色快路径"
                        "(直方图主色+LUT+连通域)。实测: 硬边平色图标 27.57→27.37 dB 而分割 "
                        "9.3→4.1s; 带渐变的 logo 37.57→34.7 dB —— 所以默认不用, 只建议"
                        "纯硬边平面图显式指定")
    g.add_argument("--flat-cov", type=float, default=0.90,
                   help="快路径判据: 主色桶需覆盖的像素占比, 低于此值走完整路线")
    g.add_argument("--flat-nb", type=int, default=5,
                   help="快路径直方图每通道位数 (5 → 32 级/通道)")
    g.add_argument("--flat-share", type=float, default=2e-4,
                   help="快路径: 一个桶要算作主色的最小像素占比")
    g.add_argument("--flat-merge", type=float, default=0.0,
                   help="快路径: 调色板归并的颜色距离阈值 (0-1)")
    g.add_argument("--edge-nms-hi", type=float, default=0.97,
                   help="edges 方法: 非极大值抑制后的滞后高阈值(能量分位)")
    g.add_argument("--edge-nms-lo", type=float, default=0.92,
                   help="edges 方法: 滞后低阈值(能量分位)")
    g.add_argument("--edge-core", type=float, default=3.0,
                   help="edges 方法: 平坦区种子要求的到边缘距离 (px)")
    g.add_argument("--edge-close", type=int, default=0,
                   help="edges 方法: 边缘闭运算次数 (补 1~2px 缺口)")
    g.add_argument("--edge-dsmooth", type=float, default=0.6,
                   help="edges 方法: 距离场平滑 σ")
    g.add_argument("--kmeans-k", type=int, default=20)
    g.add_argument("--kmeans-iters", type=int, default=16)
    g.add_argument("--pre-smooth", type=float, default=1.6, help="量化前高斯平滑 σ")
    g.add_argument("--thresh", type=float, default=12.0, help="RAG 合并阈值 (0-255 色距)")
    g.add_argument("--merge-thresh2", type=float, default=10.0,
                   help="连通域层面再次 RAG 合并的阈值 (0=关闭)")
    g.add_argument("--detail-chroma", type=float, default=12.0,
                   help="彩色细缝线细节层的 Lab 色度阈值 (0=关闭)")
    g.add_argument("--detail-scale", type=float, default=1.0,
                   help="色度图高斯平滑 σ")
    g.add_argument("--detail-min-area", type=int, default=40)
    g.add_argument("--protect-sat", type=float, default=0.0,
                   help="平均饱和度(rgb 极差)高于此值的彩色区域不参与合并/吸收 "
                        "(0=关闭; 金色缝线已由细节层单独处理)")
    g.add_argument("--merge-passes", type=int, default=4)
    g.add_argument("--min-area", type=int, default=600)
    g.add_argument("--min-width", type=int, default=0,
                   help="标签开运算核宽(px), 0=关闭; 会压缩细长真实特征, 慎用")
    g.add_argument("--denoise", type=float, default=0.0,
                   help="保边平滑 sigma_color (0=关闭; 输入有噪声时建议 0.03~0.05)")
    g.add_argument("--contour-tol", type=float, default=0.75)
    g.add_argument("--contour-smooth", type=float, default=1.0)
    g.add_argument("--fit", choices=["auto", "cr", "bezier"], default="auto",
                   help="曲线拟合: auto=按区域几何度自动选择(几何区 bezier/复杂区 cr); "
                        "cr=Douglas-Peucker+Catmull-Rom; "
                        "bezier=保角最优贝塞尔(直线输出 l, 保尖角)")
    g.add_argument("--fit-tol", type=float, default=0.10,
                   help="bezier 拟合的最大允许误差 (px)")
    g.add_argument("--corner-deg", type=float, default=62.0,
                   help="bezier 拟合的角点判定角度")
    g.add_argument("--auto-geo-frac", type=float, default=0.5,
                   help="--fit auto 的几何度门槛 (区域里几何像素占比)")
    g.add_argument("--tex-sigma", type=float, default=2.5,
                   help="纹理图高通尺度 σ")
    g.add_argument("--tex-smooth", type=float, default=6.0,
                   help="纹理图能量平滑 σ")
    g.add_argument("--tex-norm", type=float, default=99.5,
                   help="纹理归一化分位")
    g.add_argument("--tex-thr", type=float, default=3.0,
                   help="判为几何区的纹理上限 (归一化后)")
    g.add_argument("--range-thr", type=float, default=0.02,
                   help="判为几何区的内部色跨上限 (0.02≈5/255)")
    g.add_argument("--tex-close", type=int, default=0)
    g.add_argument("--tex-open", type=int, default=0)
    g.add_argument("--snap-w", type=float, default=3.0,
                   help="hybrid 方法: 边缘线在高程里的加成权重 (越大越贴边缘)")
    g.add_argument("--snap-sigma", type=float, default=1.0,
                   help="hybrid 方法: 边缘线加成前的平滑 σ")
    g.add_argument("--snap-erode", type=int, default=2,
                   help="hybrid 方法: 颜色区域种子的腐蚀半径 (给边界留吸附余地)")
    g.add_argument("--snap-band", type=float, default=3.0,
                   help="hybrid 方法: 边界最多允许移动多少像素 (0=不限制)")
    g.add_argument("--snap-mode", choices=["refine", "watershed", "off"],
                   default="refine",
                   help="边界吸附方式: refine=轮廓点沿法向亚像素吸到边缘能量脊线 "
                        "(局部、保拓扑, 默认); watershed=标记控制分水岭重切边界; off=不吸附")
    g.add_argument("--snap-shift", type=float, default=1.5,
                   help="refine 模式: 单点最大吸附位移 (px)")
    g.add_argument("--snap-sub", choices=["half", "peak"], default="half",
                   help="亚像素定心: half=亮度剖面 50%% 交点(更准), peak=能量峰值")
    g.add_argument("--snap-step", type=float, default=0.15,
                   help="refine 模式: 法向剖面采样步长 (px)")
    g.add_argument("--aa-levels", type=int, default=0,
                   help="抗锯齿过渡带重建级数 (每侧几条; 0=关闭)。原图边界有 2~3px"
                        "抗锯齿过渡, 九成平方误差集中在这条带上; 但实测把边界定准"
                        "以后, 用同色阶梯逼近斜坡反而略降 PSNR (见 README), 故默认关")
    g.add_argument("--aa-width", type=float, default=2.5,
                   help="抗锯齿过渡带总宽 (px), 应≈原图边界过渡宽度")
    g.add_argument("--aa-smooth", type=float, default=0.8,
                   help="距离场平滑 σ (越小越贴原始边界)")
    g.add_argument("--aa-tol", type=float, default=0.15,
                   help="过渡带轮廓的贝塞尔拟合容差 (px)")
    g.add_argument("--aa-geo-thr", type=float, default=-1.0,
                   help="过渡带准入的几何度阈值 (<0 时用 --auto-geo-frac)")
    g.add_argument("--aa-min-radius", type=float, default=1.4,
                   help="色块最大内切半径小于此值就不做过渡带 (避免两侧偏移交叉)")
    g.add_argument("--aa-profile", choices=["shape", "uniform"], default="shape",
                   help="过渡带分级方式: shape=按实测过渡剖面做最优量化, uniform=等宽")
    g.add_argument("--aa-min-area", type=int, default=8,
                   help="过渡带单段最小像素数")
    g.add_argument("--aa-synthetic", action=argparse.BooleanOptionalAction, default=False,
                   help="在**合成边界**上也生成抗锯齿过渡带. 合成边界 = 自适应细化从同一个原始区域"
                        "切出来的两个子区域之间的分界(颜色天然连续, 过渡带基本是白花的字节); "
                        "背景边界/其它原始区域边界/真实图像边缘不受影响, 始终保留过渡带. "
                        "默认关闭(=省掉合成边界的过渡带); --aa-synthetic 恢复旧行为(逐字节)")
    g.add_argument("--aa-edge-dedup", action=argparse.BooleanOptionalAction, default=True,
                   help="把**同一个原始区域**的祖先一路带到细化树的每一层, 用它去重: 细化递归切开后, "
                        "同一原始区域的后代之间(哪怕不是直接父子, 而是叔侄/堂兄弟)的分界都算合成边界, "
                        "只保留一份过渡带. 关闭时只认一层父子关系(旧行为, 逐字节). "
                        "真实图像边缘的过渡带不受影响 (apple@4: 1223→1043 KB, PSNR/MAE/JUMP 不变)")
    g.add_argument("--aa-edge-canon", action=argparse.BooleanOptionalAction, default=False,
                   help="[实验, 默认关闭] 更激进的去重: 同一原始区域的一组后代里, **真实图像边缘**上只"
                        "允许面积最大的那个后代生成过渡带, 其余后代只保留合成边界上的过渡带. "
                        "实测会改变真实边缘的过渡带(每个后代各自负责自己那一段边缘), 因此不建议开启")
    g.add_argument("--contour-min-area", type=float, default=30.0)

    g = p.add_argument_group("渐变")
    g.add_argument("--grad-stops", type=int, default=10)
    g.add_argument("--grad-min-gain", type=float, default=0.12,
                   help="渐变验收门槛: 线性渐变至少要消掉这个比例的平方误差, 否则该区域退回纯色. "
                        "平滑渐变图(如柔和 logo/渲染图)可降到 0.02 左右, 显著减少色块台阶")
    g.add_argument("--auto-gradient", action="store_true",
                   help="按图像平滑度自动放宽渐变门槛(平滑图自动降门槛), 无需手调")
    g.add_argument("--grad-min-range", type=float, default=0.004,
                   help="启用线性渐变的色跨下限 (0.004≈1/255; 本图渐变极缓)")
    g.add_argument("--grad-radial", action=argparse.BooleanOptionalAction, default=True,
                   help="允许径向渐变(中心+半径)候选: 与线性用同一误差增益判据, 只有当径向残差"
                        "比最优线性小 --grad-radial-margin 时才采用 (--no-grad-radial 关闭)")
    g.add_argument("--grad-radial-margin", type=float, default=GRAD_RADIAL_MARGIN,
                   help="径向采用的残差优势门槛: 径向残差需比线性残差至少小这个比例")
    g.add_argument("--merge-grad", action=argparse.BooleanOptionalAction, default=True,
                   help="梯度感知区域合并: 相邻色块的并集若能仍被单个渐变(线性/径向)解释, 就合并成"
                        "更大更平滑的区域, 消除色块台阶 (--no-merge-grad 关闭)")
    g.add_argument("--merge-grad-tol", type=float, default=0.05,
                   help="合并判据: 并集的单渐变残差 <= 两个独立拟合残差之和 ×(1+该值)")
    g.add_argument("--merge-grad-passes", type=int, default=4,
                   help="梯度感知合并的迭代轮数 (每轮固定扫描顺序, 结果确定)")

    g = p.add_argument_group("自适应细化 (误差驱动)")
    g.add_argument("--adaptive-refine", action=argparse.BooleanOptionalAction, default=True,
                   help="误差驱动自适应细化: 逐区域量测当前填充(平色/线性/径向)的残差, 残差超阈值"
                        "就按误差最大方向二分并递归重拟合, 把固定的 SVG 元素预算花在最需要的地方, "
                        "消除平滑/光泽图上的可见平色块 (--no-adaptive-refine 恢复旧行为; 与旧版"
                        " --no-grad-radial --no-merge-grad 一样逐字节复原旧输出)")
    g.add_argument("--refine-err", type=float, default=0.02,
                   help="细化的残差阈值(满量程比例, 0.02≈5/255): 区域内部(腐蚀后的核心)中最大通道"
                        "误差超过该值的像素数达到 max(16, max(32, refine-min-area/4)/2) 以上(即"
                        "确实存在一块可见的误差)才考虑二分; 局部判据很关键, 大区域上的小块误差几乎"
                        "不抬高全局 RMS")
    g.add_argument("--refine-min-area", type=int, default=400,
                   help="参与细化的最小区域面积; 小于该值的区域保持原样 (防止追着噪点/细缝走)")
    g.add_argument("--refine-max-depth", type=int, default=6,
                   help="递归细化的最大深度 (0=只做一轮不递归)")
    g.add_argument("--refine-gain", type=float, default=0.20,
                   help="二分被接受所需的最小残差下降比例 (子区域面积加权 RMS <= (1-该值)×父区域); "
                        "这是对噪声/纹理的天然保护, 也是生长速度的阀门")
    g.add_argument("--refine-budget", type=int, default=0,
                   help="细化新增区域数的上限; 0=自动 (max(64, min(256, 区域数)))")
    g.add_argument("--refine-order", choices=("merge-first", "refine-first"), default="merge-first",
                   help="细化与梯度感知合并的先后: merge-first=先合并再细化(默认); "
                        "refine-first=先在原始分区上细化, 再对细化后的分区做梯度合并")
    g.add_argument("--dump-refined-labels", default="",
                   help="把细化后的标签图写成 .npy (诊断 / 复现 refine-first 顺序); 空=不写")

    g = p.add_argument_group("网格渐变 (SVG 2 mesh; 默认关闭)")
    g.add_argument("--grad-mesh", action=argparse.BooleanOptionalAction, default=False,
                   help="允许把 SVG 2 网格渐变 (Coons patch mesh) 当作大块平滑区域的填充候选: "
                        "只有残余误差比现有线性/径向按 --grad-mesh-margin 更优时才采用. "
                        "**兼容性警告**: 网格渐变需要支持 SVG 2 mesh 的渲染器; 主流浏览器"
                        "(Chrome/Firefox/Safari)与 cairosvg 都不支持, 而且它们也不支持 SVG 2 的 "
                        "paint 回退列表, 所以只写 fill=\"url(#meshN) url(#gN)\" 时这些渲染器会解析"
                        "失败而完全不画该区域(实测 cairosvg 输出透明, 不是回退色). 因此默认还会在"
                        "网格路径下面再画一遍回退填充(--grad-mesh-underlay), 让这些渲染器看到回退"
                        "渐变; 网格本身默认关闭")
    g.add_argument("--grad-mesh-margin", type=float, default=0.10,
                   help="采用网格所需的相对优势: 网格全域 RMS 需比现有填充的 RMS 小这个比例")
    g.add_argument("--grad-mesh-patches", type=int, default=2,
                   help="网格每边的 patch 数 (2 → 3x3 个颜色停靠点)")
    g.add_argument("--grad-mesh-min-area", type=int, default=4000,
                   help="只对面积不小于该值的区域尝试网格 (小区域单渐变足够)")
    g.add_argument("--grad-mesh-underlay", action=argparse.BooleanOptionalAction, default=True,
                   help="在 mesh 路径下面再画一遍回退填充 (默认开). cairosvg/浏览器不支持 SVG 2 的 "
                        "paint 回退列表, 只写 fill=\"url(#meshN) url(#gN)\" 时它们会解析失败而根本不画"
                        "该区域(实测 cairosvg 输出透明); 加一层回退路径后这些渲染器能看到回退填充, "
                        "支持 mesh 的渲染器仍会把网格画在上面. --no-grad-mesh-underlay 只保留 paint "
                        "回退列表的纯 SVG 2 写法")

    g = p.add_argument_group("笔触 (结构张量流线)")
    g.add_argument("--no-strokes", action="store_true",
                   help="只出几何层; --preset logo 默认即开启")
    g.add_argument("--spacing", type=float, default=10.0, help="笔触间距")
    g.add_argument("--occ-ratio", type=float, default=0.45, help="占位半径/间距")
    g.add_argument("--stroke-step", type=float, default=3.0)
    g.add_argument("--stroke-max", type=float, default=120.0)
    g.add_argument("--stroke-min", type=float, default=24.0)
    g.add_argument("--stroke-nodes", type=int, default=12)
    g.add_argument("--stroke-width", type=float, default=7.5)
    g.add_argument("--stroke-grid-units", action="store_true",
                   help="笔触长度参数按描摹网格px解释(旧行为); 默认按原生像素, 自动乘 --scale")
    g.add_argument("--stroke-alpha", type=float, default=0.6)
    g.add_argument("--stroke-jitter", type=float, default=0.012)
    g.add_argument("--stroke-activity-ref", type=float, default=0.012,
                   help="笔触透明度=stroke-alpha×min(1,残差/该值); 残差来自'图像-底填'")
    g.add_argument("--stroke-alpha-min", type=float, default=0.03,
                   help="低于此透明度的笔触直接丢弃")
    g.add_argument("--coh-min", type=float, default=0.12)
    g.add_argument("--max-turn", type=float, default=70.0, help="单步最大转折角(度)")

    g = p.add_argument_group("边缘 / 缝线 (本图的金色缝线已由色块层重建, 故默认关闭)")
    g.add_argument("--edge-mode", choices=["off", "chroma", "energy", "both"], default="off",
                   help="off=不画; chroma=Lab 色度细线; energy=结构张量能量脊线")
    g.add_argument("--edge-chroma", type=float, default=9.0, help="Lab 色度阈值 (chroma 模式)")
    g.add_argument("--edge-hi", type=float, default=0.975, help="高能量分位 (energy 模式)")
    g.add_argument("--edge-lo", type=float, default=0.90, help="滞后低阈值分位 (energy 模式)")
    g.add_argument("--edge-coh", type=float, default=0.45, help="相干性下限 (energy 模式)")
    g.add_argument("--edge-width", type=float, default=2.0)
    g.add_argument("--edge-alpha", type=float, default=0.9)
    g.add_argument("--edge-min-area", type=int, default=24)
    g.add_argument("--edge-min-len", type=int, default=8)
    g.add_argument("--edge-tol", type=float, default=0.6, help="骨架路径简化容差")
    pre, _ = p.parse_known_args(argv)
    if pre.preset:
        p.set_defaults(**PRESETS[pre.preset])
    return p.parse_args(argv)


# ======================================================================
# 9. output post-processing: slimming / paths / directories
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
    global _QUIET
    args = parse_args(argv)
    # Gradient gate: explicit value, or auto-tuned from image smoothness (--auto-gradient).
    global GRAD_MIN_GAIN
    _argv = list(sys.argv[1:] if argv is None else argv)
    # Was --aa-levels given by the user? (a preset value does not count as explicit)
    _aa_explicit = any(_a == "--aa-levels" or _a.startswith("--aa-levels=") for _a in _argv)
    _re_explicit = any(_a == "--refine-err" or _a.startswith("--refine-err=") for _a in _argv)
    if args.auto_gradient:
        from PIL import Image as _Im
        with _Im.open(args.src) as _im:
            _a = np.asarray(_im.convert("RGB").resize((64, 64))).astype(np.float64) / 255.0
        _g = float(np.abs(np.diff(_a, axis=0)).mean() + np.abs(np.diff(_a, axis=1)).mean())
        GRAD_MIN_GAIN = 0.02 if _g > 0.008 else args.grad_min_gain
        if _g > 0.008:
            # Smooth/glossy images: be aggressive about merging patches into large gradient-filled
            # regions -- this is what removes the visible colour steps the user sees. Measured on
            # apple.png @scale4: >=40px plateaus 35.5% -> 26.6% (source 24.8%), JUMP% back to the
            # source level, for about 0.5 dB of PSNR. Texture-rich photos should use --no-merge-grad,
            # where merging costs ~0.5 dB and radial-only is strictly better.
            args.merge_grad_tol = max(args.merge_grad_tol, 0.3)
            args.merge_grad_passes = max(args.merge_grad_passes, 8)
        print(f"      · 自动渐变路由: 平滑度 {_g:.4f} → 渐变门槛 {GRAD_MIN_GAIN}",
              file=sys.stderr)
        # Smooth images: their remaining hard steps are the k-means patch boundaries, so also turn on
        # the boundary anti-aliasing staircase (3 levels) unless the user pinned --aa-levels. Tied to
        # the smooth-image feature stack so that --no-grad-radial --no-merge-grad still reproduces the
        # legacy output byte for byte for every other flag combination.
        if (_g > 0.008 and not _aa_explicit and int(args.aa_levels) <= 0
                and (args.grad_radial or args.merge_grad)):
            args.aa_levels = 3
            print("      · 平滑图: 边界抗锯齿过渡带自动开启 (--aa-levels 3, 可用 --aa-levels 0 关闭)",
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
                print(f"      · 平滑图: 自适应细化误差门槛收紧 {_re0} → {args.refine_err} "
                      f"(可用 --refine-err 覆盖)", file=sys.stderr)
    else:
        GRAD_MIN_GAIN = args.grad_min_gain
    _QUIET = not args.verbose
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
        sys.exit(f"[错误] 找不到输入文件: {args.src}")
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
            log(f"      · 小图自动缩参 k={k:.3f} (参考盘面 {AUTOSCALE_REF:.0f}px): "
                f"min_area={args.min_area}  spacing={args.spacing:.1f}px")
    if args.scale != 1.0:
        im = im.resize((max(32, int(W0 * args.scale)), max(32, int(H0 * args.scale))),
                       Image.LANCZOS)
    W, H = im.size
    rgb = np.asarray(im).astype(np.float64) / 255.0
    log(f"[{el()}] 读入 {args.src}  {W}x{H}"
          + (f" (原图 {W0}x{H0})" if args.scale != 1.0 else ""))

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
            log(f"[{el()}] 结构张量: GPU {gpu.device_name()} "
                f"(显存内, 与 CPU 同双精度累加口径)")
        except Exception as _e:
            print(f"      ! GPU 结构张量失败({_e}), 退回 CPU", file=sys.stderr)
            tf = tc = None
    if tf is None:
        tf = structure_tensor(rgb_s, args.sigma_d, args.sigma_i)
        tc = structure_tensor(rgb_s, args.cs_sigma_d, args.cs_sigma_i)
    log(f"[{el()}] 结构张量: 细尺度(σd={args.sigma_d},σi={args.sigma_i}) "
          f"平均相干={tf['coh'].mean():.3f} | 粗尺度(σd={args.cs_sigma_d},"
          f"σi={args.cs_sigma_i}) 平均相干={tc['coh'].mean():.3f}")

    # ---------------- segmentation ----------------
    # --labels-cache: cache the segmentation result. When tuning stroke/gradient/fitting parameters there is no need to rerun segmentation,
    # and it is off by default so default results are unaffected.
    geo_px = None
    if args.labels_cache and os.path.exists(args.labels_cache):
        labels = np.load(args.labels_cache).astype(np.int32)
        log(f"[{el()}] 载入分割缓存 {args.labels_cache} "
              f"({int(labels.max()) + 1} 个色块)")
    else:
        if args.method == "colors":
            labels = segment_colors(rgb_s, rgb255, args, rng)
        elif args.method == "edges":
            labels = segment_edges(rgb_s, tf, args, rng)
        elif args.method == "hybrid":
            if args.seg in ("auto", "flat"):
                labels = segment_flat(rgb_s, rgb255, args, rng)
                if labels is None and args.seg == "flat":
                    print("      ! 平色快路径不适用, 退回完整路线", file=sys.stderr)
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
        log(f"[{el()}] 梯度感知合并: {_n_mg0} → {_n_mg1} 个色块 "
            f"(-{_n_mg0 - _n_mg1}, {time.time() - _t_mg:.1f}s)")
    n_lab = int(labels.max()) + 1
    areas = np.bincount(labels.ravel(), minlength=n_lab)
    log(f"[{el()}] 分割: {n_lab} 个色块, 面积 {int(areas.min())}~{int(areas.max())} px")
    # shared partition-routing criteria (used by curve fitting / boundary snapping / anti-aliasing bands)
    geo_r, tex_r_all, rng_r_all, _t_norm = classify_regions(
        rgb_s, labels, areas, args.tex_sigma, args.tex_smooth, args.tex_norm,
        args.tex_thr, args.range_thr, args.min_area)
    if args.verbose:
        _keep = [int(li) for li in np.nonzero(areas >= args.min_area)[0]]
        _ngeo = sum(1 for li in _keep if geo_r[li])
        _head = ", ".join("#%d:%s(纹理%.1e/色跨%.3f)"
                          % (li, "几何" if geo_r[li] else "复杂",
                             tex_r_all[li], rng_r_all[li]) for li in _keep[:8])
        print("      · 分区路由(统一判据): %d/%d 个色块判为几何区%s"
              % (_ngeo, len(_keep), ("; 例: " + _head) if _head else ""))
    # energy threshold for subpixel edge snapping (the same threshold as the edge map)
    thr_ref = float(np.quantile(tf["energy"], args.edge_nms_hi))
    _REFINE_STATS.update(pts=0, moved=0, sum=0.0)

    # ---------------- background region ----------------
    border = np.zeros((H, W), bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    bl = np.bincount(labels[border].ravel(), minlength=n_lab)
    bg_label = int(np.argmax(bl))
    if labels[1, 1] == bg_label:
        pass
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
            log(f"[{el()}] 批量簿记: 不可用(像素不在 1/255 网格 / 标签过多), 回退逐区域")
        else:
            log(f"[{el()}] 批量簿记: {len(areas)} 个标签的掩码+中位色一次算完 "
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
        log(f"[{el()}] 逐区域阶段: {len(_order)} 个候选区域, {_jobs} 进程并行")
        with multiprocessing.get_context("fork").Pool(_jobs) as _pool:
            for _r, _msg, _st in _pool.imap(_par_region_one, _order, chunksize=4):
                if _r is not None:
                    regions.append(_r)
                if _msg:
                    log(_msg)
                _REFINE_STATS["pts"] += _st["pts"]
                _REFINE_STATS["moved"] += _st["moved"]
                _REFINE_STATS["sum"] += _st["sum"]
        log(f"      · 并行组装 {len(regions)} 个区域用了 {time.time() - _t_par:.1f}s")
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
                q = f"{grad['quality']:.2f}({grad['axis']})" if grad else "平色"
                log(f"      · 区域#{li:<2d} area={int(areas[li]):>7d} {to_hex(col)} "
                      f"渐变质量={q} 拟合={use_fit}(几何度{geo_frac:.2f})")

        log(f"      · 串行组装 {len(regions)} 个区域用了 {time.time() - _t_ser:.1f}s")

    # ---- error-driven adaptive refinement (see refine_regions) ----
    # Gated on the gradient feature stack as well, so that the legacy flat-colour route
    # (--no-grad-radial --no-merge-grad) still reproduces the old output byte for byte.
    if args.adaptive_refine and (args.grad_radial or args.merge_grad):
        _t_rf = time.time()
        _lab_ref = labels.copy()
        regions, _rst = refine_regions(rgb_s, tc, regions, args, tf, thr_ref, labels_out=_lab_ref)
        _extra = f", {len(regions) - _rst['n0']:+d}" if _rst["n1"] != _rst["n0"] else ""
        log(f"[{el()}] 自适应细化: {_rst['n0']} → {_rst['n1']} 个区域{_extra} "
            f"(二分 {_rst['split']} 次, 其中背景 {_rst['bg_split']}, 最大深度 {_rst['depth']}, "
            f"评估 {_rst['eval']}, 放弃 {_rst['rejected']}, 预算 {_rst['budget']}, "
            f"用时 {time.time() - _t_rf:.1f}s)")
        if args.merge_grad and args.refine_order == "refine-first":
            _t_rm = time.time()
            _mg, _m0, _m1 = merge_gradient_regions(rgb_s, _lab_ref, tc, args)
            if _m1 < _m0:
                regions = regroup_regions(rgb_s, regions, _lab_ref, _mg, args, tc, tf, thr_ref)
                _lab_ref = _mg
            log(f"[{el()}] 细化后梯度合并(refine-first): {_m0} → {_m1} 个色块, "
                f"重组为 {len(regions)} 个区域 ({time.time() - _t_rm:.1f}s)")
        labels = _lab_ref                      # keep AA bands / debug consistent with the refined ids
        if args.dump_refined_labels:
            np.save(args.dump_refined_labels, labels)

    # ---- mesh-gradient candidate (SVG 2 meshgradient; opt-in, see section 3.7) ----
    if args.grad_mesh:
        _t_ms = time.time()
        _ms = apply_mesh_fills(rgb_s, regions, args)
        _mr0 = _ms["rms0"] / _ms["n"] if _ms["n"] else 0.0
        _mr1 = _ms["rms1"] / _ms["n"] if _ms["n"] else 0.0
        log(f"[{el()}] 网格渐变候选: {_ms['n']}/{_ms['tried']} 个大区域采用 mesh "
            f"(平均 RMS {_mr0:.4f} → {_mr1:.4f}, 最大改善 {_ms['gain_max'] * 100:.1f}%, "
            f"每边 {args.grad_mesh_patches} patch, 用时 {time.time() - _t_ms:.1f}s)")
        if args.verbose and _ms["n"]:
            print("      · 注意: mesh 填充带 paint 回退列表 url(#meshN) url(#gN); "
                  "cairosvg/主流浏览器不支持 mesh 也不支持回退列表")

    # ---- anti-aliasing transition bands: hard edges -> the source image's 2~3px transition (aa_band_regions) ----
    if args.aa_levels > 0 and args.aa_width > 0:
        t_aa = time.time()
        aa_segs = aa_band_regions(rgb_s, regions, labels, args, tf, thr_ref)
        regions.extend(aa_segs)
        if aa_segs:
            print("[%s] 抗锯齿过渡带: %d 段 (每侧 %d 级, 总宽 %.1fpx), 覆盖 %d px, "
                  "跳过合成边界 %d 段, 跳过重复真实边缘 %d 段 | %.1fs"
                  % (el(), len(aa_segs), args.aa_levels, args.aa_width,
                     sum(a["area"] for a in aa_segs), _AA_STATS["skipped_syn"],
                     _AA_STATS["skipped_canon"], time.time() - t_aa))

    # ---- detail layer: colored thin seam lines (golden strokes) ----
    # k-means is very unfriendly to "thin and colored" narrow bands (they often get merged into the neighbouring large region), yet such seam lines
    # are exactly the key elements of a logo, so they are extracted separately as one vector layer using Lab chroma (drawn on top).
    if _REFINE_STATS["moved"] > 0 and args.verbose:
        _m = _REFINE_STATS["moved"]
        log(f"      · 亚像素边缘吸附: {_m}/{_REFINE_STATS['pts']} 个轮廓点吸到边缘上, "
              f"平均位移 {_REFINE_STATS['sum'] / _m:.3f}px (上限 {args.snap_shift}px)")
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
        log(f"[{el()}] 细节层(Lab 色度>{args.detail_chroma}): {n_det} 条彩色缝线 "
              f"(候选分量 {len(_ks)}, {_jobs_det} 进程, 用时 {time.time() - _t_det:.1f}s)")

    n_aa = sum(1 for r in regions if r.get("aa"))
    n_grad = sum(1 for r in regions if r["grad"] is not None)
    n_rad = sum(1 for r in regions if r["grad"] is not None and r["grad"].get("kind") == "radial")
    log(f"[{el()}] 区域矢量: {len(regions) - n_aa} 条路径 (另有抗锯齿带 {n_aa} 段), "
          f"其中 {n_grad} 个带渐变 (线性 {n_grad - n_rad} / 径向 {n_rad})")

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
        print(f"      活动度 residual: 均值={act[act>0].mean()*255:.2f}/255 "
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
                        log(f"      · 区域#{_g['idx']} (area={_g['area']}) "
                              f"→ {len(_g['strokes'])} 笔触")
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
                    log(f"      · 区域#{r['idx']} (area={r['area']}) → {len(items)} 笔触")
    n_strokes = sum(len(g["strokes"]) for g in stroke_groups)
    log(f"[{el()}] 笔触: {n_strokes} 条 (间距 {args.spacing}px)")

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
        log(f"[{el()}] 边缘/缝线: {len(edges)} 条 ({args.edge_mode} 模式)")
    else:
        log(f"[{el()}] 边缘/缝线: 关闭 (色块层已包含金色缝线)")

    # ---------------- emit SVG ----------------
    meta = {"bg_fill": to_hex(bg_col), "src": os.path.basename(args.src),
            "desc": (f"structure-tensor vector tracing | fine(σd={args.sigma_d},σi={args.sigma_i}) "
                     f"coarse(σd={args.cs_sigma_d},σi={args.cs_sigma_i}) | regions={len(regions)} "
                     f"gradients={n_grad} strokes={n_strokes} edges={len(edges)}")}
    svg = build_svg(W0, H0, regions, stroke_groups, edges, meta, view_box=(W, H),
                    mesh_underlay=bool(args.grad_mesh_underlay))
    raw_kb = len(svg.encode()) / 1024
    if args.compress != "off":
        svg = slim_svg(svg, prec_contour=1,
                       prec_stroke=(0 if args.compress == "tight" else 1),
                       drop_clip=(args.compress == "tight"))
    ensure_dir(args.out)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(svg)
    out_kb = os.path.getsize(args.out) / 1024
    extra = "" if args.compress == "off" else f"，{args.compress} 瘦身 {raw_kb:.0f}→{out_kb:.0f} KB"
    log(f"[{el()}] 写出 {args.out} ({out_kb:.1f} KB{extra})")

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
        log(f"      · 调试图 {args.debug}: 线段方向=等照度线, 色相=相干性, 绿=色块边界")

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
                log(f"      · 预览 {args.preview}: 渲回原生 {W0}x{H0}, MAE={mae:.2f}/255 PSNR={psnr:.2f} dB")
        except ImportError:
            print("      · 未安装 cairosvg, 跳过预览渲染")
        except Exception as exc:
            log(f"      · 预览渲染失败: {exc}")

    # ---------------- final summary ----------------
    log(f"[{el()}] 完成")
    if W != W0:
        grid = f"  →  网格 {W}x{H} (--scale {args.scale:g})"
    else:
        grid = ""
    print("  " + "-" * 62)
    print(f"  输入   {args.src}  {W0}x{H0}{grid}")
    print(f"  分区   {len(regions)} 区域 · {n_grad} 渐变 · {n_strokes} 笔触 · {len(edges)} 缝线")
    print(f"  矢量   {args.out}  {out_kb:.0f} KB" + (f"   ({psnr:.2f} dB)" if psnr else ""))
    if gz_path:
        print(f"  压缩   {gz_path}  {os.path.getsize(gz_path)/1024:.0f} KB")
    if args.preview and os.path.exists(args.preview):
        print(f"  预览   {args.preview}")
    if args.debug:
        print(f"  调试   {args.debug}")
    print("  " + "-" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
