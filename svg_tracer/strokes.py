"""Isophote streamlines: even-spaced line integration, ribbon geometry and along-line colour sampling."""
from __future__ import annotations

import math

import numpy as np

from .geometry import bilin, polyline_to_bezier_d

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


