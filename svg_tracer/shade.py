"""Stacked translucent radial gradients (gradient boosting) that correct the final composite on smooth shading."""
from __future__ import annotations

import io
import math
import time

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import measure

from .geometry import fnum

# ======================================================================
# 3.8  stacked translucent radial gradients ("gradient boosting" for smooth shading)
# ======================================================================
# Per-region fills are fitted one region at a time, so two neighbouring fills disagree by 1..6/255
# along their shared boundary. The eye reads those steps as flat polygonal patches (Mach banding)
# long before the overall pixel error looks bad: on apple.png @scale4 the smooth-area error is only
# ~1.4/255, yet 1018 of the 1165 region boundaries that sit in a locally smooth part of the source
# carry a step >= 1/255 and 626 of them >= 4/255.
#
# This stage stacks N translucent radial gradients on top of the finished artwork to cancel the
# residual shading. A layer composites as
#
#     new = (1 - a) * cur + a * C
#
# with a continuous alpha a(t) and a continuous radial colour C(t). Both are continuous in space, so a
# boundary step d of the current render can only be *attenuated* -- to (1 - a_at_boundary) * d -- and
# never cancelled by a smooth correction. That single fact drives the design:
#   * alpha must be large exactly where the boundaries are, so the disks are placed on the large
#     low-gradient areas and their colour model is fitted there;
#   * alpha is not free but a decreasing ramp A(t) = eps * clip((1 - t) / (1 - t_edge), 0, 1), so a
#     layer has no visible element edge at t = 1 and overlaps its neighbours smoothly;
#   * a layer is only accepted when it strictly reduces the smooth-weighted error of its whole disk,
#     so the stack is a boosting sequence of small improvements rather than one big gamble.
#
# Well-posedness: colour and alpha are inseparable in the composite (only a * (C - cur) is
# observable), so alpha is fixed by the parametrisation above and only the radial colour curve C(t),
# piecewise linear on --shade-stops knots, is solved for by weighted least squares. C is clamped to
# the local target percentile range so the model cannot extrapolate into a halo where a disk overlaps
# an edge, and non-smooth pixels inside a disk may not move by more than --shade-halo.
#
# Output: one <radialGradient> per layer plus one full-canvas <rect> per layer inside a single
# <g id="shade"> group drawn last. Plain SVG 1.1 (gradientUnits="userSpaceOnUse" + stop-opacity), so
# cairosvg and mainstream browsers paint it.


def _shade_alpha(t, eps, tedge):
    """Decreasing alpha ramp: eps at the centre, 0 at t = 1, so the element has no visible edge."""
    return eps * np.clip((1.0 - t) / max(1.0 - tedge, 1e-6), 0.0, 1.0)


def _shade_basis(t, knots):
    """Piecewise linear hat basis on the colour knots (a partition of unity)."""
    m = len(knots)
    B = np.zeros((t.size, m))
    for j in range(m):
        lo = knots[j - 1] if j > 0 else -1e9
        hi = knots[j + 1] if j < m - 1 else 1e9
        B[:, j] = np.clip(np.minimum((t - lo) / max(knots[j] - lo, 1e-9),
                                     (hi - t) / max(hi - knots[j], 1e-9)), 0.0, 1.0)
    return B


def _shade_win(cx, cy, r, W, H):
    return (max(0, int(cy - r) - 1), min(H, int(cy + r) + 2),
            max(0, int(cx - r) - 1), min(W, int(cx + r) + 2))


def _shade_smooth_weight(src255, gtol, ttol):
    """0..1 weight: 1 inside large low-gradient areas, 0 on texture and hard edges.

    `gmag` is the smoothed gradient magnitude (an edge detector) and `tex` the smoothed energy of the
    detail removed by a 2.5px low pass (a texture detector, which is what keeps the layers off grass,
    noise and JPEG ringing while still allowing them on a shallow colour ramp).
    """
    gray = src255 @ np.array([0.299, 0.587, 0.114])
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = (gray[:, 2:] - gray[:, :-2]) * 0.5
    gy[1:-1, :] = (gray[2:, :] - gray[:-2, :]) * 0.5
    gmag = ndi.gaussian_filter(np.hypot(gx, gy), 1.5)
    tex = ndi.gaussian_filter(((src255 - ndi.gaussian_filter(src255, (2.5, 2.5, 0.0))) ** 2).sum(-1),
                              6.0)
    return (np.clip(1.0 - (gmag / max(gtol, 1e-6)) ** 2, 0.0, 1.0)
            * np.clip(1.0 - (tex / max(ttol, 1e-6)) ** 2, 0.0, 1.0))


def _shade_candidates(rw, n, grid, margin):
    """N non-maximum-suppressed peaks of `rw` on a coarse grid (cheap and deterministic)."""
    H, W = rw.shape
    gh, gw = H // grid, W // grid
    if gh < 1 or gw < 1:
        return [], []
    bs = rw[:gh * grid, :gw * grid].reshape(gh, grid, gw, grid).max(axis=(1, 3))
    out, tmp = [], bs.copy()
    rr = max(1, int(margin // grid))
    for _ in range(max(0, n)):
        k = int(np.argmax(tmp))
        iy, ix = divmod(k, bs.shape[1])
        if tmp[iy, ix] <= 1e-9:
            break
        out.append((iy * grid + grid // 2, ix * grid + grid // 2))
        tmp[max(0, iy - rr):iy + rr + 1, max(0, ix - rr):ix + rr + 1] = -1.0
    return [o[0] for o in out], [o[1] for o in out]


def _shade_fit_layer(src, cur, w, cx, cy, r, knots, eps, tedge, tmin, halo_cap, tau):
    """Best radial colour curve for one candidate disk, or None when the disk is not usable.

    `src`/`cur` are (H,W,3) in 0..255, `w` the smooth weight, `tmin` the share of non-smooth pixels a
    disk interior may contain. The colour curve solves the weighted least squares problem
    min_C sum w * a^2 * (C(t) - u)^2 with u = cur + (src - cur) / a, i.e. the colour that would make
    the composite equal the target; a is the fixed alpha ramp, hence the a^2 in the weight.
    """
    H, W = src.shape[:2]
    y0, y1, x0, x1 = _shade_win(cx, cy, r, W, H)
    if x1 - x0 < 6 or y1 - y0 < 6:
        return None
    YY, XX = np.mgrid[y0:y1, x0:x1]
    d = np.hypot(XX - cx, YY - cy)
    inside = d <= r
    if inside.sum() < 40:
        return None
    wsub = w[y0:y1, x0:x1]
    t = np.clip(d / r, 0.0, 1.0).ravel()
    # A disk interior that touches a hard edge is rejected outright: the radial model would have to
    # repaint that edge, which is exactly the artefact this stage must not create.
    inner0 = d <= 0.7 * r
    if inner0.sum() > 0 and float((wsub[inner0] < 0.3).mean()) > tmin:
        return None
    a = _shade_alpha(t, eps, tedge)
    ww = (wsub * inside).ravel()
    wt = ww * a * a
    if wt.sum() < 1e-6:
        return None
    cur_w = cur[y0:y1, x0:x1].reshape(-1, 3)
    src_w = src[y0:y1, x0:x1].reshape(-1, 3)
    r0 = src_w - cur_w
    err0 = float((ww[:, None] * r0 * r0).sum())
    if err0 <= 1e-6:
        return None
    B = _shade_basis(t, knots)
    u = cur_w + r0 / np.maximum(a, 1e-4)[:, None]
    M = B.T @ (B * wt[:, None])
    M = M + np.eye(M.shape[0]) * (1e-8 * np.trace(M) / M.shape[0] + 1e-12)
    col = np.stack([np.linalg.solve(M, B.T @ (wt * u[:, k])) for k in range(3)], 1)
    inr = t < 0.8
    if inr.sum() > 20:
        lo = np.percentile(src_w[inr], 1, axis=0) - 2.0
        hi = np.percentile(src_w[inr], 99, axis=0) + 2.0
        col = np.clip(col, lo, hi)
    col = np.clip(col, 0.0, 255.0)
    new = (1.0 - a[:, None]) * cur_w + a[:, None] * (B @ col)
    err1 = float((ww[:, None] * (src_w - new) ** 2).sum())
    # Halo guard: the smooth weight ignores hard edges, so the colour model is not allowed to move
    # non-smooth pixels inside the disk (that is precisely what a visible halo would be).
    nsm = inside.ravel() & (ww < 0.3)
    if nsm.any() and float(np.sqrt(((new - cur_w) ** 2)[nsm].mean())) > halo_cap:
        return None
    if err1 > err0 * (1.0 + tau):
        return None
    return {"cx": float(cx), "cy": float(cy), "r": float(r), "eps": float(eps),
            "tedge": float(tedge), "knots": [float(k) for k in knots], "col": col,
            "red": err0 - err1, "gain": 1.0 - err1 / err0}


def _shade_apply(cur, lay):
    """Composite one fitted layer into `cur` in place (same maths cairosvg performs)."""
    H, W = cur.shape[:2]
    y0, y1, x0, x1 = _shade_win(lay["cx"], lay["cy"], lay["r"], W, H)
    YY, XX = np.mgrid[y0:y1, x0:x1]
    d = np.hypot(XX - lay["cx"], YY - lay["cy"])
    if not (d <= lay["r"]).any():
        return
    t = np.clip(d / lay["r"], 0.0, 1.0)
    a = _shade_alpha(t.ravel(), lay["eps"], lay["tedge"])[:, None]
    col = _shade_basis(t.ravel(), np.asarray(lay["knots"])) @ lay["col"]
    sub = cur[y0:y1, x0:x1]
    cur[y0:y1, x0:x1] = (1.0 - a.reshape(sub.shape[:2] + (1,))) * sub \
        + a.reshape(sub.shape[:2] + (1,)) * col.reshape(sub.shape[:2] + (3,))


def _shade_emit(layers, W, H):
    """(defs, group) strings for one <radialGradient> + one full-canvas <rect> per layer."""
    defs = []
    body = ['<g id="shade">']
    for i, ly in enumerate(layers):
        gid = f"shd{i}"
        stops = []
        for j, t in enumerate(ly["knots"]):
            al = _shade_alpha(np.array([t]), ly["eps"], ly["tedge"])[0]
            c = [int(round(min(255.0, max(0.0, v)))) for v in ly["col"][j]]
            stops.append(f'<stop offset="{fnum(t, 3)}" stop-color="#{c[0]:02x}{c[1]:02x}{c[2]:02x}" '
                         f'stop-opacity="{al:.4f}"/>')
        defs.append(f'<radialGradient id="{gid}" gradientUnits="userSpaceOnUse" '
                    f'cx="{fnum(ly["cx"])}" cy="{fnum(ly["cy"])}" r="{fnum(ly["r"])}">'
                    + "".join(stops) + "</radialGradient>")
        body.append(f'<rect x="0" y="0" width="{fnum(W)}" height="{fnum(H)}" fill="url(#{gid})"/>')
    body.append("</g>")
    return defs, "\n".join(body)


def build_shade_stack(src255, base_svg, W, H, args, log_fn=None):
    """Fit the stack on top of `base_svg`; returns (defs, group, stats) or None.

    `base_svg` is the finished artwork *without* the shade group, rendered at the tracing grid so the
    residual is measured on exactly the pixels this stage will correct.
    """
    def _log(msg):
        if log_fn is not None:
            log_fn(msg)

    try:
        import cairosvg
    except ImportError:
        _log("      · cairosvg is not installed, skipping the shade-layer fit (--shade-blobs needs it to measure the base residual)")
        return None
    try:
        png = cairosvg.svg2png(bytestring=base_svg.encode("utf-8"),
                               output_width=W, output_height=H)
        cur = np.asarray(Image.open(io.BytesIO(png)).convert("RGB")).astype(np.float64)
    except Exception as exc:  # pragma: no cover - renderer/environment dependent
        _log(f"      · base render failed, skipping the shade layers: {exc}")
        return None
    if cur.shape[:2] != (H, W):
        _log(f"      · base size {cur.shape[1]}x{cur.shape[0]} != grid {W}x{H}, skipping the shade layers")
        return None

    src = np.clip(src255, 0.0, 255.0)

    def _psnr(a):
        mse = float(((a - src) ** 2).mean())
        return 99.0 if mse <= 0 else 10 * math.log10(255.0 ** 2 / mse)

    psnr0 = _psnr(cur)
    w = _shade_smooth_weight(src, args.shade_smooth_gtol, args.shade_smooth_ttol)
    # Only large low-gradient areas take part: small smooth islands are texture, and correcting them
    # would spend layers on patches far too small to read as flat.
    lb = measure.label(w > 0.5, connectivity=2)
    sizes = np.bincount(lb.ravel())
    keep = [int(k) for k in np.nonzero(sizes >= args.shade_min_area)[0] if int(k) != 0]
    if not keep:
        _log("      · shade layers: no smooth area large enough, skipping")
        return None
    w = w * np.isin(lb, keep)
    area = int((w > 0.5).sum())
    rmax = float(args.shade_radius)
    radii = [rmax * k for k in (0.17, 0.29, 0.46, 0.67, 1.0)]
    radii = [r for r in radii if r >= 8.0]
    knots = np.linspace(0.0, 1.0, max(2, int(args.shade_stops)))
    epss = [float(x) for x in str(args.shade_eps).split(",") if x.strip()]
    if not epss:
        epss = [0.35]
    layers = []
    t0 = time.time()
    for _ in range(max(0, int(args.shade_layers))):
        rule = np.sqrt(((src - cur) ** 2).sum(-1)) * w
        ys, xs = _shade_candidates(ndi.gaussian_filter(rule, 6.0), int(args.shade_cand),
                                  4, args.shade_margin)
        best = None
        for cy, cx in zip(ys, xs):
            for r in radii:
                for eps in epss:
                    lay = _shade_fit_layer(src, cur, w, cx, cy, r, knots, eps, args.shade_tedge,
                                           args.shade_non_smooth, args.shade_halo, 0.0)
                    if lay is None or lay["gain"] < args.shade_gain:
                        continue
                    if best is None or lay["red"] > best["red"]:
                        best = lay
        if best is None:
            break
        _shade_apply(cur, best)
        layers.append(best)
    if not layers:
        _log("      · shade layers: no acceptable candidate (the base residual is already smooth), skipping")
        return None
    defs, group = _shade_emit(layers, W, H)
    cov = sum(math.pi * ly["r"] ** 2 for ly in layers) / max(1.0, float(area))
    stats = {"n": len(layers), "area": area, "cov": cov, "secs": time.time() - t0,
             "psnr": _psnr(cur), "psnr0": psnr0}
    return defs, group, stats


