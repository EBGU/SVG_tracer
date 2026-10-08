"""Per-region linear/radial gradient fitting and gradient-aware spatial merging of colour patches."""
from __future__ import annotations


import numpy as np
from scipy import ndimage as ndi

from .segment import _adjacent_pairs
from .state import EPS, log
from . import state

# ======================================================================
# per-region gradient fitting (the structure tensor supplies the gradient axis)
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
            if better and rad["range"] >= min_range and rad["gain"] >= state.GRAD_MIN_GAIN:
                return {"kind": "radial", "cx": rad["cx"], "cy": rad["cy"], "r": rad["r"],
                        "stops": rad["stops"], "quality": rad["gain"], "axis": "radial",
                        "range": rad["range"]}
    # flat color vs linear gradient: the gradient must remove at least 12% of the squared error and span a wide enough color range, otherwise flat color is cheaper
    if r["range"] < min_range or r["gain"] < state.GRAD_MIN_GAIN:
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
      (1) color-field PCA / least squares                        - globally optimal linear direction, never degenerates
      (2) structure-tensor energy-weighted mean gradient ∇I      - the physical direction given by local edges
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
            log(f"        · merge pass {_p + 1}: {len(pairs)} adjacent pairs → {len(acc)} merges")
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


