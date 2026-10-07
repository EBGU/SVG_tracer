"""Anti-aliasing transition bands, region-adjacency helpers and the SVG document assembly."""
from __future__ import annotations

import os

import numpy as np
from scipy import ndimage as ndi
from skimage import measure

from .contours import _REFINE_STATS, bilin_arr, mask_to_paths
from .geometry import fit_bezier_d, fnum, to_hex
from .state import EPS

# ======================================================================
# 7. SVG assembly
# ======================================================================

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
              shade=None) -> str:
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
        if r["grad"] is not None:
            gid = f"g{i}"
            g = r["grad"]
            if g.get("kind") == "radial":
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
            r["fill"] = f"url(#{gid})"
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

    if shade is not None:
        # Stacked translucent radial gradients (section 3.8): painted last so they correct the final
        # composite, including whatever the stroke and edge layers added.
        defs.extend(shade[0])
        body.append(shade[1])

    head = ('<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'width="{width}" height="{height}" viewBox="0 0 {vw} {vh}">\n'
            f"<title>{meta.get('src', 'input')} · 结构张量描摹 (structure-tensor tracing)</title>\n"
            f'<desc>{meta["desc"]}</desc>\n')
    return head + "<defs>\n" + "\n".join(defs) + "\n</defs>\n" + "\n".join(body) + "\n</svg>\n"


