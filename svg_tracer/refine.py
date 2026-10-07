"""Error-driven adaptive refinement: regions whose current fill leaves a visible local error are split and re-fitted."""
from __future__ import annotations

import heapq
import math

import numpy as np
from scipy import ndimage as ndi

from .contours import region_path_d
from .geometry import to_hex
from .gradient import fit_region_gradient, predict_region
from .state import _REFINE_DEBUG

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


