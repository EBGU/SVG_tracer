"""Region segmentation and the shared partition-routing criterion (classify_regions): k-means colour quantisation, RAG merging, the flat/edge/hybrid routes and texture maps."""
from __future__ import annotations

import sys
import time

import numpy as np
from scipy import ndimage as ndi
from skimage import filters
from skimage import measure
from skimage.color import rgb2lab
from skimage.segmentation import find_boundaries
from . import gpu

from .state import EPS, log

# ======================================================================
# region segmentation
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
        mass = d2.sum()
        if mass <= 0.0:
            # k-means++ draws the next centroid from d2 / sum(d2): a point that already coincides
            # exactly with one of the chosen centroids has d2 == 0 and can never be picked. When the
            # whole residual mass is 0 there is no candidate left, i.e. the image has fewer distinct
            # colours than the requested k, so seeding would hand numpy an all-zero probability
            # vector ("Probabilities do not sum to 1"). Stop seeding and keep the centroids that
            # were actually found -- one per distinct colour, which is exactly what the k-means++
            # objective asks for on such an input.
            break
        # d2 / sum(d2) is the exact k-means++ distribution and always sums to 1 for mass > 0, so the
        # seeding can never be handed an invalid p. The historical "+ EPS" in the denominator only
        # guarded against 0/0: it de-normalised p to sum to 1 - EPS/mass, which numpy rejects as soon
        # as the residual mass is small (below about 1e-4 -- reachable on small canvases, where the
        # smoothed colours are all nearly identical) even though the mass is strictly positive.
        idx.append(int(rng.choice(len(sample), p=d2 / mass)))
        d2 = np.minimum(d2, ((sample - sample[idx[-1]]) ** 2).sum(1))
    cen = sample[idx].copy()
    k = len(cen)                     # < the requested k only for few-colour (degenerate) inputs

    if use_gpu:
        try:
            return gpu.lloyd(sample, feats, cen, k, iters)
        except Exception as _e:
            print(f"      ! GPU k-means failed ({_e}), falling back to CPU", file=sys.stderr)

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
        log(f"      · k-means(k={args.kmeans_k}) {t1-t:.1f}s → RAG merging "
              f"{t2-t1:.1f}s → connected-component split {t3-t2:.1f}s → connected-component re-merge "
              f"{t4-t3:.1f}s, {labels.max()+1} color regions")
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
            log(f"      · flat-color fast path not applicable: {len(peaks)} dominant colors, coverage {cov*100:.1f}% "
                  f"(needs ≥{args.flat_cov*100:.0f}%), taking the full route")
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
        log(f"      · flat-color fast path: {len(peaks)} dominant colors ({cov*100:.1f}% coverage) → "
              f"{len(cen)} palette colors → LUT {t1-t:.2f}s → connected components {t2-t1:.2f}s → "
              f"RAG merging {t3-t2:.2f}s → {labels.max()+1} color regions")
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
        print(f"      ! warning: edge pixel share {frac*100:.1f}% (thresholds hi={hi:.3g} lo={lo:.3g}) "
              f"is unreasonable, please tune --edge-nms-hi/--edge-nms-lo", file=sys.stderr)
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
        print(f"      ! warning: only {n_mark} flat-region seeds found, segmentation would degenerate into a single blob over the whole image; "
              f"please decrease --edge-core or tune --edge-nms-lo", file=sys.stderr)
    labels = watershed(-d, markers).astype(np.int32) - 1
    if args.verbose:
        log(f"      · edges (NMS + hysteresis): edge pixels {edge.mean()*100:.2f}% → "
              f"{n_mark} seeds → watershed {time.time()-t:.1f}s")
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
        info = ", ".join(f"#{li}:{'geo' if geo_r[li] else 'complex'}"
                         f"(tex{tex_r[li]:.1e}/spread{rng_r[li]:.3f})" for li in keep)
        log(f"      · partition routing: edges {edge.mean()*100:.2f}% | snapping mode {args.snap_mode}"
              f" | geometric-region pixels {geo_px.mean()*100:.1f}%"
              + (f", boundary moved {n_moved} pixels" if args.snap_mode == "watershed" else ""))
        print(f"        [{info}] (geometricity criterion: texture<{args.tex_thr} and spread<{args.range_thr})"
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

