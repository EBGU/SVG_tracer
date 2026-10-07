"""Fine-scale edge ridges to skeletonised vector strokes (seam lines)."""
from __future__ import annotations


import numpy as np
from skimage import measure

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


