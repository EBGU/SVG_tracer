"""Geometry and number-formatting helpers: hex colours, fixed-point numbers, bilinear sampling, polyline/cubic-Bezier conversion and least-squares curve fitting."""
from __future__ import annotations


import numpy as np

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


