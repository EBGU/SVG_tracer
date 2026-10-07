"""Degenerate inputs must not crash: fewer distinct colours than ``--kmeans-k`` and tiny canvases.

Root cause of the pre-existing crash (``svg_tracer/segment.py``, k-means++ seeding): the next
centroid was drawn with ``p = d2 / (d2.sum() + EPS)`` where ``d2`` is the residual squared distance
to the nearest already-chosen centroid.  That expression fails in two ways:

* ``d2.sum() == 0``: a point that coincides exactly with a chosen centroid has ``d2 == 0`` and can
  never be selected, so once every sample point is covered -- i.e. the image has at most as many
  distinct colours as centroids already picked, which is always the case for a uniform image and for
  any image with fewer colours than ``k`` -- the numerator is 0 everywhere and ``p`` is an all-zero
  vector.  ``EPS`` turns the 0/0 into 0 rather than nan, it does not make ``p`` a distribution, and
  ``numpy.random.Generator.choice`` rejects it with "Probabilities do not sum to 1".
* ``d2.sum() > 0`` but tiny: the ``+ EPS`` de-normalises ``p`` to sum to ``1 - EPS/mass``, which is
  outside numpy's ``sqrt(eps)`` tolerance as soon as ``mass`` drops below about ``1e-4``.  That is
  reachable on ordinary inputs at small canvases (e.g. a 128 px anti-aliased disc with ``--scale 1``,
  where all the smoothed colours are nearly identical), so it was not only about flat images.

The seeding now computes the exact k-means++ distribution ``d2 / d2.sum()`` -- valid for every
positive mass, so it can never be rejected -- and stops seeding when the mass is exactly 0, keeping
one centroid per distinct colour.  A second, unrelated pre-existing bug tripped images narrower or
shorter than 2 px: the dead ``if labels[1, 1] == bg_label: pass`` in the background detection, which
is gone.

The inputs are generated into ``tmp_path``, so this file needs nothing from the gitignored
``inputs/`` and stays in the fast tier.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from PIL import Image

RED = (200, 30, 30)
RED_HEX = "#c81e1e"
BLUE = (40, 40, 230)
YELLOW = (240, 220, 30)
TINY_SHAPES = [(1, 1), (2, 1), (1, 2), (2, 2), (3, 3)]
FILL_RE = re.compile(r'fill="(#[0-9a-f]{6})"')


def _save(path, array) -> str:
    Image.fromarray(np.asarray(array, dtype=np.uint8)).save(path)
    return str(path)


def _solid(tmp_path, w, h, col=RED) -> str:
    return _save(tmp_path / f"solid{w}x{h}.png", np.full((h, w, 3), col, np.uint8))


def _bands(tmp_path, w, h, cols) -> str:
    """A vertical band per colour, so the image has exactly len(cols) distinct colours."""
    a = np.empty((h, w, 3), np.uint8)
    edges = np.linspace(0, w, len(cols) + 1).astype(int)
    for i, c in enumerate(cols):
        a[:, edges[i]:edges[i + 1]] = c
    return _save(tmp_path / f"bands{w}_{len(cols)}.png", a)


def _disc(tmp_path, size=128, r=40) -> str:
    """A black disc on white: anti-aliased edges, i.e. a perfectly normal input.

    128 px at --scale 1 is the configuration where the old ``+ EPS`` normalisation produced
    ``p.sum() == 1 - EPS/mass`` with a tiny but strictly positive residual mass, so numpy rejected
    it even though the image has plenty of distinct colours.
    """
    yy, xx = np.mgrid[0:size, 0:size]
    a = np.full((size, size, 3), 255, np.uint8)
    a[np.hypot(yy - (size - 1) / 2.0, xx - (size - 1) / 2.0) <= r] = 0
    return _save(tmp_path / f"disc{size}.png", a)


def _run(trace, args, out) -> str:
    """Run the CLI with the cheap common flags and return the SVG text."""
    r = trace([*args, "--gpu", "auto", "--scale", "1", "--jobs", "1", "--no-preview"], out)
    assert r.returncode == 0, r.stderr[-2500:]
    return out.read_text(encoding="utf-8")


def _fills(text) -> set:
    return set(FILL_RE.findall(text))


def _rgb(hexcol):
    return np.array([int(hexcol[i:i + 2], 16) for i in (1, 3, 5)], float)


def _nearest(fill, palette) -> float:
    return float(min(np.abs(_rgb(fill) - np.asarray(c, float)).max() for c in palette))


def _has(fills, col, tol=40) -> bool:
    """True when some fill in the SVG is within `tol` (max channel distance) of `col`."""
    return any(_nearest(f, [col]) <= tol for f in fills)


def test_single_colour_is_one_solid_region(trace, tmp_path):
    """A uniform image must produce a valid SVG that paints the whole canvas in that colour."""
    img = _solid(tmp_path, 16, 16)
    out = tmp_path / "solid.svg"
    text = _run(trace, ["--in", img], out)

    root = ET.fromstring(text)                       # raises on malformed XML
    assert root.tag.endswith("svg")
    assert _fills(text) == {RED_HEX}, "a uniform input must only ever use the input colour"
    canvas_w, canvas_h = float(root.get("width")), float(root.get("height"))
    rects = [e for e in root.iter() if e.tag.endswith("rect") and e.get("fill") == RED_HEX]
    assert rects, "the solid background rect is missing"
    assert float(rects[0].get("width")) >= canvas_w
    assert float(rects[0].get("height")) >= canvas_h


def test_fewer_distinct_colours_than_k_keeps_them(trace, tmp_path):
    """k-means++ must stop seeding instead of crashing, and keep one centroid per real colour."""
    img = _bands(tmp_path, 16, 16, [RED, BLUE])
    out = tmp_path / "two.svg"
    text = _run(trace, ["--in", img, "--kmeans-k", "20"], out)

    ET.fromstring(text)
    fills = _fills(text)
    assert len(fills) >= 2, f"both colours must survive, got {fills}"
    assert _has(fills, RED) and _has(fills, BLUE), f"input colours missing from {fills}"


def test_k_far_above_the_colour_count(trace, tmp_path):
    """--kmeans-k 100 on a three-colour image is the same degenerate seeding, 99 times over."""
    img = _bands(tmp_path, 18, 12, [RED, BLUE, YELLOW])
    out = tmp_path / "three.svg"
    text = _run(trace, ["--in", img, "--kmeans-k", "100"], out)

    ET.fromstring(text)
    fills = _fills(text)
    assert len(fills) >= 3, f"all three colours must survive, got {fills}"
    assert all(_has(fills, c) for c in (RED, BLUE, YELLOW)), f"colours missing from {fills}"


@pytest.mark.parametrize("w,h", TINY_SHAPES, ids=[f"{w}x{h}" for w, h in TINY_SHAPES])
def test_tiny_canvas(trace, tmp_path, w, h):
    """1x1 up to 3x3 canvases: no crash, well-formed XML, canvas painted in the input colour."""
    img = _solid(tmp_path, w, h)
    out = tmp_path / f"tiny{w}x{h}.svg"
    text = _run(trace, ["--in", img], out)

    root = ET.fromstring(text)
    assert _fills(text) == {RED_HEX}
    assert any(e.tag.endswith("rect") for e in root.iter())


def test_anti_aliased_disc_is_not_treated_as_degenerate(trace, tmp_path):
    """Black disc on white: a normal input, so it must still segment into its two real colours.

    It also covers the tiny-but-positive residual mass: with --scale 1 the smoothed colours along
    the rim are all nearly identical, and the historical ``+ EPS`` normalisation made that seeding
    step raise instead of drawing from a valid distribution.
    """
    img = _disc(tmp_path, 128, 40)
    out = tmp_path / "disc.svg"
    text = _run(trace, ["--in", img], out)

    ET.fromstring(text)
    fills = _fills(text)
    assert len(fills) >= 2, f"the disc should keep both colours, got {fills}"
    assert _has(fills, (0, 0, 0), 60) and _has(fills, (255, 255, 255), 60), fills


def test_kmeans_seeding_guard_unit():
    """The guard itself: no crash, one centroid per distinct colour, k unchanged when possible."""
    from svg_tracer.segment import _kmeans

    # one distinct colour, k=20 -> a single centroid, every pixel labelled 0
    same = np.tile(np.array([12.0, 3.0, 4.0]), (6, 1))
    cen, lab = _kmeans(same, 20, 4, np.random.default_rng(0))
    assert cen.shape == (1, 3)
    assert np.allclose(cen[0], [12.0, 3.0, 4.0])
    assert set(lab.tolist()) == {0}

    # three distinct colours, k=20 -> three centroids (not 20, and not a ValueError)
    cols = np.array([[0.0, 0.0, 0.0], [50.0, -20.0, 30.0], [10.0, 60.0, -40.0]])
    cen2, lab2 = _kmeans(np.repeat(cols, 4, axis=0), 20, 4, np.random.default_rng(1))
    assert cen2.shape == (3, 3)
    assert set(lab2.tolist()) <= {0, 1, 2}

    # plenty of distinct colours: the seeding still uses exactly the requested k
    feats = np.random.default_rng(2).normal(size=(400, 3))
    cen3, lab3 = _kmeans(feats, 5, 4, np.random.default_rng(3))
    assert cen3.shape == (5, 3)
    assert set(lab3.tolist()) <= set(range(5))
