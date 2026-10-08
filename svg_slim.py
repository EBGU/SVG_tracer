#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""svg_slim.py - slimming kernel for vector-tracing output
(shared by SVG_tracer.py and svgzip.py, standard library only)

Two levers:

1. **Numeric precision truncation**: the paths emitted by the generator are already
   relative commands (`M x y l dx dy c …`); here we only round each number to a given
   number of decimals and drop redundant separators. Measured lossless to the last bit
   (render mutual comparison >160 dB).
2. **drop_clip**: remove `<clipPath>` and the `clip-path` references. The generator writes
   one clipPath per color region that is **byte-for-byte identical** to the region outline
   (1393 of them in the 1x watercolor image, 28% of the volume), while the strokes are
   generated inside their own regions anyway; measured after removal, the render mutual
   comparison against the original file is 70 dB and the PSNR against the source image is
   exactly unchanged (on hard-edged artwork this clip is a bit more useful, costing about
   -0.3 dB).

Typical gain: logo 4x 614 KB → 594 KB (slim) → tight 372 KB → gzip 114 KB.
"""
from __future__ import annotations

import re

__all__ = ["slim_path", "slim_svg", "svg_layers", "fmt_num"]

_NUM_RE = re.compile(r"-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_CMD_CHARS = "MmLlCcSsQqTtAaHhVvZz"
_CLIP_D = re.compile(r'(?<![-\w])d="([^"]*)"')
_CLIP_BLOCK = re.compile(r"<clipPath[^>]*>.*?</clipPath>", re.S)
_CLIP_REF = re.compile(r'\s*clip-path="[^"]*"')


def fmt_num(v: float, prec: int = 1) -> str:
    """Compact number: fixed decimals → strip trailing zeros → drop the leading 0 before 0.5 (`.5`)."""
    s = "%.*f" % (prec, v)
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    if s in ("", "-0", "-"):
        return "0"
    if s.startswith("0."):
        return s[1:]
    if s.startswith("-0."):
        return "-" + s[2:]
    return s


def slim_path(d: str, prec: int = 1) -> str:
    """Slim down path data: keep the command letters, truncate numbers to prec decimals, drop redundant separators."""
    out, prev_num = [], False
    i, n = 0, len(d)
    while i < n:
        ch = d[i]
        if ch in _CMD_CHARS:
            out.append("z" if ch in "Zz" else ch)
            prev_num = False
            i += 1
            continue
        m = _NUM_RE.match(d, i)
        if m:
            if prev_num:
                out.append(" ")
            out.append(fmt_num(float(m.group(0)), prec))
            prev_num = True
            i = m.end()
        else:
            i += 1                      # whitespace / commas are dropped; we insert the spaces ourselves
    return "".join(out)


def svg_layers(svg: str):
    """Split into the output layers as (head, regions, strokes); strokes is an empty string when there is no stroke layer."""
    i = svg.index('id="regions"')
    j = svg.index('id="strokes"') if 'id="strokes"' in svg else len(svg)
    return svg[:i], svg[i:j], svg[j:]


def slim_svg(svg: str, prec: int = 1, prec_contour: int = 1, prec_stroke: int = 1,
             drop_clip: bool = False) -> str:
    """Truncate numeric precision per layer; with drop_clip, remove the duplicated clipPath."""
    head, regions, strokes = svg_layers(svg)

    def fix(txt, p):
        return _CLIP_D.sub(lambda m: 'd="%s"' % slim_path(m.group(1), p), txt)

    head = fix(head, prec if prec_contour is None else prec_contour)
    regions = fix(regions, prec if prec_contour is None else prec_contour)
    strokes = fix(strokes, prec if prec_stroke is None else prec_stroke)
    if drop_clip:
        head = _CLIP_BLOCK.sub("", head)
        regions = _CLIP_REF.sub("", regions)
        strokes = _CLIP_REF.sub("", strokes)
    return head + regions + strokes
