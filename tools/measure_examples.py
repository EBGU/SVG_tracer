#!/usr/bin/env python3
"""Measure the shipped examples: bytes, gzip size, layer counts and PSNR against the source.

    python tools/measure_examples.py            # print one JSON line per example, plus a summary
    python tools/measure_examples.py --check     # exit 1 when a measured value differs from the
                                                 # numbers the README documents

The numbers this prints are the ones quoted in the README's "Examples" and "Shipped examples"
sections.  The layer counters walk the ``<g id="regions">`` / ``<g id="strokes">`` groups, so they
report the paths the SVG actually contains; the CLI's ``areas`` counter walks its internal region
list and can differ by one, because the full-canvas background may or may not occupy an entry of its
own.

The rendering needs cairosvg; without it every PSNR column is skipped.
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import sys

import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (example file, source image, documented bytes, documented gzip KB)
CASES = [
    ("openai_traced_scale1.svg", "inputs/openai.png", 201398, 51.9),
    ("water_lilies_traced_scale1.svg", "inputs/water_lilies.jpg", 5138855, 826.6),
    ("wave_traced_scale1.svg", "inputs/wave.jpg", 18363702, 3314.5),
    ("apple_traced_scale1.svg", "inputs/apple.png", 137379, 31.4),
    ("apple_traced_scale4.svg", "inputs/apple.png", 1068341, 242.5),
]


def psnr(a, b) -> float:
    mse = float(((a - b) ** 2).mean())
    return 99.0 if mse <= 0 else 10 * np.log10(255.0 ** 2 / mse)


def render(url, w, h):
    import cairosvg
    png = cairosvg.svg2png(url=url, output_width=w, output_height=h)
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), np.float32)


def _group(text: str, gid: str) -> str:
    """The whole body of ``<g id="gid">``, up to the next top-level group (or the document end).

    Stopping at the first ``</g>`` would undercount: the layers are emitted as one group per region,
    so the strokes group alone contains thousands of nested ``</g>`` tags.
    """
    i = text.find(f'<g id="{gid}"')
    if i < 0:
        return ""
    j = text.find('<g id="', i + 1)
    return text[i:] if j < 0 else text[i:j]


def layers(text: str):
    """(regions, gradients, strokes, seams) -- layer sizes as the CLI counts them."""
    regions = _group(text, "regions").count("<path")
    gradients = text.count("<linearGradient") + text.count("<radialGradient")
    strokes = _group(text, "strokes").count("<path")
    seams = _group(text, "edges").count("<path")
    return regions, gradients, strokes, seams


def measure() -> dict:
    out = {}
    for name, src, _, _ in CASES:
        path = os.path.join(REPO, "examples", name)
        if not os.path.isfile(path):
            out[name] = {"missing": True}
            continue
        blob = open(path, "rb").read()
        text = blob.decode("utf-8")
        reg, grad, strk, seam = layers(text)
        rec = {"bytes": len(blob), "kb": round(len(blob) / 1024.0, 1),
               "gzip_kb": round(len(gzip.compress(blob, 9)) / 1024.0, 1),
               "regions": reg, "gradients": grad, "strokes": strk, "seams": seam}
        try:
            ref = np.asarray(Image.open(os.path.join(REPO, src)).convert("RGB"), np.float32)
            h, w = ref.shape[:2]
            rec["psnr_native"] = round(psnr(render(path, w, h), ref), 2)
            rec["size_native"] = f"{w}x{h}"
        except Exception as exc:  # cairosvg absent, or a missing source image
            rec["psnr_error"] = repr(exc)[:160]
        out[name] = rec
        print(f"{name:38s} {json.dumps(rec)}", flush=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Measure the shipped examples")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 when a measured value differs from the documented one")
    args = ap.parse_args(argv)

    out = measure()
    with open("/tmp/example_metrics.json", "w") as fh:
        json.dump(out, fh, indent=2)

    if not args.check:
        return 0
    bad = []
    for name, _src, want_bytes, want_gzip in CASES:
        rec = out[name]
        if rec.get("missing"):
            bad.append(f"{name}: missing")
            continue
        if rec["bytes"] != want_bytes:
            bad.append(f"{name}: bytes {rec['bytes']} != documented {want_bytes}")
        if abs(rec["gzip_kb"] - want_gzip) > 1.5:
            bad.append(f"{name}: gzip {rec['gzip_kb']} KB != documented {want_gzip} KB")
    if bad:
        print("\n[check] differences from the README:")
        for line in bad:
            print("  " + line)
        return 1
    print("\n[check] every measured value matches the README")
    return 0


if __name__ == "__main__":
    sys.exit(main())
