#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""svgzip.py - small SVG slimming / compression utility
(shares the kernel svg_slim.py with SVG_tracer.py --compress)

The slimming is exactly the same as --compress in SVG_tracer.py; the difference is that
this script targets an **existing** SVG:

  python svgzip.py examples/openai_traced_scale1.svg         # → examples/openai_traced_scale1_slim.svg
  python svgzip.py in.svg out/tight.svg --tight --svgz        # also drop duplicate clipPath and emit .svgz
  python svgzip.py in.svg --ref inputs/openai.png             # render back at native size to check PSNR

By default it only truncates numeric precision (lossless to the last bit). --prec 0 lowers
the precision of stroke coordinates only; measured almost lossless while saving about 1/3 of
the volume, it is the best-value setting. --tight additionally drops the clipPath that
duplicates the region outlines, giving the smallest volume, but strokes then spill outside
their own color regions and the cost grows with the scale factor (logo 2x -1.94 dB, 4x
-5.74 dB, watercolor -0.01 dB).
"""
from __future__ import annotations

import argparse
import gzip
import io
import math
import os
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svg_slim import slim_path, slim_svg, svg_layers, fmt_num  # noqa: F401  (re-exported for callers)

__version__ = "1.0.0"

EXAMPLES = """\
Examples
--------
  python svgzip.py examples/openai_traced_scale1.svg            # precision trimming (lossless)
  python svgzip.py examples/openai_traced_scale1.svg --tight --svgz   # more aggressive + emit .svgz
  python svgzip.py in.svg out/x.svg --ref inputs/openai.png     # render back at native size and verify
"""


def psnr(a, b) -> float:
    """PSNR (dB) of two float images."""
    mse = float(((a - b) ** 2).mean())
    return 99.0 if mse <= 0 else 10 * math.log10(255.0 ** 2 / mse)


def render(path: str, w: int, h: int):
    import cairosvg
    import numpy as np
    from PIL import Image
    png = cairosvg.svg2png(url=path, output_width=w, output_height=h)
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), np.float32)


def mb(n: int) -> str:
    return f"{n / 1e6:.2f} MB" if n >= 1e6 else f"{n / 1e3:.1f} KB"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="svgzip.py", description="SVG slimming / compression (numeric precision trimming + clipPath dedup + gzip)",
        epilog=EXAMPLES, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("src", help="input SVG")
    ap.add_argument("dst", nargs="?", default=None,
                    help="output SVG; default <input name>_slim.svg / _tight.svg")
    ap.add_argument("--prec", type=int, default=1, help="numeric precision of the stroke layer; 0 = integer grid only, almost lossless")
    ap.add_argument("--prec-contour", type=int, default=1, help="numeric precision of the contour/gradient layer; setting 0 drops quality noticeably")
    ap.add_argument("--drop-clip", action="store_true",
                    help="remove the clipPath and its clip-path references (smallest volume, but strokes spill outside their own color regions)")
    ap.add_argument("--tight", action="store_true",
                    help="equals --prec 0 --drop-clip (smallest volume: logo 2x -1.9 dB / 4x -5.7 dB, watercolor -0.01 dB)")
    ap.add_argument("--svgz", action="store_true", help="also write out .svgz (gzip -9)")
    ap.add_argument("--svgz-only", action="store_true", help="write out .svgz only, no SVG")
    ap.add_argument("--ref", default="", help="reference image for verification (if given, render back at native size and compare PSNR)")
    ap.add_argument("--verify", type=int, default=0, help="verification render width; 0 = the reference image's width")
    ap.add_argument("--no-render", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--version", action="version", version=f"svgzip {__version__}")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not os.path.isfile(args.src):
        sys.exit(f"[error] {args.src} not found")
    orig = open(args.src, encoding="utf-8").read()
    n0 = len(orig.encode())

    prec_stroke = 0 if args.tight else args.prec
    out = slim_svg(orig, prec_contour=args.prec_contour, prec_stroke=prec_stroke,
                   drop_clip=args.drop_clip or args.tight)
    tag = "tight" if (args.drop_clip or args.tight) else "slim"
    dst = args.dst
    if not dst:
        root, ext = os.path.splitext(args.src)
        ext = ext or ".svg"
        dst = f"{root}_{tag}{ext}"

    try:
        ET.fromstring(out)
    except ET.ParseError as exc:
        sys.exit(f"[error] the slimmed XML is invalid (writing out abandoned): {exc}")

    n1 = len(out.encode())
    if not args.svgz_only:
        with open(dst, "w", encoding="utf-8") as f:
            f.write(out)
        print(f"slimming({tag})  {args.src}: {mb(n0)} → {dst}: {mb(n1)}  ({100.0 * n1 / n0:.0f}%)")
    if args.svgz or args.svgz_only:
        gz = (dst[:-4] if dst.lower().endswith(".svg") else dst) + ".svgz"
        # mtime=0: makes the .svgz reproducible (otherwise the gzip header carries a build timestamp)
        with open(gz, "wb") as fh, gzip.GzipFile(
                fileobj=fh, mode="wb", compresslevel=9, mtime=0) as f:
            f.write(out.encode())
        nz = os.path.getsize(gz)
        print(f"gzip     {gz}: {mb(nz)}  (original {100.0 * nz / n0:.0f}%)")

    if args.ref and not args.no_render:
        import numpy as np
        from PIL import Image
        ref = np.asarray(Image.open(args.ref).convert("RGB"), np.float32)
        w = args.verify or ref.shape[1]
        h = int(round(w * ref.shape[0] / ref.shape[1]))
        a, b = render(args.src, w, h), render(dst if not args.svgz_only else args.src, w, h)
        print(f"verify   {w}x{h}  slimmed vs reference {psnr(b, ref):.2f} dB"
              f" | original file vs reference {psnr(a, ref):.2f} dB | mutual {psnr(a, b):.1f} dB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
