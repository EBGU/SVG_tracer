#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""selfcheck.py —— self-check: run the whole pipeline on a synthetic small image and
assert quality / compression round-trip / path resolution

    python selfcheck.py            # about 20~40 seconds
    python selfcheck.py --keep     # keep the temp directory, handy for inspecting intermediate artifacts

Covers:
  · number formatting, idempotence and XML validity of slim_path / slim_svg (compression kernel regression)
  · resolve_io path completion (bare filename → inputs/, automatic naming → out/)
  · end-to-end pipeline (logo_trace.py subprocess): valid XML, all layers present, correct preview size, .svgz decodable
  · interior (gradient-free) MAE —— a robust indicator of whether the solid/gradient fills were drawn correctly
  · compression round-trip: render slim / tight back at native size and compare against the original file's render

A note on the low PSNR threshold: the synthetic image is a 256px hard-edged figure, and the
1px band along its boundary alone accounts for several percent of all pixels, with a contrast
that spans the full range. An absolute 1px boundary error is a large fraction on a small
image, so overall PSNR is naturally only a little over 20 dB (the same tool reaches 40 dB on
the 1254px logo). What really decides "was it drawn correctly" is the interior MAE (measured
1.5/255).
"""
from __future__ import annotations

import argparse
import gzip
import io
import os
import re
import shlex
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from svg_slim import slim_path, slim_svg           # noqa: E402
import logo_trace                                  # noqa: E402

SCALE = 2              # 2x grid for the self-check: 512², fast enough yet still benefits from subpixel localization
PSNR_MIN = 21.0        # end-to-end PSNR lower bound (measured about 22.4 dB)
EXTRA_ARGS: list = []  # extra arguments passed through to logo_trace.py (filled in by --tool-args)
INTERIOR_MAE = 3.0     # upper bound on the interior (gradient-free) mean absolute error (measured about 1.6/255)
MUTUAL_MIN = 30.0      # lower bound on the render mutual comparison before vs after compression

_failed: list[str] = []


def check(name: str, cond: bool, detail="", note="") -> bool:
    line = ("  ok   " if cond else "  FAIL ") + name + (f"   {note}" if note else "")
    if not cond:
        line += f"   [{detail}]"
    print(line)
    if not cond:
        _failed.append(name)
    return cond


def section(title: str) -> None:
    print(f"\n== {title}")


# ----------------------------------------------------------------------
# synthetic test image
# ----------------------------------------------------------------------
def make_image(path: str, n: int = 256) -> None:
    """White background + orange circle + dark diagonal line + one linear gradient + one small high-saturation patch."""
    im = Image.new("RGB", (n, n), (255, 255, 255))
    d = ImageDraw.Draw(im)
    d.ellipse([n * 0.12, n * 0.10, n * 0.55, n * 0.53], fill=(228, 92, 26))
    d.line([n * 0.04, n * 0.95, n * 0.96, n * 0.06], fill=(40, 40, 48), width=max(2, n // 85))
    d.rectangle([n * 0.60, n * 0.62, n * 0.95, n * 0.92], fill=(20, 60, 200))
    d.rectangle([n * 0.10, n * 0.66, n * 0.26, n * 0.78], fill=(0, 190, 160))
    a = np.asarray(im).astype(np.float32)
    y0, y1, x0, x1 = int(n * 0.62), int(n * 0.92), int(n * 0.60), int(n * 0.95)
    t = np.linspace(0.0, 1.0, x1 - x0, dtype=np.float32)[None, :, None]
    a[y0:y1, x0:x1] = ((1 - t) * np.array([20, 60, 200], np.float32)
                       + t * np.array([30, 190, 220], np.float32))
    Image.fromarray(a.astype(np.uint8)).save(path)


# ----------------------------------------------------------------------
# 1. compression kernel
# ----------------------------------------------------------------------
def test_slim_units() -> None:
    section("压缩内核")
    got = slim_path("M100.0 200.0l-1.40 .60c0 0 0 0 0 0z", 1)
    check("slim_path 数字紧凑化", got == "M100 200l-1.4 .6c0 0 0 0 0 0z", got)
    check("slim_path 整数精度", slim_path("M1.6 2.4l3.5 0", 0) == "M2 2l4 0", slim_path("M1.6 2.4l3.5 0", 0))

    svg = ('<svg xmlns="http://www.w3.org/2000/svg">'
           '<defs><clipPath id="c0"><path d="M0 0 L10.00 0 L10.0 10.00 Z"/></clipPath></defs>'
           '<g id="regions"><path d="M0 0 L10.00 0 L10.0 10.00 Z"/></g>'
           '<g id="strokes" clip-path="url(#c0)"><path d="M0.50 0.50 L9.50 9.50"/></g></svg>')
    s1, s2 = slim_svg(svg), slim_svg(slim_svg(svg))
    ET.fromstring(s1)
    check("slim_svg 幂等", s1 == s2, "两次瘦身结果不同")
    check("slim_svg 更小且保留分层", len(s1) < len(svg) and 'id="regions"' in s1 and 'id="strokes"' in s1)
    t1 = slim_svg(svg, prec_stroke=0, drop_clip=True)
    ET.fromstring(t1)
    check("tight: clipPath 与引用都被删", "<clipPath" not in t1 and "clip-path" not in t1)
    check("tight: 笔触坐标取整", 'd="M0 0L10 10"' in t1, t1)


# ----------------------------------------------------------------------
# 2. path resolution
# ----------------------------------------------------------------------
def test_resolve_io(tmp: str) -> None:
    section("路径解析 (resolve_io)")
    os.makedirs(os.path.join(tmp, "inputs"), exist_ok=True)
    open(os.path.join(tmp, "inputs", "foo.png"), "wb").write(b"x")
    cwd = os.getcwd()
    os.chdir(tmp)
    try:
        ns = type("A", (), {"src": "foo.png", "out": None, "preview": None, "debug": None})()
        logo_trace.resolve_io(ns)
        check("裸文件名 → inputs/", ns.src == os.path.join("inputs", "foo.png"), ns.src)
        check("默认输出 → out/<名>_traced.svg", ns.out == os.path.join("out", "foo_traced.svg"), ns.out)
        check("预览自动命名", ns.preview == os.path.join("out", "foo_preview.png"), ns.preview)
        check("调试图默认关闭", ns.debug == "", ns.debug)
        check("输出目录已建", os.path.isdir("out"))
        ns2 = type("A", (), {"src": os.path.join("inputs", "foo.png"), "out": "x.svg",
                             "preview": "", "debug": "d.png"})()
        logo_trace.resolve_io(ns2)
        check("显式路径不被改写", (ns2.out, ns2.preview, ns2.debug) == ("x.svg", "", "d.png"))
    finally:
        os.chdir(cwd)


# ----------------------------------------------------------------------
# 3. end-to-end pipeline
# ----------------------------------------------------------------------
def run_pipeline(tmp: str):
    png = os.path.join(tmp, "syn.png")
    svg = os.path.join(tmp, "syn.svg")
    prev = os.path.join(tmp, "syn_preview.png")
    make_image(png)
    cmd = [sys.executable, os.path.join(HERE, "logo_trace.py"),
           "--in", png, "--out", svg, "--preview", prev, "--scale", str(SCALE),
           "--gzip", "--compress", "slim", "--quiet", *EXTRA_ARGS]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print((r.stdout or "").strip()[-700:])
    check("管线退出码 0", r.returncode == 0, (r.stderr or "")[-300:])
    check("写出 SVG", os.path.isfile(svg))
    txt = open(svg, encoding="utf-8").read() if os.path.isfile(svg) else ""
    try:
        ET.fromstring(txt)
        ok = True
    except ET.ParseError as exc:
        ok = False
        print("   ", exc)
    check("SVG XML 合法", ok)
    check("分层结构齐全", all(k in txt for k in ('id="regions"', "<defs>")))
    n0 = Image.open(png).size
    if ok:
        root = ET.fromstring(txt.split("?>")[-1])
        vb = root.get("viewBox")
        check("画布=原生尺寸、viewBox=描摹网格",
              root.get("width") == str(n0[0]) and root.get("height") == str(n0[1])
              and vb == f"0 0 {n0[0]*SCALE} {n0[1]*SCALE}",
              f'{root.get("width")}x{root.get("height")} viewBox={vb}')
    check("预览 PNG 存在", os.path.isfile(prev))
    if os.path.isfile(prev):
        check("预览渲回原生尺寸", Image.open(prev).size == Image.open(png).size,
              f"{Image.open(prev).size} vs {Image.open(png).size}")
    gz = svg + "z"
    check(".svgz 存在", os.path.isfile(gz))
    if os.path.isfile(gz):
        check(".svgz 解压一致", gzip.open(gz, "rt", encoding="utf-8").read() == txt)
    m = re.search(r"\((\d+\.\d+) dB\)", r.stdout or "")
    ps = float(m.group(1)) if m else -1.0
    check(f"端到端 PSNR ≥ {PSNR_MIN} dB", ps >= PSNR_MIN, f"实测 {ps} dB",
          note=f"实测 {ps:.2f} dB")
    check("摘要标注原生尺寸", f"{n0[0]}x{n0[1]}" in (r.stdout or ""), f"找不到 {n0[0]}x{n0[1]}")
    check(f"摘要标注描摹网格", f"{n0[0]*SCALE}x{n0[1]*SCALE}" in (r.stdout or ""),
          f"找不到 {n0[0]*SCALE}x{n0[1]*SCALE}")

    if os.path.isfile(prev):
        a = np.asarray(Image.open(png).convert("RGB"), np.float64)
        b = np.asarray(Image.open(prev).convert("RGB"), np.float64)
        if a.shape == b.shape:
            g = np.abs(np.gradient(a.mean(2))).sum(0)
            interior = g < 0.5
            mae = float(np.abs(a - b).mean(2)[interior].mean())
            check(f"内部(无梯度)MAE ≤ {INTERIOR_MAE}/255", mae <= INTERIOR_MAE,
                  f"实测 {mae:.2f}", note=f"实测 {mae:.2f}/255（占 {100*interior.mean():.0f}% 像素）")
    return svg


# ----------------------------------------------------------------------
# 4. compression round-trip
# ----------------------------------------------------------------------
def test_compress_roundtrip(svg_path: str) -> None:
    section("压缩往返（渲回原生尺寸）")
    raw = open(svg_path, encoding="utf-8").read()
    w = len(raw.encode())
    n = 256
    try:
        import cairosvg
    except ImportError:
        print("  skip  未安装 cairosvg")
        return

    def render(text: str):
        png = cairosvg.svg2png(bytestring=text.encode(), output_width=n, output_height=n)
        return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), np.float32)

    def psnr(a, b):
        mse = float(((a - b) ** 2).mean())
        return 99.0 if mse <= 0 else 10 * np.log10(255.0 ** 2 / mse)

    base = render(raw)
    for tag, kw in (("slim", {}), ("tight", dict(prec_stroke=0, drop_clip=True))):
        out = slim_svg(raw, **kw)
        ET.fromstring(out)
        m = psnr(base, render(out))
        check(f"{tag} 体积不增", len(out.encode()) <= w, f"{w} → {len(out.encode())}")
        check(f"{tag} 渲染互比 ≥ {MUTUAL_MIN} dB", m >= MUTUAL_MIN, f"{m:.1f} dB")
        print(f"        {tag}: {w/1024:.1f} KB → {len(out.encode())/1024:.1f} KB, 互比 {m:.1f} dB")


def main() -> int:
    ap = argparse.ArgumentParser(description="logo_trace 自检")
    ap.add_argument("--keep", action="store_true", help="保留临时目录")
    ap.add_argument("--tool-args", default="", metavar="ARGS",
                    help='透传给 logo_trace.py 的额外参数，例如 --tool-args "--gpu on"')
    args = ap.parse_args()
    global EXTRA_ARGS
    EXTRA_ARGS = shlex.split(args.tool_args)
    tmp = tempfile.mkdtemp(prefix="logotrace_selfcheck_")
    print(f"自检开始（临时目录 {tmp}）" + (f" | 透传: {EXTRA_ARGS}" if EXTRA_ARGS else ""))
    try:
        test_slim_units()
        test_resolve_io(tmp)
        svg = run_pipeline(tmp)
        test_compress_roundtrip(svg)
    finally:
        if not args.keep:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
    print()
    if _failed:
        print(f"✗ 自检失败 {len(_failed)} 项: " + ", ".join(_failed))
        return 1
    print("✓ 自检全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
