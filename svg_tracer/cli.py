"""Command line interface, presets, the parallel per-region stages and the end-to-end tracing pipeline."""
from __future__ import annotations

import argparse
import gzip
import math
import os
import sys
import time
import warnings

import numpy as np
from PIL import Image
from PIL import ImageDraw
from scipy import ndimage as ndi
from skimage import filters
from skimage import measure
from skimage.color import rgb2lab
from skimage.restoration import denoise_bilateral
from skimage.segmentation import find_boundaries
from svg_slim import slim_svg
from . import gpu

from .contours import _REFINE_STATS, region_path_d
from .edges import skeleton_paths
from .geometry import polyline_to_bezier_d, to_hex
from .gradient import GRAD_RADIAL_MARGIN, fit_region_gradient, merge_gradient_regions, predict_region
from .refine import refine_regions, regroup_regions
from .segment import classify_regions, merge_small_regions, segment_colors, segment_edges, segment_flat, segment_hybrid
from .shade import build_shade_stack
from .state import AUTOSCALE_REF, EPS, __version__, log
from .strokes import make_streamlines, ribbon_d, sample_color
from .svg_out import _AA_STATS, aa_band_regions, build_svg
from .tensor import structure_tensor
from . import state

EXAMPLES = """\
示例
----
  # 硬边平面图（logo / 图标 / 插画）：2 倍网格 + 亚像素定位 + 保角贝塞尔，一行搞定
  python logo_trace.py --in logo.png --preset logo

  # 写意绘画 / 照片：细节预设（色块小而多、笔触密而细）
  python logo_trace.py --in water-lilies-29.jpg --preset painting

  # 只要几何层（不要笔触）并顺带出 .svgz
  python logo_trace.py --in logo.png --preset logo --no-strokes --gzip

  # 追高保真：4 倍网格（5016²，约 20 分钟）
  python logo_trace.py --in logo.png --preset logo --scale 4 --compress slim --gzip

  # 自检：合成小图跑通全流程并断言质量
  python selfcheck.py

输入默认为 inputs/<名>，输出默认写到 out/；裸文件名会自动到 inputs/ 下找。
"""

# ======================================================================
# 8. main pipeline
# ======================================================================
# Presets: parameter combinations for different art styles (explicit command-line arguments win over presets)
PRESETS = {
    # Hard-edged flat artwork (logo / icon / illustration): trace on an enlarged grid, subpixel boundary localization, conformal Beziers in geometric regions
    # These are exactly the defaults; the preset's extra job is to also set --scale 2.0, so that a one-line call works
    "logo": dict(
        seg="full",
        no_strokes=True,   # logo mode emits only the geometric layer by default
        kmeans_k=20, min_area=600, contour_tol=0.75, contour_smooth=1.0,
        snap_mode="refine", snap_sub="half", fit="auto", fit_tol=0.10,
        aa_levels=0, grad_stops=10, detail_chroma=12.0,
    ),
    # Freehand painting / photographic: many small regions, contours hugging the detail, dense thin strokes
    "painting": dict(
        seg="full",
        kmeans_k=48, thresh=8.0, merge_thresh2=6.0, merge_passes=4,
        min_area=180, min_width=0, contour_tol=0.5, contour_smooth=1.3,
        contour_min_area=12.0, grad_stops=12, grad_min_range=0.003,
        detail_chroma=0.0, spacing=7.0, stroke_width=4.0, stroke_alpha=0.5,
        stroke_nodes=10, stroke_min=14.0, stroke_max=90.0, coh_min=0.10,
        stroke_activity_ref=0.02, stroke_alpha_min=0.02, aa_levels=0,
        compress="tight",   # watercolor-like images barely lose quality (measured -0.01~-0.11 dB) while saving about 1/3 of the volume
    ),
}


class _Fmt(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """Show the defaults while keeping the example line breaks from the epilog."""


# ---- parallelism for the per-region stage: fork + copy-on-write (large arrays are neither copied nor pickled) ----
_PAR_CTX = {}


def _n_jobs(args, n_items, min_items=24):
    """How many processes to use for the per-region stage.

    0/negative = auto: min(cores, 16); with too few candidate regions (default <24) it is fixed at 1
    -- so that the fork and pool startup cost does not eat the gain, and so that small images keep
    behaving exactly as they always have. Without fork (non-POSIX) it uses 1.
    """
    import multiprocessing
    if "fork" not in multiprocessing.get_all_start_methods():
        return 1
    want = int(getattr(args, "jobs", 0) or 0)
    if want <= 0:
        want = min(os.cpu_count() or 1, 16)
    if int(n_items) < int(min_items):
        return 1
    return max(1, min(want, int(n_items)))


def batch_bookkeeping(labels, rgb_s, use_gpu=True):
    """Compute the two rule-dense sub-steps of the per-region stage over the whole image in one batch; the result is bit-identical to the per-region serial version.

    1) inner: the original does binary_erosion(labels[win]==li, iterations=3) per region. The default
       structuring element is a 3x3 cross, so three iterations = a diamond (L1 radius 3)
       neighbourhood; hence the whole-image equivalent is "all labels in this pixel's diamond
       neighbourhood are identical", obtained with min/max over a radius-3 diamond footprint. The
       border is treated as 0 (erosion's border_value): using cval=-1 / cval=nl makes border pixels
       fail both the min and the max test, which is exactly equivalent.
    2) cols: per-region median color. When pixel values lie on the 1/255 grid the median is a rank
       statistic: odd counts take the middle rank, even counts the average of the two middle ranks;
       with a 256-bucket histogram + cumsum it can be computed in O(1) per label, averaging in
       numpy's operation order ((b_lo/255 + b_hi/255)/2), hence bit-identical. When the image has
       been resampled (scale != 1) the pixels are not on the 1/255 grid and np.median cannot be
       reproduced exactly from a histogram -- fall back.

    Returns (inner_all, cols, backend); when unavailable (None, None, "").
    """
    q = rgb_s * 255.0
    if not np.array_equal(q, np.rint(q)):
        return None, None, ""
    nl = int(labels.max()) + 1
    # With too few labels, whole-image filtering costs more than "each region computing just its own patch", so skip batching entirely.
    if nl > 60000 or nl < 256:
        return None, None, ""
    yy, xx = np.mgrid[-3:4, -3:4]
    dia = (np.abs(yy) + np.abs(xx)) <= 3          # 3 iterations of cross erosion == this diamond
    qq = np.rint(q).astype(np.int32)

    # GPU batching only pays off when "very large + many labels": measured at 1580 labels/1.1Mpx the CPU batch takes 0.4s,
    # while the cupy path takes 3.4s (its first run also pays about 18s for kernel compilation, cached on disk afterwards). So the threshold is set very high.
    if use_gpu and nl >= 4000 and labels.size >= 8000000:
        try:
            import cupy as cp
            from cupyx.scipy import ndimage as gndi
            gl = cp.asarray(labels)
            gdia = cp.asarray(dia)
            gmn = gndi.minimum_filter(gl, footprint=gdia, mode="constant", cval=-1)
            gmx = gndi.maximum_filter(gl, footprint=gdia, mode="constant", cval=nl)
            ginner = (gl == gmn) & (gl == gmx)
            gq = cp.asarray(qq)
            glf = gl.ravel().astype(cp.int64)
            gcin = cp.bincount(gl[ginner].ravel(), minlength=nl)
            gtot = cp.bincount(gl.ravel(), minlength=nl)
            gcols = cp.zeros((nl, 3), dtype=cp.float64)
            for c in range(3):
                qc = gq[..., c].ravel().astype(cp.int64)
                ha = cp.bincount(glf * 256 + qc, minlength=nl * 256).reshape(nl, 256)
                hb = cp.bincount(glf[ginner.ravel()] * 256 + qc[ginner.ravel()],
                                 minlength=nl * 256).reshape(nl, 256)
                use = gcin >= 200
                h = cp.where(use[:, None], hb, ha)
                tot = cp.where(use, gcin, gtot).astype(cp.int64)
                cs = cp.cumsum(h, axis=1)
                bl = (cs <= ((tot - 1) // 2)[:, None]).sum(1).astype(cp.float64)
                bh = (cs <= (tot // 2)[:, None]).sum(1).astype(cp.float64)
                gcols[:, c] = (bl / 255.0 + bh / 255.0) / 2.0
            out = (cp.asnumpy(ginner), cp.asnumpy(gcols), "GPU")
            cp._default_memory_pool.free_all_blocks()
            return out
        except Exception:
            pass

    mn = ndi.minimum_filter(labels, footprint=dia, mode="constant", cval=-1)
    mx = ndi.maximum_filter(labels, footprint=dia, mode="constant", cval=nl)
    inner_all = (labels == mn) & (labels == mx)
    lf = labels.ravel()
    lfi = lf.astype(np.int64)
    cin = np.bincount(labels[inner_all].ravel(), minlength=nl)
    tot_all = np.bincount(lf, minlength=nl)
    keep = inner_all.ravel()
    cols = np.zeros((nl, 3))
    for c in range(3):
        qc = qq[..., c].ravel()
        ha = np.bincount(lfi * 256 + qc, minlength=nl * 256).reshape(nl, 256)
        hb = np.bincount(lfi[keep] * 256 + qc[keep], minlength=nl * 256).reshape(nl, 256)
        use = cin >= 200
        h = np.where(use[:, None], hb, ha)
        tot = np.where(use, cin, tot_all).astype(np.int64)
        cs = np.cumsum(h, axis=1)
        bl = (cs <= ((tot - 1) // 2)[:, None]).sum(1).astype(np.float64)
        bh = (cs <= (tot // 2)[:, None]).sum(1).astype(np.float64)
        cols[:, c] = (bl / 255.0 + bh / 255.0) / 2.0
    return inner_all, cols, "CPU"


def _par_region_one(li):
    """Parallel unit: process one region (bit-identical to the serial version).

    Region masks do not overlap and no rng is used anywhere, so the units share no state; the parent
    writes regions back in the original order, so the output bytes match the serial version. Child
    processes inherit the large arrays in `_PAR_CTX` directly through fork.
    """
    P = _PAR_CTX
    args, labels, areas = P["args"], P["labels"], P["areas"]
    rgb_s, tc, tf = P["rgb_s"], P["tc"], P["tf"]
    thr_ref, bg_label, geo_r = P["thr_ref"], P["bg_label"], P["geo_r"]
    H, W, _MG, objs = P["H"], P["W"], P["MG"], P["objs"]
    _REFINE_STATS["pts"] = 0
    _REFINE_STATS["moved"] = 0
    _REFINE_STATS["sum"] = 0.0
    sl = objs[li] if 0 <= li < len(objs) else None
    if sl is None:
        win, y0, x0 = (slice(0, 0), slice(0, 0)), 0, 0
    else:
        y0 = max(0, sl[0].start - _MG)
        y1 = min(H, sl[0].stop + _MG)
        x0 = max(0, sl[1].start - _MG)
        x1 = min(W, sl[1].stop + _MG)
        win = (slice(y0, y1), slice(x0, x1))
    mask = labels[win] == li
    if int(areas[li]) < args.min_area:      # equals mask.sum(), but in O(1)
        return None, "", dict(_REFINE_STATS)
    _inb = P.get("inner_all")
    if _inb is not None:
        inner = _inb[win] & mask          # the whole-image batch result must be restricted back to this region
        if int(inner.sum()) < 200:
            inner = mask
        col = P["cols"][li]
    else:
        inner = ndi.binary_erosion(mask, iterations=3)
        if inner.sum() < 200:
            inner = mask
        col = np.median(rgb_s[win][inner], axis=0)
    if li == bg_label:
        return ({"idx": li, "mask": mask, "area": int(areas[li]), "d": "",
                 "fill": to_hex(col), "grad": None, "bg": True,
                 "geo": float(geo_r[li]), "win": win, "off": (y0, x0), "col": col},
                "", dict(_REFINE_STATS))
    m = ndi.binary_dilation(mask, iterations=1)
    geo_frac = float(geo_r[li])
    use_fit = args.fit
    if use_fit == "auto":
        use_fit = "bezier" if geo_frac >= args.auto_geo_frac else "cr"
    rf = None
    if args.snap_mode == "refine" and geo_frac >= args.auto_geo_frac:
        rf = (tf, thr_ref, args.snap_shift, args.snap_step, args.snap_sub)
    d = region_path_d(m, args.contour_tol, args.contour_smooth, args.contour_min_area,
                      fit=use_fit, fit_tol=args.fit_tol, corner_deg=args.corner_deg,
                      refine=rf, offset=(y0, x0))
    if not d:
        return None, "", dict(_REFINE_STATS)
    grad = fit_region_gradient(rgb_s, inner, tc, args.grad_stops,
                               args.grad_min_range, off=(y0, x0),
                               allow_radial=args.grad_radial,
                               radial_margin=args.grad_radial_margin)
    msg = ""
    if args.verbose:
        q = f"{grad['quality']:.2f}({grad['axis']})" if grad else "平色"
        msg = (f"      · 区域#{li:<2d} area={int(areas[li]):>7d} {to_hex(col)} "
               f"渐变质量={q} 拟合={use_fit}(几何度{geo_frac:.2f})")
    return ({"idx": li, "mask": mask, "area": int(areas[li]), "d": d,
             "fill": to_hex(col), "grad": grad, "bg": False,
             "geo": geo_frac, "fit": use_fit, "win": win, "off": (y0, x0), "col": col},
            msg, dict(_REFINE_STATS))


def _par_act_one(i):
    """Parallel unit: stroke-activity residual for region i.

    Instead of writing into a shared array it hands (rows, cols, vals) back to the parent to be
    written in the original order -- regions do not overlap, so the write-back order does not affect
    the result, making it bit-identical to the serial version.
    """
    P = _PAR_CTX
    r = P["regions"][i]
    if r["bg"] or r.get("aa"):
        return None
    rgb_s = P["rgb_s"]
    inner = ndi.binary_erosion(r["mask"], iterations=3)
    if inner.sum() < 50:
        return None
    rows, cols_, pred = predict_region(P["shape"], inner, r["grad"], r.get("off", (0, 0)))
    obs = rgb_s[rows, cols_]
    if pred is None:
        pred = np.median(obs, axis=0)[None, :]
    return rows, cols_, np.abs(obs - pred).mean(1)


def _par_detail_one(k):
    """Parallel unit: one colored thin seam line (no rng, bit-identical to the serial version).

    Computed only inside the bounding-box window given by ``find_objects(lb + 1)``. It used to start
    with ``mk = P["lb"] == k``, building an (H,W) boolean array over the **whole image**, after which
    erosion / dilation / color sampling each scanned the whole image again; microbenchmark (wave's
    real size 3859x2594) 364.7ms per component versus 1.053ms for the windowed version -- **346x**.
    """
    P = _PAR_CTX
    args, rgb_s = P["args"], P["rgb_s"]
    tc, tf, thr_ref = P["tc"], P["tf"], P["thr_ref"]
    H, W, MG, objs = P["H"], P["W"], P["MG"], P["objs"]
    sl = objs[k] if 0 <= k < len(objs) else None
    if sl is None:
        win, y0, x0 = (slice(0, 0), slice(0, 0)), 0, 0
    else:
        y0 = max(0, sl[0].start - MG)
        y1 = min(H, sl[0].stop + MG)
        x0 = max(0, sl[1].start - MG)
        x1 = min(W, sl[1].stop + MG)
        win = (slice(y0, y1), slice(x0, x1))
    mk = P["lb"][win] == k
    inner = ndi.binary_erosion(mk, iterations=2)
    if inner.sum() < 20:
        inner = mk
    col = np.median(rgb_s[win][inner], axis=0)
    _t_rp = time.time()
    d = region_path_d(ndi.binary_dilation(mk, iterations=1),
                      args.contour_tol * 0.5,
                      args.contour_smooth * 0.6, args.contour_min_area,
                      fit=("bezier" if args.fit == "auto" else args.fit),
                      fit_tol=args.fit_tol,
                      corner_deg=args.corner_deg,
                      refine=((tf, thr_ref, args.snap_shift, args.snap_step)
                              if args.snap_mode == "refine" else None),
                      offset=(y0, x0))
    if not d:
        return None, ""
    grad = fit_region_gradient(rgb_s, inner, tc, args.grad_stops,
                               args.grad_min_range, off=(y0, x0),
                               allow_radial=args.grad_radial,
                               radial_margin=args.grad_radial_margin)
    # The long tail of the detail layer is "the contour DP of a few very large components": there are very few components, so a line is always reported here
    # (including the component area and the region_path_d time), so the next round can pinpoint which component and how many seconds.
    _dt = time.time() - _t_rp
    q = f"{grad['quality']:.2f}" if grad else "平色"
    msg = (f"      · 缝线#{k} area={int(mk.sum()):>7d} 窗口={mk.shape[0]}x{mk.shape[1]} "
           f"轮廓={_dt:.1f}s {to_hex(col)} 渐变={q}")
    if args.verbose:
        msg += f" (候选分量中第 {k} 个)"
    return ({"idx": f"d{k}", "mask": mk, "area": int(mk.sum()),
             "d": d, "fill": to_hex(col), "grad": grad,
             "bg": False, "detail": True, "win": win, "off": (y0, x0)}, msg)


def _par_stroke_one(i):
    """Parallel unit: strokes for one region (streamline tracing + color sampling).

    In parallel mode each region uses a "deterministic derived seed", independent of scheduling
    order and process count → reproducible for the same arguments and process count. The cost: the
    random jitter grid differs from the serial version (shared rng), so --jobs 1 is the setting that
    is byte-identical to out/.
    """
    P = _PAR_CTX
    args, tc, fg = P["args"], P["tc"], P["fg"]
    act, rgb_s = P["act"], P["rgb_s"]
    H, W = P["H"], P["W"]
    r = P["sregs"][i]
    rng = np.random.default_rng((int(P["seed"]) * 1000003 + int(i) + 1) & 0x7FFFFFFF)
    win = r.get("win")
    if win is None:
        sub = r["mask"] & fg
    else:
        sub_w = r["mask"] & fg[win]
        if sub_w.sum() < 400:
            return None
        sub = np.zeros((H, W), bool)
        sub[win] = sub_w
    if sub.sum() < 400:
        return None
    segs = make_streamlines(tc, sub, args, rng)
    if not segs:
        return None
    items = []
    for pts, wid, _c in segs:
        iy = np.clip(pts[:, 1].astype(int), 0, H - 1)
        ix = np.clip(pts[:, 0].astype(int), 0, W - 1)
        a = float(act[iy, ix].mean())
        alpha = args.stroke_alpha * min(1.0, a / (args.stroke_activity_ref + EPS))
        if alpha < args.stroke_alpha_min:
            continue  # the base fill is accurate enough -> no stroke (avoids creating artifacts in flat areas)
        col = sample_color(rgb_s, pts, args.stroke_jitter, rng)
        items.append((ribbon_d(pts, wid), col, min(alpha, 0.95)))
    return {"idx": r["idx"], "d": r["d"], "area": r["area"], "strokes": items}


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="logo_trace.py",
        description="基于多尺度结构张量的位图描摹 / 矢量化工具"
                    "（结构张量 → 分区路由 → SVG）",
        epilog=EXAMPLES, formatter_class=_Fmt)
    p.add_argument("--version", action="version", version=f"logotrace {__version__}")

    g = p.add_argument_group("输入输出")
    g.add_argument("--in", dest="src", default="inputs/logo.png",
                   help="输入位图; 给裸文件名会自动到 inputs/ 下找")
    g.add_argument("--out", default=None,
                   help="输出 SVG; 默认 out/<输入名>_traced.svg")
    g.add_argument("--preview", nargs="?", const="auto", default=None,
                   help="预览 PNG; 不带值=自动命名到 out/")
    g.add_argument("--no-preview", dest="preview", action="store_const", const="",
                   help="不渲染预览")
    g.add_argument("--debug", nargs="?", const="auto", default=None,
                   help="结构张量调试图; 默认不输出, 不带值=自动命名到 out/")
    g.add_argument("--gzip", action="store_true",
                   help="同时输出 .svgz (gzip -9; 浏览器 / <img> 可直接用)")
    g.add_argument("--compress", choices=["off", "slim", "tight"], default="slim",
                   help="SVG 瘦身: off=原样; slim=数字精度裁剪(逐位无损, 可再 gzip); "
                        "tight=再多删掉重复 clipPath(体积最小, 但笔触会越出所属色块: "
                        "logo 2 倍实测 -1.9 dB / 4 倍 -5.7 dB, 水彩 -0.01 dB; "
                        "**--preset painting 默认就是 tight**). "
                        "硬边图形还想更小, 用 svgzip.py 的 --prec 0(几乎无损, 体积再降约 1/3)")

    g = p.add_argument_group("运行")
    g.add_argument("--scale", type=float, default=4.0,
                   help="描摹前 Lanczos 放大倍数 (硬边图形 2.0 起收益明显)")
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--no-autoscale", dest="no_autoscale", action="store_true",
                   help=f"关掉小图自动缩参 (默认开启: 盘面 < {AUTOSCALE_REF:.0f}px 时按 "
                        f"k=max(W,H)/{AUTOSCALE_REF:.0f} 缩小 min_area/spacing 等绝对像素参数)")
    g.add_argument("--verbose", dest="verbose", action="store_true", default=True,
                   help=argparse.SUPPRESS)
    g.add_argument("--quiet", dest="verbose", action="store_false",
                   help="只打印结束摘要")

    g = p.add_argument_group("预设")
    g.add_argument("--preset", choices=sorted(PRESETS), default=None,
                   help="参数预设 (命令行显式参数优先): "
                        "logo=硬边平面图(2 倍网格 + 亚像素定位 + 保角贝塞尔); "
                        "painting=写意/照片(色块小而多、笔触密而细)")

    g = p.add_argument_group("结构张量")
    g.add_argument("--sigma-d", type=float, default=1.0, help="细尺度 微分尺度 σd")
    g.add_argument("--sigma-i", type=float, default=2.5, help="细尺度 积分尺度 σi")
    g.add_argument("--cs-sigma-d", type=float, default=2.0, help="粗尺度 σd (笔触流场)")
    g.add_argument("--cs-sigma-i", type=float, default=12.0, help="粗尺度 σi (笔触流场)")

    g = p.add_argument_group("分割")
    g.add_argument("--method", choices=["colors", "edges", "hybrid", "watershed"],
                   default="hybrid",
                   help="hybrid=分区路由(默认): 颜色分割定区域身份, 纯色几何区把边界亚像素"
                        "吸附到边缘并走保角贝塞尔, 笔触/渐变复杂区走原来的做法; "
                        "colors=纯颜色分割(Catmull-Rom); edges=纯边缘检测分水岭; "
                        "watershed=能量分水岭")
    g.add_argument("--labels-cache", default="",
                   help="分割结果缓存文件 (.npy): 存在则载入, 否则算完保存; 空=不用")
    g.add_argument("--gpu", choices=["off", "auto", "on"], default="auto",
                   help="结构张量+k-means 是否上 GPU(默认 auto)。auto=可用就用(cupy 或自研 "
                        "CUDA 后端 cuda_backend.py), 没有就静默退回 CPU; off=只用 CPU; "
                        "auto=可用就用(自研 CUDA 后端 cuda_backend.py, 或 cupy), 否则静默"
                        "退回 CPU; on=必须有 GPU。实测: 张量阶段 1.7~2.6×, 端到端因 float32 "
                        "张量变成 37.54 dB (CPU 37.57), 且大图上收益更明显")
    g.add_argument("--jobs", type=int, default=0,
                   help="逐区域阶段的并行进程数 (fork + 写时复制, 大数组不拷贝、不 pickle)。"
                        "0=自动: min(核数,16), 候选区域少于 24 个时用 1; 1=关闭并行。"
                        "无 rng 的环节(区域矢量/活动度/细节层)并行后输出逐字节不变; "
                        "笔触层并行时改用每区域确定性种子, 所以 --jobs 1 才是与 out/ "
                        "逐字节相同的口径")
    g.add_argument("--batch-book", choices=("auto", "off"), default="auto",
                   help="逐区域阶段的规则密集子步骤(掩码腐蚀 / 逐区域中位色)改成整图批量: "
                        "auto=能算就算(GPU 上优先 cupy), off=沿用逐区域循环。批量版与"
                        "逐区域版逐字节相同(1/255 网格上中位数有闭式解); 图经过重采样"
                        "(scale != 1)时像素不在该网格上, 自动回退到逐区域。")
    g.add_argument("--seg", choices=["full", "auto", "flat"], default="full",
                   help="分割路线。full=完整路线(默认, k-means+RAG); auto/flat=平色快路径"
                        "(直方图主色+LUT+连通域)。实测: 硬边平色图标 27.57→27.37 dB 而分割 "
                        "9.3→4.1s; 带渐变的 logo 37.57→34.7 dB —— 所以默认不用, 只建议"
                        "纯硬边平面图显式指定")
    g.add_argument("--flat-cov", type=float, default=0.90,
                   help="快路径判据: 主色桶需覆盖的像素占比, 低于此值走完整路线")
    g.add_argument("--flat-nb", type=int, default=5,
                   help="快路径直方图每通道位数 (5 → 32 级/通道)")
    g.add_argument("--flat-share", type=float, default=2e-4,
                   help="快路径: 一个桶要算作主色的最小像素占比")
    g.add_argument("--flat-merge", type=float, default=0.0,
                   help="快路径: 调色板归并的颜色距离阈值 (0-1)")
    g.add_argument("--edge-nms-hi", type=float, default=0.97,
                   help="edges 方法: 非极大值抑制后的滞后高阈值(能量分位)")
    g.add_argument("--edge-nms-lo", type=float, default=0.92,
                   help="edges 方法: 滞后低阈值(能量分位)")
    g.add_argument("--edge-core", type=float, default=3.0,
                   help="edges 方法: 平坦区种子要求的到边缘距离 (px)")
    g.add_argument("--edge-close", type=int, default=0,
                   help="edges 方法: 边缘闭运算次数 (补 1~2px 缺口)")
    g.add_argument("--edge-dsmooth", type=float, default=0.6,
                   help="edges 方法: 距离场平滑 σ")
    g.add_argument("--kmeans-k", type=int, default=20)
    g.add_argument("--kmeans-iters", type=int, default=16)
    g.add_argument("--pre-smooth", type=float, default=1.6, help="量化前高斯平滑 σ")
    g.add_argument("--thresh", type=float, default=12.0, help="RAG 合并阈值 (0-255 色距)")
    g.add_argument("--merge-thresh2", type=float, default=10.0,
                   help="连通域层面再次 RAG 合并的阈值 (0=关闭)")
    g.add_argument("--detail-chroma", type=float, default=12.0,
                   help="彩色细缝线细节层的 Lab 色度阈值 (0=关闭)")
    g.add_argument("--detail-scale", type=float, default=1.0,
                   help="色度图高斯平滑 σ")
    g.add_argument("--detail-min-area", type=int, default=40)
    g.add_argument("--protect-sat", type=float, default=0.0,
                   help="平均饱和度(rgb 极差)高于此值的彩色区域不参与合并/吸收 "
                        "(0=关闭; 金色缝线已由细节层单独处理)")
    g.add_argument("--merge-passes", type=int, default=4)
    g.add_argument("--min-area", type=int, default=600)
    g.add_argument("--min-width", type=int, default=0,
                   help="标签开运算核宽(px), 0=关闭; 会压缩细长真实特征, 慎用")
    g.add_argument("--denoise", type=float, default=0.0,
                   help="保边平滑 sigma_color (0=关闭; 输入有噪声时建议 0.03~0.05)")
    g.add_argument("--contour-tol", type=float, default=0.75)
    g.add_argument("--contour-smooth", type=float, default=1.0)
    g.add_argument("--fit", choices=["auto", "cr", "bezier"], default="auto",
                   help="曲线拟合: auto=按区域几何度自动选择(几何区 bezier/复杂区 cr); "
                        "cr=Douglas-Peucker+Catmull-Rom; "
                        "bezier=保角最优贝塞尔(直线输出 l, 保尖角)")
    g.add_argument("--fit-tol", type=float, default=0.10,
                   help="bezier 拟合的最大允许误差 (px)")
    g.add_argument("--corner-deg", type=float, default=62.0,
                   help="bezier 拟合的角点判定角度")
    g.add_argument("--auto-geo-frac", type=float, default=0.5,
                   help="--fit auto 的几何度门槛 (区域里几何像素占比)")
    g.add_argument("--tex-sigma", type=float, default=2.5,
                   help="纹理图高通尺度 σ")
    g.add_argument("--tex-smooth", type=float, default=6.0,
                   help="纹理图能量平滑 σ")
    g.add_argument("--tex-norm", type=float, default=99.5,
                   help="纹理归一化分位")
    g.add_argument("--tex-thr", type=float, default=3.0,
                   help="判为几何区的纹理上限 (归一化后)")
    g.add_argument("--range-thr", type=float, default=0.02,
                   help="判为几何区的内部色跨上限 (0.02≈5/255)")
    g.add_argument("--tex-close", type=int, default=0)
    g.add_argument("--tex-open", type=int, default=0)
    g.add_argument("--snap-w", type=float, default=3.0,
                   help="hybrid 方法: 边缘线在高程里的加成权重 (越大越贴边缘)")
    g.add_argument("--snap-sigma", type=float, default=1.0,
                   help="hybrid 方法: 边缘线加成前的平滑 σ")
    g.add_argument("--snap-erode", type=int, default=2,
                   help="hybrid 方法: 颜色区域种子的腐蚀半径 (给边界留吸附余地)")
    g.add_argument("--snap-band", type=float, default=3.0,
                   help="hybrid 方法: 边界最多允许移动多少像素 (0=不限制)")
    g.add_argument("--snap-mode", choices=["refine", "watershed", "off"],
                   default="refine",
                   help="边界吸附方式: refine=轮廓点沿法向亚像素吸到边缘能量脊线 "
                        "(局部、保拓扑, 默认); watershed=标记控制分水岭重切边界; off=不吸附")
    g.add_argument("--snap-shift", type=float, default=1.5,
                   help="refine 模式: 单点最大吸附位移 (px)")
    g.add_argument("--snap-sub", choices=["half", "peak"], default="half",
                   help="亚像素定心: half=亮度剖面 50%% 交点(更准), peak=能量峰值")
    g.add_argument("--snap-step", type=float, default=0.15,
                   help="refine 模式: 法向剖面采样步长 (px)")
    g.add_argument("--aa-levels", type=int, default=0,
                   help="抗锯齿过渡带重建级数 (每侧几条; 0=关闭)。原图边界有 2~3px"
                        "抗锯齿过渡, 九成平方误差集中在这条带上; 但实测把边界定准"
                        "以后, 用同色阶梯逼近斜坡反而略降 PSNR (见 README), 故默认关")
    g.add_argument("--aa-width", type=float, default=2.5,
                   help="抗锯齿过渡带总宽 (px), 应≈原图边界过渡宽度")
    g.add_argument("--aa-smooth", type=float, default=0.8,
                   help="距离场平滑 σ (越小越贴原始边界)")
    g.add_argument("--aa-tol", type=float, default=0.15,
                   help="过渡带轮廓的贝塞尔拟合容差 (px)")
    g.add_argument("--aa-geo-thr", type=float, default=-1.0,
                   help="过渡带准入的几何度阈值 (<0 时用 --auto-geo-frac)")
    g.add_argument("--aa-min-radius", type=float, default=1.4,
                   help="色块最大内切半径小于此值就不做过渡带 (避免两侧偏移交叉)")
    g.add_argument("--aa-profile", choices=["shape", "uniform"], default="shape",
                   help="过渡带分级方式: shape=按实测过渡剖面做最优量化, uniform=等宽")
    g.add_argument("--aa-min-area", type=int, default=8,
                   help="过渡带单段最小像素数")
    g.add_argument("--aa-synthetic", action=argparse.BooleanOptionalAction, default=False,
                   help="在**合成边界**上也生成抗锯齿过渡带. 合成边界 = 自适应细化从同一个原始区域"
                        "切出来的两个子区域之间的分界(颜色天然连续, 过渡带基本是白花的字节); "
                        "背景边界/其它原始区域边界/真实图像边缘不受影响, 始终保留过渡带. "
                        "默认关闭(=省掉合成边界的过渡带); --aa-synthetic 恢复旧行为(逐字节)")
    g.add_argument("--aa-edge-dedup", action=argparse.BooleanOptionalAction, default=True,
                   help="把**同一个原始区域**的祖先一路带到细化树的每一层, 用它去重: 细化递归切开后, "
                        "同一原始区域的后代之间(哪怕不是直接父子, 而是叔侄/堂兄弟)的分界都算合成边界, "
                        "只保留一份过渡带. 关闭时只认一层父子关系(旧行为, 逐字节). "
                        "真实图像边缘的过渡带不受影响 (apple@4: 1223→1043 KB, PSNR/MAE/JUMP 不变)")
    g.add_argument("--aa-edge-canon", action=argparse.BooleanOptionalAction, default=False,
                   help="[实验, 默认关闭] 更激进的去重: 同一原始区域的一组后代里, **真实图像边缘**上只"
                        "允许面积最大的那个后代生成过渡带, 其余后代只保留合成边界上的过渡带. "
                        "实测会改变真实边缘的过渡带(每个后代各自负责自己那一段边缘), 因此不建议开启")
    g.add_argument("--contour-min-area", type=float, default=30.0)

    g = p.add_argument_group("渐变")
    g.add_argument("--grad-stops", type=int, default=10)
    g.add_argument("--grad-min-gain", type=float, default=0.12,
                   help="渐变验收门槛: 线性渐变至少要消掉这个比例的平方误差, 否则该区域退回纯色. "
                        "平滑渐变图(如柔和 logo/渲染图)可降到 0.02 左右, 显著减少色块台阶")
    g.add_argument("--auto-gradient", action="store_true",
                   help="按图像平滑度自动放宽渐变门槛(平滑图自动降门槛), 无需手调")
    g.add_argument("--grad-min-range", type=float, default=0.004,
                   help="启用线性渐变的色跨下限 (0.004≈1/255; 本图渐变极缓)")
    g.add_argument("--grad-radial", action=argparse.BooleanOptionalAction, default=True,
                   help="允许径向渐变(中心+半径)候选: 与线性用同一误差增益判据, 只有当径向残差"
                        "比最优线性小 --grad-radial-margin 时才采用 (--no-grad-radial 关闭)")
    g.add_argument("--grad-radial-margin", type=float, default=GRAD_RADIAL_MARGIN,
                   help="径向采用的残差优势门槛: 径向残差需比线性残差至少小这个比例")
    g.add_argument("--merge-grad", action=argparse.BooleanOptionalAction, default=True,
                   help="梯度感知区域合并: 相邻色块的并集若能仍被单个渐变(线性/径向)解释, 就合并成"
                        "更大更平滑的区域, 消除色块台阶 (--no-merge-grad 关闭)")
    g.add_argument("--merge-grad-tol", type=float, default=0.05,
                   help="合并判据: 并集的单渐变残差 <= 两个独立拟合残差之和 ×(1+该值)")
    g.add_argument("--merge-grad-passes", type=int, default=4,
                   help="梯度感知合并的迭代轮数 (每轮固定扫描顺序, 结果确定)")

    g = p.add_argument_group("自适应细化 (误差驱动)")
    g.add_argument("--adaptive-refine", action=argparse.BooleanOptionalAction, default=True,
                   help="误差驱动自适应细化: 逐区域量测当前填充(平色/线性/径向)的残差, 残差超阈值"
                        "就按误差最大方向二分并递归重拟合, 把固定的 SVG 元素预算花在最需要的地方, "
                        "消除平滑/光泽图上的可见平色块 (--no-adaptive-refine 恢复旧行为; 与旧版"
                        " --no-grad-radial --no-merge-grad 一样逐字节复原旧输出)")
    g.add_argument("--refine-err", type=float, default=0.02,
                   help="细化的残差阈值(满量程比例, 0.02≈5/255): 区域内部(腐蚀后的核心)中最大通道"
                        "误差超过该值的像素数达到 max(16, max(32, refine-min-area/4)/2) 以上(即"
                        "确实存在一块可见的误差)才考虑二分; 局部判据很关键, 大区域上的小块误差几乎"
                        "不抬高全局 RMS")
    g.add_argument("--refine-min-area", type=int, default=400,
                   help="参与细化的最小区域面积; 小于该值的区域保持原样 (防止追着噪点/细缝走)")
    g.add_argument("--refine-max-depth", type=int, default=6,
                   help="递归细化的最大深度 (0=只做一轮不递归)")
    g.add_argument("--refine-gain", type=float, default=0.20,
                   help="二分被接受所需的最小残差下降比例 (子区域面积加权 RMS <= (1-该值)×父区域); "
                        "这是对噪声/纹理的天然保护, 也是生长速度的阀门")
    g.add_argument("--refine-budget", type=int, default=0,
                   help="细化新增区域数的上限; 0=自动 (max(64, min(256, 区域数)))")
    g.add_argument("--refine-order", choices=("merge-first", "refine-first"), default="merge-first",
                   help="细化与梯度感知合并的先后: merge-first=先合并再细化(默认); "
                        "refine-first=先在原始分区上细化, 再对细化后的分区做梯度合并")
    g.add_argument("--dump-refined-labels", default="",
                   help="把细化后的标签图写成 .npy (诊断 / 复现 refine-first 顺序); 空=不写")

    g = p.add_argument_group("笔触 (结构张量流线)")
    g.add_argument("--no-strokes", action="store_true",
                   help="只出几何层; --preset logo 默认即开启")
    g.add_argument("--spacing", type=float, default=10.0, help="笔触间距")
    g.add_argument("--occ-ratio", type=float, default=0.45, help="占位半径/间距")
    g.add_argument("--stroke-step", type=float, default=3.0)
    g.add_argument("--stroke-max", type=float, default=120.0)
    g.add_argument("--stroke-min", type=float, default=24.0)
    g.add_argument("--stroke-nodes", type=int, default=12)
    g.add_argument("--stroke-width", type=float, default=7.5)
    g.add_argument("--stroke-grid-units", action="store_true",
                   help="笔触长度参数按描摹网格px解释(旧行为); 默认按原生像素, 自动乘 --scale")
    g.add_argument("--stroke-alpha", type=float, default=0.6)
    g.add_argument("--stroke-jitter", type=float, default=0.012)
    g.add_argument("--stroke-activity-ref", type=float, default=0.012,
                   help="笔触透明度=stroke-alpha×min(1,残差/该值); 残差来自'图像-底填'")
    g.add_argument("--stroke-alpha-min", type=float, default=0.03,
                   help="低于此透明度的笔触直接丢弃")
    g.add_argument("--coh-min", type=float, default=0.12)
    g.add_argument("--max-turn", type=float, default=70.0, help="单步最大转折角(度)")

    g = p.add_argument_group("边缘 / 缝线 (本图的金色缝线已由色块层重建, 故默认关闭)")
    g.add_argument("--edge-mode", choices=["off", "chroma", "energy", "both"], default="off",
                   help="off=不画; chroma=Lab 色度细线; energy=结构张量能量脊线")
    g.add_argument("--edge-chroma", type=float, default=9.0, help="Lab 色度阈值 (chroma 模式)")
    g.add_argument("--edge-hi", type=float, default=0.975, help="高能量分位 (energy 模式)")
    g.add_argument("--edge-lo", type=float, default=0.90, help="滞后低阈值分位 (energy 模式)")
    g.add_argument("--edge-coh", type=float, default=0.45, help="相干性下限 (energy 模式)")
    g.add_argument("--edge-width", type=float, default=2.0)
    g.add_argument("--edge-alpha", type=float, default=0.9)
    g.add_argument("--edge-min-area", type=int, default=24)
    g.add_argument("--edge-min-len", type=int, default=8)
    g.add_argument("--edge-tol", type=float, default=0.6, help="骨架路径简化容差")

    g = p.add_argument_group("着色层 (叠加半透明径向渐变 / gradient boosting; 实验性, 默认关闭)")
    g.add_argument("--shade-blobs", action=argparse.BooleanOptionalAction, default=False,
                   help="overlay stacked translucent radial gradients to cancel the 1..6/255 steps "
                        "between neighbouring region fills (the flat polygonal patches / Mach banding "
                        "that survive even when the pixel error is small). Every layer is a smooth "
                        "radial colour curve inside a soft disk that may only live in a large "
                        "low-gradient area; a layer is kept only while it strictly reduces the "
                        "smooth-weighted residual error of its disk, so the stack is a boosting "
                        "sequence. Plain SVG 1.1 (radialGradient + stop-opacity), so cairosvg and "
                        "mainstream browsers paint it identically. Costs one extra cairosvg render "
                        "plus about 1 s per layer per megapixel (measured: 50 s for 48 layers at "
                        "1000x1248).")
    g.add_argument("--shade-layers", type=int, default=48,
                   help="maximum number of stacked layers (the fit stops early once no candidate can "
                        "improve the smooth residual any more)")
    g.add_argument("--shade-radius", type=float, default=0.0,
                   help="largest disk radius in native px; 0 = auto = 0.19*max(width,height). The five "
                        "radii tried are 0.17/0.29/0.46/0.67/1.0 of it, and --scale is applied")
    g.add_argument("--shade-min-area", type=int, default=0,
                   help="only smooth connected areas of at least this many tracing-grid px^2 are "
                        "corrected; 0 = auto = (0.05*max(width,height)*--scale)^2")
    g.add_argument("--shade-eps", default="0.2,0.35,0.5",
                   help="candidate centre alphas of a disk (stronger alpha attenuates the boundary "
                        "steps more, but leans harder on the radial colour model)")
    g.add_argument("--shade-stops", type=int, default=4,
                   help="colour stops per layer (the radial colour curve is piecewise linear on them)")
    g.add_argument("--shade-tedge", type=float, default=0.55,
                   help="radius fraction that still has full alpha; the ramp to 0 runs from there to r")
    g.add_argument("--shade-smooth-gtol", type=float, default=2.5,
                   help="gradient magnitude (0..255 per px) at which the smooth weight reaches 0")
    g.add_argument("--shade-smooth-ttol", type=float, default=2.0,
                   help="detail energy at which the texture weight reaches 0 (keeps layers off noisy "
                        "areas)")
    g.add_argument("--shade-non-smooth", type=float, default=0.35,
                   help="largest share of non-smooth pixels a disk interior may contain")
    g.add_argument("--shade-halo", type=float, default=6.0,
                   help="largest RMS colour move (0..255) a layer may apply to non-smooth pixels "
                        "inside its disk (halo guard)")
    g.add_argument("--shade-gain", type=float, default=0.0,
                   help="minimum relative reduction of a disk's weighted residual error required to "
                        "keep the layer")
    g.add_argument("--shade-cand", type=int, default=8,
                   help="candidate disk centres evaluated per layer")
    g.add_argument("--shade-margin", type=float, default=28.0,
                   help="non-maximum-suppression radius between candidate centres, in grid px")
    pre, _ = p.parse_known_args(argv)
    if pre.preset:
        p.set_defaults(**PRESETS[pre.preset])
    return p.parse_args(argv)


# ======================================================================
# 9. output post-processing: slimming / paths / directories
# ======================================================================


def ensure_dir(path: str) -> None:
    """Make sure the parent directory of the output file exists."""
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)


def resolve_io(args) -> None:
    """Complete the input path (a bare filename automatically looks in inputs/) and auto-name the three outputs (default out/)."""
    if not os.path.isfile(args.src):
        alt = os.path.join("inputs", args.src)
        if os.path.isfile(alt):
            args.src = alt
    stem = os.path.splitext(os.path.basename(args.src))[0] or "out"

    def expand(val, suffix):
        if val is None or val == "auto":
            return os.path.join("out", stem + suffix)
        return val

    if not args.out:
        args.out = os.path.join("out", stem + "_traced.svg")
    args.preview = expand(args.preview, "_preview.png")
    args.debug = "" if args.debug is None else expand(args.debug, "_debug.png")
    for pth in (args.out, args.preview, args.debug):
        if pth:
            ensure_dir(pth)


def main(argv=None):
    args = parse_args(argv)
    # Gradient gate: explicit value, or auto-tuned from image smoothness (--auto-gradient).
    _argv = list(sys.argv[1:] if argv is None else argv)
    # Was --aa-levels given by the user? (a preset value does not count as explicit)
    _aa_explicit = any(_a == "--aa-levels" or _a.startswith("--aa-levels=") for _a in _argv)
    _re_explicit = any(_a == "--refine-err" or _a.startswith("--refine-err=") for _a in _argv)
    if args.auto_gradient:
        from PIL import Image as _Im
        with _Im.open(args.src) as _im:
            _a = np.asarray(_im.convert("RGB").resize((64, 64))).astype(np.float64) / 255.0
        _g = float(np.abs(np.diff(_a, axis=0)).mean() + np.abs(np.diff(_a, axis=1)).mean())
        state.GRAD_MIN_GAIN = 0.02 if _g > 0.008 else args.grad_min_gain
        if _g > 0.008:
            # Smooth/glossy images: be aggressive about merging patches into large gradient-filled
            # regions -- this is what removes the visible colour steps the user sees. Measured on
            # apple.png @scale4: >=40px plateaus 35.5% -> 26.6% (source 24.8%), JUMP% back to the
            # source level, for about 0.5 dB of PSNR. Texture-rich photos should use --no-merge-grad,
            # where merging costs ~0.5 dB and radial-only is strictly better.
            args.merge_grad_tol = max(args.merge_grad_tol, 0.3)
            args.merge_grad_passes = max(args.merge_grad_passes, 8)
        print(f"      · 自动渐变路由: 平滑度 {_g:.4f} → 渐变门槛 {state.GRAD_MIN_GAIN}",
              file=sys.stderr)
        # Smooth images: their remaining hard steps are the k-means patch boundaries, so also turn on
        # the boundary anti-aliasing staircase (3 levels) unless the user pinned --aa-levels. Tied to
        # the smooth-image feature stack so that --no-grad-radial --no-merge-grad still reproduces the
        # legacy output byte for byte for every other flag combination.
        if (_g > 0.008 and not _aa_explicit and int(args.aa_levels) <= 0
                and (args.grad_radial or args.merge_grad)):
            args.aa_levels = 3
            print("      · 平滑图: 边界抗锯齿过渡带自动开启 (--aa-levels 3, 可用 --aa-levels 0 关闭)",
                  file=sys.stderr)
        # Smooth images: the residual that is left after merging is a *graded* residual (the fill is
        # 1-2/255 off over broad shallow ramps), so the refinement error gate can be tightened
        # without chasing sensor noise. Measured on apple.png @scale4: --refine-err 0.008 + the auto
        # budget takes >=40px plateaus from 26.6% to 23.9% (source 24.8%) and JUMP% from 11.16 to
        # 11.09 (source 11.20), with PSNR 33.32 -> 33.50 dB. Textured photos keep --refine-err.
        if _g > 0.008 and not _re_explicit and args.adaptive_refine:
            _re0 = float(args.refine_err)
            args.refine_err = min(_re0, 0.008)
            if args.refine_err < _re0:
                print(f"      · 平滑图: 自适应细化误差门槛收紧 {_re0} → {args.refine_err} "
                      f"(可用 --refine-err 覆盖)", file=sys.stderr)
    else:
        state.GRAD_MIN_GAIN = args.grad_min_gain
    state.set_quiet(not args.verbose)
    resolve_io(args)
    if args.scale != 1.0 and not args.stroke_grid_units:
        # Length-like stroke parameters are interpreted in "native pixels": multiply by --scale automatically.
        # Otherwise the spacing would be in grid units and the stroke density would grow with scale² (59->9567 strokes at 4x, volume x4).
        for _nm in ("spacing", "stroke_min", "stroke_max", "stroke_width"):
            setattr(args, _nm, getattr(args, _nm) * args.scale)
    t0 = time.time()
    rng = np.random.default_rng(args.seed)

    def el():
        return f"{time.time()-t0:6.1f}s"

    # ---------------- read the image ----------------
    if not os.path.isfile(args.src):
        sys.exit(f"[错误] 找不到输入文件: {args.src}")
    im0 = Image.open(args.src).convert("RGB")
    im = im0
    W0, H0 = im.size
    # ---------------- auto-scaling of parameters for small images ----------------
    # The defaults are calibrated for a ~1254px canvas (min_area=600px², spacing=10px ...). When the image is markedly smaller,
    # these absolute pixel quantities become relatively too large: regions are over-merged and strokes become thick bands smeared over the picture.
    # Here k = max(W,H)/AUTOSCALE_REF scales length-like parameters by k and area-like ones by k²; for k >= 1 nothing changes,
    # so large canvases (logo 1254², watercolor 1511x1600) keep exactly the historical parameters and results.
    if not getattr(args, "no_autoscale", False):
        k = max(W0, H0) / AUTOSCALE_REF
        if k < 1.0:
            for _nm in ("spacing", "stroke_min", "stroke_max", "stroke_width", "stroke_step",
                        "contour_tol", "snap_band", "snap_sigma", "snap_shift", "edge_width",
                        "edge_tol", "tex_sigma", "tex_smooth", "pre_smooth", "detail_scale",
                        "aa_width", "aa_smooth", "aa_min_radius"):
                if hasattr(args, _nm):
                    setattr(args, _nm, getattr(args, _nm) * k)
            for _nm in ("min_area", "detail_min_area", "edge_min_area", "edge_min_len",
                        "contour_min_area", "aa_min_area"):
                if hasattr(args, _nm):
                    setattr(args, _nm, max(1, int(round(getattr(args, _nm) * k * k))))
            log(f"      · 小图自动缩参 k={k:.3f} (参考盘面 {AUTOSCALE_REF:.0f}px): "
                f"min_area={args.min_area}  spacing={args.spacing:.1f}px")
    if args.scale != 1.0:
        im = im.resize((max(32, int(W0 * args.scale)), max(32, int(H0 * args.scale))),
                       Image.LANCZOS)
    W, H = im.size
    rgb = np.asarray(im).astype(np.float64) / 255.0
    log(f"[{el()}] 读入 {args.src}  {W}x{H}"
          + (f" (原图 {W0}x{H0})" if args.scale != 1.0 else ""))

    # ---------------- edge-preserving smoothing ----------------
    rgb_s = rgb
    if args.denoise > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rgb_s = denoise_bilateral(rgb, sigma_color=args.denoise, sigma_spatial=2,
                                      channel_axis=-1)
    rgb255 = np.clip(rgb_s, 0, 1) * 255.0

    # ---------------- structure tensor ----------------
    tf = tc = None
    if gpu.enabled(args):
        try:
            tf = gpu.structure_tensor(rgb_s, args.sigma_d, args.sigma_i, eps=EPS)
            tc = gpu.structure_tensor(rgb_s, args.cs_sigma_d, args.cs_sigma_i, eps=EPS)
            log(f"[{el()}] 结构张量: GPU {gpu.device_name()} "
                f"(显存内, 与 CPU 同双精度累加口径)")
        except Exception as _e:
            print(f"      ! GPU 结构张量失败({_e}), 退回 CPU", file=sys.stderr)
            tf = tc = None
    if tf is None:
        tf = structure_tensor(rgb_s, args.sigma_d, args.sigma_i)
        tc = structure_tensor(rgb_s, args.cs_sigma_d, args.cs_sigma_i)
    log(f"[{el()}] 结构张量: 细尺度(σd={args.sigma_d},σi={args.sigma_i}) "
          f"平均相干={tf['coh'].mean():.3f} | 粗尺度(σd={args.cs_sigma_d},"
          f"σi={args.cs_sigma_i}) 平均相干={tc['coh'].mean():.3f}")

    # ---------------- segmentation ----------------
    # --labels-cache: cache the segmentation result. When tuning stroke/gradient/fitting parameters there is no need to rerun segmentation,
    # and it is off by default so default results are unaffected.
    geo_px = None
    if args.labels_cache and os.path.exists(args.labels_cache):
        labels = np.load(args.labels_cache).astype(np.int32)
        log(f"[{el()}] 载入分割缓存 {args.labels_cache} "
              f"({int(labels.max()) + 1} 个色块)")
    else:
        if args.method == "colors":
            labels = segment_colors(rgb_s, rgb255, args, rng)
        elif args.method == "edges":
            labels = segment_edges(rgb_s, tf, args, rng)
        elif args.method == "hybrid":
            if args.seg in ("auto", "flat"):
                labels = segment_flat(rgb_s, rgb255, args, rng)
                if labels is None and args.seg == "flat":
                    print("      ! 平色快路径不适用, 退回完整路线", file=sys.stderr)
            if args.seg == "flat" and labels is None:
                labels, geo_px = segment_hybrid(rgb_s, rgb255, tf, args, rng)
            elif args.seg == "full":
                labels, geo_px = segment_hybrid(rgb_s, rgb255, tf, args, rng)
            elif labels is None:
                labels, geo_px = segment_hybrid(rgb_s, rgb255, tf, args, rng)
        else:
            from skimage.feature import peak_local_max
            from skimage.segmentation import watershed
            elev = ndi.gaussian_filter(tf["energy"], 1.2)
            elev /= elev.max() + EPS
            coords = peak_local_max(-elev, min_distance=9, exclude_border=False,
                                    labels=np.ones(elev.shape, bool))
            markers = np.zeros(elev.shape, np.int32)
            markers[tuple(coords.T)] = np.arange(1, coords.shape[0] + 1)
            labels = watershed(elev, markers, compactness=0.0005).astype(np.int32)
        if args.labels_cache:
            np.save(args.labels_cache, labels)
    labels = merge_small_regions(labels, rgb_s, args.min_area,
                                 core_radius=max(0, (args.min_width - 1) // 2),
                                 protect_sat=args.protect_sat)
    # gradient-aware spatial merging: adjacent patches whose union is still one gradient become a
    # single (larger, smoother) region. Off -> labels untouched, so every downstream byte is unchanged.
    if getattr(args, "merge_grad", False):
        _t_mg = time.time()
        labels, _n_mg0, _n_mg1 = merge_gradient_regions(rgb_s, labels, tc, args)
        log(f"[{el()}] 梯度感知合并: {_n_mg0} → {_n_mg1} 个色块 "
            f"(-{_n_mg0 - _n_mg1}, {time.time() - _t_mg:.1f}s)")
    n_lab = int(labels.max()) + 1
    areas = np.bincount(labels.ravel(), minlength=n_lab)
    log(f"[{el()}] 分割: {n_lab} 个色块, 面积 {int(areas.min())}~{int(areas.max())} px")
    # shared partition-routing criteria (used by curve fitting / boundary snapping / anti-aliasing bands)
    geo_r, tex_r_all, rng_r_all, _t_norm = classify_regions(
        rgb_s, labels, areas, args.tex_sigma, args.tex_smooth, args.tex_norm,
        args.tex_thr, args.range_thr, args.min_area)
    if args.verbose:
        _keep = [int(li) for li in np.nonzero(areas >= args.min_area)[0]]
        _ngeo = sum(1 for li in _keep if geo_r[li])
        _head = ", ".join("#%d:%s(纹理%.1e/色跨%.3f)"
                          % (li, "几何" if geo_r[li] else "复杂",
                             tex_r_all[li], rng_r_all[li]) for li in _keep[:8])
        print("      · 分区路由(统一判据): %d/%d 个色块判为几何区%s"
              % (_ngeo, len(_keep), ("; 例: " + _head) if _head else ""))
    # energy threshold for subpixel edge snapping (the same threshold as the edge map)
    thr_ref = float(np.quantile(tf["energy"], args.edge_nms_hi))
    _REFINE_STATS.update(pts=0, moved=0, sum=0.0)

    # ---------------- background region ----------------
    border = np.zeros((H, W), bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    bl = np.bincount(labels[border].ravel(), minlength=n_lab)
    bg_label = int(np.argmax(bl))
    bg_mask = labels == bg_label
    bg_col = np.median(rgb_s[bg_mask], axis=0) if bg_mask.any() else np.ones(3)

    # ---------------- assembling regions ----------------
    # Each region is computed only inside a "bounding box + safety margin" window. Outside the window nothing affects the local operators here
    # (erosion/dilation/median/contour/pixel indexing all depend only on the region's surroundings), so the result is bit-identical to the full-image version,
    # while the complexity drops from O(H*W) per region to O(region area) per region.
    _objs = ndi.find_objects(labels + 1)   # index k corresponds to label k -- including label 0!
    # The margin must cover the adaptive cropping inside mask_to_paths: dilation 1 + pad 3 + mar(3*smooth+3);
    # if it is too small, the Gaussian smoothing sees the "window border" instead of the real all-zero background and the contour coordinates are no longer bit-identical.
    _MG = 8 + int(3 * args.contour_smooth) + 4

    def _win_of(li):
        sl = _objs[li] if 0 <= li < len(_objs) else None
        if sl is None:
            return (slice(0, 0), slice(0, 0)), 0, 0
        y0 = max(0, sl[0].start - _MG)
        y1 = min(H, sl[0].stop + _MG)
        x0 = max(0, sl[1].start - _MG)
        x1 = min(W, sl[1].stop + _MG)
        return (slice(y0, y1), slice(x0, x1)), y0, x0

    def _recon(r):
        """Window mask → full-image-coordinate mask (only called where a full-image tensor is genuinely needed)."""
        win = r.get("win")
        if win is None:
            return r["mask"]
        full = np.zeros((H, W), bool)
        full[win] = r["mask"]
        return full

    # The rule-dense sub-steps (mask erosion / per-region median color) are batch-computed once over the whole image, preferring the GPU.
    # Byte-identical to the per-region version (see the derivation in batch_bookkeeping), and it falls back automatically when the image was resampled.
    _inner_all = _cols_bk = None
    _bk_backend = ""
    if getattr(args, "batch_book", "auto") != "off":
        _t_bk = time.time()
        _inner_all, _cols_bk, _bk_backend = batch_bookkeeping(labels, rgb_s,
                                                             use_gpu=gpu.enabled(args))
        if _inner_all is None:
            log(f"[{el()}] 批量簿记: 不可用(像素不在 1/255 网格 / 标签过多), 回退逐区域")
        else:
            log(f"[{el()}] 批量簿记: {len(areas)} 个标签的掩码+中位色一次算完 "
                f"({time.time() - _t_bk:.1f}s, {_bk_backend})")

    regions = []
    _order = [int(x) for x in np.argsort(-areas)]
    _jobs = _n_jobs(args, len(_order))
    if _jobs > 1:
        # fork: the large arrays labels/rgb_s/tc/tf are shared by copy-on-write, neither copied nor pickled;
        # imap returns in the original order, so the order of regions is exactly that of the serial version.
        import multiprocessing
        _PAR_CTX.clear()
        _PAR_CTX.update(dict(args=args, labels=labels, areas=areas, rgb_s=rgb_s, tc=tc,
                             tf=tf, thr_ref=thr_ref, bg_label=bg_label, geo_r=geo_r,
                             H=H, W=W, MG=_MG, objs=_objs,
                             inner_all=_inner_all, cols=_cols_bk))
        _t_par = time.time()
        log(f"[{el()}] 逐区域阶段: {len(_order)} 个候选区域, {_jobs} 进程并行")
        with multiprocessing.get_context("fork").Pool(_jobs) as _pool:
            for _r, _msg, _st in _pool.imap(_par_region_one, _order, chunksize=4):
                if _r is not None:
                    regions.append(_r)
                if _msg:
                    log(_msg)
                _REFINE_STATS["pts"] += _st["pts"]
                _REFINE_STATS["moved"] += _st["moved"]
                _REFINE_STATS["sum"] += _st["sum"]
        log(f"      · 并行组装 {len(regions)} 个区域用了 {time.time() - _t_par:.1f}s")
    else:
        _t_ser = time.time()
        for li in np.argsort(-areas):
            li = int(li)
            win, y0, x0 = _win_of(li)
            mask = labels[win] == li
            if int(areas[li]) < args.min_area:      # equals mask.sum(), but in O(1)
                continue
            if _inner_all is not None:
                inner = _inner_all[win] & mask    # same as above: restrict back to this region
                if int(inner.sum()) < 200:
                    inner = mask
                col = _cols_bk[li]
            else:
                inner = ndi.binary_erosion(mask, iterations=3)
                if inner.sum() < 200:
                    inner = mask
                col = np.median(rgb_s[win][inner], axis=0)
            if li == bg_label:
                regions.append({"idx": li, "mask": mask, "area": int(areas[li]), "d": "",
                                "fill": to_hex(col), "grad": None, "bg": True,
                                "geo": float(geo_r[li]), "win": win, "off": (y0, x0), "col": col})
                continue
            m = ndi.binary_dilation(mask, iterations=1)
            geo_frac = float(geo_r[li])
            use_fit = args.fit
            if use_fit == "auto":
                use_fit = "bezier" if geo_frac >= args.auto_geo_frac else "cr"
            rf = None
            if args.snap_mode == "refine" and geo_frac >= args.auto_geo_frac:
                rf = (tf, thr_ref, args.snap_shift, args.snap_step, args.snap_sub)
            d = region_path_d(m, args.contour_tol, args.contour_smooth, args.contour_min_area,
                              fit=use_fit, fit_tol=args.fit_tol, corner_deg=args.corner_deg,
                              refine=rf, offset=(y0, x0))
            if not d:
                continue
            # rgb_s/tc are passed as the full image and off only converts pixel indices to absolute coordinates -> values are bit-identical to the full-image version
            grad = fit_region_gradient(rgb_s, inner, tc, args.grad_stops,
                                       args.grad_min_range, off=(y0, x0),
                                       allow_radial=args.grad_radial,
                                       radial_margin=args.grad_radial_margin)
            regions.append({"idx": li, "mask": mask, "area": int(areas[li]), "d": d,
                            "fill": to_hex(col), "grad": grad, "bg": False,
                            "geo": geo_frac, "fit": use_fit, "win": win, "off": (y0, x0),
                            "col": col})
            if args.verbose:
                q = f"{grad['quality']:.2f}({grad['axis']})" if grad else "平色"
                log(f"      · 区域#{li:<2d} area={int(areas[li]):>7d} {to_hex(col)} "
                      f"渐变质量={q} 拟合={use_fit}(几何度{geo_frac:.2f})")

        log(f"      · 串行组装 {len(regions)} 个区域用了 {time.time() - _t_ser:.1f}s")

    # ---- error-driven adaptive refinement (see refine_regions) ----
    # Gated on the gradient feature stack as well, so that the legacy flat-colour route
    # (--no-grad-radial --no-merge-grad) still reproduces the old output byte for byte.
    if args.adaptive_refine and (args.grad_radial or args.merge_grad):
        _t_rf = time.time()
        _lab_ref = labels.copy()
        regions, _rst = refine_regions(rgb_s, tc, regions, args, tf, thr_ref, labels_out=_lab_ref)
        _extra = f", {len(regions) - _rst['n0']:+d}" if _rst["n1"] != _rst["n0"] else ""
        log(f"[{el()}] 自适应细化: {_rst['n0']} → {_rst['n1']} 个区域{_extra} "
            f"(二分 {_rst['split']} 次, 其中背景 {_rst['bg_split']}, 最大深度 {_rst['depth']}, "
            f"评估 {_rst['eval']}, 放弃 {_rst['rejected']}, 预算 {_rst['budget']}, "
            f"用时 {time.time() - _t_rf:.1f}s)")
        if args.merge_grad and args.refine_order == "refine-first":
            _t_rm = time.time()
            _mg, _m0, _m1 = merge_gradient_regions(rgb_s, _lab_ref, tc, args)
            if _m1 < _m0:
                regions = regroup_regions(rgb_s, regions, _lab_ref, _mg, args, tc, tf, thr_ref)
                _lab_ref = _mg
            log(f"[{el()}] 细化后梯度合并(refine-first): {_m0} → {_m1} 个色块, "
                f"重组为 {len(regions)} 个区域 ({time.time() - _t_rm:.1f}s)")
        labels = _lab_ref                      # keep AA bands / debug consistent with the refined ids
        if args.dump_refined_labels:
            np.save(args.dump_refined_labels, labels)

    # ---- anti-aliasing transition bands: hard edges -> the source image's 2~3px transition (aa_band_regions) ----
    if args.aa_levels > 0 and args.aa_width > 0:
        t_aa = time.time()
        aa_segs = aa_band_regions(rgb_s, regions, labels, args, tf, thr_ref)
        regions.extend(aa_segs)
        if aa_segs:
            print("[%s] 抗锯齿过渡带: %d 段 (每侧 %d 级, 总宽 %.1fpx), 覆盖 %d px, "
                  "跳过合成边界 %d 段, 跳过重复真实边缘 %d 段 | %.1fs"
                  % (el(), len(aa_segs), args.aa_levels, args.aa_width,
                     sum(a["area"] for a in aa_segs), _AA_STATS["skipped_syn"],
                     _AA_STATS["skipped_canon"], time.time() - t_aa))

    # ---- detail layer: colored thin seam lines (golden strokes) ----
    # k-means is very unfriendly to "thin and colored" narrow bands (they often get merged into the neighbouring large region), yet such seam lines
    # are exactly the key elements of a logo, so they are extracted separately as one vector layer using Lab chroma (drawn on top).
    if _REFINE_STATS["moved"] > 0 and args.verbose:
        _m = _REFINE_STATS["moved"]
        log(f"      · 亚像素边缘吸附: {_m}/{_REFINE_STATS['pts']} 个轮廓点吸到边缘上, "
              f"平均位移 {_REFINE_STATS['sum'] / _m:.3f}px (上限 {args.snap_shift}px)")
    if args.detail_chroma > 0:
        _t_det = time.time()
        lab = rgb2lab(np.clip(rgb_s, 0, 1))
        chroma = ndi.gaussian_filter(np.hypot(lab[..., 1], lab[..., 2]),
                                     args.detail_scale)
        dm = ndi.binary_closing(chroma > args.detail_chroma, np.ones((3, 3), bool))
        lb = measure.label(dm, connectivity=2)
        szs = np.bincount(lb.ravel(), minlength=1)
        _ks = [int(k) for k in np.nonzero(szs >= args.detail_min_area)[0] if int(k) != 0]
        _PAR_CTX.clear()
        _PAR_CTX.update(args=args, rgb_s=rgb_s, tc=tc, tf=tf, thr_ref=thr_ref, lb=lb,
                        H=H, W=W, MG=_MG, objs=ndi.find_objects(lb + 1))
        # Every component of the detail layer is heavy enough (whole-image filtering + contours), so the threshold is lowered to 2: the component count is often far below 24,
        # and the default threshold would degrade it to serial -- measured on the full image, those two stages were literally "only 1 of 16 workers busy".
        _jobs_det = _n_jobs(args, len(_ks), min_items=2)
        _det = []
        if _jobs_det > 1:
            import multiprocessing
            with multiprocessing.get_context("fork").Pool(_jobs_det) as _pool:
                for _r, _msg in _pool.imap(_par_detail_one, _ks, chunksize=4):
                    if _r is not None:
                        _det.append(_r)
                    if _msg:
                        log(_msg)
        else:
            for _k in _ks:
                _r, _msg = _par_detail_one(_k)
                if _r is not None:
                    _det.append(_r)
                if _msg:
                    log(_msg)
        regions.extend(_det)
        n_det = len(_det)
        log(f"[{el()}] 细节层(Lab 色度>{args.detail_chroma}): {n_det} 条彩色缝线 "
              f"(候选分量 {len(_ks)}, {_jobs_det} 进程, 用时 {time.time() - _t_det:.1f}s)")

    n_aa = sum(1 for r in regions if r.get("aa"))
    n_grad = sum(1 for r in regions if r["grad"] is not None)
    n_rad = sum(1 for r in regions if r["grad"] is not None and r["grad"].get("kind") == "radial")
    log(f"[{el()}] 区域矢量: {len(regions) - n_aa} 条路径 (另有抗锯齿带 {n_aa} 段), "
          f"其中 {n_grad} 个带渐变 (线性 {n_grad - n_rad} / 径向 {n_rad})")

    # ---- stroke activity: residual between the image and the per-region base fill (gradient/flat color), deciding where strokes are needed ----
    act = np.zeros((H, W), np.float32)
    _PAR_CTX.clear()
    _PAR_CTX.update(args=args, regions=regions, rgb_s=rgb_s, shape=rgb.shape)
    _jobs_act = _n_jobs(args, len(regions))
    if _jobs_act > 1:
        import multiprocessing
        with multiprocessing.get_context("fork").Pool(_jobs_act) as _pool:
            _it = _pool.imap(_par_act_one, range(len(regions)), chunksize=8)
            for _res in _it:
                if _res is not None:
                    _rows, _cols, _vals = _res
                    act[_rows, _cols] = _vals
    else:
        for _i in range(len(regions)):
            _res = _par_act_one(_i)
            if _res is not None:
                _rows, _cols, _vals = _res
                act[_rows, _cols] = _vals
    act = ndi.gaussian_filter(act, 2.5)
    if args.verbose and act.max() > 0:
        print(f"      活动度 residual: 均值={act[act>0].mean()*255:.2f}/255 "
              f"p99={np.percentile(act[act>0],99)*255:.2f}/255")

    # ---------------- strokes ----------------
    stroke_groups = []
    if not args.no_strokes:
        bg_mask2 = _recon(regions[0]) if regions and regions[0]["bg"] else bg_mask
        fg = ~bg_mask2
        fg = ndi.binary_erosion(fg, iterations=1)
        _sregs = sorted([r for r in regions if not r["bg"] and not r.get("aa")],
                        key=lambda z: -z["area"])
        _jobs_str = _n_jobs(args, len(_sregs))
        if _jobs_str > 1:
            # parallel: each child builds its own rng from (seed, index), so the result is scheduling-independent and reproducible;
            # but the jitter pattern differs from the serial version (shared rng) -- use --jobs 1 to reproduce out/ byte for byte.
            import multiprocessing
            _PAR_CTX.clear()
            _PAR_CTX.update(args=args, tc=tc, fg=fg, act=act, rgb_s=rgb_s, H=H, W=W,
                            sregs=_sregs, seed=int(args.seed or 0))
            with multiprocessing.get_context("fork").Pool(_jobs_str) as _pool:
                for _g in _pool.imap(_par_stroke_one, range(len(_sregs)), chunksize=2):
                    if _g is None:
                        continue
                    stroke_groups.append(_g)
                    if args.verbose:
                        log(f"      · 区域#{_g['idx']} (area={_g['area']}) "
                              f"→ {len(_g['strokes'])} 笔触")
        else:
            for r in sorted([r for r in regions if not r["bg"] and not r.get("aa")],
                            key=lambda z: -z["area"]):
                # The sub.sum()<400 test is computed inside the window first (exactly equivalent to the full image, and O(window));
                # only regions that really need streamlines pay the one-off O(H*W) restoration -- make_streamlines needs the whole-image
                # tensor field anyway, so the per-region O(H*W) cost is inherent to it.
                win = r.get("win")
                if win is None:
                    sub = r["mask"] & fg
                else:
                    sub_w = r["mask"] & fg[win]
                    if sub_w.sum() < 400:
                        continue
                    sub = np.zeros((H, W), bool)
                    sub[win] = sub_w
                if sub.sum() < 400:
                    continue
                segs = make_streamlines(tc, sub, args, rng)
                if not segs:
                    continue
                items = []
                for pts, wid, _c in segs:
                    iy = np.clip(pts[:, 1].astype(int), 0, H - 1)
                    ix = np.clip(pts[:, 0].astype(int), 0, W - 1)
                    a = float(act[iy, ix].mean())
                    alpha = args.stroke_alpha * min(1.0, a / (args.stroke_activity_ref + EPS))
                    if alpha < args.stroke_alpha_min:
                        continue  # the base fill is accurate enough -> no stroke (avoids creating artifacts in flat areas)
                    col = sample_color(rgb_s, pts, args.stroke_jitter, rng)
                    items.append((ribbon_d(pts, wid), col, min(alpha, 0.95)))
                stroke_groups.append({"idx": r["idx"], "d": r["d"], "strokes": items})
                if args.verbose:
                    log(f"      · 区域#{r['idx']} (area={r['area']}) → {len(items)} 笔触")
    n_strokes = sum(len(g["strokes"]) for g in stroke_groups)
    log(f"[{el()}] 笔触: {n_strokes} 条 (间距 {args.spacing}px)")

    # ---------------- edges / seam lines ----------------
    edges = []
    if args.edge_mode != "off":
        fgm = ~bg_mask
        em = np.zeros((H, W), bool)
        chroma = None
        if args.edge_mode in ("chroma", "both"):
            lab = rgb2lab(np.clip(rgb_s, 0, 1))
            chroma = np.hypot(lab[..., 1], lab[..., 2])
            em |= (ndi.gaussian_filter(chroma, 0.8) > args.edge_chroma) & fgm
        if args.edge_mode in ("energy", "both"):
            e = tf["energy"]
            vals = e[fgm]
            hi = float(np.quantile(vals, args.edge_hi)) if vals.size else 0.0
            lo = float(np.quantile(vals, args.edge_lo)) if vals.size else 0.0
            em |= (filters.apply_hysteresis_threshold(e, lo, hi) & fgm
                   & (tf["coh"] > args.edge_coh))
        lb = measure.label(em, connectivity=2)
        sizes = np.bincount(lb.ravel())
        keep = np.nonzero(sizes >= args.edge_min_area)[0]
        keep = keep[keep != 0]
        em = np.isin(lb, keep) if keep.size else np.zeros_like(em)
        from skimage import morphology   # lazy import: needed in this one place only, and importing at startup costs 2.6s
        skel = morphology.skeletonize(em)
        for p in skeleton_paths(skel, min_len=args.edge_min_len, tol=args.edge_tol):
            col = sample_color(rgb_s, p, 0.0, rng)
            iy = np.clip(p[:, 1].astype(int), 0, H - 1)
            ix = np.clip(p[:, 0].astype(int), 0, W - 1)
            if chroma is not None:
                ch = float(np.median(chroma[iy, ix]))
                wq = args.edge_width * (0.75 + 0.5 * min(1.0, max(0.0, ch / args.edge_chroma - 1.0)))
            else:
                wq = args.edge_width * (0.8 + 0.8 * min(1.0, float(np.median(tf["coh"][iy, ix]))))
            edges.append((polyline_to_bezier_d(p, closed=False), col, wq, args.edge_alpha))
        log(f"[{el()}] 边缘/缝线: {len(edges)} 条 ({args.edge_mode} 模式)")
    else:
        log(f"[{el()}] 边缘/缝线: 关闭 (色块层已包含金色缝线)")

    # ---------------- emit SVG ----------------
    meta = {"bg_fill": to_hex(bg_col), "src": os.path.basename(args.src),
            "desc": (f"structure-tensor vector tracing | fine(σd={args.sigma_d},σi={args.sigma_i}) "
                     f"coarse(σd={args.cs_sigma_d},σi={args.cs_sigma_i}) | regions={len(regions)} "
                     f"gradients={n_grad} strokes={n_strokes} edges={len(edges)}")}
    svg = build_svg(W0, H0, regions, stroke_groups, edges, meta, view_box=(W, H))
    sh_stats = None
    if args.shade_blobs:
        # Defaults that depend on the picture: a "smooth blob" is a fraction of the canvas, not a
        # fixed number of pixels, so the radius and the minimum area follow the native size.
        if args.shade_radius <= 0:
            args.shade_radius = max(48.0, 0.19 * max(W0, H0) * args.scale)
        if args.shade_min_area <= 0:
            args.shade_min_area = max(900, int(round((0.05 * max(W0, H0) * args.scale) ** 2)))
        _sh = build_shade_stack(rgb255, svg, W, H, args, log)
        if _sh is not None:
            svg = build_svg(W0, H0, regions, stroke_groups, edges, meta, view_box=(W, H),
                            shade=_sh[:2])
            sh_stats = _sh[2]
            log(f"[{el()}] 着色层: {sh_stats['n']} 层叠加径向渐变 "
                f"(平滑区 {sh_stats['area']}px, 盘面积/平滑区 {sh_stats['cov']:.1f}x, "
                f"底图 {sh_stats['psnr0']:.2f} → 加层后 {sh_stats['psnr']:.2f} dB, "
                f"用时 {sh_stats['secs']:.1f}s)")
    raw_kb = len(svg.encode()) / 1024
    if args.compress != "off":
        svg = slim_svg(svg, prec_contour=1,
                       prec_stroke=(0 if args.compress == "tight" else 1),
                       drop_clip=(args.compress == "tight"))
    ensure_dir(args.out)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(svg)
    out_kb = os.path.getsize(args.out) / 1024
    extra = "" if args.compress == "off" else f"，{args.compress} 瘦身 {raw_kb:.0f}→{out_kb:.0f} KB"
    log(f"[{el()}] 写出 {args.out} ({out_kb:.1f} KB{extra})")

    gz_path = ""
    if args.gzip:
        gz_path = (args.out[:-4] if args.out.lower().endswith(".svg") else args.out) + ".svgz"
        # mtime=0: makes the .svgz reproducible (otherwise the gzip header carries a build timestamp)
        with open(gz_path, "wb") as fh, gzip.GzipFile(
                fileobj=fh, mode="wb", compresslevel=9, mtime=0) as f:
            f.write(svg.encode())
        log(f"      · gzip {gz_path} ({os.path.getsize(gz_path)/1024:.1f} KB)")

    # ---------------- structure tensor debug image ----------------
    if args.debug:
        base = 0.42 * rgb_s + 0.58
        img = Image.fromarray((np.clip(base, 0, 1) * 255).astype(np.uint8))
        dr = ImageDraw.Draw(img)
        step = 14
        ys, xs = np.mgrid[step // 2:H:step, step // 2:W:step]
        ys, xs = ys.ravel(), xs.ravel()
        cohv = tc["coh"][ys, xs]
        k = cohv > 0.15
        ys, xs, cohv = ys[k], xs[k], cohv[k]
        L = 5.0
        for y, x, c in zip(ys.tolist(), xs.tolist(), cohv.tolist()):
            ang = math.atan2(tc["ty"][y, x], tc["tx"][y, x])
            dx, dy = math.cos(ang) * L, math.sin(ang) * L
            dr.line([x - dx, y - dy, x + dx, y + dy],
                    fill=(int(255 * c), int(90 * (1 - c)), int(255 * (1 - c))), width=1)
        by, bx = np.nonzero(find_boundaries(labels, mode="outer"))
        for y, x in zip(by[::2].tolist(), bx[::2].tolist()):
            img.putpixel((x, y), (0, 160, 0))
        img.save(args.debug)
        log(f"      · 调试图 {args.debug}: 线段方向=等照度线, 色相=相干性, 绿=色块边界")

    # ---------------- preview + fidelity ----------------
    psnr = None
    if args.preview:
        try:
            import cairosvg
            # Render back at **native size** before comparing: that way the self-reported PSNR and comparisons across --scale use the same basis
            cairosvg.svg2png(url=args.out, write_to=args.preview,
                             output_width=W0, output_height=H0)
            prev = np.asarray(Image.open(args.preview).convert("RGB")).astype(np.float64)
            ref = np.asarray(im0).astype(np.float64)
            if prev.shape == ref.shape:
                mae = float(np.abs(prev - ref).mean())
                mse = float(((prev - ref) ** 2).mean())
                psnr = 99.0 if mse <= 0 else 10 * math.log10(255.0 ** 2 / mse)
                log(f"      · 预览 {args.preview}: 渲回原生 {W0}x{H0}, MAE={mae:.2f}/255 PSNR={psnr:.2f} dB")
        except ImportError:
            print("      · 未安装 cairosvg, 跳过预览渲染")
        except Exception as exc:
            log(f"      · 预览渲染失败: {exc}")

    # ---------------- final summary ----------------
    log(f"[{el()}] 完成")
    if W != W0:
        grid = f"  →  网格 {W}x{H} (--scale {args.scale:g})"
    else:
        grid = ""
    print("  " + "-" * 62)
    print(f"  输入   {args.src}  {W0}x{H0}{grid}")
    print(f"  分区   {len(regions)} 区域 · {n_grad} 渐变 · {n_strokes} 笔触 · {len(edges)} 缝线")
    print(f"  矢量   {args.out}  {out_kb:.0f} KB" + (f"   ({psnr:.2f} dB)" if psnr else ""))
    if sh_stats:
        print(f"  着色   {sh_stats['n']} 层径向渐变 (平滑区 {sh_stats['area']}px, 覆盖 "
              f"{sh_stats['cov']:.1f}x)")
    if gz_path:
        print(f"  压缩   {gz_path}  {os.path.getsize(gz_path)/1024:.0f} KB")
    if args.preview and os.path.exists(args.preview):
        print(f"  预览   {args.preview}")
    if args.debug:
        print(f"  调试   {args.debug}")
    print("  " + "-" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
