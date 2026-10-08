"""GPU backend (optional): cupy implementations of the structure tensor and k-means;
falls back to the in-house CUDA backend when cupy is unavailable.

Design principles
-----------------
1. **Optional**: enabled only when a GPU backend is available and --gpu permits it;
   otherwise it falls back entirely to the numpy path.
   The structure tensor prefers cupy (separable filtering from cupyx.scipy.ndimage), and
   k-means prefers the **in-house CUDA kernel**
   -- on the same 4090 the latter measured faster: at pipeline scale 150k points 99 ms vs
   cupy 118 ms, over the full image 1.5M points 128 ms vs 163 ms (pure Lloyd body 43 ms vs
   92 ms); each serves as the other's fallback, and with neither it falls back to numpy.
   The same SVG_tracer.py therefore runs in both the SVG_tracer (pure CPU) and
   SVG_tracer-gpu (with cupy) environments, with no branching code.
2. **Arrays stay in device memory**: each stage does H2D / D2H only once at its start and
   end. On this machine the GPU link measured H2D 7.2 GB/s / D2H 6.0 GB/s in a quiet
   window (under heavy load the same link is an order of magnitude slower); "one transfer
   per step" would eat the entire gain back.
3. **float32 in device memory**: the 4090's FP64 is only 1/64 of FP32 (about 1.3 TFLOP/s),
   so float64 would turn this stage into a net loss; values are promoted back to float64
   on the way out, leaving downstream dtype and precision behavior unchanged.
4. **Failures are not fatal**: the caller handles try/except and falls back to CPU if the GPU misbehaves.

Measured (1254², single 4090, FP32, compared against scipy float32 at the same precision):
    σ=2.5 two-pass separable filter: 73.9 ms → 0.044 ms (about 1700×)
    σ=12.0                        : 210.2 ms → 0.175 ms (about 1200×)
    H2D/D2H quiet-window measured 7.2 / 6.0 GB/s (under heavy load we historically measured 6.3 MB one-way in 8.7 ms ≈ 0.7 GB/s)
"""
from __future__ import annotations

import os

import numpy as np

_CUPY = None
_TRIED = False
_CUDA = None
_CUDATRIED = False


def _cupy():
    """Lazily import cupy (cached); SVG_TRACER_GPU=0 forces it off."""
    global _CUPY, _TRIED
    if not _TRIED:
        _TRIED = True
        if os.environ.get("SVG_TRACER_GPU", "1") not in ("0", ""):
            try:
                import cupy
                cupy.zeros(1, dtype=cupy.float32)      # really initialize the context
                _CUPY = cupy
            except Exception:
                _CUPY = None
    return _CUPY


def _cuda():
    """In-house CUDA backend (cuda_backend.py, nvcc + ctypes); does not require cupy."""
    global _CUDA, _CUDATRIED
    if not _CUDATRIED:
        _CUDATRIED = True
        if os.environ.get("SVG_TRACER_GPU", "1") not in ("0", ""):
            try:
                import cuda_backend
                _CUDA = cuda_backend if cuda_backend.available() else None
            except Exception:
                _CUDA = None
    return _CUDA


def available() -> bool:
    """True when either cupy or the in-house CUDA backend is available."""
    return _cupy() is not None or _cuda() is not None


def has_kmeans() -> bool:
    """cupy has its own version; otherwise consult the **real** availability of the
    in-house CUDA backend's k-means (including its self-check).

    Honesty: in the in-house backend, k-means availability is decoupled from the structure
    tensor (cuda_backend.kmeans_available()). A failed structure-tensor self-check will not
    make k-means report itself available, and a failed k-means self-check will not switch
    the structure tensor off.
    """
    if _cupy() is not None:
        return True
    cd = _cuda()
    if cd is None:
        return False
    try:
        return bool(cd.kmeans_available())
    except Exception:
        return False


def enabled(args) -> bool:
    """Determined jointly by --gpu auto/on/off and cupy availability."""
    mode = getattr(args, "gpu", "auto")
    if mode == "off":
        return False
    if mode == "on" and not available():
        raise RuntimeError("--gpu on but cupy is unavailable (use the SVG_tracer-gpu environment, or set --gpu auto)")
    return available()


def device_name() -> str:
    cp = _cupy()
    if cp is None:
        cd = _cuda()
        return f"{cd.device_name()} (in-house CUDA)" if cd is not None else "none"
    try:
        dev = cp.cuda.Device()
        p = cp.cuda.runtime.getDeviceProperties(dev.id)
        nm = p["name"]
        return nm.decode() if isinstance(nm, bytes) else str(nm)
    except Exception:
        return "cupy"


def structure_tensor(rgb: np.ndarray, sigma_d: float, sigma_i: float,
                     color: bool = True, eps: float = 1e-12,
                     dtype=None) -> dict:
    """Mathematically equivalent to SVG_tracer.structure_tensor; returns float64.

    The cupy path keeps float32 (it is only a fallback nowadays). Note that it is not worse
    than float64 -- after switching to f64, logo actually dropped from 37.62 to 37.40 dB,
    which shows quality is **chaotically sensitive** to this tensor, not "the more accurate
    the better".
    """
    # Prefer the in-house CUDA kernel: it "transfers float32 + accumulates in double" and
    # measured almost bit-identical to scipy (watercolor scale1 differs by 1 byte only);
    # cupy's separable filtering rounds differently and drifts the downstream k-means/RAG
    # cascade -- watercolor measured 22.06 vs CPU 24.00 dB, and switching to float64 did not
    # save it either (37.40 vs f32's 37.62, showing quality is chaotically sensitive to this
    # tensor, not "the more accurate the better"). cupy is demoted to a fallback: for
    # machines without nvcc.
    try:
        _c = _cuda()
    except Exception:
        _c = None
    if _c is not None:
        return _c.structure_tensor(rgb, sigma_d, sigma_i, color=color, eps=eps)
    cp = _cupy()
    if cp is None:
        raise RuntimeError("structure tensor: neither the in-house CUDA backend (nvcc) nor cupy is available")
    from cupyx.scipy import ndimage as cndi

    cd = cp.float32 if dtype is not np.float64 else cp.float64
    x = cp.asarray(rgb, dtype=cd)
    chans = [x[..., i] for i in range(3)] if color else [x.mean(2)]
    shape = rgb.shape[:2]
    jxx = cp.zeros(shape, cd)
    jyy = cp.zeros(shape, cd)
    jxy = cp.zeros(shape, cd)
    for c in chans:
        gx = cndi.gaussian_filter(c, sigma_d, order=(0, 1), mode="nearest")
        gy = cndi.gaussian_filter(c, sigma_d, order=(1, 0), mode="nearest")
        jxx += cndi.gaussian_filter(gx * gx, sigma_i, mode="nearest")
        jyy += cndi.gaussian_filter(gy * gy, sigma_i, mode="nearest")
        jxy += cndi.gaussian_filter(gx * gy, sigma_i, mode="nearest")
    nch = float(len(chans))
    jxx /= nch
    jyy /= nch
    jxy /= nch

    tr = jxx + jyy
    dif = jxx - jyy
    tmp = cp.sqrt(dif * dif + 4.0 * jxy * jxy)
    l1 = 0.5 * (tr + tmp)
    l2 = 0.5 * (tr - tmp)

    # Principal eigenvector: of the two candidates take the one with the larger magnitude (same criterion as the CPU version)
    ax, ay = jxy, l1 - jxx
    bx, by = l1 - jyy, jxy
    use_b = cp.hypot(bx, by) > cp.hypot(ax, ay)
    vx = cp.where(use_b, bx, ax)
    vy = cp.where(use_b, by, ay)
    nrm = cp.hypot(vx, vy)
    nrm = cp.where(nrm < eps, cd(1.0), nrm)
    vx = vx / nrm
    vy = vy / nrm

    luma = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
    coh = (l1 - l2) / (l1 + l2 + eps)
    out = {"l1": l1, "l2": l2, "coh": coh, "energy": l1, "luma": luma,
           "gx": vx, "gy": vy, "tx": -vy, "ty": vx}
    host = {k: cp.asnumpy(v).astype(np.float64) for k, v in out.items()}
    del x, jxx, jyy, jxy, l1, l2, tr, dif, tmp, vx, vy, nrm, luma, coh, out
    cp.get_default_memory_pool().free_all_blocks()
    return host


def lloyd(sample: np.ndarray, feats: np.ndarray, cen: np.ndarray,
          k: int, iters: int):
    """Lloyd iterations + final full-image assignment, all running in device memory;
    returns the (cen, labels) host arrays.

    Consistent with the CPU version: centroids are fitted on the subsample, then the full
    image is assigned once at the end; distances use the gemm expansion.

    Backend choice: **prefer the in-house CUDA kernel** -- on the same machine it measured
    faster than cupy at both real scales (150k points 99 ms vs 118 ms; 1.5M points 128 ms vs
    163 ms), and its inertia relative difference against the CPU reference implementation is
    smaller (order 1e-12 vs cupy's 1e-10). When the in-house kernel is unavailable or raises,
    it falls back to cupy; only when both are unavailable does it raise, letting the caller
    (SVG_tracer._kmeans via try/except) fall back to CPU.
    """
    cd = _cuda()
    if cd is not None and callable(getattr(cd, "lloyd", None)):
        ok = True
        try:
            ok = bool(cd.kmeans_available())
        except Exception:
            ok = False
        if ok:
            try:
                return cd.lloyd(sample, feats, cen, k, iters)
            except Exception:
                pass                    # fall back to cupy if the in-house kernel misbehaves
    cp = _cupy()
    if cp is None:
        raise RuntimeError("the GPU backend for k-means is unavailable (neither cupy nor the in-house CUDA backend is available)")

    ss = cp.asarray(sample, dtype=cp.float32)
    cc = cp.asarray(cen, dtype=cp.float32)

    def assign(x):
        cn = (cc * cc).sum(1)
        return (cn[None, :] - 2.0 * (x @ cc.T)).argmin(1).astype(cp.int32)

    lab = assign(ss)
    for _ in range(int(iters)):
        cnt = cp.bincount(lab, minlength=k).astype(cp.float32)
        new = cp.empty_like(cc)
        for c in range(cc.shape[1]):
            new[:, c] = cp.bincount(lab, weights=ss[:, c], minlength=k) / cp.maximum(cnt, 1)
        cc = new
        lab = assign(ss)
    out = cp.asnumpy(assign(cp.asarray(feats, dtype=cp.float32))).astype(np.int32)
    cen_out = cp.asnumpy(cc).astype(np.float64)
    del ss, cc, lab
    cp.get_default_memory_pool().free_all_blocks()
    return cen_out, out
