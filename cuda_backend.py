"""In-house CUDA backend: separable Gaussian convolution for the structure tensor +
k-means (Lloyd) (pure ctypes + nvcc, no dependency on cupy/torch).

Design
------
* The GPU only does **separable convolution** and **products**; element-wise algebra such as
  the eigendecomposition stays on the host in numpy (the same formula as the CPU version).
* Each channel plane is transferred only once at each end (H2D/D2H): ``lt_st_begin`` -> per
  channel ``lt_st_channel`` (internally: two derivative passes + 3 products + 3 sigma_i
  smoothing passes + accumulation, all in device memory) -> ``lt_st_end`` fetches
  jxx/jyy/jxy in one go. This avoids "one transfer per step".
* Storage/transfer use float32, but the convolution's **weighted sums accumulate in double**
  (consistent with scipy: intermediates float32, accumulation float64). This pipeline is
  transfer/host bound (9 taps and 97 taps take the same time), so the cost of FP64 is not
  measurable; accumulating in float32 only would raise the direction error of gx/gy from
  ~1e-4 to ~1.2e-3. Everything is promoted to float64 before returning.
* k-means (``lloyd``) likewise "upload once, transfer nothing back inside the loop": the
  float32 features and the float64 initial centroids are each uploaded once, then iters
  rounds of "assign -> mean -> zero empty clusters -> reassign" run in device memory, and
  only the centroids and the full-image labels are transferred back once at the end.
  Distances are squared Euclidean, **accumulated in double**, with semantics aligned item by
  item with ``logo_trace._kmeans`` (including zeroing empty-cluster centroids and taking the
  smallest cluster id on ties).
* Device buffers are cached and reused by (W,H) / (N,F) capacity, so cudaMalloc is not called
  on every invocation.
* Missing nvcc / failed compilation / failed **structure tensor** self-check -> ``available()``
  returns False without raising; failure details are in ``last_error()`` and ``build_log()``
  (first 20 lines of the build log). k-means has an **independent** self-check and switch,
  ``kmeans_available()``: its failure only makes ``lloyd`` fall back to CPU and does not
  affect available().
* Boundary mode = clamp (equivalent to scipy ``mode="nearest"``).

Mathematically equivalent to ``logo_trace.structure_tensor`` (the same kernels and formulas),
with exactly the same keys:
    l1, l2, coh, energy(=l1), luma, gx, gy, tx, ty
"""
from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import threading

import numpy as np

# ----------------------------------------------------------------------
# paths
# ----------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_BUILD_DIR = os.path.join(_HERE, "build")
_SRC_PATH = os.path.join(_BUILD_DIR, "logotrace_cuda.cu")
_SO_PATH = os.path.join(_BUILD_DIR, "liblogotrace_cuda.so")
_NVCC = os.environ.get("LOGO_TRACE_NVCC", "/usr/local/cuda/bin/nvcc")
_ARCH = os.environ.get("LOGO_TRACE_CUDA_ARCH", "sm_89")

# ----------------------------------------------------------------------
# CUDA source (written to build/logotrace_cuda.cu, then compiled)
# ----------------------------------------------------------------------
_CUDA_SOURCE = r"""
// Self-developed separable Gaussian FIR (float32, clamp boundary) + structure tensor channel pipeline
//   lt_rows : along x (columns); kernel = kx
//   lt_cols : along y (rows); kernel = ky, optional accumulation
// The three kernels (sigma_d smoothing k0 / sigma_d first derivative k1 / sigma_i smoothing ks) live in __constant__,
// and the device buffers are cached and reused by capacity. Each channel is transferred only once at each end.
#include <cuda_runtime.h>
#include <stdio.h>

#define LT_MAXR 1023
#define LT_TAPS (2 * LT_MAXR + 1)
#define LT_NBUF 8

__constant__ float cK0[LT_TAPS];
__constant__ float cK1[LT_TAPS];
__constant__ float cKS[LT_TAPS];

__device__ __forceinline__ float lt_tap(int w, int i) {
    if (w == 0) return cK0[i];
    if (w == 1) return cK1[i];
    return cKS[i];
}

__global__ void lt_rows(const float* __restrict__ in, float* __restrict__ out,
                        int W, int H, int w, int R) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    const float* row = in + (size_t)y * W;
    // Double-precision accumulation: consistent with scipy (intermediates stored as float32, convolution sums in double),
    // and this pipeline is transfer/host bound (9 taps and 97 taps take the same time), so the extra cost of FP64 is unmeasurable.
    double s = 0.0;
    for (int i = -R; i <= R; ++i) {
        int xx = x + i;
        xx = xx < 0 ? 0 : (xx >= W ? W - 1 : xx);
        s += (double)lt_tap(w, i + R) * (double)row[xx];
    }
    out[(size_t)y * W + x] = (float)s;
}

__global__ void lt_cols(const float* __restrict__ in, float* __restrict__ out,
                        int W, int H, int w, int R, int accum) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    double s = 0.0;
    for (int i = -R; i <= R; ++i) {
        int yy = y + i;
        yy = yy < 0 ? 0 : (yy >= H ? H - 1 : yy);
        s += (double)lt_tap(w, i + R) * (double)in[(size_t)yy * W + x];
    }
    size_t o = (size_t)y * W + x;
    out[o] = accum ? (float)((double)out[o] + s) : (float)s;
}

__global__ void lt_prod(const float* __restrict__ a, const float* __restrict__ b,
                        float* __restrict__ o, float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = a[i] * b[i] * scale;
}

static float* g_b[LT_NBUF] = {0, 0, 0, 0, 0, 0, 0, 0};
static size_t g_cap = 0;
static int g_ready = 0;
static char g_name[256] = "?";
static char g_err[512] = "";

static void set_err(const char* m) { snprintf(g_err, sizeof(g_err), "%s", m); }

extern "C" const char* lt_last_error(void) { return g_err; }
extern "C" const char* lt_device_name(void) { return g_name; }
extern "C" int lt_device_count(void) { int n = 0; cudaGetDeviceCount(&n); return n; }
extern "C" int lt_max_radius(void) { return LT_MAXR; }

extern "C" int lt_init(void) {
    g_err[0] = 0;
    if (g_ready) return 0;
    int nd = 0;
    cudaError_t e = cudaGetDeviceCount(&nd);
    if (e != cudaSuccess) { set_err(cudaGetErrorString(e)); return -1; }
    if (nd < 1) { set_err("no CUDA device"); return -1; }
    if ((e = cudaSetDevice(0)) != cudaSuccess) { set_err(cudaGetErrorString(e)); return -1; }
    cudaDeviceProp p;
    if ((e = cudaGetDeviceProperties(&p, 0)) != cudaSuccess) { set_err(cudaGetErrorString(e)); return -1; }
    snprintf(g_name, sizeof(g_name), "%s", p.name);
    g_ready = 1;
    return 0;
}

extern "C" void lt_free_buffers(void) {
    for (int i = 0; i < LT_NBUF; ++i) {
        if (g_b[i]) { cudaFree(g_b[i]); g_b[i] = 0; }
    }
    g_cap = 0;
}

// 8 device buffers reused by capacity:
//   lt_sep_conv : b0=in, b1=tmp
//   structure tensor: b0=channel, b1=convolution temp, b2=gx, b3=gy, b4=product, b5..b7=jxx/jyy/jxy
static int ensure_bufs(size_t n) {
    if (g_b[0] && n <= g_cap) return 0;
    lt_free_buffers();
    for (int i = 0; i < LT_NBUF; ++i) {
        cudaError_t e = cudaMalloc((void**)&g_b[i], n * sizeof(float));
        if (e != cudaSuccess) {
            set_err(cudaGetErrorString(e));
            lt_free_buffers();
            return -1;
        }
    }
    g_cap = n;
    return 0;
}

static void set_sym(int which, const float* k, int r) {
    if (which == 0) cudaMemcpyToSymbol(cK0, k, (size_t)(2 * r + 1) * sizeof(float));
    else if (which == 1) cudaMemcpyToSymbol(cK1, k, (size_t)(2 * r + 1) * sizeof(float));
    else cudaMemcpyToSymbol(cKS, k, (size_t)(2 * r + 1) * sizeof(float));
}

// ---------- one standalone separable convolution: 1 H2D + rows + cols + 1 D2H ----------
extern "C" int lt_sep_conv(const void* h_in, void* h_out, int W, int H,
                           const float* kx, int rx,
                           const float* ky, int ry) {
    g_err[0] = 0;
    if (!g_ready) { set_err("lt_init() not called"); return -1; }
    if (W <= 0 || H <= 0) { set_err("bad size"); return -2; }
    if (rx < 0 || ry < 0 || rx > LT_MAXR || ry > LT_MAXR) { set_err("radius out of range"); return -2; }
    if (rx == 0 && ry == 0) { set_err("both radii are zero"); return -2; }

    size_t n = (size_t)W * (size_t)H;
    if (ensure_bufs(n) != 0) return -3;

    set_sym(0, kx, rx);
    set_sym(1, ky, ry);
    cudaError_t e = cudaMemcpy(g_b[0], h_in, n * sizeof(float), cudaMemcpyHostToDevice);
    if (e != cudaSuccess) { set_err(cudaGetErrorString(e)); return -5; }

    dim3 th(32, 8);
    dim3 bl((W + 31) / 32, (H + 7) / 8);
    float* src = g_b[0];
    float* dst = g_b[1];
    if (rx > 0) { lt_rows<<<bl, th>>>((const float*)src, dst, W, H, 0, rx); float* t = src; src = dst; dst = t; }
    if (ry > 0) { lt_cols<<<bl, th>>>((const float*)src, dst, W, H, 1, ry, 0); float* t = src; src = dst; dst = t; }
    e = cudaGetLastError();
    if (e != cudaSuccess) { set_err(cudaGetErrorString(e)); return -6; }
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { set_err(cudaGetErrorString(e)); return -6; }
    e = cudaMemcpy(h_out, src, n * sizeof(float), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) { set_err(cudaGetErrorString(e)); return -7; }
    return 0;
}

// ---------- structure tensor channel pipeline ----------
// begin: clear the jxx/jyy/jxy accumulators
extern "C" int lt_st_begin(int W, int H) {
    g_err[0] = 0;
    if (!g_ready) { set_err("lt_init() not called"); return -1; }
    if (W <= 0 || H <= 0) { set_err("bad size"); return -2; }
    size_t n = (size_t)W * (size_t)H;
    if (n >= (size_t)2147483647) { set_err("image too large"); return -2; }
    if (ensure_bufs(n) != 0) return -3;
    for (int i = 5; i <= 7; ++i) {
        cudaError_t e = cudaMemset(g_b[i], 0, n * sizeof(float));
        if (e != cudaSuccess) { set_err(cudaGetErrorString(e)); return -4; }
    }
    return 0;
}

// one channel plane: a single H2D, after which gx/gy/3 products/sigma_i smoothing/accumulation all stay in device memory
//   k0/r0: sigma_d smoothing kernel; k1/r1: sigma_d first-derivative kernel; ks/rs: sigma_i smoothing kernel
//   scale : 1 / number of channels
extern "C" int lt_st_channel(const void* h_in, int W, int H,
                             const float* k0, int r0,
                             const float* k1, int r1,
                             const float* ks, int rs,
                             float scale) {
    g_err[0] = 0;
    if (!g_ready) { set_err("lt_init() not called"); return -1; }
    if (r0 < 1 || r1 < 1 || rs < 1) { set_err("radius must be >= 1"); return -2; }
    if (r0 > LT_MAXR || r1 > LT_MAXR || rs > LT_MAXR) { set_err("radius out of range"); return -2; }
    size_t n = (size_t)W * (size_t)H;
    if (ensure_bufs(n) != 0) return -3;

    set_sym(0, k0, r0);
    set_sym(1, k1, r1);
    set_sym(2, ks, rs);
    cudaError_t e = cudaMemcpy(g_b[0], h_in, n * sizeof(float), cudaMemcpyHostToDevice);
    if (e != cudaSuccess) { set_err(cudaGetErrorString(e)); return -5; }

    dim3 th(32, 8);
    dim3 bl((W + 31) / 32, (H + 7) / 8);
    int nb = (int)((n + 255) / 256);

    // gx = rows(x, k1) -> cols(y, k0)   [order=(0,1)]
    lt_rows<<<bl, th>>>(g_b[0], g_b[1], W, H, 1, r1);
    lt_cols<<<bl, th>>>(g_b[1], g_b[2], W, H, 0, r0, 0);
    // gy = rows(x, k0) -> cols(y, k1)   [order=(1,0)]
    lt_rows<<<bl, th>>>(g_b[0], g_b[1], W, H, 0, r0);
    lt_cols<<<bl, th>>>(g_b[1], g_b[3], W, H, 1, r1, 0);

    const float* A[3] = {g_b[2], g_b[3], g_b[2]};
    const float* B[3] = {g_b[2], g_b[3], g_b[3]};
    float* J[3] = {g_b[5], g_b[6], g_b[7]};
    for (int t = 0; t < 3; ++t) {
        lt_prod<<<nb, 256>>>(A[t], B[t], g_b[4], scale, (int)n);
        lt_rows<<<bl, th>>>(g_b[4], g_b[1], W, H, 2, rs);
        lt_cols<<<bl, th>>>(g_b[1], J[t], W, H, 2, rs, 1);
    }
    e = cudaGetLastError();
    if (e != cudaSuccess) { set_err(cudaGetErrorString(e)); return -6; }
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { set_err(cudaGetErrorString(e)); return -6; }
    return 0;
}

// end: fetch jxx/jyy/jxy (each (H,W) float32)
extern "C" int lt_st_end(void* h_jxx, void* h_jyy, void* h_jxy, int W, int H) {
    g_err[0] = 0;
    if (!g_ready) { set_err("lt_init() not called"); return -1; }
    size_t n = (size_t)W * (size_t)H;
    cudaError_t e = cudaMemcpy(h_jxx, g_b[5], n * sizeof(float), cudaMemcpyDeviceToHost);
    if (e == cudaSuccess) e = cudaMemcpy(h_jyy, g_b[6], n * sizeof(float), cudaMemcpyDeviceToHost);
    if (e == cudaSuccess) e = cudaMemcpy(h_jxy, g_b[7], n * sizeof(float), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) { set_err(cudaGetErrorString(e)); return -7; }
    return 0;
}

// ======================================================================
// k-means (Lloyd) -- features and initial centroids are uploaded only once, after which
//   "assign(sample) -> accumulate means -> zero empty clusters -> reassign" runs for iters rounds entirely in device memory,
//   with no transfer back to the host per round; only the centroids + full-image labels are transferred back once at the end.
// aligned step by step with the CPU reference in logo_trace._kmeans:
//   * the initial centroids are passed in from the host (k-means++ runs on the host); the device does not resample and has no randomness
//   * before entering the loop, sample is assigned once (corresponding to the CPU's lab_sub = assign(sample))
//   * per-round order: accumulate statistics with the previous round's labels -> compute means -> reassign
//   * empty cluster (cnt==0): the whole centroid row becomes 0 (corresponding to the CPU's new = np.zeros_like(cen)), not keeping the old value
//   * distance = squared Euclidean; accumulated term by term in double and compared pointwise in double (float32 accumulation is not accurate enough)
//   * on ties take the smallest cluster id (consistent with np.argmin)
//   * at the end, reassign over the **whole image** once before returning (corresponding to the CPU's return cen, assign(feats))
// ======================================================================
#define LT_KM_MAXKF 4096   // upper bound on dynamic shared memory: assign needs k*f and update needs k*(f+1) doubles

static float*  g_km_feat = 0;   // ns*F
static float*  g_km_full = 0;   // nf*F
static int*    g_km_lab  = 0;   // max(ns,nf)
static double* g_km_cen  = 0;   // k*F (always double: the CPU centroids are float64 too)
static double* g_km_sum  = 0;   // k*F
static double* g_km_cnt  = 0;   // k
static size_t  g_km_cf = 0, g_km_cff = 0, g_km_cl = 0;

static void km_free_all(void) {
    if (g_km_feat) { cudaFree(g_km_feat); g_km_feat = 0; }
    if (g_km_full) { cudaFree(g_km_full); g_km_full = 0; }
    if (g_km_lab)  { cudaFree(g_km_lab);  g_km_lab = 0; }
    if (g_km_cen)  { cudaFree(g_km_cen);  g_km_cen = 0; }
    if (g_km_sum)  { cudaFree(g_km_sum);  g_km_sum = 0; }
    if (g_km_cnt)  { cudaFree(g_km_cnt);  g_km_cnt = 0; }
    g_km_cf = g_km_cff = g_km_cl = 0;
}

extern "C" void lt_km_free(void) { km_free_all(); }
extern "C" int lt_km_maxkf(void) { return LT_KM_MAXKF; }

// Mean: each block first accumulates a local k*(f+1) accumulator in shared memory (intra-block atomic adds),
// then merges it into the global one at the end of the block (k*(f+1) global atomics per block), avoiding a global atomic per point.
__global__ void lt_km_update(const float* __restrict__ feat, const int* __restrict__ lab,
                             double* __restrict__ sum, double* __restrict__ cnt,
                             int n, int f, int k) {
    extern __shared__ double sm[];      // [0,k*f) per-cluster feature sums; [k*f,k*f+k) per-cluster counts
    int kf = k * f;
    for (int i = threadIdx.x; i < kf + k; i += blockDim.x) sm[i] = 0.0;
    __syncthreads();
    int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        int c = lab[i];
        if (c < 0 || c >= k) continue;
        const float* p = feat + (size_t)i * f;
        double* s = sm + c * f;
        for (int j = 0; j < f; ++j) atomicAdd(s + j, (double)p[j]);
        atomicAdd(sm + kf + c, 1.0);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < kf + k; i += blockDim.x) {
        double v = sm[i];
        if (v != 0.0) atomicAdd((i < kf) ? (sum + i) : (cnt + (i - kf)), v);
    }
}

__global__ void lt_km_center(const double* __restrict__ sum, const double* __restrict__ cnt,
                             double* __restrict__ cen, int k, int f) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k * f) return;
    double c = cnt[i / f];
    cen[i] = (c > 0.0) ? (sum[i] / c) : 0.0;   // empty cluster -> 0 (same as the CPU's zeros_like)
}

__global__ void lt_km_assign(const float* __restrict__ feat, const double* __restrict__ cen,
                             int* __restrict__ lab, int n, int f, int k) {
    extern __shared__ double sh[];      // k*f: centroids copied to shared memory and reused point by point
    int kf = k * f;
    for (int i = threadIdx.x; i < kf; i += blockDim.x) sh[i] = cen[i];
    __syncthreads();
    int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        const float* p = feat + (size_t)i * f;
        double best = 0.0;
        int bc = 0;
        for (int c = 0; c < k; ++c) {
            const double* q = sh + c * f;
            double d = 0.0;
            for (int j = 0; j < f; ++j) { double t = (double)p[j] - q[j]; d += t * t; }
            if (c == 0 || d < best) { best = d; bc = c; }   // strict <: on ties take the smallest cluster id
        }
        lab[i] = bc;
    }
}

static int km_ensure(size_t nf_feat, size_t nf_full, size_t nlab, size_t kf) {
    cudaError_t e;
    if (nf_feat > g_km_cf) {
        if (g_km_feat) { cudaFree(g_km_feat); g_km_feat = 0; }
        g_km_cf = 0;
        e = cudaMalloc((void**)&g_km_feat, nf_feat * sizeof(float));
        if (e != cudaSuccess) { set_err(cudaGetErrorString(e)); km_free_all(); return -1; }
        g_km_cf = nf_feat;
    }
    if (nf_full > g_km_cff) {
        if (g_km_full) { cudaFree(g_km_full); g_km_full = 0; }
        g_km_cff = 0;
        e = cudaMalloc((void**)&g_km_full, nf_full * sizeof(float));
        if (e != cudaSuccess) { set_err(cudaGetErrorString(e)); km_free_all(); return -1; }
        g_km_cff = nf_full;
    }
    if (nlab > g_km_cl) {
        if (g_km_lab) { cudaFree(g_km_lab); g_km_lab = 0; }
        g_km_cl = 0;
        e = cudaMalloc((void**)&g_km_lab, nlab * sizeof(int));
        if (e != cudaSuccess) { set_err(cudaGetErrorString(e)); km_free_all(); return -1; }
        g_km_cl = nlab;
    }
    if (g_km_cen == 0) {
        e = cudaMalloc((void**)&g_km_cen, LT_KM_MAXKF * sizeof(double));
        if (e != cudaSuccess) { set_err(cudaGetErrorString(e)); km_free_all(); return -1; }
        e = cudaMalloc((void**)&g_km_sum, LT_KM_MAXKF * sizeof(double));
        if (e != cudaSuccess) { set_err(cudaGetErrorString(e)); km_free_all(); return -1; }
        e = cudaMalloc((void**)&g_km_cnt, LT_KM_MAXKF * sizeof(double));
        if (e != cudaSuccess) { set_err(cudaGetErrorString(e)); km_free_all(); return -1; }
    }
    if (kf > LT_KM_MAXKF) { set_err("k*f exceeds shared limit"); return -2; }
    return 0;
}

// One full Lloyd run: returns (k,F) float64 centroids + nf int32 full-image labels.
extern "C" int lt_km_lloyd(const void* h_sample, int ns,
                           const void* h_feats, int nf,
                           const void* h_cen, int k, int f, int iters,
                           void* h_cen_out, void* h_lab_out) {
    g_err[0] = 0;
    if (!g_ready) { set_err("lt_init() not called"); return -1; }
    if (ns <= 0 || nf <= 0 || k <= 0 || f <= 0 || iters < 0) { set_err("bad k-means args"); return -2; }
    if ((size_t)k * (size_t)(f + 1) > (size_t)LT_KM_MAXKF) {
        set_err("k*(f+1) too large for shared memory");
        return -2;
    }
    size_t kf = (size_t)k * (size_t)f;
    size_t nsf = (size_t)ns * (size_t)f;
    size_t nff = (size_t)nf * (size_t)f;
    size_t nlab = (size_t)(ns > nf ? ns : nf);
    if (km_ensure(nsf, nff, nlab, kf) != 0) return -3;

    cudaError_t e = cudaMemcpy(g_km_feat, h_sample, nsf * sizeof(float), cudaMemcpyHostToDevice);
    if (e == cudaSuccess) e = cudaMemcpy(g_km_cen, h_cen, kf * sizeof(double), cudaMemcpyHostToDevice);
    if (e != cudaSuccess) { set_err(cudaGetErrorString(e)); return -5; }

    int ab = (ns + 255) / 256; if (ab > 8192) ab = 8192; if (ab < 1) ab = 1;
    int fb = (nf + 255) / 256; if (fb > 8192) fb = 8192; if (fb < 1) fb = 1;
    int ub = (ns + 255) / 256; if (ub > 2048) ub = 2048; if (ub < 1) ub = 1;
    int cb = (int)((kf + 255) / 256); if (cb < 1) cb = 1;
    size_t sha = kf * sizeof(double);
    size_t shu = (kf + (size_t)k) * sizeof(double);

    lt_km_assign<<<ab, 256, sha>>>(g_km_feat, g_km_cen, g_km_lab, ns, f, k);   // assign once before the loop
    for (int it = 0; it < iters; ++it) {
        e = cudaMemset(g_km_sum, 0, sha);
        if (e == cudaSuccess) e = cudaMemset(g_km_cnt, 0, (size_t)k * sizeof(double));
        if (e != cudaSuccess) { set_err(cudaGetErrorString(e)); return -4; }
        lt_km_update<<<ub, 256, shu>>>(g_km_feat, g_km_lab, g_km_sum, g_km_cnt, ns, f, k);
        lt_km_center<<<cb, 256>>>(g_km_sum, g_km_cnt, g_km_cen, k, f);
        lt_km_assign<<<ab, 256, sha>>>(g_km_feat, g_km_cen, g_km_lab, ns, f, k);
    }
    e = cudaMemcpy(g_km_full, h_feats, nff * sizeof(float), cudaMemcpyHostToDevice);
    if (e != cudaSuccess) { set_err(cudaGetErrorString(e)); return -5; }
    lt_km_assign<<<fb, 256, sha>>>(g_km_full, g_km_cen, g_km_lab, nf, f, k);   // final full-image assignment

    e = cudaGetLastError();
    if (e != cudaSuccess) { set_err(cudaGetErrorString(e)); return -6; }
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { set_err(cudaGetErrorString(e)); return -6; }
    e = cudaMemcpy(h_cen_out, g_km_cen, kf * sizeof(double), cudaMemcpyDeviceToHost);
    if (e == cudaSuccess) e = cudaMemcpy(h_lab_out, g_km_lab, (size_t)nf * sizeof(int), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) { set_err(cudaGetErrorString(e)); return -7; }
    return 0;
}
"""

# ----------------------------------------------------------------------
# Gaussian kernels (identical to scipy.ndimage._gaussian_kernel1d; clamp boundary)
# ----------------------------------------------------------------------
def _radius(sigma) -> int:
    """Radius for scipy's default truncate=4.0."""
    return int(4.0 * float(sigma) + 0.5)


def _kernel(sigma, order: int) -> np.ndarray:
    """order=0: smoothing; order=1: first derivative (correlation kernel, numerically identical to scipy)."""
    r = _radius(sigma)
    x = np.arange(-r, r + 1, dtype=np.float64)
    phi = np.exp(-0.5 * x * x / (float(sigma) * float(sigma)))
    phi /= phi.sum()
    if order == 0:
        return phi
    if order == 1:
        return (-x / (float(sigma) * float(sigma))) * phi
    raise ValueError("order 只支持 0 或 1, 得到 %r" % (order,))


# ----------------------------------------------------------------------
# Host reference implementation (float64, clamp): used by the self-check; also the fallback path when sigma degenerates (radius 0)
# ----------------------------------------------------------------------
def _np_sep_conv(a: np.ndarray, kx: np.ndarray, ky: np.ndarray) -> np.ndarray:
    """Separable correlation: apply kx along x, then ky along y; clamp boundary (equivalent to scipy mode='nearest')."""
    out = np.asarray(a, dtype=np.float64)
    rx = (len(kx) - 1) // 2
    ry = (len(ky) - 1) // 2
    if rx >= 1:
        p = np.pad(out, ((0, 0), (rx, rx)), mode="edge")
        acc = np.zeros_like(out)
        for i, v in enumerate(kx):
            acc += v * p[:, i:i + out.shape[1]]
        out = acc
    else:
        out = out * float(kx[0])
    if ry >= 1:
        p = np.pad(out, ((ry, ry), (0, 0)), mode="edge")
        acc = np.zeros_like(out)
        for i, v in enumerate(ky):
            acc += v * p[i:i + out.shape[0], :]
        out = acc
    else:
        out = out * float(ky[0])
    return out


def _np_tensor_planes(rgb: np.ndarray, sigma_d, sigma_i, color: bool):
    """Host float64 reference implementation of (jxx, jyy, jxy) (same formula as the CPU version)."""
    a = np.asarray(rgb, dtype=np.float64)
    chans = [a[..., 0], a[..., 1], a[..., 2]] if color else [a.mean(axis=2)]
    kd0 = _kernel(sigma_d, 0)
    kd1 = _kernel(sigma_d, 1)
    ki = _kernel(sigma_i, 0)
    H, W = a.shape[:2]
    jxx = np.zeros((H, W))
    jyy = np.zeros((H, W))
    jxy = np.zeros((H, W))
    for c in chans:
        gx = _np_sep_conv(c, kd1, kd0)
        gy = _np_sep_conv(c, kd0, kd1)
        jxx += _np_sep_conv(gx * gx, ki, ki)
        jyy += _np_sep_conv(gy * gy, ki, ki)
        jxy += _np_sep_conv(gx * gy, ki, ki)
    n_ch = float(len(chans))
    return jxx / n_ch, jyy / n_ch, jxy / n_ch


# ----------------------------------------------------------------------
# lazy build + load
# ----------------------------------------------------------------------
_LOCK = threading.Lock()
_LIB = None
_STATE = {"tried": False, "ok": False, "err": "", "log": [], "cmd": ""}
# k-means uses its **own** self-check state: its self-check failure must not affect the structure tensor (available() is decided by the structure tensor alone)
_KM_STATE = {"tried": False, "ok": False, "err": ""}

_c_void_p = ctypes.c_void_p


def _find_nvcc():
    if os.path.sep in _NVCC:
        return _NVCC if (os.path.isfile(_NVCC) and os.access(_NVCC, os.X_OK)) else None
    return shutil.which(_NVCC)


def _write_source():
    """Only rewrite the source on disk when its content changed, so that each import does not refresh the mtime and trigger a rebuild."""
    os.makedirs(_BUILD_DIR, exist_ok=True)
    try:
        with open(_SRC_PATH, "r", encoding="utf-8") as f:
            old = f.read()
    except OSError:
        old = None
    if old != _CUDA_SOURCE:
        with open(_SRC_PATH, "w", encoding="utf-8") as f:
            f.write(_CUDA_SOURCE)


def _so_is_fresh() -> bool:
    try:
        return os.path.getmtime(_SO_PATH) >= os.path.getmtime(_SRC_PATH)
    except OSError:
        return False


def _compile(nvcc) -> None:
    cmd = [nvcc, "-O3", "-arch=" + _ARCH, "-shared", "-Xcompiler", "-fPIC",
           "-o", _SO_PATH, _SRC_PATH]
    _STATE["cmd"] = " ".join(cmd)
    pr = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if pr.returncode != 0:
        _STATE["log"] = ((pr.stderr or "") + (pr.stdout or "")).splitlines()[:20]
        raise RuntimeError("nvcc 编译失败 rc=%d" % pr.returncode)


def _bind(lib) -> None:
    lib.lt_init.argtypes = []
    lib.lt_init.restype = ctypes.c_int
    lib.lt_device_name.argtypes = []
    lib.lt_device_name.restype = ctypes.c_char_p
    lib.lt_device_count.argtypes = []
    lib.lt_device_count.restype = ctypes.c_int
    lib.lt_last_error.argtypes = []
    lib.lt_last_error.restype = ctypes.c_char_p
    lib.lt_max_radius.argtypes = []
    lib.lt_max_radius.restype = ctypes.c_int
    lib.lt_free_buffers.argtypes = []
    lib.lt_free_buffers.restype = None
    lib.lt_sep_conv.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
    lib.lt_sep_conv.restype = ctypes.c_int
    lib.lt_st_begin.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.lt_st_begin.restype = ctypes.c_int
    lib.lt_st_channel.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                  ctypes.c_void_p, ctypes.c_int,
                                  ctypes.c_void_p, ctypes.c_int,
                                  ctypes.c_void_p, ctypes.c_int, ctypes.c_float]
    lib.lt_st_channel.restype = ctypes.c_int
    lib.lt_st_end.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                              ctypes.c_int, ctypes.c_int]
    lib.lt_st_end.restype = ctypes.c_int
    lib.lt_km_free.argtypes = []
    lib.lt_km_free.restype = None
    lib.lt_km_maxkf.argtypes = []
    lib.lt_km_maxkf.restype = ctypes.c_int
    lib.lt_km_lloyd.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
                                ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                ctypes.c_void_p, ctypes.c_void_p]
    lib.lt_km_lloyd.restype = ctypes.c_int


def _last_error(lib) -> str:
    try:
        m = lib.lt_last_error()
        return m.decode("utf-8", "replace") if m else "?"
    except Exception:                                            # noqa: BLE001
        return "?"


def _np_lloyd_ref(sample, feats, cen, k, iters):
    """The Lloyd body of ``logo_trace._kmeans`` (float64), used only for the comparison
    self-check.

    Line by line it mirrors the CPU version's assign / bincount / zeros_like; the only
    difference is that the CPU version draws its k-means++ initialization from an rng, while
    here the initial centroids are supplied by the caller (the device side does not resample
    either).
    """
    cen = np.array(cen, dtype=np.float64, copy=True)

    def assign(f):
        """Expand (x-c)^2 into gemm form (same formulation as the CPU version)."""
        out = np.empty(len(f), np.int32)
        cn = (cen * cen).sum(1)
        for i in range(0, len(f), 200000):
            blk = f[i:i + 200000]
            out[i:i + 200000] = (cn[None, :] - 2.0 * (blk @ cen.T)).argmin(1)
        return out

    lab = assign(sample)
    for _ in range(iters):
        cnt = np.bincount(lab, minlength=k).astype(np.float64)
        new = np.zeros_like(cen)
        for c in range(cen.shape[1]):
            new[:, c] = np.bincount(lab, weights=sample[:, c], minlength=k) / np.maximum(cnt, 1)
        cen = new
        lab = assign(sample)
    return cen, assign(feats)


def _self_test(lib) -> None:
    """End-to-end self-check (small random image): both sep_conv and the structure-tensor pipeline must match the host numpy reference."""
    rng = np.random.default_rng(7)
    W, H = 11, 9
    sig_d, sig_i = 1.0, 2.0
    kd0 = _kernel(sig_d, 0)
    kd1 = _kernel(sig_d, 1)
    ks = _kernel(sig_i, 0)

    def pf(a):
        return np.ascontiguousarray(a, dtype=np.float32)

    # 1) single separable convolution: direction + clamp boundary
    a = rng.random((H, W))
    out = np.zeros((H, W), np.float32)
    rc = lib.lt_sep_conv(pf(a).ctypes.data_as(_c_void_p), out.ctypes.data_as(_c_void_p),
                         W, H, pf(kd1).ctypes.data_as(_c_void_p), (len(kd1) - 1) // 2,
                         pf(kd0).ctypes.data_as(_c_void_p), (len(kd0) - 1) // 2)
    if rc != 0:
        raise RuntimeError("lt_sep_conv rc=%d (%s)" % (rc, _last_error(lib)))
    exp = _np_sep_conv(a, kd1, kd0)
    if not np.allclose(out, exp, atol=1e-5, rtol=1e-5):
        raise RuntimeError("lt_sep_conv 自检不符 max|d|=%.3g" % np.abs(out - exp).max())

    # 2) structure-tensor pipeline: jxx/jyy/jxy (multi-channel Di Zenzo)
    img = rng.random((H, W, 3))
    jx, jy, jz = _np_tensor_planes(img, sig_d, sig_i, True)
    gx = np.empty((H, W), np.float32)
    gy = np.empty((H, W), np.float32)
    gz = np.empty((H, W), np.float32)
    rc = lib.lt_st_begin(W, H)
    if rc != 0:
        raise RuntimeError("lt_st_begin rc=%d (%s)" % (rc, _last_error(lib)))
    for i in range(3):
        rc = lib.lt_st_channel(pf(img[..., i]).ctypes.data_as(_c_void_p), W, H,
                               pf(kd0).ctypes.data_as(_c_void_p), (len(kd0) - 1) // 2,
                               pf(kd1).ctypes.data_as(_c_void_p), (len(kd1) - 1) // 2,
                               pf(ks).ctypes.data_as(_c_void_p), (len(ks) - 1) // 2,
                               ctypes.c_float(1.0 / 3.0))
        if rc != 0:
            raise RuntimeError("lt_st_channel rc=%d (%s)" % (rc, _last_error(lib)))
    rc = lib.lt_st_end(gx.ctypes.data_as(_c_void_p), gy.ctypes.data_as(_c_void_p),
                       gz.ctypes.data_as(_c_void_p), W, H)
    if rc != 0:
        raise RuntimeError("lt_st_end rc=%d (%s)" % (rc, _last_error(lib)))
    for nm, got, want in (("jxx", gx, jx), ("jyy", gy, jy), ("jxy", gz, jz)):
        d = np.abs(got - want).max()
        if d > 1e-5 * max(1.0, np.abs(want).max()):
            raise RuntimeError("结构张量自检不符 %s max|d|=%.3g" % (nm, d))


def _km_self_test(lib) -> None:
    """k-means self-check: compare the partition **pixel by pixel** against the CPU reference
    (not merely "does it run").

    Two cases:
      * normal: initial centroids = the 5 true cluster centers -> the partition must match pixel by pixel;
      * empty cluster: the 5th initial centroid is thrown very far from the data -> it must be empty on the first round.
        Verifies that CPU/GPU handle empty clusters identically (the whole centroid row becomes 0,
        neither "keep the old value" nor "reseed").
    We deliberately do **not** build the empty cluster by making "two initial centroids coincide
    exactly": there the distances are strictly equal, and the CPU's gemm expansion (BLAS) and
    np.argmin's tie-breaking order are themselves inconsistent -- that is a spurious floating-point
    tie, not a semantic difference.
    """
    rngk = np.random.default_rng(20240)
    kk, ff = 5, 4
    blobs = rngk.normal(0.0, 1.0, (kk, ff))
    blobs *= 45.0 / np.linalg.norm(blobs, axis=1)[:, None]   # each cluster center sits at distance 45 from the origin, within-cluster sigma 0.4
    samp = np.repeat(blobs, 140, 0) + rngk.normal(0.0, 0.4, (kk * 140, ff))
    full = np.repeat(blobs, 300, 0) + rngk.normal(0.0, 0.4, (kk * 300, ff))
    far = np.full(ff, 1000.0)
    cases = (("常规", blobs.copy()),
             ("空簇", np.vstack([blobs[:kk - 1], far[None, :]])))
    for tag, cen0 in cases:
        it_try = 6
        s32 = np.ascontiguousarray(samp, dtype=np.float32)
        f32 = np.ascontiguousarray(full, dtype=np.float32)
        c0 = np.ascontiguousarray(cen0, dtype=np.float64)
        ref_c, ref_l = _np_lloyd_ref(samp, full, cen0, kk, it_try)
        got_c, got_l = _km_lloyd_raw(lib, s32, f32, c0, kk, it_try)
        nbad = int((ref_l != got_l).sum())
        if nbad:
            raise RuntimeError("k-means 自检不符 (%s): 划分不一致 %d/%d" % (tag, nbad, len(ref_l)))
        dc = float(np.abs(got_c - ref_c).max())
        if dc > 1e-4:
            raise RuntimeError("k-means 自检不符 (%s): 质心 max|d|=%.3g" % (tag, dc))
    if not np.array_equal(got_c[kk - 1], np.zeros(ff)):
        raise RuntimeError("k-means 自检不符 (空簇): 空簇质心应为全 0, 实得 %r" % (got_c[kk - 1],))


def _init() -> bool:
    global _LIB
    with _LOCK:
        if _STATE["tried"]:
            return _STATE["ok"]
        _STATE["tried"] = True
        try:
            nvcc = _find_nvcc()
            if nvcc is None:
                raise RuntimeError("找不到可执行的 nvcc (%s)" % _NVCC)
            _write_source()
            if not _so_is_fresh():
                _compile(nvcc)
            lib = ctypes.CDLL(_SO_PATH)
            _bind(lib)
            rc = lib.lt_init()
            if rc != 0:
                raise RuntimeError("lt_init rc=%d (%s)" % (rc, _last_error(lib)))
            _self_test(lib)
            _LIB = lib
            _STATE["ok"] = True
        except Exception as ex:                                   # noqa: BLE001
            _STATE["err"] = "%s: %s" % (type(ex).__name__, ex)
            _LIB = None
            _STATE["ok"] = False
        return _STATE["ok"]


# ----------------------------------------------------------------------
# public API
# ----------------------------------------------------------------------
def available() -> bool:
    """nvcc available + compiles + runs (end-to-end self-check passes). The result is cached;
    any failure returns False.

    Covers the **structure tensor** pipeline only (this function is its switch); k-means
    availability is separate, see ``kmeans_available()``, and the two do not affect each other.
    """
    return _init()


def kmeans_available() -> bool:
    """Whether the in-house CUDA k-means (``lloyd``) is available: requires ``available()``
    and a passing k-means self-check.

    The self-check compares the partition pixel by pixel against the CPU reference on the same
    input (including empty-cluster semantics). A failure only makes k-means fall back to CPU;
    it does **not** affect ``available()`` / ``structure_tensor``.
    """
    if not available():
        return False
    with _LOCK:
        if _KM_STATE["tried"]:
            return _KM_STATE["ok"]
        _KM_STATE["tried"] = True
        try:
            _km_self_test(_LIB)
            _KM_STATE["ok"] = True
        except Exception as ex:                                   # noqa: BLE001
            _KM_STATE["err"] = "%s: %s" % (type(ex).__name__, ex)
            _KM_STATE["ok"] = False
        return _KM_STATE["ok"]


def device_name() -> str:
    """GPU name; returns '无' when unavailable."""
    if not available():
        return "无"
    try:
        return _LIB.lt_device_name().decode("utf-8", "replace")
    except Exception:                                            # noqa: BLE001
        return "无"


def last_error() -> str:
    """Reason for the most recent initialization failure (empty string means there was none)."""
    return _STATE["err"]


def kmeans_last_error() -> str:
    """Reason for the most recent k-means self-check failure (empty string means there was none)."""
    return _KM_STATE["err"]


def build_log() -> list:
    """First 20 lines of nvcc output when the build failed."""
    return list(_STATE["log"])


def build_cmd() -> str:
    """The nvcc command line used for this build."""
    return _STATE["cmd"]


def free_buffers() -> None:
    """Free the cached device buffers (they are reallocated automatically on the next call)."""
    if _LIB is not None:
        try:
            _LIB.lt_free_buffers()
        except Exception:                                        # noqa: BLE001
            pass


def free_kmeans_buffers() -> None:
    """Free the device buffers cached for k-means."""
    if _LIB is not None:
        try:
            _LIB.lt_km_free()
        except Exception:                                        # noqa: BLE001
            pass


def _km_maxkf() -> int:
    """Upper bound on k*(F+1) allowed by the device (dynamic shared memory limit)."""
    try:
        return int(_LIB.lt_km_maxkf())
    except Exception:                                            # noqa: BLE001
        return 4096


def _km_lloyd_raw(lib, smp32, fl32, c0, k: int, iters: int):
    """Low-level call (does not consult available(), does not take the lock): shared by the public API and the self-check; raises on failure."""
    f = int(smp32.shape[1])
    cen_out = np.empty((k, f), dtype=np.float64)
    lab_out = np.empty((int(fl32.shape[0]),), dtype=np.int32)
    rc = lib.lt_km_lloyd(smp32.ctypes.data_as(_c_void_p), ctypes.c_int(int(smp32.shape[0])),
                         fl32.ctypes.data_as(_c_void_p), ctypes.c_int(int(fl32.shape[0])),
                         c0.ctypes.data_as(_c_void_p), ctypes.c_int(int(k)),
                         ctypes.c_int(f), ctypes.c_int(int(iters)),
                         cen_out.ctypes.data_as(_c_void_p), lab_out.ctypes.data_as(_c_void_p))
    if rc != 0:
        raise RuntimeError("lt_km_lloyd rc=%d (%s)" % (rc, _last_error(lib)))
    return cen_out, lab_out


def lloyd(sample: np.ndarray, feats: np.ndarray, cen: np.ndarray, k: int, iters: int):
    """Lloyd iterations + final full-image assignment, all in device memory (same signature
    and same return value as ``gpu_backend.lloyd``).

    * Semantics match the CPU path of ``logo_trace._kmeans``: the centroids are fitted on
      ``sample`` (the caller computes the initial centroids with k-means++ and passes them in),
      and ``feats`` is finally assigned once over the whole image.
    * Features (float32) and initial centroids (float64) are uploaded only once at the start and
      are never sent back to the host inside the loop; distances are squared Euclidean,
      accumulated term by term in double; empty-cluster centroids become 0 (same as the CPU's
      zeros_like).
    * Backend unavailable / parameters or device memory over the limit / any runtime failure ->
      raises RuntimeError, and the caller (``logo_trace._kmeans`` via try/except) silently falls
      back to CPU.

    Returns ``(cen float64 (k,F), labels int32 (len(feats),))``.
    """
    if not kmeans_available():
        raise RuntimeError("CUDA k-means 后端不可用: %s" % (_KM_STATE["err"] or "unknown"))
    k = int(k)
    iters = int(iters)
    smp = np.ascontiguousarray(sample, dtype=np.float32)
    fl = np.ascontiguousarray(feats, dtype=np.float32)
    c0 = np.ascontiguousarray(cen, dtype=np.float64)
    if smp.ndim != 2 or fl.ndim != 2 or smp.shape[0] < 1 or fl.shape[0] < 1:
        raise ValueError("sample/feats 必须是非空的二维数组, 得到 %r / %r" % (smp.shape, fl.shape))
    f = int(smp.shape[1])
    if f != int(fl.shape[1]) or c0.shape != (k, f):
        raise ValueError("列数/质心形状不符: sample%r feats%r cen%r k=%d"
                         % (smp.shape, fl.shape, c0.shape, k))
    if k < 1 or iters < 0:
        raise ValueError("k/iters 非法: k=%d iters=%d" % (k, iters))
    lim = _km_maxkf()
    if k * (f + 1) > lim:
        raise ValueError("k*(F+1)=%d 超过设备 shared 上限 %d" % (k * (f + 1), lim))
    if not (np.isfinite(smp).all() and np.isfinite(c0).all()):
        raise ValueError("sample/cen 含 NaN/Inf")
    return _km_lloyd_raw(_LIB, smp, fl, c0, k, iters)


def _conv(src32: np.ndarray, W: int, H: int, kx: np.ndarray, ky: np.ndarray) -> np.ndarray:
    """One separable convolution (kx along x, ky along y, clamp), returns float32 (H,W). A general-purpose primitive."""
    rx = (len(kx) - 1) // 2
    ry = (len(ky) - 1) // 2
    if rx == 0 and ry == 0:
        return np.ascontiguousarray(src32 * np.float32(kx[0] * ky[0]), dtype=np.float32)
    out = np.empty((H, W), dtype=np.float32)
    kx32 = np.ascontiguousarray(kx, dtype=np.float32)
    ky32 = np.ascontiguousarray(ky, dtype=np.float32)
    rc = _LIB.lt_sep_conv(src32.ctypes.data_as(_c_void_p), out.ctypes.data_as(_c_void_p),
                          ctypes.c_int(W), ctypes.c_int(H),
                          kx32.ctypes.data_as(_c_void_p), ctypes.c_int(rx),
                          ky32.ctypes.data_as(_c_void_p), ctypes.c_int(ry))
    if rc != 0:
        raise RuntimeError("lt_sep_conv rc=%d (%s)" % (rc, _last_error(_LIB)))
    return out


def _eigen(jxx: np.ndarray, jyy: np.ndarray, jxy: np.ndarray,
           rgb: np.ndarray, eps: float) -> dict:
    """Eigendecomposition + output dict (corresponds line by line to logo_trace.structure_tensor)."""
    tr = jxx + jyy
    dif = jxx - jyy
    tmp = np.sqrt(dif * dif + 4.0 * jxy * jxy)
    l1 = 0.5 * (tr + tmp)
    l2 = 0.5 * (tr - tmp)

    # Principal eigenvector: of the two candidates take the one with the larger magnitude (same criterion as the CPU version)
    ax, ay = jxy, l1 - jxx
    bx, by = l1 - jyy, jxy
    use_b = np.hypot(bx, by) > np.hypot(ax, ay)
    vx = np.where(use_b, bx, ax)
    vy = np.where(use_b, by, ay)
    nrm = np.hypot(vx, vy)
    nrm = np.where(nrm < eps, 1.0, nrm)
    vx = vx / nrm
    vy = vy / nrm

    luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    coh = (l1 - l2) / (l1 + l2 + eps)
    return {"l1": l1, "l2": l2, "coh": coh, "energy": l1, "luma": luma,
            "gx": vx, "gy": vy, "tx": -vy, "ty": vx}


def structure_tensor(rgb: np.ndarray, sigma_d: float, sigma_i: float,
                     color: bool = True, eps: float = 1e-12) -> dict:
    """Mathematically equivalent to ``logo_trace.structure_tensor`` (the GPU only performs the
    separable convolutions and products).

    Parameters
    ----
    rgb     : (H,W,3) float array (float64, values in [0,1])
    sigma_d : differentiation scale; sigma_i : tensor integration scale
    color   : True uses the three-channel Di Zenzo; False uses rgb.mean(2)
    eps     : denominator guard for coh / normalization

    Returns
    ----
    A dict with exactly the same keys as the CPU version, all float64 (H,W):
        l1, l2, coh, energy(=l1), luma, gx, gy, tx, ty
    """
    if not available():
        raise RuntimeError("CUDA 后端不可用: %s" % (_STATE["err"] or "unknown"))

    a = np.asarray(rgb, dtype=np.float64)
    if a.ndim != 3 or a.shape[2] < 3:
        raise ValueError("rgb 必须是 (H,W,3), 得到 %r" % (a.shape,))
    H, W = int(a.shape[0]), int(a.shape[1])
    chans = [a[..., 0], a[..., 1], a[..., 2]] if color else [a.mean(axis=2)]

    kd0 = _kernel(sigma_d, 0)
    kd1 = _kernel(sigma_d, 1)
    ki = _kernel(sigma_i, 0)
    r0, r1, rs = (len(kd0) - 1) // 2, (len(kd1) - 1) // 2, (len(ki) - 1) // 2

    if r0 < 1 or r1 < 1 or rs < 1 or (H * W) >= 2147483647:
        # sigma too small (radius 0) or image too large: fall back to the host float64 reference implementation (equally correct)
        jxx, jyy, jxy = _np_tensor_planes(a, sigma_d, sigma_i, color)
        return _eigen(jxx, jyy, jxy, a, eps)

    lib = _LIB

    def pf(v):
        return np.ascontiguousarray(v, dtype=np.float32)

    kd0_32, kd1_32, ki_32 = pf(kd0), pf(kd1), pf(ki)
    rc = lib.lt_st_begin(ctypes.c_int(W), ctypes.c_int(H))
    if rc != 0:
        raise RuntimeError("lt_st_begin rc=%d (%s)" % (rc, _last_error(lib)))
    scale = ctypes.c_float(1.0)      # scaling stays on the host (exactly the same order as the CPU's jxx/=n_ch)
    for c in chans:
        rc = lib.lt_st_channel(pf(c).ctypes.data_as(_c_void_p), ctypes.c_int(W), ctypes.c_int(H),
                               kd0_32.ctypes.data_as(_c_void_p), ctypes.c_int(r0),
                               kd1_32.ctypes.data_as(_c_void_p), ctypes.c_int(r1),
                               ki_32.ctypes.data_as(_c_void_p), ctypes.c_int(rs),
                               scale)
        if rc != 0:
            raise RuntimeError("lt_st_channel rc=%d (%s)" % (rc, _last_error(lib)))
    j32 = [np.empty((H, W), dtype=np.float32) for _ in range(3)]
    rc = lib.lt_st_end(j32[0].ctypes.data_as(_c_void_p), j32[1].ctypes.data_as(_c_void_p),
                       j32[2].ctypes.data_as(_c_void_p), ctypes.c_int(W), ctypes.c_int(H))
    if rc != 0:
        raise RuntimeError("lt_st_end rc=%d (%s)" % (rc, _last_error(lib)))

    n_ch = float(len(chans))
    return _eigen(j32[0].astype(np.float64) / n_ch, j32[1].astype(np.float64) / n_ch,
                  j32[2].astype(np.float64) / n_ch, a, eps)
