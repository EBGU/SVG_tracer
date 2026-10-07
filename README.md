# SVG_tracer —— bitmap-to-vector tracing (structure tensor + partition routing)

> **SVG_tracer** is the project name. The Python module keeps its historical name: the program is
> `logo_trace.py`, and the console entry point installed from `pyproject.toml` is `svg-tracer`
> (equivalent to `python logo_trace.py`). Helper modules (`svg_slim.py`, `svgzip.py`,
> `gpu_backend.py`, `cuda_backend.py`, `selfcheck.py`) are unchanged.

It reads stroke direction and gradient orientation with a **multi-scale structure tensor** and fixes
boundary positions with **edge detection**, automatically deciding per region which of the two
drawing styles to use, and emits a self-contained, compression-friendly `.svg` / `.svgz`:

| Input | Result (rendered back at native size, compared with the source) | Size |
| --- | --- | --- |
| `inputs/logo.png` (1254², solid colors + geometry + golden seam lines) | **40.36 dB** (37.57 dB on a 1x grid) | 461 KB → gzip **146 KB** → rounded strokes **98.6 KB** (≈lossless)|
| `inputs/logo.png` + `--scale 4` (high fidelity) | **43.43 dB** | 594 KB → gzip **197 KB** → rounded strokes **161 KB** (≈lossless)|
| `inputs/water-lilies-29.jpg` (1511×1600 freehand oil painting) | 24.00 dB | 8.12 MB → gzip **1.90 MB** → tight **0.92 MB** |

Three key points:

- **Automatic division of labour within one image**: for each region it measures the "high-frequency
  texture + interior color spread"; flat-looking parts take the **edge detection + subpixel snapping +
  conformal Bezier** route, and textured parts automatically fall back to **color segmentation +
  Catmull-Rom + structure-tensor streamline strokes**.
- **Almost all of the error sits in the 1px band along the boundary** (3.00% of the pixels but about
  92% of the squared error), so the improvements focus on "getting the boundary right": 50% intensity
  crossing for centering, and tracing on an enlarged grid.
- **The output is naturally compression-friendly**: the paths are already relative commands, so
  precision can be trimmed losslessly, and after gzip it usually compresses to 1/3 ~ 1/8.

---

## Quick start

```bash
conda create -y -n logotrace python=3.11 && conda activate logotrace
pip install numpy scipy scikit-image pillow cairosvg

python logo_trace.py --in logo.png --preset logo     # a bare input name is looked up in inputs/, output goes to out/
python selfcheck.py                                  # self-check: all green means the environment is fine (20~60 s)
```

That command produces `out/logo_traced.svg` (461 KB, 40.36 dB) and `out/logo_traced_preview.png`.
A summary is printed at the end of the console run:

```
  输入   inputs/logo.png  1254x1254  →  网格 2508x2508 (--scale 2)
  分区   52 区域 · 39 渐变 · 428 笔触 · 0 缝线
  矢量   out/logo_traced.svg  461 KB   (40.36 dB)
  压缩   out/logo_traced.svgz  146 KB
  预览   out/logo_traced_preview.png
```

### Common task recipes

| I want to | Command |
| --- | --- |
| Hard-edged flat artwork (logo / icon / illustration) | `python logo_trace.py --in logo.png --preset logo` |
| Only the geometric layer, no strokes | add `--no-strokes` |
| Chase high fidelity (4x grid, about 20 minutes) | add `--scale 4 --compress slim --gzip`, then `svgzip.py … --prec 0` (161 KB, 43.43 dB) |
| Freehand painting / photos | `python logo_trace.py --in water-lilies-29.jpg --preset painting` |
| The smallest possible file | add `--compress slim --gzip`, then `svgzip.py … --prec 0` (98.6 KB, ≈lossless); `--tight` goes further (72 KB, but −1.9 dB) |
| Try parameters first, without preview or logs | add `--no-preview --quiet` |
| Compress an **existing** SVG | `python svgzip.py out/logo_traced_scale4.svg --prec 0 --svgz` (lossless 161 KB); add `--tight` for even smaller |
| Baseline: the original pure color segmentation | `--method colors --fit cr --snap-mode off` |
| Baseline: the pure edge-detection route | `--method edges --fit bezier` |
| Run segmentation only once while tuning strokes | add `--labels-cache /tmp/lab.npy` the first time, then reuse it |

## Examples

All four inputs in `inputs/` traced with the current code, on a quiet machine, with the shared flags
`--gpu auto --jobs 64 --no-preview` — plus `--auto-gradient` for `apple.png`, the one smooth/glossy input
(sizes are the plain `.svg` files, no `.svgz`; the per-run logs are not stored in the repository,
`examples/RUNLOG.txt` keeps the machine-readable summary):

| Input | Preset + scale | SVG | Regions · gradients · strokes | Wall clock |
| --- | --- | --- | --- | --- |
| `inputs/apple.png` | `--preset logo --scale 4 --auto-gradient` | [apple_traced_scale4.svg](examples/apple_traced_scale4.svg) — 1043 KB (1,068,362 B) | 383 · 151 · 0 | 65 s |
| `inputs/openai.png` | `--preset logo --scale 1` | [openai_traced_scale1.svg](examples/openai_traced_scale1.svg) — 197 KB (201,419 B) | 31 · 0 · 0 | 37 s |
| `inputs/water_lilies.jpg` | `--preset painting --scale 1` | [water_lilies_traced_scale1.svg](examples/water_lilies_traced_scale1.svg) — 5018 KB (5,138,876 B) | 1169 · 1077 · 9812 | 119 s |
| `inputs/wave.jpg` | `--preset painting --scale 1` | [wave_traced_scale1.svg](examples/wave_traced_scale1.svg) — 17,933 KB (18,363,723 B) | 3453 · 2201 · 34,799 | 288 s |

```bash
python logo_trace.py --in inputs/apple.png       --preset logo     --scale 4 --gpu auto --jobs 64 --no-preview --auto-gradient --out examples/apple_traced_scale4.svg
python logo_trace.py --in inputs/openai.png      --preset logo     --scale 1 --gpu auto --jobs 64 --no-preview --out examples/openai_traced_scale1.svg
python logo_trace.py --in inputs/water_lilies.jpg --preset painting --scale 1 --gpu auto --jobs 64 --no-preview --out examples/water_lilies_traced_scale1.svg
python logo_trace.py --in inputs/wave.jpg        --preset painting --scale 1 --gpu auto --jobs 64 --no-preview --out examples/wave_traced_scale1.svg
```

The two logo examples stay small (openai is fully flat: 0 gradients, 0 strokes); the two paintings are
mostly gradients and strokes, which is where the adaptive refinement and the gradient-aware merging earn
their keep. Gradient counts include both linear and radial fits.

`apple.png` is smooth/glossy, so it is traced with `--auto-gradient` (gradient gate 0.12 → 0.02, merge
tolerance 0.05 → 0.3 over 8 passes, `--aa-levels 3`, `--refine-err 0.008`). Measured against the input
at its native 250×312 render: the ≥40 px plateau share drops from 31.4 % to 23.7 % horizontally and
from 28.5 % to 24.3 % vertically, and JUMP % from 11.71/12.64 to 11.11/11.96, for 0.9 dB of PSNR
(34.21 → 33.35) and 152 KB. The other three examples were re-checked with the same flag and keep the
plain command: `openai.png` comes out byte-identical with it (md5 `52b15cb8…`), while on
`water_lilies.jpg` it costs 1.95 dB PSNR (23.78 → 21.83), raises MAE from 12.11 to 15.00/255 and
quadruples the ≥40 px vertical plateau share (1.4 % → 5.8 %), and on `wave.jpg` it costs 1.37 dB PSNR
(21.10 → 19.73 at 1280×861) only to shave a plateau share that was about 1 % to begin with — the route
is for smooth artwork, not for texture-rich photos (see the `--auto-gradient` row above).

## Directory layout

```
SVG_tracer/                (the Python module keeps the name logo_trace.py)
├── logo_trace.py      main program (single file, 100+ parameters documented in Chinese)
├── svg_slim.py        slimming kernel (numeric precision trimming / clipPath dedup, standard library only)
├── svgzip.py          slimming / compression CLI (same kernel as --compress)
├── gpu_backend.py     GPU dispatch layer (prefers cupy, falls back to cuda_backend)
├── cuda_backend.py    in-house CUDA structure tensor (on-the-fly nvcc compile + ctypes, no cupy needed)
├── build/             the liblogotrace_cuda.so compiled from the module above (auto-generated on first run)
├── selfcheck.py       self-check (synthetic small image through the whole pipeline + compression round-trip + path resolution)
├── README.md          this file
├── docs/
│   └── experiments.md algorithm details and every ablation experiment (look here to dig deeper)
├── inputs/            input images used by the examples: apple.png, openai.png, water_lilies.jpg, wave.jpg
├── examples/          the four regenerated example SVGs + RUNLOG.txt (see "Examples")
└── out/               all artifacts of a normal run: *.svg / *.svgz / *_preview.png / comparison figures
```

**Input/output conventions**: passing a bare filename to `--in` (such as `logo.png`) looks it up under
`inputs/`; when `--out` is omitted the result is written to `out/<input-name>_traced.svg`; previews and
debug images follow the same rule (`out/<input-name>_preview.png` / `_debug.png`). Missing directories
are created automatically.

## Presets

`--preset` simply bundles a set of defaults (parameters given explicitly on the command line win):

| Preset | For | Key differences |
| --- | --- | --- |
| `logo` | Hard-edged flat artwork, icons, illustrations | `--scale 2` (trace on a 2x grid) + subpixel localization + conformal Beziers (`--fit-tol 0.10`), with the 50% crossing for subpixel centering |
| `painting` | Oil paintings / photos and other freehand art | `k=48`, `--min-area 180`, `--spacing 7`, full-range gradients, transition bands off |
| (none given) | General | The same geometry/boundary parameters as `logo`, but a **1x grid** (fast and small, PSNR 2.8 dB lower) |

What `--scale` means is "trace on the enlarged grid": the canvas is still written at native size
(`width/height=1254` + `viewBox="0 0 2508 2508"`), so the default display size is unchanged while
zooming in looks sharper. For hard-edged artwork every step up pays off (1→1.5→2→4x:
37.57 / 39.21 / 40.36 / 43.43 dB), whereas freehand art has diminishing returns (+0.71 dB for 2.2x the
size); see [docs/experiments.md](docs/experiments.md).

**Automatic parameter scaling for small images**: the defaults are calibrated for a ~1254px canvas
(`min_area=600px²`, `spacing=10px` …). When the image is markedly smaller these absolute pixel
quantities become relatively too large (regions are over-merged and strokes smear into wide bands), so
the tool automatically multiplies length-like parameters by `k=max(W,H)/1254` and area-like ones by
`k²` (nothing changes for k ≥ 1, so images of 1254² and above keep exactly the historical results).
Disable it with `--no-autoscale`.

## Common parameters

The complete list is in `python logo_trace.py --help` (100+ entries, grouped into "input/output /
runtime / presets / structure tensor / segmentation / gradients / strokes / edges").
The ones you actually reach for day to day are usually these:

| Parameter | Default | Effect |
| --- | --- | --- |
| `--in / --out` | `inputs/logo.png` / `out/<name>_traced.svg` | Input and output; a bare filename is looked up in `inputs/` |
| `--preset` | empty | `logo` / `painting`, see the table above |
| `--scale` | 1.0 | Tracing grid magnification. For hard-edged artwork use 2.0 (+2.80 dB) or 4.0 (+3.07 dB) |
| `--compress` | **slim** | `off` as-is / `slim` numeric precision trimming + rounded strokes (bit-lossless) / `tight` = the two above plus duplicate clipPath removal (smallest, but strokes escape their regions, see "Output size and compression"; **the compression cost depends on the file itself, so always measure it on your own file**: measured from only **−0.11 dB** for `--tight` on the full-image painting setting and **−0.83 dB** for `--prec 0` trimming, to **−1.9 dB** for `--tight` on logo; `slim` behaves the same way, saving 10% on the trimmed variant and 0% on the full-image painting variant** **`--preset painting` defaults to `tight`** (watercolor measures −0.01 dB with about 1/5 of the size saved).|
| `--gzip` | off | Also write `.svgz` (gzip -9), usable directly by browsers and `<img>` |
| `--no-strokes` | off | Keep only the geometric layer (regions + gradients); much smaller, good for purely flat artwork |
| `--method` | **hybrid** | `hybrid` partition routing / `colors` pure color segmentation / `edges` pure edge watershed / `watershed` |
| `--seg` | **full** | Segmentation route: `full` the complete route (k-means + multi-round RAG merging, the quality baseline) / `flat` (with `auto` as its alias, the two are equivalent) the flat-color fast path. For hard-edged flat icons 9.3→4.1 s with only −0.20 dB PSNR; **a logo with gradients loses 2.9 dB (37.57→34.71), so the default is `full`** |
| `--gpu` | **auto** | Use the GPU (**now on by default**): **the structure tensor prefers the in-house CUDA kernel** (transfers float32 + accumulates in double, almost bit-identical to scipy: the same watercolor parameters differ by only a few bytes), falling back to cupy when nvcc is absent (`cupyx.scipy.ndimage` separable filtering, 13~20× faster than CPU), and **k-means uses the in-house CUDA kernel** (nvcc+ctypes, measured at pipeline scale 150k points 99 ms vs cupy 118 ms, full image 1.5M points 128 ms vs 163 ms, 3.0~3.6× versus CPU, with cupy as fallback). `auto` uses it when it can / `on` forces it / `off` is pure CPU. Quiet-window end-to-end **8.4 → 7.2 s (1.17×)**; but the output is only "floating-point close" to the `out/` deliverables rather than byte-identical (logo: CPU 599 strokes/37.57 dB, in-house 598/37.66 dB, cupy 594/37.62 dB). **To reproduce the deliverables byte for byte, pass `--gpu off` explicitly** (on a machine without a GPU, `auto` already means pure CPU) |
| `--jobs` | **0=auto** | Number of parallel processes for the per-region stage (fork + copy-on-write: large arrays are neither copied nor pickled). 0=auto `min(cores,16)`, or 1 when there are fewer than 24 candidate regions; 1=off. **Watercolor 1× measured end-to-end 510.7 → 249.9 s (2.04×)** (8 processes, load≈198 at the time). **The stages without rng (region vectors / stroke activity / colored detail layer) produce byte-identical output when parallelized**; the stroke layer switches to a deterministic per-region seed when parallel (reproducible for the same arguments and process count). So **the byte-identical-to-`out/` setting is `--jobs 1 --gpu off`** |
| `--batch-book` | **auto** | Rewrites the rule-based sub-steps of the per-region stage (mask erosion / per-region median color) as whole-image batches: `auto` batches whenever it can (GPU preferred, cupy), `off` keeps the per-region loop. The batched version is **byte-identical** to the per-region version (the median has a closed form on the 1/255 grid); resampled images with `scale != 1` fall back automatically, and with < 256 labels it is skipped too (whole-image filtering is more expensive than per-region then) |
| `--fit` | **auto** | `auto` picks from the region's geometricity; `bezier` conformal optimal (good for straight edges and sharp corners); `cr` Catmull-Rom |
| `--fit-tol / --contour-smooth` | 0.10 / 1.0 | **Must be tuned together**: tightening either one alone makes things worse; tightening both gains +0.47 dB |
| `--snap-mode / --snap-sub` | **refine / half** | Subpixel boundary snapping in geometric regions; `half`=the 50% intensity crossing (4× more accurate than the old `peak`, +0.74 dB) |
| `--aa-levels / --aa-width` | **0** / 2.5 | Boundary transition-band reconstruction. Once localization is accurate it gives **no benefit**, so it is off by default. **New: for a smooth image `--auto-gradient` now defaults `--aa-levels` to 3** (the smoothness test is the one `--auto-gradient` already runs); an explicit `--aa-levels N` always wins |
| `--aa-synthetic / --aa-edge-dedup / --aa-edge-canon` | off / **on** / off | Which anti-aliasing bands are rebuilt once `--aa-levels > 0` (all three are no-ops while it is 0). Adaptive refinement cuts one original region into several children; the border between two children of the **same** original region is *synthetic* (the colors are continuous across it by construction), so a transition band there is pure byte cost. `--aa-synthetic` re-enables bands on synthetic borders (the old byte-for-byte behavior; default **off** = skip them). `--aa-edge-dedup` (**on by default**) pushes the "same original region" tag down the whole refinement tree, so uncle/cousin borders also count as synthetic instead of only one parent-child step -- measured apple `--scale 4`: **1223 → 1043 KB** with identical PSNR/MAE. `--aa-edge-canon` is a more aggressive **experimental** variant (**off by default**): keep a real-image-edge band only for the largest descendant of an original region, and drop it for the others -- it deletes real-edge AA, so it is a per-image opt-in |
| `--auto-gradient / --grad-min-gain` | off / 0.12 | `--grad-min-gain` is the gradient acceptance gate: a linear fit must remove at least this fraction of the squared error or the region falls back to a flat fill; for smooth imagery (soft logos, rendered/near-gradient art) 0.02 ≈ removes the visible color steps. `--auto-gradient` measures image smoothness on a 64² thumbnail and lowers the gate to 0.02 by itself on smooth images (and also defaults `--aa-levels` to 3), so the gate does not have to be hand-tuned. Keep it on for smooth/glossy inputs: the shipped `examples/apple_traced_scale4.svg` is generated with it (≥40 px plateau share 31.4 % → 23.7 % against 24.8 % for the source, paid for with 0.9 dB PSNR). Its smoothness test fires on the other three examples too, where it is a no-op (`openai.png`, byte-identical) or a clear loss (`water_lilies.jpg`: −1.95 dB PSNR, ≥40 px vertical plateaus 1.4 % → 5.8 %), so it intentionally stays a flag — see `## Examples` |
| `--grad-radial / --grad-radial-margin` | **on** / 0.05 | Besides the two linear axes, also fit a **radial** candidate (centre + radius, `<radialGradient>`, same `--grad-stops` count) with the **same error/gain criterion**; it is adopted only when its residual beats the best linear residual by this margin, so genuinely linear ramps keep their linear fit. Candidates for the centre: region centroid, bounding-box centre, and the least-squares intersection of the local gradient lines (rejected when ill-conditioned). `--no-grad-radial` disables |
| `--merge-grad / --merge-grad-tol / --merge-grad-passes` | **on** / 0.05 / 4 | Gradient-aware **spatial** region merging: k-means boundaries are hard steps by construction, so adjacent patches are re-merged whenever a single gradient (linear or radial) over the union still passes the acceptance gate **and** the union's mean squared error stays within `1 + tol` of the area-weighted mean of the two separate fills' errors. The second test protects hard-edged artwork (two flat regions have near-zero separate error, so a gradient that only bridges their step is rejected). Fixed scan order + label-indexed sample seeds make it deterministic. `--no-merge-grad` disables |
| `--adaptive-refine` + `--refine-err / --refine-min-area / --refine-max-depth / --refine-gain / --refine-budget / --refine-order` | **on** / 0.02 / 400 / 6 / 0.20 / 0=auto / `merge-first` | **Error-driven adaptive refinement** (new): each region measures the residual of its current fill (flat / linear / radial), and a region whose core contains a *local* patch of visible error is bisected along the worst-error axis and re-fitted recursively, so a fixed SVG element budget is spent where it is actually needed and the flat color steps that used to show on smooth or glossy artwork disappear. `--refine-err` is the residual threshold (0.02 ≈ 5/255) and the trigger is local (a small patch of error barely moves a big region's global RMS); `--refine-min-area 400` keeps noise and thin seams out; `--refine-max-depth 6` bounds the recursion; `--refine-gain 0.20` requires the area-weighted child RMS to beat the parent by 20% (this is the natural protection against texture/noise and the growth valve); `--refine-budget 0` = auto `max(64, min(256, #regions))`; `--refine-order` picks refine before/after gradient merging (`merge-first` default, `refine-first` refines the raw partition first). `--no-adaptive-refine` restores the old behavior (byte-for-byte together with `--no-grad-radial --no-merge-grad`) |
| `--grad-mesh` + `--grad-mesh-margin / --grad-mesh-patches / --grad-mesh-min-area / --grad-mesh-underlay` | off / 0.10 / 2 / 4000 / on | **SVG 2 mesh gradients** (Coons patches) as one more fill candidate for large smooth regions; adopted only when the full-domain RMS beats the current linear/radial fill by `--grad-mesh-margin` (0.10). `--grad-mesh-patches 2` means 3×3 color stops per side, and only regions at least `--grad-mesh-min-area` (4000 px²) are tried. **Compatibility caveat:** `<meshgradient>` needs an SVG 2 mesh-capable renderer -- mainstream browsers (Chrome/Firefox/Safari) and cairosvg ignore it, and they also do not support SVG 2's paint fallback list, so a bare `fill="url(#meshN) url(#gN)"` makes them drop the region entirely (measured: cairosvg paints it transparent). The tool therefore always emits a plain fallback path underneath when `--grad-mesh-underlay` is on (**default**), so non-mesh renderers still paint correctly; `--no-grad-mesh-underlay` writes the pure SVG 2 paint-fallback form only. Mesh is **off by default** |
| `--kmeans-k / --thresh` | 20 / 12 | Region count and RAG merge threshold (use 48 / 8 for freehand art) |
| `--min-area / --contour-tol` | 600 / 0.75 | Minimum region area and contour simplification tolerance |
| `--tex-thr / --range-thr` | 3.0 / 0.02 | Texture upper bound and interior color-spread upper bound for calling a region "solid-color geometric" |
| `--spacing / --stroke-width` | 10 / 7.5 | Stroke spacing and width (**interpreted in native pixels**, multiplied by `--scale` automatically) |
| `--stroke-grid-units` | off | Return to the old behavior: stroke lengths interpreted in **tracing-grid** units (at 4x, 599 → 9567 strokes, 6.5 MB) |
| `--no-preview / --quiet` | — | Do not render a preview / print only the final summary |
| `--labels-cache` | empty | `.npy` cache of the segmentation result, so tuning stroke/fitting parameters does not rerun segmentation. **Note: running with the cache and running in full are not necessarily byte-identical** -- segmentation consumes the global `rng` internally, and skipping it changes the jitter grid of the stroke stage (measured on watercolor: strokes 9679→9668, 8516347→8509314 bytes). Using the cache for tuning is fine, but **use the same basis on both sides when benchmarking** |
| `--no-autoscale` | off | Turn off automatic parameter scaling for small images |

Threads are capped at 4 by default (`LOGO_TRACE_THREADS` overrides this): this is a shared large
machine, and letting BLAS/OpenMP thread by core count drags a single step out to 60 s through
oversubscription.

## Speed: three acceleration routes and measurements

The three switches can be stacked; the default settings are chosen so that "the output matches the
delivered byte-for-byte baseline".

| Switch | What it does | Measured (scale 1, `inputs/logo.png`, 192-core machine) |
| --- | --- | --- |
| `--seg full` (default) | k-means + multi-round RAG merging + partition routing, the quality baseline | 1× 37.57 dB / 442 KB; 2× 40.36 dB (byte-identical to `out/logo_traced.svg`) |
| `--seg flat` | Flat-color fast path: histogram dominant colors → palette → LUT nearest color → connected components → RAG, skipping the slow k-means/RAG re-segmentation | Hard-edged flat icons: segmentation 9.3 → 4.1 s, 27.57 → 27.37 dB (−0.20); **logo with gradients: 37.57 → 34.71 dB (−2.86), do not use** |
| The per-region stage (**default**, no switch) | Bounding-box restructuring: each region does `labels == li` / erosion / median / contour extraction inside its own bounding box (+ margin) instead of over the whole image | Watercolor 1× (same window, same cache): region-vector stage **195.8 → 16.0 s = 12.2×**, end-to-end 323.6 → 95.6 s = 3.4×, and the artifact is **byte-identical** (md5 `58f3185593`); logo 1× is byte-identical as well -- it changes no output at all, so it is simply on by default |
| `--gpu auto` (**default**) | Use the GPU: **the structure tensor goes through cupy** (`cupyx.scipy.ndimage` separable filtering), **k-means goes through the in-house CUDA kernel** (`cuda_backend.py`, nvcc+ctypes; cupy as fallback) | Same quiet window: tensor 2.0→0.7 s, segmentation 4.6→3.2 s, end-to-end **8.4→7.2 s (1.17×)**, PSNR 37.57(CPU)/37.66(in-house)/37.62(cupy) dB (strokes 599/598/594); **`--scale 4` is where it shines**: tensor 32.2→4.0 s (8.1×), end-to-end 105.4→75.7 s (1.39×), PSNR 43.43→43.38 (−0.05 dB) |
| `--jobs` (**auto by default**) | Parallelize the per-region stage with fork: region vectors, stroke activity and the colored detail layer each use "compute in parallel + parent writes back in the original order" | The stages without rng are **byte-identical** (logo 4262 candidate labels, 8 processes and the serial baseline share md5 `7a3363aa9d`; watercolor 1× with cache 9668 strokes / ~230 seam lines give `f9bf95b3dfb9` for both jobs1 and jobs8). **Same-window A/B: end-to-end 510.7 → 249.9 s (2.04×)**, of which region vectors 2.68× and strokes 2.03× (load≈198 at the time; a quiet machine will be higher); the stroke layer uses a deterministic per-region seed when parallel |
| `--batch-book` (**auto by default**) | Turns the two most rule-based steps of the per-region stage (mask erosion, per-region median color) into one whole-image pass, GPU first with a CPU fallback. **Byte-identical** to the per-region version (measured at the same md5 on both logo and watercolor, and watercolor also matches the pre-optimization snapshot). But it only covers about 7% of that stage's time (64.9% is in `refine_contour`, Amdahl limit ≈1.07×), so its role is "a free, zero-risk optimization" rather than a primary speedup |

**Why `--gpu` is now on by default** (`auto`): use it when it can, and a machine without a GPU
silently falls back to pure CPU, so it never fails in any environment.
The price is that **the output goes from "byte-identical to the `out/` deliverables" to
"floating-point close"** (logo 1×: CPU 599 strokes/37.57 dB, in-house kernel 598/37.66 dB,
cupy 594/37.62 dB; a controlled experiment -- forcing k-means back to CPU and keeping only the GPU
structure tensor -- produced output **byte-identical** to full GPU, showing that the entire
difference comes from the float32 structure-tensor path and that the k-means backend itself has zero
effect on the output). **To reproduce `out/` byte for byte (for regression comparison), pass
`--gpu off` explicitly**; `--scale 4` is where it pays off most (tensor 32.2 → 4.0 s = 8.1×,
end-to-end 105.4 → 75.7 s = 1.39×, PSNR only 0.05 dB lower at 43.43 → 43.38), while for images like
watercolor, where "segmentation and the per-region stage take 99% of the time", it is almost
imperceptible (95.6 → 95.3 s). It also has to shuttle data back and forth between the GPU and the host
(quiet-window measurements H2D 7.2 GB/s / D2H 6.0 GB/s; under heavy load the same link is an order of
magnitude slower, which is why the backend insists on keeping intermediate results in device memory
and never does "one transfer per step").

**How cupy and the in-house kernel divide the work**: the `logotrace-gpu` environment has
`cupy-cuda12x` installed (pip is configured for the Tsinghua mirror + the Aliyun extra-index, and a
local proxy from `~/network.sh` is also available), so **the structure tensor prefers cupy**; in the
`logotrace` environment, which has no cupy, the same path automatically switches to the in-house CUDA
kernel -- `cuda_backend.py` compiles on the fly into `build/liblogotrace_cuda.so` using the system
`nvcc` (the first run waits for the compile, afterwards the cache is used), and silently falls back to
CPU if compilation fails or nvcc is missing. **k-means is the other way round: the in-house kernel is
preferred with cupy as fallback** -- on the same machine the in-house kernel is faster at both
pipeline scale (150k points 99 ms vs cupy 118 ms) and full-image scale (1.5M points 128 ms vs 163 ms).
Neither route changes the default output; details and the full measurement table are in
[experiments.md §9.3.2](docs/experiments.md).

**Acceptance (measured on this machine, compared step by step before and after the changes)**

All four deliverables rerun with **`--gpu off --jobs 1`** **reproduce byte for byte** the files in
`out/` (identical md5) -- in other words, under **that** setting these acceleration changes have zero
effect on the output. Note that the defaults have changed: `--gpu auto` turns the output into
"floating-point close" (see above), and `--jobs 0` is byte-identical only in the stages without rng,
using a per-region seed when the stroke layer is parallel (with very many strokes the jitter can
change the count: at logo settings yielding 4863 strokes, jobs1/jobs8 give 4863/4902, though the
parallelism itself is deterministic, see [experiments.md §9.6](docs/experiments.md)). **For
regression/reproduction use `--gpu off --jobs 1` explicitly**; `--seg flat` is still off by default
(it loses 2.9 dB).

| Item | Result |
| --- | --- |
| `selfcheck.py` | **all passing**; the "end-to-end PSNR ≥ 21.0 dB" check measures **22.36 dB**, **bit-identical** to the self-check from before the acceleration work |
| logo 1× | **byte-identical to the verified output from before the changes** (12 regions · 10 gradients · 599 strokes · 442 KB · 37.57 dB) |
| logo 2× | **byte-identical to `out/logo_traced.svg`** (471925 bytes, i.e. 40.36 dB) -- **re-verified on the final version including the per-region restructuring** |
| watercolor 1× | **byte-identical to `out/water-lilies-29_traced.svg`** (8516347 bytes, i.e. 24.00 dB; 2064 regions · 9679 strokes · 1511 gradients) |
| per-region restructuring | logo 1× vs `/tmp/t2.svg`, and watercolor vs the **unmodified stable version (same cache setting)**, are both **byte-identical** (watercolor md5 `58f3185593`); same-window measurement of the region-vector stage **12.2×** |
| GPU backend | Structure tensor: 9 keys × 2 parameter sets × 1.57M pixels, worst criterion **9.7e-05 < 1e-03**, **0 sign flips**; k-means: partition mismatches **0 / 1,500,000**, relative inertia difference **1.487e-10**. `--gpu on` end-to-end **37.66 dB** (`--gpu off` 37.57) -- a controlled experiment shows this difference comes entirely from the float32 structure-tensor path, and the k-means backend itself has zero byte-level effect. **First measurement of the cupy backend**: structure tensor 13~20× vs CPU, k-means 2.6~3.6× vs CPU, `--gpu on` end-to-end **7.2 s (1.17×)**, PSNR 37.62 dB. **Re-verified after `--gpu` became `auto` by default**: logo 1× = 451799 B / 598 strokes / **37.66 dB** (`--gpu off` is 452118 B / 599 / 37.57), wave.jpg (3859×2594, 10 million pixels) tensor stage 10.4 s |
| Per-region multiprocessing (`--jobs`) | The three rng-free stages (region vectors / stroke activity / colored detail layer) are **byte-identical** to the serial version with 8 processes: logo 1× (4262 candidate labels) shares md5 `7a3363aa9d`; with `--detail-chroma 2 --no-strokes`, jobs1/jobs8 both give `3dc0dd0f04`. The stroke layer switches to a deterministic per-region seed when parallel (running the same command twice gives the same md5); details in [experiments.md §9.6](docs/experiments.md) |

**Where the long runs (freehand art) get stuck**: watercolor 1× is 8.12 MB / 24.00 dB / 2064 regions
+ 9679 strokes, and its time is **not** in the structure tensor (30 s) or RAG merging (a dozen
seconds) but in the **per-region stage** -- every region did `labels == li`, erosion, median and
contour extraction over the whole image, and 2064 regions × 4.3M pixels is tens of billions of
operations. **It now computes only inside that region's own bounding box** (`ndi.find_objects` + a
safety margin), the artifact is **byte-identical** to before, and the per-region stage measured
**195.8 → 16.0 s (12.2×)** with the cached end-to-end run going 323.6 → 95.6 s; this is **default
behavior** and needs no switch (it changes no bytes, unlike the default-off `--seg flat` and the
default-on `--gpu auto`).
Utilization fluctuates (192 cores), and the same code with the same parameters has measured anywhere
from 10 s to 115 s, so the numbers above give ratios and stage shares wherever possible.

## Output size and compression

`svg_slim.py` provides three levers, and both `--compress` and `svgzip.py` use them:

1. **Numeric precision trimming** (`slim`, default): the paths emitted by the generator are already
   relative commands (`M x y l dx dy c …`), so it only tightens the number of decimals and drops
   redundant separators (including rounding stroke coordinates to the integer grid) → 2x 479→461 KB,
   4x 614→594 KB, with **bit-identical rendering** (mutual comparison >160 dB).
2. **Lowering stroke precision on its own** (`--prec 0`): stroke coordinates keep only the integer
   grid, which is **measured to be almost lossless** (2x +0.03 dB, 4x ±0), shrinking the size by a
   further ~20% -- the best-value setting for hard-edged artwork, and the shipped `*_prec0.svgz` is
   exactly this.
3. **Removing duplicate clipPath** (`--drop-clip`): the generator writes every region a clipPath that
   is **byte-for-byte identical** to the region outline (1393 of them in the 1x watercolor image, 28%
   of the size). It shrinks the size the most, but **strokes escape the region they belong to**, and
   the cost grows with the magnification: logo 1x −0.28 dB, 2x −1.94 dB, 4x **−5.74 dB**; watercolor
   only −0.01 dB (small strokes inside large regions mostly stay inside). `--tight` = 2 + 3.

| Processing | logo 2x | logo 4x | Quality |
| --- | --- | --- | --- |
| Original file (`--compress off`) | 479 KB | 614 KB | 40.36 / 43.43 dB |
| `slim` (lossless, shipped default) | **461 KB** | **594 KB** | bit-identical |
| `slim` + gzip (`.svgz`) | 146 KB | 197 KB | lossless |
| **+ rounded strokes** (`_prec0.svgz`) | **98.6 KB** | **161 KB** | 40.39 / 43.43 dB (≈lossless)|
| + clipPath removal (`_tight.svgz`) | **72 KB** | **114 KB** | 38.42 / **37.69 dB** ⚠️ |

Freehand art like watercolor is even more worth compressing: 8.12 MB → `slim`+gzip **1.90 MB** →
`tight`+gzip **0.92 MB** (8.8×, at a cost of −0.01 dB).
**4x high fidelity drops from 6505 KB to 161 KB (40×) without losing any quality** (`slim`+gzip is
33×). Enabling gzip on the server or serving the `.svgz` directly both work.

> ⚠️ Do not use `--tight` together with high magnification: 4x tight is only 37.69 dB -- barely above
> the 37.57 dB of 1x uncompressed -- yet 2.7 dB worse than 2x `slim` at 40.36 dB. The high fidelity
> bought by magnification gets eaten back by "escaping strokes".

## Self-check

```bash
python selfcheck.py            # about 20~60 seconds (depending on machine load); --keep keeps the temp directory
python selfcheck.py --tool-args "--gpu on"    # pass a switch through to the main program to verify the switch itself also passes the self-check (same for --seg flat)
```

It synthesizes a 256² hard-edged image in a temp directory (circle / diagonal line / gradient patch /
high-saturation patch), runs the real subprocess pipeline, and checks:

- The slimming kernel: numeric compaction, idempotence, valid XML, and that `tight` really removes the clipPath;
- Path resolution: bare filename → `inputs/`, automatic naming → `out/`, explicit paths left untouched;
- End to end: exit code, valid SVG XML, all layers present, **the preview rendered back at native size**, matching `.svgz` decompression, PSNR ≥ 21 dB;
- **Interior (gradient-free) MAE ≤ 3/255** -- a robust indicator of "were the regions and gradients drawn correctly" (measures 0.61);
- Compression round-trip: after `slim` / `tight`, render back at native size and compare against the original file's render, mutual comparison ≥ 30 dB.

> Note that the PSNR in the self-check is only around 22 dB: the synthetic image is a 256px
> hard-edged figure, and the 1px band along its boundary is a few percent of all pixels with a
> contrast that spans the full range, so an absolute 1px boundary error is a large fraction on a small
> image. The same tool reaches 40 dB on the 1254px logo.
> So "was it drawn correctly" should be judged by the interior MAE.

## Deliverables list (`out/`)

| File | Description |
| --- | --- |
| `logo_traced.svg` | **Main deliverable**: 461 KB, `width/height=1254` + `viewBox="0 0 2508 2508"`, self-contained with no external links |
| `logo_traced.svgz` / `_prec0.svgz` / `_tight.svgz` | gzip **146 KB** / rounded strokes **98.6 KB** (≈lossless, 40.39 dB) / tight **72 KB** (−1.9 dB)|
| `logo_traced_preview.png` | Preview rendered back at native 1254² (MAE 0.67/255, **40.36 dB**) |
| `logo_traced_preview_2x.png` | Preview at the 2x grid resolution (2508²), for inspecting detail |
| `logo_traced_scale4.svg` / `.svgz` / `_prec0.svgz` / `_tight.svgz` | High-fidelity variant: 594 KB / **197 KB** / **161 KB** (≈lossless) / 114 KB (−5.7 dB ⚠️), rendering back at native size to **43.43 dB** |
| `logo_traced_scale4_preview.png` | Preview of the above rendered back at native 1254² |
| `logo_traced_scale4_nostrokes.svg` / `.svgz` / `_preview.png` | 4x **geometric layer only** (no strokes): 223 KB / 73 KB, **43.08 dB** -- to see how much "grid refinement" contributes on its own |
| `logo_scale_curve.png` | 1 / 1.5 / 2 / 4x tracing curve: 3x zoom at the golden seam line + error ×8 |
| `logo_compare.png` | Comparison: full image / three 3x zooms + error ×8 (original method vs partition routing + 2x grid) |
| `logo_tensor_debug.png` | Structure tensor diagnostics: short segments = isophotes (stroke direction), hue = coherence, green = region boundaries |
| `water-lilies-29_traced.svg` | Watercolor detail preset: 8.12 MB (2064 regions + 9679 strokes), 24.00 dB |
| `water-lilies-29_traced.svgz` / `_tight.svgz` | **1.90 MB** / **0.92 MB** |
| `water-lilies-29_traced_scale15.svg` / `.svgz` / `_tight.svgz` | 1.5x variant: 18.07 MB / 4.20 MB / **2.06 MB**, 24.71 dB |
| `water-lilies-29_preview.png`, `_preview_2x.png`, `_scale15_preview.png`, `_compare.png` | Watercolor previews and comparison figures |

Timings (4 threads): **the load average on this machine sits at 197~200 all year (192 cores), and the
same code with the same parameters has measured anywhere from 10 s to 115 s**, so both a "quiet
window" and a "high load" setting are given here; do not read the high-load numbers as the
algorithmic cost:

| Task | Quiet window | High load (other jobs running at the same time) |
| --- | --- | --- |
| logo 1× (`--scale 1`) | **10.3 s** (tensor 2.1 / k-means 1.0 / segmentation to 5.1 / strokes to 9.4) | 70–120 s |
| logo 2× (`--scale 2`) | not measured separately | 347 s (428 s with four concurrent jobs) |
| logo 4× (5016²) | — | about 22 minutes |
| watercolor 1× | not measured separately | 922 s (segmentation 278 s, region vectors 794.7 s, strokes 911.9 s) |
| watercolor 1.5× | — | about 79 minutes |

`--gpu on` in a quiet window takes logo 1× from 10.3 s down to 8.6 s (see the "Speed" section above).

## Which route to use when

- **Hard-edged flat artwork** (logos, icons, QR codes, line art): `--preset logo`. Partition routing
  classifies 96.8% of the pixels as geometric regions, automatically taking the subpixel snapping +
  conformal Bezier route; `--method edges` also suits this kind of image (fewer, cleaner regions) but
  with lower color fidelity than partition routing.
- **Photos / oil paintings / watercolors**: `--preset painting`. Partition routing classifies 0% of the
  regions as geometric, so everything goes through color segmentation + Catmull-Rom + streamline
  strokes; the gains come from **detail density** (`k`, `min_area`, `spacing`), not boundary accuracy.
- **Both small and accurate**: `--preset logo --scale 2 --compress slim --gzip`, then
  `svgzip.py out/logo_traced.svg --prec 0 --svgz` → **98.6 KB / 40.39 dB** (4.8× smaller than the
  original file, with no loss of quality).
- **One notch more accurate**: `--scale 4` (+3.07 dB, 594 KB, about 22 minutes under high load).

## Limitations and caveats

- **Vector boundaries are not feathered**: the source image's anti-aliasing ramp is approximated with
  hard-edged fills, so error necessarily remains in the boundary band; going an order of magnitude
  lower would require giving the vector boundaries transparency/feathering (SVG's `mask` / `opacity`),
  which this tool does not do.
- **PSNR is naturally low on small images**: an absolute 1px boundary error is 0.4% of the frame on a
  256px image and 0.08% on a 1254px image, so the same tool only reaches the low 20s dB on small
  images. On small images prefer the interior MAE, or add `--scale` to raise the relative accuracy.
- **What `--scale` buys is finer boundary geometry, not new detail**: it traces on the Lanczos-upscaled
  image. A source image with enough pixels (or an SVG/PDF source) always beats upscaling after the
  fact.
- **Use `--min-width` with care**: the label opening eats genuinely thin, long features (such as
  golden seam lines).
- **`--grad-mesh` output needs an SVG 2 mesh-capable renderer**: browsers and cairosvg ignore
  `<meshgradient>`, and they do not implement SVG 2's paint fallback list either, so without help they
  would drop those regions completely. The tool always emits a plain fallback path underneath
  (`--grad-mesh-underlay`, on by default), so non-mesh renderers still paint the region correctly --
  the mesh is simply not shown in them. Mesh gradients remain off unless `--grad-mesh` is given.
- **`--aa-edge-canon` is a per-image opt-in and is OFF by default**: it is the aggressive variant of
  the AA dedup that deletes real-edge transition bands, keeping only the largest descendant per
  original region; enable it only after checking the result on that specific image.
- **Keep the thread cap on a shared machine**: 4 by default; using all 192 cores makes BLAS thread
  oversubscription stall a single step for 60 s.
- The 1.5x watercolor variant was run with the **old defaults** (stroke spacing in grid units, 21230
  strokes); rerunning it with the current defaults would cut the size noticeably (estimated on the
  order of ~10 MB, not measured).

---

## Going deeper

The algorithm (structure tensor / edge detection / subpixel snapping / conformal Beziers / partition
routing criteria) and all the ablation experiments (method comparisons, error decomposed by distance
to edge, 2D fit-smooth sweeps, tracing-grid curves, per-layer byte attribution, compression
measurements) are collected in **[docs/experiments.md](docs/experiments.md)**.
