"""Multi-scale structure tensor: Di Zenzo multi-channel gradients, eigen-decomposition, coherence and isophote orientation."""
from __future__ import annotations


import numpy as np
from scipy import ndimage as ndi

from .state import EPS

# ======================================================================
# structure tensor
# ======================================================================
def structure_tensor(rgb: np.ndarray, sigma_d: float, sigma_i: float,
                     color: bool = True) -> dict:
    """Multi-channel (Di Zenzo) structure tensor and its eigendecomposition.

    Returns l1, l2, coh, energy(=l1), gx/gy (principal eigenvector = gradient direction),
        tx/ty (minor eigenvector = stroke/isophote direction)
    """
    chans = [rgb[..., i] for i in range(3)] if color else [rgb.mean(2)]
    jxx = np.zeros(rgb.shape[:2], np.float64)
    jyy = np.zeros_like(jxx)
    jxy = np.zeros_like(jxx)
    for c in chans:
        gx = ndi.gaussian_filter(c, sigma_d, order=(0, 1), mode="nearest")
        gy = ndi.gaussian_filter(c, sigma_d, order=(1, 0), mode="nearest")
        jxx += ndi.gaussian_filter(gx * gx, sigma_i, mode="nearest")
        jyy += ndi.gaussian_filter(gy * gy, sigma_i, mode="nearest")
        jxy += ndi.gaussian_filter(gx * gy, sigma_i, mode="nearest")
    n_ch = len(chans)
    jxx /= n_ch
    jyy /= n_ch
    jxy /= n_ch

    tr = jxx + jyy
    dif = jxx - jyy
    tmp = np.sqrt(dif * dif + 4.0 * jxy * jxy)
    l1 = 0.5 * (tr + tmp)
    l2 = 0.5 * (tr - tmp)

    # Principal eigenvector: of the two candidates take the one with the larger magnitude, to avoid degeneracy
    ax, ay = jxy, l1 - jxx
    bx, by = l1 - jyy, jxy
    use_b = np.hypot(bx, by) > np.hypot(ax, ay)
    vx = np.where(use_b, bx, ax)
    vy = np.where(use_b, by, ay)
    nrm = np.hypot(vx, vy)
    nrm[nrm < 1e-12] = 1.0
    vx /= nrm
    vy /= nrm

    luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    coh = (l1 - l2) / (l1 + l2 + EPS)
    return {"l1": l1, "l2": l2, "coh": coh, "energy": l1, "luma": luma,
            "gx": vx, "gy": vy, "tx": -vy, "ty": vx}


