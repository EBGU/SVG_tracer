"""GPU backend re-exports: cupy helpers plus the self-built CUDA kernel in gpu_backend/cuda_backend."""
from __future__ import annotations

import gpu_backend as _backend
from gpu_backend import (  # noqa: F401
    available,
    device_name,
    enabled,
    has_kmeans,
    lloyd,
    structure_tensor,
)

__all__ = ["available", "device_name", "enabled", "has_kmeans", "lloyd",
           "structure_tensor", "_backend"]
