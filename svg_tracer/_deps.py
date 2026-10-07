"""Third-party dependency bootstrap: the single friendly error message for a missing scikit-image."""
from __future__ import annotations

import sys

try:
    from skimage import filters, measure                    # noqa: F401
    from skimage.color import rgb2lab                       # noqa: F401
    from skimage.restoration import denoise_bilateral       # noqa: F401
    from skimage.segmentation import find_boundaries        # noqa: F401
except Exception as exc:  # pragma: no cover - only without scikit-image
    sys.exit(f"[错误] 需要 scipy 与 scikit-image: {exc}")
