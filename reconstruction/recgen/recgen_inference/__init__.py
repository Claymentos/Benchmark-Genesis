"""recgen_inference: single-view and multi-view 3D reconstruction from RGB-D.

Quick start:
    >>> from recgen_inference import build_recgen, generate
    >>> pipeline = build_recgen.build("recgen_base.multiview_stereo")
    >>> result = generate(pipeline, rgb, depth, mask, intrinsics)
    >>> result.save("./out")
"""

from reconstruction.recgen.recgen_inference import build_recgen
from reconstruction.recgen.recgen_inference._result import RecGenResult
from reconstruction.recgen.recgen_inference.inference import generate, generate_multiview

__version__ = "0.1.0"

__all__ = [
    "build_recgen",
    "generate",
    "generate_multiview",
    "RecGenResult",
    "__version__",
]
