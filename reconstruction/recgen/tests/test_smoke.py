"""Smoke tests for recgen_inference.

Level 1 (test_imports): pure import checks, no GPU, no checkpoints.
Level 2 (test_preprocessing): runs preprocessing on real example data, no GPU.
"""

import numpy as np
import pytest
from pathlib import Path

from reconstruction.recgen.recgen_inference import RecGenResult, generate
from reconstruction.recgen.recgen_inference.recgen_modules.utils import pose_utils, random_utils

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


# ---------------------------------------------------------------------------
# Level 1: Import tests
# ---------------------------------------------------------------------------

def test_core_imports():
    from reconstruction.recgen.recgen_inference import build_recgen
    from reconstruction.recgen.recgen_inference.preprocessing import (
        preprocess_view,
        pointmap_from_depth,
        apply_mask_erosion,
        crop_to_bounding_box,
        normalize_depth,
    )


def test_pipeline_imports():
    from reconstruction.recgen.recgen_inference.recgen_modules.pipelines import RecGenPipeline
    from reconstruction.recgen.recgen_inference.recgen_modules.pipelines.samplers import (
        FlowEulerSampler,
        FlowEulerCfgSampler,
        FlowEulerGuidanceIntervalSampler,
    )


def test_module_imports():
    import reconstruction.recgen.recgen_inference.recgen_modules.modules.sparse as sp
    from reconstruction.recgen.recgen_inference.recgen_modules.modules.sparse.transformer import (
        ModulatedSparseTransformerCrossBlock,
        SparseTransformerBlock,
    )
    from reconstruction.recgen.recgen_inference.recgen_modules.modules.norm import LayerNorm32
    from reconstruction.recgen.recgen_inference.recgen_modules.modules.utils import zero_module
    from reconstruction.recgen.recgen_inference.recgen_modules.modules.spatial import patchify, unpatchify
    from reconstruction.recgen.recgen_inference.recgen_modules.modules.transformer import (
        AbsolutePositionEmbedder,
        ModulatedTransformerCrossBlock,
    )


def test_model_imports():
    from reconstruction.recgen.recgen_inference.recgen_modules.models.sparse_structure_pose_flow import SparseStructurePoseFlowModel
    from reconstruction.recgen.recgen_inference.recgen_modules.models.structured_latent_cond_flow import SLatCondFlowModel
    from reconstruction.recgen.recgen_inference.recgen_modules.models.sparse_elastic_mixin import SparseTransformerElasticMixin
    from reconstruction.recgen.recgen_inference.recgen_modules.models.sparse_structure_flow import TimestepEmbedder


def test_utils_imports():
    from reconstruction.recgen.recgen_inference.recgen_modules.utils import render_utils
    from reconstruction.recgen.recgen_inference.recgen_modules.utils.render_utils import render_video


def test_no_top_level_trellis_or_recgen():
    """The restructure moved trellis under recgen_inference. Old top-level names
    must not leak back in via site-packages or namespace packages."""
    import importlib
    for name in ("trellis", "recgen"):
        try:
            importlib.import_module(name)
        except ImportError:
            continue
        raise AssertionError(
            f"{name!r} is importable at the top level; the restructure should have nested it under recgen_inference"
        )


# ---------------------------------------------------------------------------
# Level 2: Preprocessing tests (no GPU)
# ---------------------------------------------------------------------------

def _load_example(idx=0):
    from PIL import Image
    rgb = np.array(Image.open(EXAMPLES_DIR / f"ex{idx}_rgb.png").convert("RGB"))
    depth_img = Image.open(EXAMPLES_DIR / f"ex{idx}_depth.png")
    depth = np.array(depth_img).astype(np.float32)
    if depth.max() > 100:
        depth = depth / 1000.0
    mask = np.array(Image.open(EXAMPLES_DIR / f"ex{idx}_mask.png"))
    if mask.max() == 1:
        mask = mask * 255
    return rgb, depth, mask


def _default_intrinsics(h, w):
    fx = fy = max(h, w)
    cx, cy = w / 2.0, h / 2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)


@pytest.mark.skipif(not EXAMPLES_DIR.exists(), reason="examples/ not found")
def test_pointmap_from_depth():
    from reconstruction.recgen.recgen_inference.preprocessing import pointmap_from_depth

    rgb, depth, mask = _load_example(0)
    h, w = depth.shape
    intrinsics = _default_intrinsics(h, w)

    pointmap = pointmap_from_depth(depth, intrinsics)
    assert pointmap.shape == (h, w, 3), f"Expected ({h},{w},3), got {pointmap.shape}"
    assert np.isfinite(pointmap[depth > 0]).all(), "Non-finite values in pointmap"


@pytest.mark.skipif(not EXAMPLES_DIR.exists(), reason="examples/ not found")
def test_mask_erosion():
    from reconstruction.recgen.recgen_inference.preprocessing import apply_mask_erosion

    _, _, mask = _load_example(0)
    eroded, info = apply_mask_erosion(mask, enabled=True)
    assert eroded.dtype == np.uint8
    assert set(np.unique(eroded)).issubset({0, 255})
    assert info["pixels_after"] <= info["pixels_before"]


@pytest.mark.skipif(not EXAMPLES_DIR.exists(), reason="examples/ not found")
def test_preprocess_view():
    from reconstruction.recgen.recgen_inference.preprocessing import preprocess_view

    rgb, depth, mask = _load_example(0)
    h, w = depth.shape
    intrinsics = _default_intrinsics(h, w)

    result = preprocess_view(
        rgb, depth, mask, intrinsics,
        quantile_drop_threshold=0.05,
    )

    assert "image" in result
    assert "pointmap" in result
    assert "mask" in result
    assert "cam2ncam" in result
    assert result["cam2ncam"].shape == (4, 4)
    assert result["pointmap"].shape[0] == 3


@pytest.mark.skipif(not EXAMPLES_DIR.exists(), reason="examples/ not found")
def test_normalize_depth_auto_detects_units():
    """uint16 mm and float32 m should produce the same metric depth map."""
    from reconstruction.recgen.recgen_inference.preprocessing import normalize_depth
    from PIL import Image

    depth_uint16 = np.array(Image.open(EXAMPLES_DIR / "ex0_depth.png"))
    assert depth_uint16.dtype == np.uint16 or depth_uint16.max() > 100

    depth_m_from_uint = normalize_depth(depth_uint16)
    depth_m_from_float = normalize_depth(depth_m_from_uint.astype(np.float32))

    assert depth_m_from_uint.dtype == np.float32
    assert 0.1 < depth_m_from_uint[depth_m_from_uint > 0].max() < 10.0
    np.testing.assert_allclose(depth_m_from_uint, depth_m_from_float, rtol=1e-6)
