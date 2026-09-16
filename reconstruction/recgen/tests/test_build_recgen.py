"""Registry / config tests for recgen_inference.build_recgen.
"""

import pytest

from reconstruction.recgen.recgen_inference import build_recgen


def test_list_models_non_empty():
    models = build_recgen.list_models()
    assert isinstance(models, list)
    assert len(models) > 0
    assert "recgen_base" in models


def test_list_checkpoints_non_empty():
    ckpts = build_recgen.list_checkpoints()
    assert isinstance(ckpts, list)
    assert len(ckpts) > 0
    assert "recgen_base.multiview_stereo" in ckpts


def test_get_config_full_name():
    cfg = build_recgen.get_config("recgen_base.multiview_stereo")
    assert cfg is not None
    assert cfg["hf_repo"] == "TRI-ML/RecGen"
    assert cfg["hf_subdir"] == ""
    for key in ("slat_ckpt", "sparse_ckpt", "slat_config", "sparse_config"):
        assert key in cfg