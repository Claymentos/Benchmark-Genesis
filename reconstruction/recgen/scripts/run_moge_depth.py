"""Estimate metric depth for a single RGB image with MoGe, in RecGen's input format.

Alternative to real sensor depth (e.g. from `extract_svo_frame.py`): useful
when the depth sensor fails on the object (glossy/textureless surfaces often
leave large holes in stereo/ToF depth). MoGe-2/3 checkpoints are metric-scale.

Must be run in an environment with `moge` installed
(https://github.com/microsoft/MoGe), pointed at via `--moge-repo` if it is
not importable directly.

Note: the `moge infer` CLI's `--maps` can crash on environments where
opencv has no OpenEXR writer support; this script uses the Python API and
writes depth as a uint16-millimetre PNG instead, matching what
`scripts/run_inference.py` expects.

Example:
    conda run -n moge python scripts/run_moge_depth.py \
        --rgb examples/test/outputs/recgen_input/frame_000000_rgb.png \
        --out examples/test/outputs/recgen_input/frame_000000_depth_mm_moge.png \
        --fov-x 100.67103576660156 \
        --checkpoint Ruicheng/moge-3-vitl
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Estimate metric depth with MoGe and save it in RecGen's format")
    p.add_argument("--rgb", required=True, help="Path to the input RGB image")
    p.add_argument("--out", required=True, help="Output path for the uint16-mm depth PNG")
    p.add_argument(
        "--fov-x", type=float, default=None,
        help="Known horizontal FOV in degrees (e.g. from a *_meta.json intrinsics_left.h_fov). "
             "Locks MoGe's internal focal length to the real camera instead of letting it guess one, "
             "which matters since RecGen back-projects this depth using the real intrinsics.",
    )
    p.add_argument("--checkpoint", default="Ruicheng/moge-3-vitl", help="HF hub repo id or local path")
    p.add_argument("--version", default="v3", choices=["v1", "v2", "v3"], help="MoGe model version")
    p.add_argument("--refine-steps", type=int, default=3, help="Sparse refinement steps (v3 only)")
    p.add_argument("--device", default="cuda", help="Device name, e.g. 'cuda' or 'cpu'")
    p.add_argument(
        "--moge-repo", default=None,
        help="Path to a local MoGe repo checkout to add to sys.path, if `moge` isn't pip-installed",
    )
    p.add_argument("--out-mask", default=None, help="Optional path to also save MoGe's valid-pixel mask PNG")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.moge_repo:
        sys.path.insert(0, args.moge_repo)

    from moge.model import import_model_class_by_version

    device = torch.device(args.device)
    model_cls = import_model_class_by_version(args.version)
    model = model_cls.from_pretrained(args.checkpoint).to(device).eval()

    image = cv2.cvtColor(cv2.imread(args.rgb), cv2.COLOR_BGR2RGB)
    image_tensor = torch.tensor(image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)

    infer_kwargs = {"fov_x": args.fov_x}
    if args.version == "v3":
        infer_kwargs["refine_steps"] = args.refine_steps

    with torch.no_grad():
        output = model.infer(image_tensor, **infer_kwargs)

    depth = output["depth"].cpu().numpy()   # (H, W) float32 metres
    mask = output["mask"].cpu().numpy()     # (H, W) bool, valid-pixel mask (already applied to `depth`)

    finite = np.isfinite(depth) & mask
    print(f"[run_moge_depth] valid px: {finite.sum()} / {depth.size}")
    print(f"[run_moge_depth] depth range (valid): "
          f"{depth[finite].min() if finite.any() else None} .. {depth[finite].max() if finite.any() else None}")

    depth_mm = np.where(finite, depth * 1000.0, 0.0)
    depth_mm = np.clip(depth_mm, 0, 65535).astype(np.uint16)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), depth_mm)
    print(f"[run_moge_depth] saved: {out_path}")

    if args.out_mask:
        mask_path = Path(args.out_mask)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(mask_path), (mask * 255).astype(np.uint8))
        print(f"[run_moge_depth] saved: {mask_path}")


if __name__ == "__main__":
    main()
