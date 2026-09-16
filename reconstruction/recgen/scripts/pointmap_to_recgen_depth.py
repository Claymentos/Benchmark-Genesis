"""Convert a do-as-i-do MoGe pointmap + intrinsics.txt into RecGen's input format.

do-as-i-do/reconstruction's own stage 2 already computes a MoGe pointmap and
intrinsics for the reference frame (`scripts/get_pointmap_dir.py`, producing
`<n>_pointmap.npy` + `<n>_intrinsics.txt`) before generate_mesh_sam3d.py ever
runs. Reuse that instead of re-running depth estimation: this script just
reshapes it into what `scripts/run_inference.py` expects (uint16-mm depth
PNG + a `fu/fv/pu/pv` YAML).

The pointmap's Z channel is used directly as depth (forward distance in the
"R3 camera space" convention, see recgen_mesh_to_layout.py's docstring) --
this assumes the packaged MoGe variant produces metric-scale points. Sanity
check this the first time you use it (e.g. via
`check_mask_depth_coverage.py`-style validation, or just eyeballing the
`overlay.png` RecGen produces) since do-as-i-do's Fast-SAM3D-bundled MoGe
checkpoint may differ from the public `Ruicheng/moge-*` ones.

Example:
    python scripts/pointmap_to_recgen_depth.py \
        --pointmap /path/to/video_dir/0028_pointmap.npy \
        --intrinsics-txt /path/to/video_dir/0028_intrinsics.txt \
        --out-depth /path/to/video_dir/0028_depth_mm.png \
        --out-intrinsics-yaml /path/to/video_dir/0028_intrinsics.yaml
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert a do-as-i-do MoGe pointmap into RecGen's depth+intrinsics input")
    p.add_argument("--pointmap", required=True, help="Path to <n>_pointmap.npy")
    p.add_argument("--intrinsics-txt", required=True, help="Path to <n>_intrinsics.txt (fx, fy, cx, cy; one per line)")
    p.add_argument("--out-depth", required=True, help="Output path for the uint16-mm depth PNG")
    p.add_argument("--out-intrinsics-yaml", required=True, help="Output path for the fu/fv/pu/pv YAML")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    pointmap = np.load(args.pointmap)
    pointmap = np.squeeze(pointmap)
    if pointmap.ndim != 3 or pointmap.shape[-1] != 3:
        raise ValueError(
            f"Expected a (H, W, 3) pointmap after squeezing leading batch dims, got shape {pointmap.shape}. "
            f"Check get_pointmap_dir.py's current output layout."
        )
    depth = pointmap[..., 2].astype(np.float32)  # forward distance, metres (assumed -- see module docstring)

    finite = np.isfinite(depth) & (depth > 0)
    print(f"[pointmap_to_recgen_depth] valid px: {finite.sum()} / {depth.size}")
    print(f"[pointmap_to_recgen_depth] depth range (valid): "
          f"{depth[finite].min() if finite.any() else None} .. {depth[finite].max() if finite.any() else None}")

    depth_mm = np.where(finite, depth * 1000.0, 0.0)
    depth_mm = np.clip(depth_mm, 0, 65535).astype(np.uint16)

    out_depth = Path(args.out_depth)
    out_depth.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_depth), depth_mm)
    print(f"[pointmap_to_recgen_depth] saved: {out_depth}")

    with open(args.intrinsics_txt) as f:
        fx, fy, cx, cy = (float(line.strip()) for line in f if line.strip())

    out_yaml = Path(args.out_intrinsics_yaml)
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(out_yaml, "w") as f:
        yaml.safe_dump({"fu": fx, "fv": fy, "pu": cx, "pv": cy}, f)
    print(f"[pointmap_to_recgen_depth] saved: {out_yaml}")


if __name__ == "__main__":
    main()
