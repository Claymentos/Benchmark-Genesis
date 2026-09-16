"""Rank frames by how much valid ZED depth falls inside the object mask.

Before picking a frame for RecGen, check this: a mask can be large and
clean while the depth sensor still has almost no valid samples under it
(e.g. a glossy/white/textureless object defeats stereo/NEURAL depth). Using
such a frame silently produces a degenerate near-zero-scale pose in RecGen
instead of an error, so it's worth screening for depth coverage first.

Example:
    python scripts/check_mask_depth_coverage.py \
        --svo /path/to/take001_grasping_cup.svo2 \
        --meta /path/to/take001_grasping_cup_meta.json \
        --masks-dir examples/test/outputs/masks \
        --mask-name cup.png \
        --min-mask-px 5000 \
        --top 15
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pyzed.sl as sl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rank frames by valid-depth coverage under their object mask")
    p.add_argument("--svo", required=True, help="Path to the .svo2 file")
    p.add_argument("--meta", required=True, help="Path to the matching *_meta.json (for depth mode/min/max)")
    p.add_argument(
        "--masks-dir", required=True,
        help="Directory of frame_NNNNNN_masks/ subfolders (as produced by a SAM2 tracking run)",
    )
    p.add_argument("--mask-name", default="cup.png", help="Mask filename inside each frame_NNNNNN_masks/ folder")
    p.add_argument("--min-mask-px", type=int, default=0, help="Skip frames whose mask has fewer than this many pixels")
    p.add_argument("--top", type=int, default=15, help="How many top frames to print")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.meta, "r") as f:
        meta = json.load(f)
    depth_cfg = meta["depth"]

    init = sl.InitParameters()
    init.set_from_svo_file(args.svo)
    init.svo_real_time_mode = False
    init.depth_mode = getattr(sl.DEPTH_MODE, depth_cfg["mode"])
    init.coordinate_units = sl.UNIT.METER
    init.depth_minimum_distance = float(depth_cfg["min"])
    init.depth_maximum_distance = float(depth_cfg["max"])

    cam = sl.Camera()
    cam.open(init)
    runtime = sl.RuntimeParameters()
    depth_mat = sl.Mat()

    frame_dirs = sorted(Path(args.masks_dir).glob("frame_*_masks"))
    results = []
    for frame_dir in frame_dirs:
        mask_path = frame_dir / args.mask_name
        m = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if m is None:
            continue
        mask_bin = m > 0
        npix = int(mask_bin.sum())
        if npix < args.min_mask_px:
            continue

        idx = int(frame_dir.name.split("_")[1])
        cam.set_svo_position(idx)
        if cam.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            continue
        cam.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
        d = depth_mat.get_data()
        finite = np.isfinite(d) & (d > 0)
        valid_in_mask = int((finite & mask_bin).sum())
        results.append((idx, npix, valid_in_mask, valid_in_mask / npix))

    cam.close()

    results.sort(key=lambda x: -x[3])
    print(f"top {args.top} frames by fraction of mask with valid depth (min_mask_px={args.min_mask_px}):")
    for idx, npix, valid, frac in results[: args.top]:
        print(f"  frame {idx:6d}  mask_px={npix:7d}  valid_depth={valid:7d}  frac={frac:.3f}")


if __name__ == "__main__":
    main()
