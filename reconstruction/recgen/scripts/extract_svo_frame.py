"""Extract a single RGB + metric depth frame from a ZED .svo2 recording.

The exported `*_depth.mp4` from `zed_record.py --depth-video` is an 8-bit
colorized visualization (lossy, not metric) and cannot be used as RecGen
depth input. This script re-runs the ZED SDK's depth engine on the raw
stereo pair stored in the .svo2 to recover real per-pixel metric depth.

Requires `pyzed` (ZED SDK Python bindings) matching the SDK version the
recording was made with (see `*_meta.json` -> `sdk_version`).

Example:
    python scripts/extract_svo_frame.py \
        --svo /path/to/take001_grasping_cup.svo2 \
        --meta /path/to/take001_grasping_cup_meta.json \
        --frame 220 \
        --out examples/test/outputs/recgen_input
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pyzed.sl as sl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract RGB + metric depth from a ZED .svo2 recording")
    p.add_argument("--svo", required=True, help="Path to the .svo2 file")
    p.add_argument("--meta", required=True, help="Path to the matching *_meta.json (for depth mode/min/max)")
    p.add_argument("--frame", type=int, required=True, help="Frame index to extract")
    p.add_argument("--out", required=True, help="Output directory")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.meta, "r") as f:
        meta = json.load(f)
    depth_cfg = meta["depth"]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_rgb = out_dir / f"frame_{args.frame:06d}_rgb.png"
    out_depth_mm = out_dir / f"frame_{args.frame:06d}_depth_mm.png"

    init = sl.InitParameters()
    init.set_from_svo_file(args.svo)
    init.svo_real_time_mode = False
    init.depth_mode = getattr(sl.DEPTH_MODE, depth_cfg["mode"])  # e.g. "NEURAL", must match the recording
    init.coordinate_units = sl.UNIT.METER
    init.depth_minimum_distance = float(depth_cfg["min"])
    init.depth_maximum_distance = float(depth_cfg["max"])

    cam = sl.Camera()
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"open failed: {status}", file=sys.stderr)
        sys.exit(1)

    cam.set_svo_position(args.frame)
    runtime = sl.RuntimeParameters()
    err = cam.grab(runtime)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"grab failed: {err}", file=sys.stderr)
        sys.exit(1)

    img = sl.Mat()
    depth = sl.Mat()
    cam.retrieve_image(img, sl.VIEW.LEFT)
    cam.retrieve_measure(depth, sl.MEASURE.DEPTH)

    rgb = cv2.cvtColor(img.get_data(), cv2.COLOR_BGRA2RGB)
    depth_m = depth.get_data().astype(np.float32)
    finite = np.isfinite(depth_m) & (depth_m > 0)
    print(f"[extract_svo_frame] valid depth px: {finite.sum()} / {depth_m.size}")

    depth_mm = np.where(finite, depth_m * 1000.0, 0.0)
    depth_mm = np.clip(depth_mm, 0, 65535).astype(np.uint16)

    cv2.imwrite(str(out_rgb), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_depth_mm), depth_mm)
    print(f"[extract_svo_frame] saved: {out_rgb}, {out_depth_mm}")

    cam.close()


if __name__ == "__main__":
    main()
