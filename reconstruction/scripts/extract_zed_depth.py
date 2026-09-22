#!/usr/bin/env python3
"""
Extract per-frame metric depth from a ZED .svo2 recording.

Writes uint16-millimetre PNGs, the same format run_moge_depth_batch.py produces, so
every downstream stage takes this as a drop-in replacement. Prefer it whenever the
recording is an SVO: MoGe infers depth from a single image, while this is what the
stereo camera actually measured. The sidecar `*_depth.mp4` is not a substitute -- it
is an 8-bit colour-mapped visualisation with 256 distinct values.

Needs pyzed, which lives in the system python rather than a conda env:
    python extract_zed_depth.py --svo take001.svo2 --out-dir depth_zed
"""

import argparse
import os

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="ZED SVO -> uint16-mm depth PNGs")
    parser.add_argument("--svo", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--depth-mode",
        default="NEURAL",
        choices=["NEURAL", "NEURAL_PLUS", "ULTRA", "QUALITY", "PERFORMANCE"],
        help="Match the mode the recording was made with (see the meta JSON)",
    )
    parser.add_argument("--fill", action="store_true", help="Let the SDK fill depth holes")
    return parser.parse_args()


def main():
    args = parse_args()

    import cv2
    import pyzed.sl as sl

    init = sl.InitParameters()
    init.set_from_svo_file(args.svo)
    init.depth_mode = getattr(sl.DEPTH_MODE, args.depth_mode)
    init.coordinate_units = sl.UNIT.MILLIMETER
    init.depth_minimum_distance = 100.0

    camera = sl.Camera()
    status = camera.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise SystemExit(f"could not open {args.svo}: {status}")

    runtime = sl.RuntimeParameters()
    runtime.enable_fill_mode = args.fill

    os.makedirs(args.out_dir, exist_ok=True)
    total = camera.get_svo_number_of_frames()
    measure = sl.Mat()
    written = 0

    for index in range(total):
        camera.set_svo_position(index)
        if camera.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            continue
        camera.retrieve_measure(measure, sl.MEASURE.DEPTH)
        depth = measure.get_data()
        finite = np.isfinite(depth) & (depth > 0)
        depth_mm = np.clip(np.where(finite, depth, 0.0), 0, 65535).astype(np.uint16)
        cv2.imwrite(os.path.join(args.out_dir, f"{index:06d}.png"), depth_mm)
        written += 1
        if index % 50 == 0 or index == total - 1:
            print(f"[zed] {index + 1}/{total} valid {100 * finite.mean():.0f}%", flush=True)

    camera.close()
    print(f"[zed] wrote {written} depth maps to {args.out_dir}")


if __name__ == "__main__":
    main()
