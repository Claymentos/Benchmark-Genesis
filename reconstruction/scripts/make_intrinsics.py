#!/usr/bin/env python3
"""
Write the camera intrinsics YAML that RecGen and the tracking stages read.

Previously this fell out of the sam-3d-objects pointmap stage; with MoGe supplying
depth directly, the intrinsics come from the recording's own calibration instead of
being re-estimated, which is both more accurate and one less model to run.

    python make_intrinsics.py --meta take001_meta.json --out intrinsics.yaml
    python make_intrinsics.py --fx 530.28 --fy 530.28 --cx 641.34 --cy 354.53 --width 1280 --out intrinsics.yaml

Prints the horizontal FOV, which MoGe wants in order to lock its focal length to the
real camera rather than guessing one.
"""

import argparse
import json
import math


def parse_args():
    parser = argparse.ArgumentParser(description="Write intrinsics YAML for the pipeline")
    parser.add_argument("--meta", default=None, help="ZED *_meta.json with an intrinsics_left block")
    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--fy", type=float, default=None)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)
    parser.add_argument("--width", type=int, default=None, help="Image width, for the FOV when --meta is absent")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    horizontal_fov = None
    if args.meta:
        with open(args.meta) as handle:
            intrinsics = json.load(handle)["intrinsics_left"]
        fx, fy, cx, cy = (intrinsics[k] for k in ("fx", "fy", "cx", "cy"))
        horizontal_fov = intrinsics.get("h_fov")
    else:
        missing = [n for n, v in (("fx", args.fx), ("fy", args.fy), ("cx", args.cx), ("cy", args.cy)) if v is None]
        if missing:
            raise SystemExit(f"provide --meta, or all of fx/fy/cx/cy (missing: {', '.join(missing)})")
        fx, fy, cx, cy = args.fx, args.fy, args.cx, args.cy

    if horizontal_fov is None:
        width = args.width or int(round(cx * 2))
        horizontal_fov = math.degrees(2 * math.atan(width / (2 * fx)))

    with open(args.out, "w") as handle:
        handle.write(f"fu: {fx}\nfv: {fy}\npu: {cx}\npv: {cy}\n")

    print(f"Wrote {args.out}")
    print(f"FOV_X={horizontal_fov}")
    print(f"FOCAL={fx}")


if __name__ == "__main__":
    main()
