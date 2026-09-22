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
    parser.add_argument(
        "--intrinsics-json",
        default=None,
        help="Calibration JSON keyed by camera id (HRDexDB cam_param/intrinsics.json)",
    )
    parser.add_argument("--camera", default=None, help="Camera id to read from --intrinsics-json")
    parser.add_argument(
        "--undistorted",
        action="store_true",
        help="Frames have already been undistorted, so use the undistort matrix",
    )
    parser.add_argument(
        "--from-moge",
        default=None,
        help="Estimate intrinsics from this image with MoGe, for footage with no calibration. "
        "Less accurate than a real calibration, so prefer --meta when a recording has one.",
    )
    parser.add_argument("--checkpoint", default="Ruicheng/moge-3-vitl")
    parser.add_argument("--version", default="v3", choices=["v1", "v2", "v3"])
    parser.add_argument("--moge-repo", default=None)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def estimate_with_moge(image_path, checkpoint, version, moge_repo):
    """Run MoGe on one image and return its predicted (fx, fy, cx, cy) in pixels."""
    import sys

    import cv2
    import torch

    if moge_repo:
        sys.path.insert(0, moge_repo)
    from moge.model import import_model_class_by_version

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = import_model_class_by_version(version).from_pretrained(checkpoint).to(device).eval()

    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    tensor = torch.tensor(image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)
    with torch.no_grad():
        output = model.infer(tensor)

    # MoGe reports intrinsics normalised by image size.
    intrinsics = output["intrinsics"].cpu().numpy()
    height, width = image.shape[:2]
    return (
        float(intrinsics[0, 0] * width),
        float(intrinsics[1, 1] * height),
        float(intrinsics[0, 2] * width),
        float(intrinsics[1, 2] * height),
    )


def main():
    args = parse_args()

    horizontal_fov = None
    if args.meta:
        with open(args.meta) as handle:
            intrinsics = json.load(handle)["intrinsics_left"]
        fx, fy, cx, cy = (intrinsics[k] for k in ("fx", "fy", "cx", "cy"))
        horizontal_fov = intrinsics.get("h_fov")
    elif args.intrinsics_json:
        with open(args.intrinsics_json) as handle:
            calibration = json.load(handle)
        if args.camera not in calibration:
            raise SystemExit(
                f"camera {args.camera!r} not in {args.intrinsics_json}; "
                f"available: {sorted(calibration)[:8]}"
            )
        entry = calibration[args.camera]
        matrix = entry["intrinsics_undistort" if args.undistorted else "original_intrinsics"]
        fx, fy = matrix[0][0], matrix[1][1]
        cx, cy = matrix[0][2], matrix[1][2]
        args.width = args.width or entry.get("width")
        distortion = entry.get("dist_params")
        if distortion and not args.undistorted and abs(distortion[0]) > 0.05:
            print(
                f"note: k1={distortion[0]:.3f} and the frames are not undistorted. "
                "The pinhole model this writes is exact only near the image centre.",
            )
    elif args.from_moge:
        fx, fy, cx, cy = estimate_with_moge(
            args.from_moge, args.checkpoint, args.version, args.moge_repo
        )
        args.width = args.width or int(round(cx * 2))
    else:
        missing = [n for n, v in (("fx", args.fx), ("fy", args.fy), ("cx", args.cx), ("cy", args.cy)) if v is None]
        if missing:
            raise SystemExit(
                f"provide --meta, --from-moge, or all of fx/fy/cx/cy (missing: {', '.join(missing)})"
            )
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
