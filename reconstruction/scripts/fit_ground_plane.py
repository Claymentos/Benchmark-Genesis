#!/usr/bin/env python3
"""
Fit the supporting plane (table) from a metric depth map, in camera coordinates.

The result gives two things the rest of the pipeline needs: the real up direction,
which is tens of degrees off the camera's own up axis for a downward-looking
recording, and the surface objects rest on.

Run in any env with numpy/opencv:
    python fit_ground_plane.py --depth 0049_depth_mm_moge.png --intrinsics intrinsics.yaml \\
        --exclude masks/cup.png masks/left_hand_0.png --out ground_plane.json
"""

import argparse
import json

import cv2
import numpy as np
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Fit the supporting plane from depth")
    parser.add_argument("--depth", required=True, help="uint16-millimetre depth PNG")
    parser.add_argument("--intrinsics", required=True, help="YAML with fu/fv/pu/pv")
    parser.add_argument("--exclude", nargs="*", default=[], help="Masks to exclude (objects, hands)")
    parser.add_argument("--out", required=True, help="Output JSON")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--threshold", type=float, default=0.01, help="Inlier distance, metres")
    parser.add_argument("--max-depth", type=float, default=2.0, help="Ignore points beyond this, metres")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.intrinsics) as handle:
        calibration = yaml.safe_load(handle)
    fu, fv, pu, pv = (float(calibration[k]) for k in ("fu", "fv", "pu", "pv"))

    depth = cv2.imread(args.depth, cv2.IMREAD_UNCHANGED).astype(np.float64) / 1000.0
    height, width = depth.shape
    ys, xs = np.mgrid[0:height, 0:width]
    points = np.stack(
        [(xs - pu) * depth / fu, (ys - pv) * depth / fv, depth], axis=-1
    ).reshape(-1, 3)

    keep = np.isfinite(points).all(axis=1) & (points[:, 2] > 0.1) & (points[:, 2] < args.max_depth)
    for mask_path in args.exclude:
        mask = cv2.imread(mask_path, 0)
        if mask is not None:
            keep &= ~(mask > 128).reshape(-1)
    points = points[keep]

    rng = np.random.default_rng(0)
    best_count, best_normal = -1, None
    for _ in range(args.iterations):
        sample = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal = normal / norm
        offset = -normal @ sample[0]
        count = int((np.abs(points @ normal + offset) < args.threshold).sum())
        if count > best_count:
            best_count, best_normal, best_offset = count, normal, offset

    inliers = np.abs(points @ best_normal + best_offset) < args.threshold
    centred = points[inliers] - points[inliers].mean(axis=0)
    # Refit on the inliers: the minimum-variance direction of the supporting points
    # is a far better normal than the one from the three-point sample that found them.
    normal = np.linalg.svd(centred, full_matrices=False)[2][2]
    if normal[1] > 0:  # camera y points down, so up has negative y
        normal = -normal
    offset = -normal @ points[inliers].mean(axis=0)

    result = {
        "normal_camera": normal.tolist(),
        "offset": float(offset),
        "inlier_fraction": float(inliers.mean()),
        "depth": args.depth,
    }
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=2)

    tilt = np.degrees(np.arccos(np.clip(normal @ np.array([0.0, -1.0, 0.0]), -1, 1)))
    print(f"Wrote {args.out}")
    print(f"  normal (camera frame): {np.round(normal, 4).tolist()}")
    print(f"  inliers: {inliers.mean() * 100:.0f}% of {len(points)} points")
    print(f"  camera pitch from level: {tilt:.1f} deg")


if __name__ == "__main__":
    main()
