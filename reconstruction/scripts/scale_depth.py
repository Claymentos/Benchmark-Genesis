#!/usr/bin/env python3
"""
Rescale a directory of uint16-mm depth PNGs by one factor, keeping the original.

Monocular depth (MoGe) is metric only up to a per-clip scale error. When the clip also
has a measured metric -- TouchAnything's mocap hands -- touchanything_hands.py
reports the scale s that takes the measured world into the depth's units. Dividing the
depth by s puts the whole scene (mesh, plane, camera and object tracks) into true
metres, so the hands keep their measured size.

The original maps are moved to `<depth-dir>_raw` once, and the scale applied so far is
kept in `<depth-dir>/scale.json`. A new --scale is relative to the depth as it is now,
so scales compose: a second pass with the refit's s ~ 1 refines the first rather than
undoing it.

Anything already computed from the depth can be carried along instead of recomputed,
because a rigid solve on points that all shrink by s shrinks its translation by s and
keeps its rotation:
  --points FILE   point tracks (`points_3d`, track_points_cotracker.py)
  --poses FILE    rigid trajectories (`translation`, pose_from_points.py and the like)

Run in the `genesis` env:
    python scale_depth.py --depth-dir depth_moge --scale 0.846 \\
        --points table_points.npz --poses camera_traj.npz
"""

import argparse
import json
import os
import shutil

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Rescale depth PNGs into true metric")
    parser.add_argument("--depth-dir", required=True)
    parser.add_argument("--scale", type=float, required=True,
                        help="Depth units per true metre, relative to the depth as it is now")
    parser.add_argument("--points", nargs="*", default=[], help="npz files whose points_3d to rescale")
    parser.add_argument("--poses", nargs="*", default=[], help="npz files whose translation to rescale")
    args = parser.parse_args()

    depth_dir = os.path.abspath(args.depth_dir.rstrip("/"))
    raw_dir = depth_dir + "_raw"
    record = os.path.join(depth_dir, "scale.json")
    applied = 1.0
    if os.path.isdir(raw_dir):
        if os.path.exists(record):
            with open(record) as handle:
                applied = float(json.load(handle)["scale"])
    else:
        shutil.move(depth_dir, raw_dir)
    total = applied * args.scale
    os.makedirs(depth_dir, exist_ok=True)

    names = sorted(f for f in os.listdir(raw_dir) if f.endswith(".png"))
    for name in names:
        depth = cv2.imread(os.path.join(raw_dir, name), cv2.IMREAD_UNCHANGED).astype(np.float64)
        scaled = np.clip(np.round(depth / total), 0, 65535).astype(np.uint16)
        cv2.imwrite(os.path.join(depth_dir, name), scaled)
    with open(record, "w") as handle:
        json.dump({"scale": total, "raw": raw_dir}, handle, indent=2)
    print(f"Wrote {len(names)} maps to {depth_dir}: raw / {total:g} "
          f"(this pass {args.scale:g}, before {applied:g})")

    for path, key in [(p, "points_3d") for p in args.points] + [(p, "translation") for p in args.poses]:
        data = dict(np.load(path))
        data[key] = data[key] / args.scale
        np.savez(path, **data)
        print(f"  {os.path.basename(path)}: {key} / {args.scale:g}")


if __name__ == "__main__":
    main()
