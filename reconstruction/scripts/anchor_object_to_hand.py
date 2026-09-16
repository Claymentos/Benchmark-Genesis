#!/usr/bin/env python3
"""
Re-anchor a tracked object to the HaWoR hand, via hand-calibrated depth.

Port of do-as-i-do's optimize_translation_scale.py to this pipeline's data. The
object's absolute depth from a monocular depth map is the least trustworthy part of
the reconstruction -- it drifts badly once the object is lifted towards the camera,
which separates hand and object even though both are individually plausible. The
hand, by contrast, comes from HaWoR with a metric articulated prior.

So the hand is treated as ground truth and the depth map is used only for the
*relative* hand-to-object offset, rescaled per frame:

    k       = hand_z(HaWoR) / hand_z(depth)     per-frame depth correction
    obj_pos = hand(HaWoR) + k * (obj(depth) - hand(depth))

Run in the `genesis` env (needs trimesh + scipy):
    python anchor_object_to_hand.py --object-traj cup_traj.npz --hand-meshes all_hand_meshes.npz \\
        --mesh recgen_out/cup/posed_mesh.obj --depth-dir depth_moge --masks-root .../masks \\
        --object cup --intrinsics intrinsics.yaml --ground-plane ground_plane.json --out cup_traj.npz
"""

import argparse
import json
import os

import numpy as np
import yaml

from track_object_v2s2r import apply_ground_constraint


def parse_args():
    parser = argparse.ArgumentParser(description="Anchor object depth to the HaWoR hand")
    parser.add_argument("--object-traj", required=True, help="cup_traj.npz from track_object_v2s2r.py")
    parser.add_argument("--hand-meshes", required=True, help="HaWoR all_hand_meshes.npz")
    parser.add_argument("--side", default="left", choices=["left", "right"])
    parser.add_argument("--mesh", required=True, help="Object mesh in reference-frame camera coords")
    parser.add_argument("--depth-dir", required=True)
    parser.add_argument("--masks-root", required=True)
    parser.add_argument("--object", required=True)
    parser.add_argument("--intrinsics", required=True)
    parser.add_argument("--ground-plane", default=None)
    parser.add_argument("--rest-speed", type=float, default=0.004)
    parser.add_argument("--max-tilt", type=float, default=30.0)
    parser.add_argument("--smooth-window", type=int, default=9)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def back_project(u, v, depth, fu, fv, pu, pv):
    """Back-project integer pixel coordinates through a metric depth map."""
    z = depth[v, u]
    keep = z > 0
    points = np.stack([(u - pu) * z / fu, (v - pv) * z / fv, z], axis=1)
    return points[keep], keep


def visible_vertices(vertices, fu, fv, pu, pv, width, height):
    """Keep the nearest mesh vertex per pixel, approximating the visible surface.

    The depth map only ever sees the front of the hand, so comparing it against all
    vertices -- the back of the hand included -- would bias the depth ratio. A
    z-buffer over the projected vertices is enough separation for a 778-vertex MANO
    mesh, without needing ray-mesh intersection.
    """
    u = np.clip((fu * vertices[:, 0] / vertices[:, 2] + pu).astype(int), 0, width - 1)
    v = np.clip((fv * vertices[:, 1] / vertices[:, 2] + pv).astype(int), 0, height - 1)
    nearest = {}
    for index, (ui, vi, z) in enumerate(zip(u, v, vertices[:, 2])):
        key = (ui, vi)
        if key not in nearest or z < vertices[nearest[key], 2]:
            nearest[key] = index
    keep = np.array(sorted(nearest.values()))
    return keep, u[keep], v[keep]


def main():
    args = parse_args()

    import trimesh

    with open(args.intrinsics) as handle:
        calibration = yaml.safe_load(handle)
    fu, fv, pu, pv = (float(calibration[k]) for k in ("fu", "fv", "pu", "pv"))

    tracked = np.load(args.object_traj)
    rotations = tracked["rotation"].copy()
    translations = tracked["translation"].copy()
    valid = tracked["valid"].astype(bool)
    n_frames = len(valid)

    hand = np.load(args.hand_meshes)
    hand_vertices = hand[f"{args.side}_vertices"]
    hand_valid = np.isfinite(hand_vertices).all(axis=(1, 2))

    centroid = np.asarray(trimesh.load(args.mesh, process=False).vertices).mean(axis=0)

    depth_files = sorted(f for f in os.listdir(args.depth_dir) if f.endswith(".png"))
    anchored = np.zeros(n_frames, dtype=bool)

    import cv2

    for t in range(n_frames):
        if not (valid[t] and hand_valid[t]):
            continue

        object_mask_path = os.path.join(
            args.masks_root, f"frame_{t:06d}_masks", f"{args.object}.png"
        )
        if not os.path.exists(object_mask_path):
            continue
        object_mask = cv2.imread(object_mask_path, 0) > 128
        if object_mask.sum() < 50:
            continue

        depth = cv2.imread(
            os.path.join(args.depth_dir, depth_files[t]), cv2.IMREAD_UNCHANGED
        ).astype(np.float64) / 1000.0
        height, width = depth.shape

        keep, u, v = visible_vertices(hand_vertices[t], fu, fv, pu, pv, width, height)
        hand_depth_points, sampled = back_project(u, v, depth, fu, fv, pu, pv)
        if sampled.sum() < 20:
            continue

        hand_real = np.median(hand_vertices[t][keep][sampled], axis=0)
        hand_from_depth = np.median(hand_depth_points, axis=0)
        if hand_from_depth[2] <= 0:
            continue

        ys, xs = np.nonzero(object_mask)
        object_points, _ = back_project(xs, ys, depth, fu, fv, pu, pv)
        if len(object_points) < 50:
            continue
        object_from_depth = np.median(object_points, axis=0)

        ratio = hand_real[2] / hand_from_depth[2]
        translations[t] = (
            hand_real + ratio * (object_from_depth - hand_from_depth) - rotations[t] @ centroid
        )
        anchored[t] = True

    if anchored.sum() >= 2:
        # Frames without a usable hand or object mask keep the tracker's own
        # translation, bridged so the two sources do not step against each other.
        order = np.flatnonzero(anchored)
        span = np.arange(order[0], order[-1] + 1)
        for axis in range(3):
            translations[span, axis] = np.interp(span, order, translations[order, axis])

    resting = np.zeros(n_frames, dtype=bool)
    if args.ground_plane:
        with open(args.ground_plane) as handle:
            normal = np.array(json.load(handle)["normal_camera"], dtype=float)
        mesh_vertices = np.asarray(trimesh.load(args.mesh, process=False).vertices)
        rotations, translations, resting = apply_ground_constraint(
            rotations, translations, valid, mesh_vertices, normal,
            args.rest_speed, args.max_tilt, args.smooth_window,
        )

    np.savez(
        args.out,
        rotation=rotations,
        translation=translations,
        valid=valid,
        inliers=tracked["inliers"],
        resting=resting,
        anchored=anchored,
        ref_frame=tracked["ref_frame"],
    )
    print(f"Wrote {args.out}: {anchored.sum()}/{n_frames} frames anchored to the hand")


if __name__ == "__main__":
    main()
