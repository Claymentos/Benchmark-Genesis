#!/usr/bin/env python3
"""
Carry a camera-relative object trajectory into the reference camera's frame.

track_object_v2s2r.py solves X_t = R_t X_ref + t_t, with X_ref in the reference
camera and X_t in the camera of frame t. For a static camera the two frames coincide
and that is the object's motion. For a moving one it mixes the object's motion with
the camera's: here the camera's own motion comes from tracking the static table the
same way (`camera_traj.npz`, T_ct<-cref), and the object pose in the reference frame
is

    T_cref<-obj(t) = T_ct<-cref^-1  T_ct<-obj(t)

Every later stage (keyframes, grasps, refinement, replay) reads the reference camera
frame, so this is the only change they need. One more thing depends on the camera:
extract_keyframes.py finds contact from the 2D pixel centroid of the object tracks,
and camera motion alone moves that centroid. The written `tracks` are therefore the
tracked points re-projected through the composed poses into the reference camera,
as a static camera would have seen them. The raw tracks are kept as `tracks_image`.
A trajectory without 2D tracks (pose_from_points.py writes none) is composed alone.

Run in the `genesis` env:
    python compose_camera_motion.py --object-traj obj_traj_cam.npz --camera-traj camera_traj.npz \\
        --intrinsics intrinsics.yaml --depth-dir depth_moge --out obj_traj.npz
"""

import argparse
import os

import cv2
import numpy as np
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Object trajectory: camera frame t -> reference camera")
    parser.add_argument("--object-traj", required=True, help="track_object_v2s2r.py output, camera-relative")
    parser.add_argument("--camera-traj", required=True, help="Static scene tracked the same way")
    parser.add_argument("--intrinsics", required=True)
    parser.add_argument("--depth-dir", default=None,
                        help="Depth the tracks were lifted with; needed only when the "
                        "trajectory carries 2D tracks to stabilise")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    obj = dict(np.load(args.object_traj))
    cam = np.load(args.camera_traj)
    if int(obj["ref_frame"]) != int(cam["ref_frame"]):
        raise SystemExit(f"reference frames differ: object {obj['ref_frame']}, camera {cam['ref_frame']}")
    with open(args.intrinsics) as handle:
        calib = yaml.safe_load(handle)
    fu, fv, pu, pv = (float(calib[k]) for k in ("fu", "fv", "pu", "pv"))

    # T_cref<-ct = (R_c^T, -R_c^T t_c); composed with the object's camera-relative pose.
    cam_R_inv = cam["rotation"].transpose(0, 2, 1)
    rotation = cam_R_inv @ obj["rotation"]
    translation = np.einsum("tij,tj->ti", cam_R_inv, obj["translation"] - cam["translation"])
    valid = obj["valid"].astype(bool) & cam["valid"].astype(bool)
    print(f"[compose] {valid.sum()}/{len(valid)} frames with both an object and a camera pose "
          f"(object {obj['valid'].sum()}, camera {cam['valid'].sum()})")

    moved = np.linalg.norm(cam["translation"][cam["valid"]], axis=1)
    turned = np.degrees(np.arccos(np.clip(
        (np.trace(cam["rotation"][cam["valid"]], axis1=1, axis2=2) - 1) / 2, -1, 1)))
    print(f"[compose] camera travel from the reference: max {moved.max() * 100:.1f} cm, "
          f"max {turned.max():.1f} deg")

    obj.update(
        rotation=rotation,
        translation=translation,
        valid=valid,
        camera_rotation=cam["rotation"],
        camera_translation=cam["translation"],
    )
    if "tracks" not in obj or args.depth_dir is None:
        np.savez(args.out, **obj)
        print(f"Wrote {args.out}")
        return

    # Stabilised tracks: the reference-frame 3D points pushed through the composed
    # poses and projected into the reference camera.
    ref = int(obj["ref_frame"])
    depth_names = sorted(f for f in os.listdir(args.depth_dir) if f.endswith(".png"))
    depth = cv2.imread(os.path.join(args.depth_dir, depth_names[ref]), cv2.IMREAD_UNCHANGED) / 1000.0
    tracks = obj["tracks"]  # (T, K, 2), frame-first
    uv = tracks[ref]
    px = np.clip(np.round(uv).astype(int), 0, [depth.shape[1] - 1, depth.shape[0] - 1])
    z = depth[px[:, 1], px[:, 0]]
    points = np.stack([(uv[:, 0] - pu) / fu * z, (uv[:, 1] - pv) / fv * z, z], axis=1)
    stable = np.einsum("tij,kj->tki", rotation, points) + translation[:, None]
    stabilised = np.stack(
        [fu * stable[..., 0] / stable[..., 2] + pu, fv * stable[..., 1] / stable[..., 2] + pv], axis=-1
    )
    # Frames without a pose, and points without depth, keep what the tracker saw.
    keep = ~valid[:, None] | (z <= 0)[None, :]
    stabilised = np.where(keep[..., None], tracks, stabilised).astype(np.float32)

    obj.update(tracks=stabilised, tracks_image=tracks)
    np.savez(args.out, **obj)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
