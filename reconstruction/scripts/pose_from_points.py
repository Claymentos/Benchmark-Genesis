#!/usr/bin/env python3
"""
Turn CoTracker's lifted points (track_points_cotracker.py) into an object trajectory.

The points are already in 3D, one cloud per frame, so the object's pose is whatever
rigid motion carries the reference frame's cloud onto each later one. Kabsch with
RANSAC solves that per frame, and RANSAC matters here: a point whose depth patch
catches the background, or that slid off the object during an occlusion, is an
outlier in 3D even when its 2D track looks fine.

This is the same rigid fit track_object_v2s2r.py performs, on CoTracker points rather
than TAPIR's, and with the model-free depth lift in place of that script's internal
one -- so it is an independent trajectory to compare with the FoundationPose one, not
a replacement for it. The shared cleanup (drop impossible frames, interpolate, smooth,
and rest the object on the table) is imported from that script rather than repeated.

--smoother kalman replaces the sliding-window smoothing with a constant-velocity
Kalman filter and RTS smoother (kalman_poses.py): it weights each frame by the
inlier count behind it, fills gaps by prediction instead of interpolation, and is
what the CoTracker pipeline runs.

The output has the same fields as the other trajectories, so the Genesis replays and
the overlay read it unchanged.

Run in the `genesis` env:
    python pose_from_points.py --points points_cotracker.npz --mesh posed_mesh.obj \\
        --ground-plane ground_plane.json --out cup_traj_points.npz
"""

import argparse
import json

import numpy as np

from kalman_poses import kalman_smooth_poses
from track_object_v2s2r import (apply_ground_constraint, clean_poses, gate_poses,
                                kabsch_ransac, load_obj_vertices)


def parse_args():
    parser = argparse.ArgumentParser(description="Rigid object pose from lifted point tracks")
    parser.add_argument("--points", required=True, help="points_cotracker.npz")
    parser.add_argument("--out", required=True)
    parser.add_argument("--mesh", default=None, help="Reference mesh, for the ground constraint")
    parser.add_argument("--ground-plane", default=None, help="ground_plane.json")
    parser.add_argument(
        "--min-points",
        type=int,
        default=12,
        help="Points a frame needs, lifted and shared with the reference frame, to be solved",
    )
    parser.add_argument("--ransac-iters", type=int, default=200)
    parser.add_argument("--ransac-thresh", type=float, default=0.012,
                        help="Inlier distance, metres")
    parser.add_argument("--max-step", type=float, default=0.06,
                        help="Largest plausible travel between frames, metres")
    parser.add_argument("--max-turn", type=float, default=20.0,
                        help="Largest plausible rotation between frames, degrees")
    parser.add_argument("--smooth-window", type=int, default=9)
    parser.add_argument(
        "--smoother",
        choices=("window", "kalman", "none"),
        default="window",
        help="How the gated per-frame solves are turned into a trajectory: a sliding "
             "window average, a constant-velocity Kalman filter with an RTS pass, or "
             "none -- gated solves only, gaps left unfilled, for a later stage to smooth "
             "(e.g. a Kalman filter after a moving camera's motion is composed out)",
    )
    parser.add_argument("--kalman-pos-noise", type=float, default=0.010,
                        help="Position measurement noise, metres (1 sigma)")
    parser.add_argument("--kalman-accel", type=float, default=0.005,
                        help="Process noise: unmodelled acceleration, metres/frame^2")
    parser.add_argument("--kalman-rot-noise", type=float, default=4.0,
                        help="Orientation measurement noise, degrees (1 sigma)")
    parser.add_argument("--kalman-ang-accel", type=float, default=2.0,
                        help="Process noise: unmodelled angular acceleration, degrees/frame^2")
    parser.add_argument("--kalman-gate", type=float, default=6.0,
                        help="Reject a measurement whose innovation is further than this "
                             "Mahalanobis distance from the prediction; loose on purpose, "
                             "since a real acceleration also surprises the motion model")
    parser.add_argument("--rest-speed", type=float, default=0.004,
                        help="Speed below which the object counts as resting, metres/frame")
    parser.add_argument("--max-tilt", type=float, default=30.0,
                        help="Tilt beyond which a resting object is not flattened, degrees")
    return parser.parse_args()


def main():
    args = parse_args()
    data = np.load(args.points)
    points_3d = data["points_3d"]
    reference = int(data["ref_frame"])
    n_frames, n_points, _ = points_3d.shape

    source_all = points_3d[reference]
    seeded = np.isfinite(source_all).all(axis=1)
    if seeded.sum() < args.min_points:
        raise SystemExit(f"only {seeded.sum()} points lifted at the reference frame")

    rotations = np.tile(np.eye(3), (n_frames, 1, 1))
    translations = np.zeros((n_frames, 3))
    valid = np.zeros(n_frames, dtype=bool)
    inliers = np.zeros(n_frames, dtype=int)
    rng = np.random.default_rng(0)

    for index in range(n_frames):
        usable = seeded & np.isfinite(points_3d[index]).all(axis=1)
        if usable.sum() < args.min_points:
            continue
        rotation, translation, mask = kabsch_ransac(
            source_all[usable], points_3d[index][usable], args.ransac_iters,
            args.ransac_thresh, rng,
        )
        if rotation is None:
            continue
        rotations[index] = rotation
        translations[index] = translation
        inliers[index] = int(mask.sum())
        valid[index] = True

    print(f"[pose] solved {valid.sum()}/{n_frames} frames from {seeded.sum()} reference points, "
          f"median {np.median(inliers[valid]) if valid.any() else 0:.0f} inliers")

    # Continuity is judged on the tracked points' centroid, not on the translation,
    # which rotation noise swings far more than the object moves (see gate_poses).
    centre = source_all[seeded].mean(axis=0)
    solved = int(valid.sum())

    if args.smoother == "kalman":
        # The jump gate first, so a wild solve is never handed to the filter: the
        # filter's own innovation gate is there for the smaller outliers that stay
        # inside a plausible frame-to-frame step, not for wild ones.
        valid = gate_poses(rotations, translations, valid, args.max_step, args.max_turn, centre=centre)
        print(f"[pose] continuity gate kept {valid.sum()}/{solved} solves")
        rotations, translations, valid = kalman_smooth_poses(
            rotations, translations, valid,
            position_sigma=args.kalman_pos_noise,
            accel_sigma=args.kalman_accel,
            rotation_sigma=args.kalman_rot_noise,
            angular_accel_sigma=args.kalman_ang_accel,
            gate=args.kalman_gate,
            confidence=inliers.astype(float),
        )
    elif args.smoother == "none":
        valid = gate_poses(rotations, translations, valid, args.max_step, args.max_turn, centre=centre)
        print(f"[pose] continuity gate kept {valid.sum()}/{solved} solves")
    else:
        kept = gate_poses(rotations, translations, valid, args.max_step, args.max_turn, centre=centre)
        print(f"[pose] continuity gate kept {kept.sum()}/{solved} solves; the rest are interpolated")
        rotations, translations, valid = clean_poses(
            rotations, translations, valid, args.max_step, args.max_turn, args.smooth_window,
            confidence=inliers.astype(float), centre=centre,
        )
    print(f"[pose] {valid.sum()}/{n_frames} frames after dropping implausible jumps and smoothing")

    resting = np.zeros(n_frames, dtype=bool)
    if args.ground_plane and args.mesh:
        with open(args.ground_plane) as handle:
            normal = np.asarray(json.load(handle)["normal_camera"], dtype=float)
        normal /= np.linalg.norm(normal)
        vertices = load_obj_vertices(args.mesh)
        rotations, translations, resting = apply_ground_constraint(
            rotations, translations, valid, vertices, normal,
            args.rest_speed, args.max_tilt, args.smooth_window,
        )
        print(f"[pose] ground constraint applied, {int(resting.sum())} frames resting")

    np.savez(
        args.out,
        rotation=rotations,
        translation=translations,
        valid=valid,
        inliers=inliers,
        resting=resting,
        ref_frame=reference,
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
