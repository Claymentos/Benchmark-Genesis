#!/usr/bin/env python3
"""
Correct a held object's depth from a measured hand, keeping what the image fixed.

Monocular depth is least reliable exactly where a small object is held up in a fist:
on TouchAnything's pick_up_screwdriver the MoGe depth of the glove is ~9 cm too close
to the camera, and the lifted screwdriver, whose pixels border it, inherits the bias
and floats 10-15 cm above the mocap hand that holds it. Its image position and
rotation are still right: they come from the 2D tracks, which the depth does not
touch.

So only the one quantity the image cannot see is corrected. In the camera of frame
t, the object is slid along the ray through its centre until its depth relative to
the fist matches the depth it had relative to the fist at grasp onset -- the frames
where the hand is already on the object but the object still rests on the table, so
its depth is trustworthy:

    delta   = mean over onset of  z_object - z_fist
    alpha_t = (z_fist_t + delta) / z_object_t          (camera of frame t)
    c'_t    = alpha_t c_t                              (the object's centre)

The centre's projection is unchanged, and so is the rotation. Outside the grasp alpha
is 1, with a smoothstep ramp at each end. The mocap hand is the reference here,
because it is measured and did touch the table on this clip exactly when it grasped
and set down.

Works on the reference-camera trajectory compose_camera_motion.py writes, with the
camera track it was composed from.

Run in the `genesis` env:
    python snap_depth_to_hand.py --object-traj obj_traj.npz --camera-traj camera_traj.npz \\
        --hand-meshes all_hand_meshes.npz --side left --mesh posed_mesh.obj \\
        --window 44-187 --onset 10 --out obj_traj_snapped.npz
"""

import argparse

import numpy as np
import trimesh

# Joints that surround a held object: the thumb, and the proximal and middle
# phalanges of the fingers. Fingertips wrap around it, the wrist sits behind it.
FIST = [2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]


def parse_args():
    parser = argparse.ArgumentParser(description="Held object's depth from the measured hand")
    parser.add_argument("--object-traj", required=True, help="Reference-camera object trajectory")
    parser.add_argument("--camera-traj", required=True, help="T_ct<-cref, as composed with")
    parser.add_argument("--hand-meshes", required=True, help="all_hand_meshes.npz (reference camera)")
    parser.add_argument("--side", default="left", choices=["left", "right"])
    parser.add_argument("--mesh", required=True, help="Object mesh the trajectory was tracked with")
    parser.add_argument("--window", required=True, help="Grasp as 'START-END', inclusive frames")
    parser.add_argument("--onset", type=int, default=10, help="Frames at the window's start that set delta")
    parser.add_argument("--ramp", type=int, default=8, help="Frames over which the correction eases in and out")
    parser.add_argument("--smooth", type=int, default=9, help="Moving-average window on alpha, frames")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    track = dict(np.load(args.object_traj))
    camera = np.load(args.camera_traj)
    hand = np.load(args.hand_meshes)
    rotation, translation = track["rotation"], track["translation"]
    valid = track["valid"].astype(bool) & hand[f"{args.side}_valid"].astype(bool)
    n_frames = len(rotation)

    centre = np.asarray(trimesh.load(args.mesh, process=False).vertices).mean(axis=0)
    position = np.einsum("tij,j->ti", rotation, centre) + translation      # reference camera
    fist = hand[f"{args.side}_joints"][:, FIST].mean(axis=1)

    def in_camera(points):
        return np.einsum("tij,tj->ti", camera["rotation"], points) + camera["translation"]

    c_cam, fist_cam = in_camera(position), in_camera(fist)

    start, end = (int(v) for v in args.window.split("-"))
    onset = np.arange(start, min(start + args.onset, end + 1))
    onset = onset[valid[onset]]
    if len(onset) == 0:
        raise SystemExit(f"no valid frame in the onset {start}-{start + args.onset - 1}")
    delta = np.mean(c_cam[onset, 2] - fist_cam[onset, 2])

    span = np.arange(start, end + 1)
    alpha = np.ones(n_frames)
    alpha[span] = (fist_cam[span, 2] + delta) / c_cam[span, 2]
    alpha[~valid] = 1.0
    if args.smooth > 1:
        kernel = np.ones(args.smooth) / args.smooth
        padded = np.pad(alpha[span], args.smooth // 2, mode="edge")
        alpha[span] = np.convolve(padded, kernel, mode="valid")[: len(span)]
    weight = np.zeros(n_frames)
    weight[span] = 1.0
    ramp = min(args.ramp, len(span) // 2)
    if ramp > 0:
        x = (np.arange(ramp) + 1) / (ramp + 1)
        edge = x * x * (3 - 2 * x)
        weight[start : start + ramp] = edge
        weight[end - ramp + 1 : end + 1] = edge[::-1]
    alpha = 1.0 + weight * (alpha - 1.0)

    moved_cam = c_cam * alpha[:, None]
    moved = np.einsum("tji,tj->ti", camera["rotation"], moved_cam - camera["translation"])
    shift = np.linalg.norm(moved - position, axis=1)
    print(f"[snap] delta (object - fist depth at onset {onset[0]}-{onset[-1]}): {delta * 100:+.1f} cm")
    print(f"[snap] {args.window}: depth scaled by {alpha[span].min():.3f}-{alpha[span].max():.3f}, "
          f"centre moved median {np.median(shift[span]) * 100:.1f} cm, max {shift.max() * 100:.1f} cm")

    track["translation"] = translation + (moved - position)
    track["depth_alpha"] = alpha
    np.savez(args.out, **track)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
