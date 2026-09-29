#!/usr/bin/env python3
"""
TouchAnything mocap hands -> `all_hand_meshes.npz` in the reference camera frame.

TouchAnything records the hands with Rokoko gloves registered to Vive wrist trackers
(`rokoko_hands.json`, 21 joints per hand in the MediaPipe/OpenPose order, already
`aligned_to_vive`), so the hand pose is measured and not estimated. HaWoR is not
needed. What the recording lacks is the video camera: it is head-mounted, not rigid
to the chest tracker (a fixed chest->camera offset leaves 34 px of median error),
and there is no calibration.

The camera motion comes from `camera_traj.npz`: the static table tracked like an
object by track_object_v2s2r.py, which gives T_ct<-cref for every frame in MoGe's
metric. This script then fits the single similarity that places the Vive world in the
reference camera,

    p_cref = s * R p_world + t,

over every frame together, from two cues:

  * the Rokoko joints, carried through T_ct<-cref and projected, against WiLoR's 2D
    joints (`wilor_hands.json`, camera frame at WiLoR's full-image focal of
    5000/256 * max(W, H), which is how its keypoints reproject onto the glove);
  * their depth against the MoGe depth sampled at those 2D joints, which pins the
    scale s to the depth the object is lifted with, so that hand and object agree
    even where MoGe's metric is off.

A per-frame PnP on the same correspondences is not an alternative: WiLoR is ~20 px
off on the black gloves, and 42 noisy points per frame leave the camera pose loose.
Seven parameters shared by 223 frames are not loose.

Writes `{side}_joints` (T, 21, 3), `{side}_vertices` (points sampled along the bones,
for keyframe_grasps.py's hand-to-surface distance) and `{side}_valid`, in OpenCV
coordinates of the reference camera, which is what every later stage reads.

Run in the `genesis` env:
    python touchanything_hands.py --episode-dir touchanything/Workbench/pick_up_screwdriver/X \\
        --camera-traj camera_traj.npz --intrinsics intrinsics.yaml --depth-dir depth_moge \\
        --out all_hand_meshes.npz --overlay hands_overlay.mp4 --frames-dir all_frames
"""

import argparse
import json
import os

import cv2
import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

SIDES = ("left", "right")
# Parent of each of the 21 joints; the wrist (0) has none.
PARENTS = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]


def parse_args():
    parser = argparse.ArgumentParser(description="TouchAnything mocap hands -> reference camera frame")
    parser.add_argument("--episode-dir", required=True, help="TouchAnything episode (rokoko/wilor json)")
    parser.add_argument("--camera-traj", required=True, help="Table tracked by track_object_v2s2r.py")
    parser.add_argument("--intrinsics", required=True)
    parser.add_argument("--depth-dir", required=True, help="Depth the object is lifted with (uint16 mm)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--transform-out", default=None, help="Fitted similarity + diagnostics (.json)")
    parser.add_argument("--frames-dir", default=None, help="With --overlay, the frames to draw on")
    parser.add_argument("--overlay", default=None, help="Video of the projected mocap joints")
    parser.add_argument("--depth-weight", type=float, default=300.0,
                        help="Pixels per metre of depth residual; balances the two cues")
    parser.add_argument("--fixed-scale", type=float, default=None,
                        help="Hold s at this value instead of fitting it (1.0 = trust Vive metric)")
    parser.add_argument("--bone-samples", type=int, default=6)
    return parser.parse_args()


def read_jsonl(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def to_matrix(rotation, translation):
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


def similarity(x):
    return np.exp(x[6]), Rotation.from_rotvec(x[:3]).as_matrix(), x[3:6]


def main():
    args = parse_args()
    with open(args.intrinsics) as handle:
        calib = yaml.safe_load(handle)
    fu, fv, pu, pv = (float(calib[k]) for k in ("fu", "fv", "pu", "pv"))

    rokoko = read_jsonl(os.path.join(args.episode_dir, "rokoko_hands.json"))
    wilor = read_jsonl(os.path.join(args.episode_dir, "wilor_hands.json"))
    n_frames = len(rokoko)

    camera = np.load(args.camera_traj)
    if len(camera["rotation"]) != n_frames:
        raise SystemExit(f"camera track has {len(camera['rotation'])} frames, mocap {n_frames}")
    cam_valid = camera["valid"].astype(bool)
    # T_ct<-cref, the table's motion as seen by the camera.
    cam_R, cam_t = camera["rotation"], camera["translation"]

    depth_names = sorted(f for f in os.listdir(args.depth_dir) if f.endswith(".png"))
    height, width = cv2.imread(os.path.join(args.depth_dir, depth_names[0]), cv2.IMREAD_UNCHANGED).shape
    wilor_focal = 5000.0 / 256.0 * max(width, height)

    world = np.zeros((n_frames, 42, 3))
    mocap_ok = np.zeros((n_frames, 2), dtype=bool)
    keypoints = np.zeros((n_frames, 42, 2))
    keypoint_ok = np.zeros((n_frames, 42), dtype=bool)
    keypoint_depth = np.full((n_frames, 2), np.nan)
    for t in range(n_frames):
        depth = cv2.imread(os.path.join(args.depth_dir, depth_names[t]), cv2.IMREAD_UNCHANGED) / 1000.0
        for k, side in enumerate(SIDES):
            joints = np.asarray(rokoko[t][f"{side}_pos"], dtype=float)
            if joints.shape == (21, 3) and np.isfinite(joints).all():
                world[t, 21 * k : 21 * k + 21] = joints
                mocap_ok[t, k] = True
            detected = np.asarray(wilor[t][f"{side}_pos"], dtype=float)
            if detected.shape != (21, 3) or not cam_valid[t]:
                continue
            uv = wilor_focal * detected[:, :2] / detected[:, 2:] + [width / 2, height / 2]
            inside = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
            keypoints[t, 21 * k : 21 * k + 21] = uv
            keypoint_ok[t, 21 * k : 21 * k + 21] = inside
            # The glove surface under the knuckles; the joint centres sit ~1 cm deeper,
            # which the median over the whole hand absorbs as a constant bias.
            px = np.round(uv[inside]).astype(int)
            samples = depth[px[:, 1], px[:, 0]]
            samples = samples[samples > 0]
            if len(samples) >= 8:
                keypoint_depth[t, k] = np.median(samples) + 0.01
    keypoint_ok &= np.repeat(mocap_ok, 21, axis=1)

    def project(x, frames):
        scale, rotation, translation = similarity(x)
        ref = scale * world[frames] @ rotation.T + translation
        cam = np.einsum("tij,tkj->tki", cam_R[frames], ref) + cam_t[frames][:, None]
        uv = np.stack([fu * cam[..., 0] / cam[..., 2] + pu, fv * cam[..., 1] / cam[..., 2] + pv], -1)
        return cam, uv

    frames = np.flatnonzero(keypoint_ok.any(axis=1))

    def residual(x):
        cam, uv = project(x, frames)
        pixel = (uv - keypoints[frames]) * keypoint_ok[frames][..., None]
        behind = np.clip(0.05 - cam[..., 2], 0, None) * 1e4
        hand_z = np.stack([np.median(cam[:, :21, 2], 1), np.median(cam[:, 21:, 2], 1)], 1)
        dz = np.nan_to_num((hand_z - keypoint_depth[frames]) * args.depth_weight)
        return np.concatenate([pixel.ravel(), behind.ravel(), dz.ravel()])

    # Initialise from a PnP on the first frame with detections, where T_ct<-cref is
    # (close to) the identity at the reference frame and composes with it elsewhere.
    t0 = frames[0]
    sel = keypoint_ok[t0]
    K = np.array([[fu, 0, pu], [0, fv, pv], [0, 0, 1.0]])
    _, rvec, tvec, _ = cv2.solvePnPRansac(world[t0][sel], keypoints[t0][sel], K, None, reprojectionError=20)
    T_ct_w = to_matrix(cv2.Rodrigues(rvec)[0], tvec[:, 0])
    T_ref_w = np.linalg.inv(to_matrix(cam_R[t0], cam_t[t0])) @ T_ct_w
    x0 = np.r_[Rotation.from_matrix(T_ref_w[:3, :3]).as_rotvec(), T_ref_w[:3, 3], 0.0]

    if args.fixed_scale is not None:
        log_s = np.log(args.fixed_scale)
        fit = least_squares(lambda y: residual(np.r_[y, log_s]), x0[:6], loss="soft_l1", f_scale=15.0)
        x = np.r_[fit.x, log_s]
    else:
        fit = least_squares(residual, x0, loss="soft_l1", f_scale=15.0)
        x = fit.x

    scale, rotation, translation = similarity(x)
    cam, uv = project(x, frames)
    err = np.linalg.norm(uv - keypoints[frames], axis=-1)[keypoint_ok[frames]]
    hand_z = np.stack([np.median(cam[:, :21, 2], 1), np.median(cam[:, 21:, 2], 1)], 1)
    dz = (hand_z - keypoint_depth[frames])[np.isfinite(keypoint_depth[frames])]
    print(f"[hands] world -> reference camera: scale {scale:.3f}, "
          f"{len(frames)} frames, {keypoint_ok.sum()} 2D joints")
    print(f"[hands] reprojection vs WiLoR: median {np.median(err):.1f} px, p90 {np.percentile(err, 90):.1f} px")
    print(f"[hands] hand depth vs MoGe: median {np.median(dz) * 100:+.1f} cm, "
          f"MAD {np.median(np.abs(dz - np.median(dz))) * 100:.1f} cm")

    ref = scale * world @ rotation.T + translation
    out = {}
    for k, side in enumerate(SIDES):
        joints = ref[:, 21 * k : 21 * k + 21]
        # Bone-length drift is how a glove glitch shows; report it rather than hide it.
        bones = np.linalg.norm(joints[:, 1:] - joints[:, PARENTS[1:]], axis=-1)
        drift = np.abs(bones / np.median(bones[mocap_ok[:, k]], axis=0) - 1)[mocap_ok[:, k]]
        print(f"[hands] {side}: {mocap_ok[:, k].sum()}/{n_frames} frames, "
              f"bone-length drift p99 {np.percentile(drift, 99) * 100:.1f}%")
        weights = np.linspace(0, 1, args.bone_samples + 1)[1:]
        along = joints[:, PARENTS[1:], None] + weights[:, None] * (
            joints[:, 1:, None] - joints[:, PARENTS[1:], None]
        )
        out[f"{side}_joints"] = joints
        out[f"{side}_vertices"] = np.concatenate([joints, along.reshape(n_frames, -1, 3)], axis=1)
        out[f"{side}_valid"] = mocap_ok[:, k]
    np.savez(args.out, **out, source="touchanything_rokoko", scale=scale)
    print(f"Wrote {args.out}")

    if args.transform_out:
        with open(args.transform_out, "w") as handle:
            json.dump({
                "T_ref_world": to_matrix(scale * rotation, translation).tolist(),
                "scale": scale,
                "median_px": float(np.median(err)),
                "median_depth_cm": float(np.median(dz) * 100),
            }, handle, indent=2)

    if args.overlay and args.frames_dir:
        _, uv_all = project(x, np.arange(n_frames))
        names = sorted(f for f in os.listdir(args.frames_dir) if f.endswith(".png"))
        writer = cv2.VideoWriter(args.overlay, cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height))
        for t, name in enumerate(names):
            image = cv2.imread(os.path.join(args.frames_dir, name))
            for k, colour in enumerate([(0, 0, 255), (255, 0, 0)]):
                pts = uv_all[t, 21 * k : 21 * k + 21]
                for j in range(1, 21):
                    a, b = pts[PARENTS[j]].astype(int), pts[j].astype(int)
                    cv2.line(image, tuple(a), tuple(b), colour, 2)
                for u, v in keypoints[t, 21 * k : 21 * k + 21][keypoint_ok[t, 21 * k : 21 * k + 21]]:
                    cv2.circle(image, (int(u), int(v)), 2, (0, 255, 0), -1)
            writer.write(image)
        writer.release()
        print(f"Wrote {args.overlay}  (lines: mocap, green: WiLoR)")


if __name__ == "__main__":
    main()
