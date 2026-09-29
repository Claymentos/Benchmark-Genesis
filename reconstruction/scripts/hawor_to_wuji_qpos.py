#!/usr/bin/env python3
"""
Convert a HaWoR hand track into a Wuji Hand joint trajectory for simulator replay.

Reads HaWoR's `all_hand_meshes.npz` (MANO joints in camera frame), retargets each
frame onto the Wuji Hand with wuji_retargeting, and writes per-frame finger joint
angles plus the wrist pose that positions the hand in the scene.

Usage:
    python hawor_to_wuji_qpos.py --hand-meshes all_hand_meshes.npz --side left --out wuji_traj.npz
"""

import argparse
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

DEFAULT_WUJI_DIR = os.environ.get(
    "WUJI_RETARGETING_DIR",
    os.path.expanduser("~/wuji-ego-mint/eval/simulate/wuji-retargeting"),
)

# HaWoR's MANO wrapper already emits the 21 joints in the MediaPipe/OpenPose layout
# the retargeter expects (it reads kp[0], kp[5], kp[9] to build the wrist frame), so
# no reordering is applied. Permuting them into "OpenPose order" scrambles the
# fingers: it drives bone-length variation over time from 4% to 31%, and the spread
# of the four MCP knuckle distances from the wrist from 5 mm to 40 mm.


def clean_keypoints(keypoints, valid, max_gap, smooth_window, hampel_sigma=4.0):
    """Reject outlier frames, fill short gaps, then smooth.

    Mirrors the three stages MINT applies to HaWoR predictions before anything
    downstream consumes them (ego_pipeline/data_cleaning/final_clean.py). Working on
    keypoints rather than MANO parameters keeps this dependency-free, and smoothing
    the keypoints also smooths the wrist frame, which is derived from them.
    """
    from scipy.signal import savgol_filter

    keypoints = keypoints.copy()
    valid = valid.copy()

    wrist = keypoints[:, 0]
    reference = np.full_like(wrist, np.nan)
    reference[valid] = wrist[valid]
    residual = np.full(len(wrist), np.nan)
    for axis in range(3):
        series = reference[:, axis]
        filled = np.interp(np.arange(len(series)), np.flatnonzero(valid), series[valid])
        reference[:, axis] = filled
    residual = np.linalg.norm(wrist - reference, axis=1)
    deviation = np.median(np.abs(residual[valid] - np.median(residual[valid])))
    valid &= residual <= np.median(residual[valid]) + hampel_sigma * 1.4826 * deviation

    known = np.flatnonzero(valid)
    if len(known) < 2:
        return keypoints, valid
    span = np.arange(known[0], known[-1] + 1)
    for joint in range(keypoints.shape[1]):
        for axis in range(3):
            keypoints[span, joint, axis] = np.interp(span, known, keypoints[known, joint, axis])

    # Only bridge gaps short enough that interpolation is defensible; longer dropouts
    # stay invalid so the replay does not invent motion across them.
    usable = np.zeros(len(valid), dtype=bool)
    usable[span] = True
    for start, end in zip(known[:-1], known[1:]):
        if end - start - 1 > max_gap:
            usable[start + 1 : end] = False

    window = min(smooth_window, len(span) - (1 - len(span) % 2))
    if window >= 5:
        keypoints[span] = savgol_filter(keypoints[span], window, 2, axis=0)
    return keypoints, usable


def parse_args():
    parser = argparse.ArgumentParser(description="HaWoR hand track -> Wuji Hand trajectory")
    parser.add_argument("--hand-meshes", required=True, help="HaWoR all_hand_meshes.npz")
    parser.add_argument("--side", default="left", choices=["left", "right"])
    parser.add_argument("--out", required=True, help="Output .npz trajectory")
    parser.add_argument("--config", default=None, help="Retargeting YAML (default: bundled glove config)")
    parser.add_argument("--wuji-dir", default=DEFAULT_WUJI_DIR, help="wuji-retargeting checkout")
    parser.add_argument("--fps", type=float, default=30.0, help="Source video frame rate")
    parser.add_argument(
        "--mirror-x",
        default="no",
        choices=["yes", "no"],
        help="Mirror camera-frame x. Off by default: HaWoR applies R_x to its joints and "
        "then multiplies by R_w2c, which for a static camera is R_x again, so the flips "
        "cancel and the stored joints are already in OpenCV camera coordinates.",
    )
    parser.add_argument("--clean", default="yes", choices=["yes", "no"])
    parser.add_argument("--max-gap", type=int, default=30, help="Longest dropout to interpolate across")
    parser.add_argument("--smooth-window", type=int, default=9, help="Savitzky-Golay window in frames")
    return parser.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, os.path.abspath(args.wuji_dir))

    from wuji_retargeting.mediapipe import (
        OPERATOR2MANO_LEFT,
        OPERATOR2MANO_RIGHT,
        estimate_frame_from_hand_points,
    )
    from wuji_retargeting.retarget import Retargeter

    config = args.config or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config",
        f"wuji_mano_{args.side}.yaml",
    )

    data = np.load(args.hand_meshes)
    joints = data[f"{args.side}_joints"]
    # HaWoR's own valid flag does not cover every dropout: frames where the
    # infiller failed carry NaN joints while still being marked valid.
    valid = data[f"{args.side}_valid"].astype(bool) & np.isfinite(joints).all(axis=(1, 2))
    keypoints = np.nan_to_num(joints)

    if args.mirror_x == "yes":
        keypoints = keypoints * np.array([-1.0, 1.0, 1.0])

    if args.clean == "yes":
        keypoints, valid = clean_keypoints(keypoints, valid, args.max_gap, args.smooth_window)

    retargeter = Retargeter.from_yaml(config, hand_side=args.side)
    operator2mano = OPERATOR2MANO_LEFT if args.side == "left" else OPERATOR2MANO_RIGHT

    joint_names = list(retargeter.optimizer.robot.dof_joint_names)

    n_frames = keypoints.shape[0]
    qpos = np.zeros((n_frames, len(joint_names)))
    wrist_pos = np.zeros((n_frames, 3))
    wrist_quat = np.zeros((n_frames, 4))

    last_qpos = None
    last_quat = np.array([1.0, 0.0, 0.0, 0.0])
    last_pos = np.zeros(3)

    for t in range(n_frames):
        if not valid[t]:
            # Holding the last solution keeps the low-pass filter from ingesting
            # the placeholder keypoints HaWoR writes for undetected frames.
            qpos[t] = last_qpos if last_qpos is not None else 0.0
            wrist_pos[t] = last_pos
            wrist_quat[t] = last_quat
            continue

        frame_kp = keypoints[t]
        qpos[t] = retargeter.retarget(frame_kp)

        rotation = estimate_frame_from_hand_points(frame_kp - frame_kp[0]) @ operator2mano
        quat_xyzw = Rotation.from_matrix(rotation).as_quat()

        wrist_pos[t] = frame_kp[0]
        wrist_quat[t] = np.r_[quat_xyzw[3], quat_xyzw[:3]]

        last_qpos, last_pos, last_quat = qpos[t], wrist_pos[t], wrist_quat[t]

    np.savez(
        args.out,
        qpos=qpos,
        wrist_pos=wrist_pos,
        wrist_quat=wrist_quat,
        valid=valid,
        joint_names=np.array(joint_names),
        side=args.side,
        fps=args.fps,
    )
    print(f"Wrote {args.out}: {n_frames} frames ({int(valid.sum())} valid), {len(joint_names)} joints")
    print(f"  joints: {joint_names}")


if __name__ == "__main__":
    main()
