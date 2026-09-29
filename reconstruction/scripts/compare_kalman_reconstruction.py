#!/usr/bin/env python3
"""
What a Kalman filter does to the reconstruction, side by side with the raw solve.

Both reconstructed trajectories are filtered with the same constant-velocity Kalman
filter and RTS pass (kalman_poses.py), and the raw and filtered results are rendered
into one video: raw on the left, filtered on the right, same camera, same frame.

  object   the tracked rigid pose is filtered directly -- rotation and translation are
           exactly what the filter is written for.

  hand     HaWoR gives a deforming 778-vertex mesh, which is not a pose. It does give a
           global (rot, trans) per frame, and removing those from the vertices drops
           their spread across the clip from 37.7 mm to 8.6 mm on this take -- the
           remainder is finger articulation. So the GLOBAL pose is filtered and put
           back, leaving articulation untouched:

               v_filtered = R_filt @ (R_raw^T @ (v_raw - t_raw)) + t_filt

           Filtering the vertices themselves would smooth the fingers as well, which is
           a different claim than smoothing the hand's motion.

A note on what "raw" has to mean here: the trackers in this repo already smooth. The
object trajectory a pipeline run leaves on disk has had a savgol pass over it, so it is
not a raw solve -- re-run track_object_v2s2r.py with --smooth-window 1 to get one.
HaWoR's meshes are raw as written.

Run in the `genesis` env:
    python compare_kalman_reconstruction.py --hand-meshes all_hand_meshes.npz \\
        --side right --object-mesh posed_mesh.obj --object-poses traj_raw.npz \\
        --frames-dir all_frames --intrinsics intrinsics.yaml --out kalman_compare.mp4
"""

import argparse
import os
import sys

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from kalman_poses import kalman_smooth_poses          # noqa: E402
from render_reconstruction import decimate, shade      # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Raw vs Kalman-filtered reconstruction")
    parser.add_argument("--hand-meshes", required=True, help="HaWoR all_hand_meshes.npz")
    parser.add_argument("--side", default="right", choices=("left", "right"))
    parser.add_argument("--object-mesh", required=True)
    parser.add_argument("--object-poses", required=True,
                        help="RAW object trajectory (tracker run with --smooth-window 1)")
    parser.add_argument("--intrinsics", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--frames-dir", default=None)
    parser.add_argument("--background", choices=("frame", "dark"), default="frame")
    parser.add_argument("--backdrop", type=float, default=0.4)
    parser.add_argument("--scale", type=float, default=0.42)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--faces", type=int, default=4000)
    parser.add_argument("--hand-colour", default="235,170,140")
    parser.add_argument("--object-colour", default="120,180,245")
    parser.add_argument("--ambient", type=float, default=0.35)
    parser.add_argument("--plot", default=None, help="Write the per-frame signals here")

    # One filter, both trajectories, so the comparison is of the same estimator.
    parser.add_argument("--kalman-pos-noise", type=float, default=0.010, help="metres")
    parser.add_argument("--kalman-accel", type=float, default=0.005, help="metres/frame^2")
    parser.add_argument("--kalman-rot-noise", type=float, default=4.0, help="degrees")
    parser.add_argument("--kalman-ang-accel", type=float, default=2.0, help="degrees/frame^2")
    parser.add_argument("--kalman-gate", type=float, default=6.0)
    return parser.parse_args()


def filtered_poses(rotations, translations, valid, args, confidence=None):
    return kalman_smooth_poses(
        rotations, translations, valid,
        position_sigma=args.kalman_pos_noise,
        accel_sigma=args.kalman_accel,
        rotation_sigma=args.kalman_rot_noise,
        angular_accel_sigma=args.kalman_ang_accel,
        gate=args.kalman_gate,
        confidence=confidence,
    )


def jitter(translations, rotations):
    """Median |second difference|: the high-frequency content smoothing is meant to remove."""
    step = np.linalg.norm(np.diff(translations, axis=0), axis=1)
    accel = np.linalg.norm(np.diff(translations, 2, axis=0), axis=1)
    angle = np.degrees((Rotation.from_matrix(rotations[:-1]).inv()
                        * Rotation.from_matrix(rotations[1:])).magnitude())
    return dict(
        step_mm=1000 * np.median(step),
        jitter_mm=1000 * np.median(accel),
        jitter_p90_mm=1000 * np.percentile(accel, 90),
        rot_jitter_deg=np.median(np.abs(np.diff(angle))),
        path_mm=1000 * step.sum(),
    )


def report(tag, before, after):
    print(f"  {tag}")
    print(f"    raw     step {before['step_mm']:6.2f} mm | jitter {before['jitter_mm']:6.2f} mm "
          f"(p90 {before['jitter_p90_mm']:6.2f}) | rot jitter {before['rot_jitter_deg']:.3f} deg "
          f"| path {before['path_mm']:7.0f} mm")
    print(f"    kalman  step {after['step_mm']:6.2f} mm | jitter {after['jitter_mm']:6.2f} mm "
          f"(p90 {after['jitter_p90_mm']:6.2f}) | rot jitter {after['rot_jitter_deg']:.3f} deg "
          f"| path {after['path_mm']:7.0f} mm")
    if before["jitter_mm"] > 0:
        print(f"    -> jitter x{before['jitter_mm'] / max(after['jitter_mm'], 1e-9):.1f} lower, "
              f"path {100 * after['path_mm'] / max(before['path_mm'], 1e-9):.0f}% of raw")


def main():
    args = parse_args()
    import cv2
    import trimesh

    calibration = yaml.safe_load(open(args.intrinsics))
    fu, fv = calibration["fu"], calibration["fv"]
    pu, pv = calibration["pu"], calibration["pv"]

    # ── hand: filter the global pose, keep the articulation ──
    hand = np.load(args.hand_meshes, allow_pickle=True)
    vertices_raw = hand[f"{args.side}_vertices"].astype(np.float64)
    hand_faces = hand[f"{args.side}_faces"].astype(np.int32)
    hand_valid = hand[f"{args.side}_valid"].astype(bool)
    hand_rot_raw = Rotation.from_rotvec(hand[f"{args.side}_rot"]).as_matrix()
    hand_trans_raw = hand[f"{args.side}_trans"].astype(np.float64)

    hand_rot_kf, hand_trans_kf, hand_valid_kf = filtered_poses(
        hand_rot_raw, hand_trans_raw, hand_valid, args)

    local = np.einsum("fji,fkj->fki", hand_rot_raw, vertices_raw - hand_trans_raw[:, None, :])
    vertices_kf = np.einsum("fij,fkj->fki", hand_rot_kf, local) + hand_trans_kf[:, None, :]

    # ── object: filter the tracked pose directly ──
    poses = np.load(args.object_poses)
    rot_raw, trans_raw = poses["rotation"], poses["translation"]
    object_valid = poses["valid"].astype(bool)
    confidence = poses["inliers"].astype(float) if "inliers" in poses.files else None
    rot_kf, trans_kf, object_valid_kf = filtered_poses(
        rot_raw, trans_raw, object_valid, args, confidence)

    print("Kalman filter on both reconstructed trajectories")
    report("hand (global wrist pose)",
           jitter(hand_trans_raw, hand_rot_raw), jitter(hand_trans_kf, hand_rot_kf))
    report("object (tracked rigid pose)",
           jitter(trans_raw, rot_raw), jitter(trans_kf, rot_kf))

    mesh = decimate(trimesh.load(args.object_mesh, process=False), args.faces)
    object_vertices = np.asarray(mesh.vertices, dtype=np.float64)
    object_faces = np.asarray(mesh.faces, dtype=np.int32)

    n_frames = min(len(vertices_raw), len(rot_raw))
    names = []
    if args.frames_dir:
        names = sorted(f for f in os.listdir(args.frames_dir) if f.endswith(".png"))
    if names:
        height, width = cv2.imread(os.path.join(args.frames_dir, names[0])).shape[:2]
    else:
        width, height = int(round(2 * pu)), int(round(2 * pv))
    panel_w, panel_h = int(round(width * args.scale)), int(round(height * args.scale))

    hand_colour = [float(v) for v in args.hand_colour.split(",")][::-1]
    object_colour = [float(v) for v in args.object_colour.split(",")][::-1]

    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (panel_w * 2, panel_h))
    if not writer.isOpened():
        raise SystemExit(f"could not open {args.out} for writing")

    def draw(canvas, meshes):
        depths, colours, polygons = [], [], []
        for vertices, faces, colour in meshes:
            z = vertices[:, 2]
            if np.mean(z > 1e-6) < 0.999:
                continue
            u = fu * vertices[:, 0] / np.maximum(z, 1e-6) + pu
            v = fv * vertices[:, 1] / np.maximum(z, 1e-6) + pv
            pixels = np.stack([u, v], axis=1) * args.scale
            face_depth, face_colour = shade(vertices, faces, colour, args.ambient)
            tri = pixels[faces]
            inside = ~((tri[:, :, 0] < 0).all(1) | (tri[:, :, 0] > panel_w).all(1) |
                       (tri[:, :, 1] < 0).all(1) | (tri[:, :, 1] > panel_h).all(1))
            depths.append(face_depth[inside])
            colours.append(face_colour[inside])
            polygons.append(np.round(tri[inside]).astype(np.int32))
        if not depths:
            return
        depth = np.concatenate(depths)
        colour = np.concatenate(colours)
        polygon = np.concatenate(polygons)
        for face in np.argsort(-depth):
            cv2.fillConvexPoly(canvas, polygon[face],
                               tuple(float(c) for c in colour[face]), lineType=cv2.LINE_AA)

    for index in range(n_frames):
        if args.background == "frame" and index < len(names):
            frame = cv2.imread(os.path.join(args.frames_dir, names[index]))
            frame = cv2.resize(frame, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
            base = (frame.astype(np.float32) * args.backdrop).astype(np.uint8)
        else:
            base = np.full((panel_h, panel_w, 3), 18, np.uint8)

        panels = []
        for label, verts, hvalid, rot, trans, ovalid in (
            ("raw", vertices_raw, hand_valid, rot_raw, trans_raw, object_valid),
            ("kalman", vertices_kf, hand_valid_kf, rot_kf, trans_kf, object_valid_kf),
        ):
            canvas = base.copy()
            meshes = []
            if hvalid[index]:
                meshes.append((verts[index], hand_faces, hand_colour))
            if ovalid[index]:
                meshes.append((object_vertices @ rot[index].T + trans[index],
                               object_faces, object_colour))
            draw(canvas, meshes)
            cv2.putText(canvas, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2, cv2.LINE_AA)
            panels.append(canvas)

        pair = np.hstack(panels)
        cv2.line(pair, (panel_w, 0), (panel_w, panel_h), (90, 90, 90), 1)
        cv2.putText(pair, f"frame {index}", (12, panel_h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(pair)

    writer.release()
    print(f"Wrote {args.out}: {n_frames} frames @ {args.fps:g} fps, {panel_w * 2}x{panel_h}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        for axis, (tag, a, b) in zip(axes, (
            ("hand wrist", hand_trans_raw, hand_trans_kf),
            ("object", trans_raw, trans_kf),
        )):
            axis.plot(1000 * np.linalg.norm(np.diff(a, 2, axis=0), axis=1),
                      lw=0.9, label="raw", color="#c0504d")
            axis.plot(1000 * np.linalg.norm(np.diff(b, 2, axis=0), axis=1),
                      lw=1.2, label="kalman", color="#4f81bd")
            axis.set_ylabel(f"{tag}\n|2nd difference| (mm)")
            axis.legend(loc="upper right")
            axis.grid(alpha=0.3)
        axes[-1].set_xlabel("frame")
        figure.tight_layout()
        figure.savefig(args.plot, dpi=120)
        print(f"Wrote {args.plot}")


if __name__ == "__main__":
    main()
