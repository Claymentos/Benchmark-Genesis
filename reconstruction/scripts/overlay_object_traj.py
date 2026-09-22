#!/usr/bin/env python3
"""
Project a tracked object trajectory back onto the source video.

The definitive accuracy check: everything else in this pipeline measures a
trajectory against itself, which says nothing about whether the pose is *right*.
Drawing the posed mesh over the frames it came from compares it against the only
ground truth available, and unlike centroid metrics it shows orientation -- an object
rocking in place barely moves its centroid but is obvious here.

Several trajectories can be overlaid at once (raw vs anchored, say) to see what each
stage changed. The object's mask contour is drawn too when available, so silhouette
disagreement is visible directly.

    python overlay_object_traj.py --frames-dir all_frames --mesh posed_mesh.obj \\
        --intrinsics intrinsics.yaml --traj cup_traj_raw.npz cup_traj.npz \\
        --labels raw anchored --masks-root video_segmentation/masks --object cup \\
        --out overlay.mp4
"""

import argparse
import os

import numpy as np
import yaml

# BGR, kept distinguishable on the mostly-neutral scenes these videos contain.
COLOURS = [(60, 220, 60), (60, 160, 255), (255, 200, 60), (200, 60, 255)]
MASK_COLOUR = (255, 255, 255)


def parse_args():
    parser = argparse.ArgumentParser(description="Overlay object trajectories on the source video")
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument(
        "--mesh",
        nargs="+",
        required=True,
        help="Object mesh; give one per trajectory when they were tracked against different meshes",
    )
    parser.add_argument("--intrinsics", required=True)
    parser.add_argument("--traj", nargs="+", required=True, help="One or more trajectory .npz files")
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--masks-root", default=None)
    parser.add_argument("--object", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--points", type=int, default=3000, help="Mesh vertices drawn per frame")
    parser.add_argument("--axis-length", type=float, default=0.05, help="Pose axis length, metres")
    parser.add_argument("--scale", type=float, default=1.0, help="Output scale factor")
    return parser.parse_args()


def project(points, fu, fv, pu, pv):
    """Pinhole projection; points behind the camera are marked invalid."""
    in_front = points[:, 2] > 1e-6
    z = np.where(in_front, points[:, 2], 1.0)
    u = fu * points[:, 0] / z + pu
    v = fv * points[:, 1] / z + pv
    return np.stack([u, v], axis=1), in_front


def main():
    args = parse_args()

    import cv2
    import trimesh

    with open(args.intrinsics) as handle:
        calibration = yaml.safe_load(handle)
    fu, fv, pu, pv = (float(calibration[k]) for k in ("fu", "fv", "pu", "pv"))

    trajectories = [np.load(path) for path in args.traj]

    meshes = args.mesh if len(args.mesh) > 1 else args.mesh * len(trajectories)
    if len(meshes) != len(trajectories):
        raise SystemExit("give either one mesh or exactly one per trajectory")
    sampled, centroids = [], []
    for path in meshes:
        mesh = trimesh.load(path, process=False)
        vertices = np.asarray(mesh.vertices)
        centroids.append(vertices.mean(axis=0))
        if len(vertices) > args.points:
            vertices = vertices[:: len(vertices) // args.points]
        sampled.append(vertices)
    labels = args.labels or [os.path.basename(p).replace(".npz", "") for p in args.traj]

    names = sorted(f for f in os.listdir(args.frames_dir) if f.lower().endswith((".png", ".jpg")))
    first = cv2.imread(os.path.join(args.frames_dir, names[0]))
    height, width = first.shape[:2]
    size = (int(width * args.scale), int(height * args.scale))

    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, size)

    for index, name in enumerate(names):
        frame = cv2.imread(os.path.join(args.frames_dir, name))

        if args.masks_root and args.object:
            mask_path = os.path.join(
                args.masks_root, f"frame_{index:06d}_masks", f"{args.object}.png"
            )
            mask = cv2.imread(mask_path, 0) if os.path.exists(mask_path) else None
            if mask is not None and (mask > 128).any():
                contours, _ = cv2.findContours(
                    (mask > 128).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(frame, contours, -1, MASK_COLOUR, 2)

        for slot, (traj, label) in enumerate(zip(trajectories, labels)):
            colour = COLOURS[slot % len(COLOURS)]
            if index >= len(traj["valid"]) or not bool(traj["valid"][index]):
                continue
            rotation = traj["rotation"][index]
            translation = traj["translation"][index]

            posed = sampled[slot] @ rotation.T + translation
            pixels, ok = project(posed, fu, fv, pu, pv)
            for u, v in pixels[ok].astype(int):
                if 0 <= u < width and 0 <= v < height:
                    frame[v, u] = colour

            # Axes at the object's own origin make orientation legible: pure spin
            # leaves the projected points looking almost unchanged.
            origin = centroids[slot] @ rotation.T + translation
            tips = (centroids[slot] + np.eye(3) * args.axis_length) @ rotation.T + translation
            centre_px, centre_ok = project(origin[None], fu, fv, pu, pv)
            tip_px, tip_ok = project(tips, fu, fv, pu, pv)
            if centre_ok[0]:
                for axis in range(3):
                    if tip_ok[axis]:
                        cv2.line(
                            frame,
                            tuple(centre_px[0].astype(int)),
                            tuple(tip_px[axis].astype(int)),
                            colour,
                            2,
                        )

            confidence = traj["confidence"][index] if "confidence" in traj.files else None
            text = f"{label}" + (f"  v={confidence:.2f}" if confidence is not None else "")
            cv2.putText(
                frame, text, (12, 30 + 26 * slot), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2
            )

        cv2.putText(
            frame, f"frame {index}", (12, height - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            MASK_COLOUR, 2,
        )
        writer.write(cv2.resize(frame, size) if args.scale != 1.0 else frame)

    writer.release()
    print(f"Wrote {args.out}: {len(names)} frames")


if __name__ == "__main__":
    main()
