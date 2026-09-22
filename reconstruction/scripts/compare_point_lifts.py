#!/usr/bin/env python3
"""
Lift one set of point tracks with two depth sources and compare them on the video.

CoTracker's tracks are 2D and know nothing about depth, so lifting the *same* tracks
with ZED stereo depth and with MoGe's monocular estimate isolates the depth: every
difference in the 3D points comes from it, not from the tracking.

Each output frame puts the two side by side -- points coloured by their own depth,
with a top-down inset of the lifted cloud. The inset is where the two part company:
a monocular estimate can place the object plausibly in the image and still bend the
cloud, or slide it along the view ray between frames.

Two numbers summarise each source:
  * how many tracked points it can lift at all (holes in the depth map lose points),
  * how much the distances between lifted points wander over time, which on a rigid
    object is pure lift error.

Run in the `genesis` env (or `sam3`; both have OpenCV):
    python compare_point_lifts.py --points points_cotracker.npz --frames-dir all_frames \\
        --depth-dirs depth_zed depth_moge --labels ZED MoGe --intrinsics intrinsics.yaml \\
        --out compare_depth_lift.mp4 --plot compare_depth_lift.png
"""

import argparse
import os

import cv2
import numpy as np
import yaml

from track_points_cotracker import lift, rigidity


def parse_args():
    parser = argparse.ArgumentParser(description="Compare depth sources behind one set of tracks")
    parser.add_argument("--points", required=True, help="points_cotracker.npz (tracks + visibility)")
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--depth-dirs", nargs=2, required=True, help="Two directories of depth PNGs")
    parser.add_argument("--labels", nargs=2, default=["ZED", "MoGe"])
    parser.add_argument("--intrinsics", required=True)
    parser.add_argument("--out", required=True, help="Side-by-side video (.mp4)")
    parser.add_argument("--plot", default=None, help="Summary figure (.png)")
    parser.add_argument("--object-mask", default=None,
                        help="masks root, to report depth on the object only")
    parser.add_argument("--object", default=None)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--depth-patch", type=int, default=2)
    parser.add_argument("--zfar", type=float, default=3.0)
    parser.add_argument("--trail", type=int, default=12)
    parser.add_argument("--scale", type=float, default=0.5, help="Scale of each panel")
    return parser.parse_args()


def lift_all(tracks, visible, depth_dir, names, calibration, patch, zfar):
    points_3d = np.full((len(names), tracks.shape[1], 3), np.nan)
    for index, name in enumerate(names):
        path = os.path.join(depth_dir, os.path.splitext(name)[0] + ".png")
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            continue
        lifted = lift(tracks[index], depth.astype(np.float64) / 1000.0, calibration, patch, zfar)
        lifted[~visible[index]] = np.nan
        points_3d[index] = lifted
    return points_3d


def panel(frame, tracks, visible, points_3d, index, limits, label, trail):
    """One source's view: points coloured by its depth, plus a top-down inset."""
    near, far, x_lo, x_hi = limits
    height, width = frame.shape[:2]
    canvas = frame.copy()
    depths = points_3d[..., 2]
    for point in range(tracks.shape[1]):
        if not visible[index, point]:
            continue
        u, v = tracks[index, point]
        if not (0 <= u < width and 0 <= v < height):
            continue
        z = depths[index, point]
        if np.isfinite(z):
            shade = int(np.clip((z - near) / max(far - near, 1e-6), 0, 1) * 255)
            colour = tuple(int(c) for c in cv2.applyColorMap(
                np.uint8([[shade]]), cv2.COLORMAP_TURBO)[0, 0])
        else:
            colour = (150, 150, 150)
        start = max(index - trail, 0)
        path = tracks[start:index + 1, point]
        for a, b in zip(path[:-1], path[1:]):
            cv2.line(canvas, tuple(a.astype(int)), tuple(b.astype(int)), colour, 1)
        cv2.circle(canvas, (int(u), int(v)), 3, colour, -1)

    inset_w, inset_h = int(width * 0.3), int(height * 0.36)
    inset = np.full((inset_h, inset_w, 3), 30, np.uint8)
    for point in range(points_3d.shape[1]):
        x, _, z = points_3d[index, point]
        if not np.isfinite(z):
            continue
        px = int((x - x_lo) / max(x_hi - x_lo, 1e-6) * (inset_w - 10)) + 5
        py = int((z - near) / max(far - near, 1e-6) * (inset_h - 10)) + 5
        if 0 <= px < inset_w and 0 <= py < inset_h:
            shade = int(np.clip((z - near) / max(far - near, 1e-6), 0, 1) * 255)
            colour = tuple(int(c) for c in cv2.applyColorMap(
                np.uint8([[shade]]), cv2.COLORMAP_TURBO)[0, 0])
            cv2.circle(inset, (px, py), 2, colour, -1)
    cv2.rectangle(inset, (0, 0), (inset_w - 1, inset_h - 1), (200, 200, 200), 1)
    cv2.putText(inset, f"top view  {near:.2f}-{far:.2f} m", (8, inset_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    canvas[10:10 + inset_h, width - inset_w - 10:width - 10] = inset

    lifted = np.isfinite(depths[index]) & visible[index]
    lines = [
        f"{label}   lifted {lifted.sum()}/{int(visible[index].sum())} visible",
        f"median depth {np.nanmedian(depths[index][lifted]):.3f} m" if lifted.any() else "no depth",
    ]
    for row, text in enumerate(lines):
        y = 36 + 36 * row
        cv2.putText(canvas, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
        cv2.putText(canvas, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return canvas


def main():
    args = parse_args()
    with open(args.intrinsics) as handle:
        values = yaml.safe_load(handle)
    calibration = tuple(float(values[k]) for k in ("fu", "fv", "pu", "pv"))

    data = np.load(args.points)
    tracks, visible = data["tracks"], data["visible"].astype(bool)
    names = sorted(
        f for f in os.listdir(args.frames_dir)
        if f.lower().endswith((".png", ".jpg")) and "_" not in f
    )[: len(tracks)]

    clouds = [
        lift_all(tracks, visible, directory, names, calibration, args.depth_patch, args.zfar)
        for directory in args.depth_dirs
    ]

    for label, cloud in zip(args.labels, clouds):
        lifted = np.isfinite(cloud[..., 2]) & visible
        spread = rigidity(cloud)
        print(f"{label:6s} lifted {lifted.sum() / max(visible.sum(), 1) * 100:5.1f}% of visible "
              f"point-frames | median depth {np.nanmedian(cloud[..., 2]):.3f} m | "
              f"pairwise distance spread median {np.median(spread) * 1000:5.1f} mm, "
              f"p90 {np.percentile(spread, 90) * 1000:6.1f} mm")

    both = np.isfinite(clouds[0][..., 2]) & np.isfinite(clouds[1][..., 2])
    if both.any():
        ratio = clouds[1][..., 2][both] / clouds[0][..., 2][both]
        print(f"{args.labels[1]} / {args.labels[0]} depth ratio: median {np.median(ratio):.3f} "
              f"(1.0 would be the same scale)")
        # After taking that scale out, what is left is shape error, not scale error.
        scaled = clouds[1][..., 2] / np.median(ratio)
        error = np.abs(scaled - clouds[0][..., 2])[both]
        print(f"depth difference after scaling: median {np.median(error) * 1000:.1f} mm, "
              f"p90 {np.percentile(error, 90) * 1000:.1f} mm")

    # One colour range per source: each is shown at its own scale, since a monocular
    # estimate has no metric scale to share.
    limits = []
    for cloud in clouds:
        finite = np.isfinite(cloud[..., 2])
        near, far = np.percentile(cloud[..., 2][finite], [2, 98]) if finite.any() else (0, 1)
        x_lo, x_hi = np.percentile(cloud[..., 0][finite], [1, 99]) if finite.any() else (-1, 1)
        limits.append((near, far, x_lo, x_hi))

    first = cv2.imread(os.path.join(args.frames_dir, names[0]))
    height, width = first.shape[:2]
    size = (int(width * args.scale), int(height * args.scale))
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                             (2 * size[0], size[1]))
    for index, name in enumerate(names):
        frame = cv2.imread(os.path.join(args.frames_dir, name))
        panels = [
            cv2.resize(panel(frame, tracks, visible, cloud, index, limit, label, args.trail), size)
            for cloud, limit, label in zip(clouds, limits, args.labels)
        ]
        strip = np.hstack(panels)
        cv2.putText(strip, f"frame {index}", (14, size[1] - 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0), 4)
        cv2.putText(strip, f"frame {index}", (14, size[1] - 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)
        writer.write(strip)
    writer.release()
    print(f"Wrote {args.out}: {len(names)} frames")

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        frames = np.arange(len(names))
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, constrained_layout=True)
        for label, cloud, colour in zip(args.labels, clouds, ("#c0504d", "#3a6ea5")):
            median_depth = np.nanmedian(cloud[..., 2], axis=1)
            axes[0].plot(frames, median_depth, color=colour, label=label)
            axes[1].plot(frames, (np.isfinite(cloud[..., 2]) & visible).sum(axis=1),
                         color=colour, label=label)
            centre = np.nanmedian(cloud, axis=1)
            speed = np.r_[0, np.linalg.norm(np.diff(centre, axis=0), axis=1)] * 1000
            axes[2].plot(frames, speed, color=colour, label=label)
        axes[0].set_ylabel("median depth (m)")
        axes[0].set_title("Same point tracks, lifted with each depth source")
        axes[1].set_ylabel("points lifted")
        axes[2].set_ylabel("cloud centre motion (mm/frame)")
        axes[2].set_xlabel("frame")
        for axis in axes:
            axis.legend(frameon=False)
            axis.spines[["top", "right"]].set_visible(False)
            axis.grid(alpha=0.25)
        fig.savefig(args.plot, dpi=120)
        print(f"Wrote {args.plot}")


if __name__ == "__main__":
    main()
