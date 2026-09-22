#!/usr/bin/env python3
"""
Check MoGe point maps (get_pointmap_dir.py) against the ZED stereo depth, on the video.

A point map projects perfectly through its own predicted intrinsics by construction,
so drawing it that way says nothing about accuracy. The measured depth and the real
calibration are the reference instead. Each output frame is a 2x2 grid:

    point-map depth (scaled)       | ZED depth, same colour range
    depth error after scaling      | point-map points through the real camera

The top row shares one colour range, so shape differences show as colour differences.
The point map is not metric, so it is scaled per frame to the ZED median before
comparing; that scale is printed on the frame, and its drift over time is the
flicker a tracker built on these maps would inherit. The bottom right draws where
each point lands when projected with the calibrated intrinsics: a correct map sends
every point back to its own pixel, so the arrows are the pixel error (mostly the
wrong focal length), and the object's points show where the map places it.

A summary figure plots the per-frame scale and focal, and the object's depth from
both sources using a single global scale -- what a tracker without stereo would see.

Run in the `genesis` env:
    python overlay_pointmap.py --frames-dir all_frames --depth-dir depth_zed \\
        --intrinsics intrinsics.yaml --masks-root video_segmentation/masks --object cup \\
        --out overlay_pointmap.mp4 --plot pointmap_accuracy.png
"""

import argparse
import os

import cv2
import numpy as np
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Overlay MoGe point maps against ZED depth")
    parser.add_argument("--frames-dir", required=True, help="Frames with <n>_pointmap.npy beside them")
    parser.add_argument("--pointmap-dir", default=None, help="Default: --frames-dir")
    parser.add_argument("--depth-dir", required=True, help="Reference depth, uint16 millimetres")
    parser.add_argument("--intrinsics", required=True, help="Calibrated intrinsics.yaml")
    parser.add_argument("--masks-root", default=None)
    parser.add_argument("--object", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--plot", default=None, help="Summary figure (png)")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--zfar", type=float, default=3.0, help="Ignore reference depth beyond this, metres")
    parser.add_argument("--max-error", type=float, default=0.05, help="Error colour-scale top, metres")
    parser.add_argument("--grid", type=int, default=48, help="Arrow spacing in the reprojection panel, pixels")
    parser.add_argument("--scale", type=float, default=0.5, help="Size of each panel relative to the frame")
    return parser.parse_args()


def colourise(values, valid, low, high, colormap):
    norm = np.clip((values - low) / max(high - low, 1e-6), 0, 1)
    image = cv2.applyColorMap((norm * 255).astype(np.uint8), colormap)
    image[~valid] = 0
    return image


def blend(frame, overlay, valid, alpha=0.65):
    out = frame.copy()
    out[valid] = (alpha * overlay[valid] + (1 - alpha) * frame[valid]).astype(np.uint8)
    return out


def label(image, lines, colour=(255, 255, 255)):
    for row, text in enumerate(lines):
        y = 34 + 34 * row
        cv2.putText(image, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
        cv2.putText(image, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)


def outline(image, mask, colour=(255, 255, 255)):
    if mask is not None and mask.any():
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, colour, 2)


def main():
    args = parse_args()
    pointmap_dir = args.pointmap_dir or args.frames_dir

    with open(args.intrinsics) as handle:
        calibration = yaml.safe_load(handle)
    fu, fv, pu, pv = (float(calibration[k]) for k in ("fu", "fv", "pu", "pv"))

    names = sorted(
        f for f in os.listdir(args.frames_dir)
        if f.lower().endswith((".png", ".jpg"))
        and os.path.exists(os.path.join(pointmap_dir, os.path.splitext(f)[0] + "_pointmap.npy"))
    )
    if not names:
        raise SystemExit(f"no <frame>_pointmap.npy found in {pointmap_dir}")

    first = cv2.imread(os.path.join(args.frames_dir, names[0]))
    height, width = first.shape[:2]
    panel = (int(width * args.scale), int(height * args.scale))
    writer = cv2.VideoWriter(
        args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (2 * panel[0], 2 * panel[1])
    )

    grid_v, grid_u = np.mgrid[args.grid // 2:height:args.grid, args.grid // 2:width:args.grid]
    series = {k: [] for k in ("frame", "scale", "focal", "abs_rel", "pixel_err",
                              "obj_zed", "obj_pm_raw", "obj_err")}

    for name in names:
        stem = os.path.splitext(name)[0]
        index = int(stem)
        frame = cv2.imread(os.path.join(args.frames_dir, name))
        points = np.load(os.path.join(pointmap_dir, f"{stem}_pointmap.npy")).astype(np.float64)
        if points.shape[0] == 3:
            points = points.transpose(1, 2, 0)
        intrinsics_path = os.path.join(pointmap_dir, f"{stem}_intrinsics.npy")
        focal = float(np.load(intrinsics_path)[0, 0]) if os.path.exists(intrinsics_path) else np.nan

        zed = cv2.imread(os.path.join(args.depth_dir, f"{stem}.png"), cv2.IMREAD_UNCHANGED)
        zed = zed.astype(np.float64) / 1000.0
        pm_z = points[..., 2]
        valid_zed = (zed > 0.001) & (zed < args.zfar)
        valid_pm = np.isfinite(pm_z) & (pm_z > 1e-6)
        both = valid_zed & valid_pm

        mask = None
        if args.masks_root and args.object:
            path = os.path.join(args.masks_root, f"frame_{index:06d}_masks", f"{args.object}.png")
            if os.path.exists(path):
                mask = cv2.imread(path, 0) > 128

        # Per-frame scale: the map's depth is only defined up to scale.
        scale = np.median(zed[both] / pm_z[both]) if both.any() else np.nan
        scaled = points * scale
        error = np.abs(scaled[..., 2] - zed)

        low, high = np.percentile(zed[valid_zed], [2, 98]) if valid_zed.any() else (0, 1)
        top_left = blend(frame, colourise(scaled[..., 2], valid_pm, low, high, cv2.COLORMAP_TURBO), valid_pm)
        top_right = blend(frame, colourise(zed, valid_zed, low, high, cv2.COLORMAP_TURBO), valid_zed)
        bottom_left = blend(
            frame, colourise(error, both, 0, args.max_error, cv2.COLORMAP_INFERNO), both, alpha=0.75
        )
        for image in (top_left, top_right, bottom_left):
            outline(image, mask)

        # Reprojection through the real camera: each point should return to its pixel.
        with np.errstate(divide="ignore", invalid="ignore"):
            proj_u = fu * points[..., 0] / points[..., 2] + pu
            proj_v = fv * points[..., 1] / points[..., 2] + pv
        bottom_right = (frame * 0.6).astype(np.uint8)
        pixel_u, pixel_v = np.meshgrid(np.arange(width), np.arange(height))
        displacement = np.hypot(proj_u - pixel_u, proj_v - pixel_v)
        for v, u in zip(grid_v.ravel(), grid_u.ravel()):
            if valid_pm[v, u] and np.isfinite(proj_u[v, u]):
                cv2.arrowedLine(
                    bottom_right, (int(u), int(v)), (int(proj_u[v, u]), int(proj_v[v, u])),
                    (80, 200, 255), 1, tipLength=0.15,
                )
        obj_zed = obj_pm = obj_err = np.nan
        if mask is not None and mask.any():
            outline(bottom_right, mask)
            vs, us = np.nonzero(mask & valid_pm)
            keep = slice(None, None, max(1, len(vs) // 1500))
            for u, v in zip(proj_u[vs[keep], us[keep]], proj_v[vs[keep], us[keep]]):
                if 0 <= u < width and 0 <= v < height:
                    cv2.circle(bottom_right, (int(u), int(v)), 2, (60, 60, 255), -1)
            object_both = mask & both
            if object_both.any():
                obj_zed = np.median(zed[object_both])
                obj_pm = np.median(pm_z[object_both])
                obj_err = np.median(error[object_both])

        abs_rel = np.median(error[both] / zed[both]) if both.any() else np.nan
        pixel_err = np.median(displacement[valid_pm]) if valid_pm.any() else np.nan

        label(top_left, [f"point map x{scale:.3f}", f"MoGe focal {focal:.0f} px"])
        label(top_right, [f"ZED depth  {low:.2f}-{high:.2f} m", f"calibrated focal {fu:.0f} px"])
        label(bottom_left, [
            f"|error| after scaling, 0-{args.max_error * 100:.0f} cm",
            f"median rel. {abs_rel * 100:.1f}%"
            + (f" | {args.object} {obj_err * 100:.1f} cm" if np.isfinite(obj_err) else ""),
        ])
        label(bottom_right, [
            "points through the real camera",
            f"median shift {pixel_err:.0f} px   frame {index}",
        ])

        tiles = [cv2.resize(t, panel) for t in (top_left, top_right, bottom_left, bottom_right)]
        writer.write(np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]))

        for key, value in zip(series, (index, scale, focal, abs_rel, pixel_err,
                                       obj_zed, obj_pm, obj_err)):
            series[key].append(value)

    writer.release()
    data = {k: np.asarray(v, dtype=float) for k, v in series.items()}
    print(f"Wrote {args.out}: {len(names)} frames")

    # A tracker without stereo gets one scale at best, not one per frame.
    global_scale = np.nanmedian(data["scale"])
    obj_pm = data["obj_pm_raw"] * global_scale
    scale_jitter = np.nanmedian(np.abs(np.diff(data["scale"]))) / global_scale
    obj_gap = np.abs(obj_pm - data["obj_zed"])
    print(f"  per-frame scale: median {global_scale:.3f}, frame-to-frame change {scale_jitter * 100:.1f}%")
    print(f"  focal: point map {np.nanmedian(data['focal']):.0f} px "
          f"({np.nanmin(data['focal']):.0f}-{np.nanmax(data['focal']):.0f}) vs calibrated {fu:.0f} px")
    print(f"  depth error after per-frame scaling: median {np.nanmedian(data['abs_rel']) * 100:.1f}% relative")
    print(f"  reprojection shift through the real camera: median {np.nanmedian(data['pixel_err']):.0f} px")
    if np.isfinite(data["obj_zed"]).any():
        print(f"  {args.object} depth, per-frame scale: median error {np.nanmedian(data['obj_err']) * 100:.1f} cm")
        print(f"  {args.object} depth, one global scale: median error {np.nanmedian(obj_gap) * 100:.1f} cm, "
              f"p90 {np.nanpercentile(obj_gap, 90) * 100:.1f} cm")

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        frames = data["frame"]
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, constrained_layout=True)
        axes[0].plot(frames, data["scale"], color="#3a6ea5")
        axes[0].axhline(global_scale, color="#888888", linestyle="--", linewidth=1)
        axes[0].set_ylabel("scale to ZED")
        axes[0].set_title("Point-map scale per frame (dashed: median)")
        axes[1].plot(frames, data["focal"], color="#3a6ea5", label="point map")
        axes[1].axhline(fu, color="#c0504d", linewidth=1.5, label="calibrated")
        axes[1].set_ylabel("focal (px)")
        axes[1].legend(frameon=False)
        if np.isfinite(data["obj_zed"]).any():
            axes[2].plot(frames, data["obj_zed"], color="#c0504d", label="ZED")
            axes[2].plot(frames, obj_pm, color="#3a6ea5", label="point map x global scale")
            axes[2].set_ylabel(f"{args.object} depth (m)")
            axes[2].legend(frameon=False)
        axes[2].set_xlabel("frame")
        for axis in axes:
            axis.spines[["top", "right"]].set_visible(False)
            axis.grid(alpha=0.25)
        fig.savefig(args.plot, dpi=120)
        print(f"Wrote {args.plot}")


if __name__ == "__main__":
    main()
