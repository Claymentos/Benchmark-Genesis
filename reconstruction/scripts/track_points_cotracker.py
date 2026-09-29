#!/usr/bin/env python3
"""
Track points on the object with CoTracker, lift them to 3D with the depth maps, and
draw the result on the source video.

Point tracks say where the object's surface goes in the image; the depth map says how
far each of those pixels is. Together they give a 3D path per point without any mesh
or pose model, which makes this an independent read on the pose trackers: their
rigid-body fit has to agree with where these points actually went.

The overlay is the check. Points are drawn on the frames they came from, coloured by
depth, and a top-down inset shows the same points lifted into the camera's 3D space,
so an implausible lift (a point sliding along the view ray, a hole in the depth map)
is visible rather than hidden in a number.

Rigidity is reported too: on a rigid object the distances between lifted points
cannot change, so the spread of each pair's distance over time measures how much of
the 3D motion is noise -- lift error, not real motion.

Run in the `sam3` env (CoTracker lives there):
    python track_points_cotracker.py --frames-dir all_frames --depth-dir depth_zed \\
        --masks-root video_segmentation/masks --object cup --ref-frame 40 \\
        --intrinsics intrinsics.yaml --out cup_points.npz --overlay overlay_points.mp4
"""

import argparse
import os

import cv2
import numpy as np
import yaml

HUB_CACHE = os.path.expanduser("~/.cache/torch/hub/facebookresearch_co-tracker_main")


def parse_args():
    parser = argparse.ArgumentParser(description="CoTracker point tracks lifted with depth")
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--depth-dir", required=True, help="Per-frame uint16-millimetre PNGs")
    parser.add_argument("--masks-root", required=True, help="video_segmentation/masks")
    parser.add_argument("--object", required=True, help="Mask name, e.g. 'cup'")
    parser.add_argument("--ref-frame", type=int, required=True, help="Frame the points start on")
    parser.add_argument("--intrinsics", required=True, help="YAML with fu/fv/pu/pv")
    parser.add_argument("--out", required=True, help="Tracks and lifted points (.npz)")
    parser.add_argument("--overlay", default=None, help="Video showing the result (.mp4)")
    parser.add_argument(
        "--cotracker-dir",
        default=os.environ.get("COTRACKER_DIR", HUB_CACHE),
        help="CoTracker checkout; the torch.hub cache holds one",
    )
    parser.add_argument(
        "--checkpoint",
        default=os.path.expanduser("~/.cache/torch/hub/checkpoints/scaled_offline.pth"),
        help="CoTracker 3 offline weights. Offline sees the whole clip at once, which "
        "keeps points on the object through an occlusion better than the online model.",
    )
    parser.add_argument("--num-points", type=int, default=400, help="Points seeded on the object")
    parser.add_argument(
        "--track-width",
        type=int,
        default=512,
        help="Width the frames are fed at; CoTracker resizes to its own working size anyway",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=100,
        help="Frames tracked per pass. The offline model holds every frame's features at "
        "once, which does not fit for a long clip, so passes overlap by one frame and each "
        "one restarts from where the last left the points.",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--depth-patch",
        type=int,
        default=2,
        help="Half-width of the depth patch median-sampled at each point, pixels. A single "
        "pixel straddling a depth edge reads the background instead of the object.",
    )
    parser.add_argument("--zfar", type=float, default=3.0, help="Ignore depth beyond this, metres")
    parser.add_argument("--trail", type=int, default=12, help="Frames of trail drawn per point")
    parser.add_argument("--scale", type=float, default=1.0, help="Overlay scale factor")
    return parser.parse_args()


def seed_points(mask, count, rng):
    """Points spread over the object's mask at the reference frame."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise SystemExit("the object's mask is empty at the reference frame")
    if len(xs) <= count:
        chosen = np.arange(len(xs))
    else:
        # A stride keeps them spread over the mask rather than clumped, which a random
        # draw does on an elongated object.
        chosen = np.linspace(0, len(xs) - 1, count).astype(int)
    return np.stack([xs[chosen], ys[chosen]], axis=1).astype(np.float32)


def track_chunked(model, video, queries_xy, ref_frame, chunk, backward=False):
    """Track points over a long clip in overlapping passes.

    Each pass runs the offline model, which sees its whole window at once, then the
    next pass starts from the positions the last one ended at. Points are carried even
    when a pass loses sight of them, so a point can come back; visibility is AND-ed, so
    a point occluded in any pass is reported occluded there.
    """
    import torch

    n_frames = video.shape[1]
    order = (
        list(range(ref_frame, n_frames)) if not backward
        else list(range(ref_frame, -1, -1))
    )
    tracks = np.zeros((len(order), len(queries_xy), 2), np.float32)
    visible = np.zeros((len(order), len(queries_xy)), bool)
    start, positions = 0, queries_xy
    while start < len(order):
        stop = min(start + chunk, len(order))
        window = video[:, order[start:stop]]
        queries = np.concatenate(
            [np.zeros((len(positions), 1), np.float32), positions], axis=1
        )
        with torch.no_grad():
            chunk_tracks, chunk_visible = model(
                window, queries=torch.from_numpy(queries).float()[None].to(window.device)
            )
        tracks[start:stop] = chunk_tracks[0].cpu().numpy()
        visible[start:stop] = chunk_visible[0].cpu().numpy().astype(bool)
        positions = tracks[stop - 1]
        del chunk_tracks, chunk_visible, window
        torch.cuda.empty_cache()
        print(f"[cotracker] {'backward' if backward else 'forward'} "
              f"frames {order[start]}-{order[stop - 1]}", flush=True)
        # Overlap by one frame so the next pass starts from a tracked position.
        start = stop - 1 if stop < len(order) else stop
    return order, tracks, visible


def lift(points, depth, calibration, patch, zfar):
    """Pixels -> 3D camera points, using the median depth of a small patch."""
    fu, fv, pu, pv = calibration
    height, width = depth.shape
    lifted = np.full((len(points), 3), np.nan)
    for index, (u, v) in enumerate(points):
        u_i, v_i = int(round(u)), int(round(v))
        if not (0 <= u_i < width and 0 <= v_i < height):
            continue
        window = depth[
            max(v_i - patch, 0):v_i + patch + 1, max(u_i - patch, 0):u_i + patch + 1
        ]
        window = window[(window > 0.001) & (window < zfar)]
        if window.size == 0:
            continue
        z = float(np.median(window))
        lifted[index] = ((u - pu) * z / fu, (v - pv) * z / fv, z)
    return lifted


def rigidity(points_3d, pairs=2000, rng=None):
    """Spread of pairwise distances over time, metres: 0 for a perfectly rigid lift."""
    n_frames, n_points, _ = points_3d.shape
    rng = rng or np.random.default_rng(0)
    first = rng.integers(0, n_points, pairs)
    second = rng.integers(0, n_points, pairs)
    keep = first != second
    distances = np.linalg.norm(
        points_3d[:, first[keep]] - points_3d[:, second[keep]], axis=2
    )
    with np.errstate(invalid="ignore"):
        spread = np.nanstd(distances, axis=0)
    return spread[np.isfinite(spread)]


def main():
    args = parse_args()

    import torch

    with open(args.intrinsics) as handle:
        values = yaml.safe_load(handle)
    calibration = tuple(float(values[k]) for k in ("fu", "fv", "pu", "pv"))

    names = sorted(
        f for f in os.listdir(args.frames_dir)
        if f.lower().endswith((".png", ".jpg")) and "_" not in f
    )
    n_frames = len(names)
    frames = [cv2.imread(os.path.join(args.frames_dir, name)) for name in names]
    height, width = frames[0].shape[:2]

    mask_path = os.path.join(
        args.masks_root, f"frame_{args.ref_frame:06d}_masks", f"{args.object}.png"
    )
    mask = cv2.imread(mask_path, 0)
    if mask is None:
        raise SystemExit(f"no mask at {mask_path}")
    queries_xy = seed_points(mask > 128, args.num_points, np.random.default_rng(0))
    print(f"[cotracker] {len(queries_xy)} points seeded on '{args.object}' "
          f"at frame {args.ref_frame}", flush=True)

    # The clip is fed at a reduced width: CoTracker resizes internally, and the offline
    # model holds every frame at once.
    scale = args.track_width / width
    small = [cv2.resize(f, (args.track_width, int(round(height * scale)))) for f in frames]
    video = torch.from_numpy(np.stack(small)[..., ::-1].copy()).permute(0, 3, 1, 2).float()
    video = video[None].cuda()

    import sys

    sys.path.insert(0, args.cotracker_dir)
    from cotracker.predictor import CoTrackerPredictor

    model = CoTrackerPredictor(
        checkpoint=args.checkpoint, v2=False, offline=True, window_len=60
    ).cuda()
    print(f"[cotracker] tracking {n_frames} frames at {args.track_width} px wide, "
          f"{args.chunk} per pass", flush=True)

    tracks = np.zeros((n_frames, len(queries_xy), 2), np.float32)
    visible = np.zeros((n_frames, len(queries_xy)), bool)
    for backward in (False, True):
        order, chunk_tracks, chunk_visible = track_chunked(
            model, video, queries_xy * scale, args.ref_frame, args.chunk, backward
        )
        tracks[order] = chunk_tracks
        visible[order] = chunk_visible
    tracks /= scale  # back to full resolution
    del video
    torch.cuda.empty_cache()

    points_3d = np.full((n_frames, len(queries_xy), 3), np.nan)
    for index in range(n_frames):
        depth_path = os.path.join(args.depth_dir, f"{os.path.splitext(names[index])[0]}.png")
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            continue
        depth = depth.astype(np.float64) / 1000.0
        lifted = lift(tracks[index], depth, calibration, args.depth_patch, args.zfar)
        lifted[~visible[index]] = np.nan
        points_3d[index] = lifted

    lifted_fraction = np.isfinite(points_3d[..., 2]) & visible
    print(f"[cotracker] visible {visible.mean() * 100:.0f}% of point-frames, "
          f"lifted {lifted_fraction.mean() * 100:.0f}% (the rest have no depth there)")
    spread = rigidity(points_3d)
    if spread.size:
        print(f"[cotracker] pairwise distance spread over time: median "
              f"{np.median(spread) * 1000:.1f} mm, p90 {np.percentile(spread, 90) * 1000:.1f} mm "
              f"(0 would be a perfectly rigid lift)")

    np.savez(
        args.out,
        tracks=tracks,
        visible=visible,
        points_3d=points_3d,
        queries=queries_xy,
        ref_frame=args.ref_frame,
        fps=args.fps,
    )
    print(f"Wrote {args.out}")

    if args.overlay:
        draw_overlay(args, frames, names, tracks, visible, points_3d, calibration)


def draw_overlay(args, frames, names, tracks, visible, points_3d, calibration):
    """Video with the tracked points, coloured by depth, and a top-down 3D inset."""
    height, width = frames[0].shape[:2]
    size = (int(width * args.scale), int(height * args.scale))
    writer = cv2.VideoWriter(
        args.overlay, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, size
    )

    depths = points_3d[..., 2]
    finite = np.isfinite(depths)
    near, far = np.percentile(depths[finite], [2, 98]) if finite.any() else (0.0, 1.0)
    # Top-down inset: x across, z into the scene, over the whole clip's extent.
    x_all = points_3d[..., 0][finite]
    x_lo, x_hi = np.percentile(x_all, [1, 99]) if finite.any() else (-1, 1)
    z_lo, z_hi = near, far
    inset_w, inset_h = int(width * 0.26), int(height * 0.34)

    for index, frame in enumerate(frames):
        canvas = frame.copy()
        for point in range(tracks.shape[1]):
            u, v = tracks[index, point]
            if not (0 <= u < width and 0 <= v < height):
                continue
            z = depths[index, point]
            if not visible[index, point]:
                continue
            if np.isfinite(z):
                shade = int(np.clip((z - near) / max(far - near, 1e-6), 0, 1) * 255)
                colour = tuple(int(c) for c in cv2.applyColorMap(
                    np.uint8([[shade]]), cv2.COLORMAP_TURBO)[0, 0])
            else:
                colour = (150, 150, 150)  # tracked, but no depth to lift it with
            start = max(index - args.trail, 0)
            trail = tracks[start:index + 1, point]
            for a, b in zip(trail[:-1], trail[1:]):
                cv2.line(canvas, tuple(a.astype(int)), tuple(b.astype(int)), colour, 1)
            cv2.circle(canvas, (int(u), int(v)), 3, colour, -1)

        # Inset: the same points seen from above, which is where a bad lift shows.
        inset = np.full((inset_h, inset_w, 3), 30, np.uint8)
        for point in range(points_3d.shape[1]):
            x, _, z = points_3d[index, point]
            if not np.isfinite(z):
                continue
            px = int((x - x_lo) / max(x_hi - x_lo, 1e-6) * (inset_w - 10)) + 5
            py = int((z - z_lo) / max(z_hi - z_lo, 1e-6) * (inset_h - 10)) + 5
            if 0 <= px < inset_w and 0 <= py < inset_h:
                shade = int(np.clip((z - near) / max(far - near, 1e-6), 0, 1) * 255)
                colour = tuple(int(c) for c in cv2.applyColorMap(
                    np.uint8([[shade]]), cv2.COLORMAP_TURBO)[0, 0])
                cv2.circle(inset, (px, py), 2, colour, -1)
        cv2.rectangle(inset, (0, 0), (inset_w - 1, inset_h - 1), (200, 200, 200), 1)
        cv2.putText(inset, "top view (x, depth)", (8, inset_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        canvas[10:10 + inset_h, width - inset_w - 10:width - 10] = inset

        lifted = np.isfinite(depths[index]) & visible[index]
        lines = [
            f"frame {index}   visible {visible[index].sum()}/{tracks.shape[1]}"
            f"   lifted {lifted.sum()}",
            f"depth {near:.2f}-{far:.2f} m"
            + (f"   median {np.nanmedian(depths[index][lifted]):.3f} m" if lifted.any() else ""),
        ]
        for row, text in enumerate(lines):
            y = 34 + 34 * row
            cv2.putText(canvas, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
            cv2.putText(canvas, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        writer.write(cv2.resize(canvas, size) if args.scale != 1.0 else canvas)

    writer.release()
    print(f"Wrote {args.overlay}: {len(frames)} frames")


if __name__ == "__main__":
    main()
