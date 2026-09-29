#!/usr/bin/env python3
"""
Per-frame rigid object pose from point tracking, following Video2Sim2Real.

Tracks 2D points on the object with BootsTAPIR, lifts them to 3D with per-frame
metric depth, and solves the frame-to-reference rigid transform with Kabsch under
RANSAC. This replaces do-as-i-do's guided pose-diffusion tracker, which needs
sam-3d-objects and ~32 GB of VRAM; nothing here needs either.

Run in the `tapnet` env:
    python track_object_v2s2r.py --frames-dir all_frames --masks-root video_segmentation/masks \\
        --depth-dir depth_moge --object cup --ref-frame 49 --intrinsics intrinsics.yaml \\
        --out object_traj.npz
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="V2S2R-style object pose tracking")
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--masks-root", required=True, help="video_segmentation/masks")
    parser.add_argument("--depth-dir", required=True, help="Per-frame uint16-mm depth PNGs")
    parser.add_argument("--object", required=True, help="Object id, e.g. 'cup'")
    parser.add_argument("--ref-frame", type=int, required=True)
    parser.add_argument("--intrinsics", required=True, help="YAML with fu/fv/pu/pv")
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint", default=os.environ.get("TAPNET_CKPT", ""))
    parser.add_argument("--num-points", type=int, default=256)
    parser.add_argument("--resize", type=int, nargs=2, default=[256, 256])
    parser.add_argument("--vis-threshold", type=float, default=0.5)
    parser.add_argument(
        "--min-points",
        type=int,
        default=15,
        help="Minimum inliers to accept a pose; below roughly 15 non-collinear "
        "correspondences the rigid solve is not meaningfully constrained",
    )
    parser.add_argument(
        "--low-confidence",
        type=float,
        default=0.1,
        help="Report how many frames fall below this normalised inlier fraction",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=48,
        help="Frames per TAPIR pass; the whole sequence at once needs more VRAM than is usually free",
    )
    parser.add_argument("--ransac-iters", type=int, default=200)
    parser.add_argument("--ransac-thresh", type=float, default=0.02, help="Inlier distance, metres")
    parser.add_argument("--max-step", type=float, default=0.06, help="Max plausible travel per frame, metres")
    parser.add_argument("--max-turn", type=float, default=20.0, help="Max plausible rotation per frame, degrees")
    parser.add_argument("--smooth-window", type=int, default=9)
    parser.add_argument("--ground-plane", default=None, help="ground_plane.json from fit_ground_plane.py")
    parser.add_argument("--mesh", default=None, help="Reference mesh, required with --ground-plane")
    parser.add_argument(
        "--rest-speed",
        type=float,
        default=0.004,
        help="Centroid speed below which the object counts as resting, metres per frame",
    )
    parser.add_argument(
        "--max-tilt",
        type=float,
        default=30.0,
        help="Largest tilt off the table normal allowed while held, degrees; raise it for pouring",
    )
    return parser.parse_args()


def kabsch(source, target):
    """Rigid transform taking `source` onto `target` (both (N, 3))."""
    source_centre = source.mean(axis=0)
    target_centre = target.mean(axis=0)
    covariance = (source - source_centre).T @ (target - target_centre)
    u, _, vt = np.linalg.svd(covariance)
    correction = np.diag([1.0, 1.0, np.sign(np.linalg.det(vt.T @ u.T))])
    rotation = vt.T @ correction @ u.T
    return rotation, target_centre - rotation @ source_centre


def kabsch_ransac(source, target, iterations, threshold, rng):
    """Kabsch with RANSAC, returning (rotation, translation, inlier_mask)."""
    best_inliers = None
    for _ in range(iterations):
        sample = rng.choice(len(source), 3, replace=False)
        try:
            rotation, translation = kabsch(source[sample], target[sample])
        except np.linalg.LinAlgError:
            continue
        residual = np.linalg.norm(source @ rotation.T + translation - target, axis=1)
        inliers = residual < threshold
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers
    if best_inliers is None or best_inliers.sum() < 3:
        return None, None, None
    rotation, translation = kabsch(source[best_inliers], target[best_inliers])
    return rotation, translation, best_inliers


def lift(points_2d, depth, fu, fv, pu, pv, patch=2):
    """Back-project pixel coordinates using a metric depth map. Returns (points, valid).

    Depth is taken as the median of a small patch: MoGe's per-pixel depth is noisy on
    the glossy mug and at silhouette edges, where a single sample can land on the
    background metres away and wreck the rigid fit.
    """
    height, width = depth.shape
    x = np.round(points_2d[:, 0]).astype(int)
    y = np.round(points_2d[:, 1]).astype(int)
    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x = np.clip(x, 0, width - 1)
    y = np.clip(y, 0, height - 1)

    z = np.zeros(len(x))
    for i, (xi, yi) in enumerate(zip(x, y)):
        window = depth[
            max(yi - patch, 0) : yi + patch + 1, max(xi - patch, 0) : xi + patch + 1
        ]
        positive = window[window > 0]
        z[i] = np.median(positive) if positive.size else 0.0

    valid = inside & (z > 0)
    return np.stack([(x - pu) * z / fu, (y - pv) * z / fv, z], axis=1), valid


def smooth_on_manifold(translations, rotations, window, confidence=None):
    """Smooth a pose sequence by weighted averaging over a sliding window.

    Rotations are averaged with scipy's Karcher/chordal mean rather than by filtering
    quaternion components independently and renormalising, which is not a manifold
    operation and distorts orientation when the window spans a large turn.

    `confidence` down-weights frames the rigid solve was unsure about, so a pose
    recovered from a handful of inliers is pulled toward its better-observed
    neighbours instead of dragging them toward it.
    """
    from scipy.spatial.transform import Rotation

    half = window // 2
    offsets = np.arange(-half, half + 1)
    kernel = np.exp(-0.5 * (offsets / (half / 2.0)) ** 2)

    n_frames = len(translations)
    smoothed_translations = np.empty_like(translations)
    smoothed_rotations = []

    for index in range(n_frames):
        neighbours = np.clip(index + offsets, 0, n_frames - 1)
        weights = kernel.copy()
        if confidence is not None:
            weights = weights * np.clip(confidence[neighbours], 1e-3, None)
        weights /= weights.sum()

        smoothed_translations[index] = weights @ translations[neighbours]
        smoothed_rotations.append(rotations[neighbours].mean(weights=weights))

    return smoothed_translations, Rotation.concatenate(smoothed_rotations)


def gate_poses(rotations, translations, valid, max_step, max_turn):
    """Drop frames whose solved pose cannot follow from the one before it.

    A near-cylindrical, textureless object leaves rotation about its axis weakly
    observable, so individual frames can solve to wild orientations while still
    reprojecting near the mask. Continuity is the constraint that catches those.

    Continuity, not distance from a reference pose: the object legitimately travels
    far (here the mug is lifted and set back down), so only an implausible jump
    between neighbouring frames indicates a bad solve.
    """
    from scipy.spatial.transform import Rotation

    valid = valid.copy()
    order = np.flatnonzero(valid)
    if len(order) < 3:
        return valid

    anchor = order[0]
    for index in order[1:]:
        gap = index - anchor
        step = np.linalg.norm(translations[index] - translations[anchor])
        turn = np.degrees(
            (Rotation.from_matrix(rotations[anchor]).inv() * Rotation.from_matrix(rotations[index])).magnitude()
        )
        if step > max_step * gap or turn > max_turn * gap:
            valid[index] = False
        else:
            anchor = index
    return valid


def clean_poses(rotations, translations, valid, max_step, max_turn, smooth_window, confidence=None):
    """Drop physically impossible frames, then interpolate and smooth what remains."""
    from scipy.spatial.transform import Rotation, Slerp

    valid = gate_poses(rotations, translations, valid, max_step, max_turn)
    order = np.flatnonzero(valid)
    if len(order) < 3:
        return rotations, translations, valid

    span = np.arange(order[0], order[-1] + 1)
    filled_translations = np.stack(
        [np.interp(span, order, translations[order, axis]) for axis in range(3)], axis=1
    )
    slerp = Slerp(order, Rotation.from_matrix(rotations[order]))
    filled_rotations = slerp(span)

    window = min(smooth_window, len(span) - (1 - len(span) % 2))
    if window >= 5:
        filled_translations, filled_rotations = smooth_on_manifold(
            filled_translations, filled_rotations, window, confidence[span] if confidence is not None else None
        )

    rotations = rotations.copy()
    translations = translations.copy()
    rotations[span] = filled_rotations.as_matrix()
    translations[span] = filled_translations
    cleaned = np.zeros_like(valid)
    cleaned[span] = True
    return rotations, translations, cleaned


def load_obj_vertices(path, max_vertices=20000):
    """Read vertex positions from an .obj, subsampled.

    Avoids a trimesh dependency in the tracking env, and the constraint only needs a
    centroid and an extreme point along one direction, for which a subsample of a
    few hundred thousand vertices is indistinguishable from the full mesh.
    """
    vertices = []
    with open(path) as handle:
        for line in handle:
            if line.startswith("v "):
                vertices.append([float(value) for value in line.split()[1:4]])
    vertices = np.array(vertices)
    if len(vertices) > max_vertices:
        step = len(vertices) // max_vertices
        vertices = vertices[::step]
    return vertices


def rotation_between(source, target):
    """Minimal rotation taking unit vector `source` onto unit vector `target`."""
    axis = np.cross(source, target)
    sine = np.linalg.norm(axis)
    cosine = float(source @ target)
    if sine < 1e-9:
        return np.eye(3) if cosine > 0 else -np.eye(3)
    skew = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    return np.eye(3) + skew + skew @ skew * ((1 - cosine) / sine**2)


def apply_ground_constraint(
    rotations, translations, valid, vertices, normal, rest_speed, max_tilt, smooth_window
):
    """Keep the object out of the table, and flat on it whenever it is at rest.

    Point tracking leaves the object free to sink through the table or hover above
    it, and leaves rotation about the object's axis barely observable. The supporting
    plane fixes both, and it is the frame Video2Sim2Real works in anyway. The plane
    offset is taken from the reference mesh rather than the depth fit, so the
    reference pose touches the table exactly by construction.
    """
    from scipy.ndimage import median_filter, uniform_filter1d
    from scipy.spatial.transform import Rotation

    rotations = rotations.copy()
    translations = translations.copy()
    offset = -float((vertices @ normal).min())
    centroid = vertices.mean(axis=0)

    positions = np.einsum("tij,j->ti", rotations, centroid) + translations
    speed = np.r_[0.0, np.linalg.norm(np.diff(positions, axis=0), axis=1)]
    resting = median_filter((speed < rest_speed).astype(float), size=9) > 0.5
    # A hard on/off rest flag steps the tilt limit from max to zero in one frame and
    # puts a visible snap in the trajectory; ramping it keeps the transition smooth.
    settled = uniform_filter1d(resting.astype(float), size=15)

    for index in np.flatnonzero(valid):
        rotation, translation = rotations[index], translations[index]

        # A near-180-degree rotation maps a cylinder onto itself, so Kabsch happily
        # returns an upside-down mug that fits the points about as well as the right
        # way up. Capping tilt (to zero while resting) removes that degeneracy while
        # leaving yaw about the normal, which the point tracks do constrain.
        limit = (1.0 - settled[index]) * np.radians(max_tilt)
        up = rotation @ normal
        tilt = float(np.arccos(np.clip(up @ normal, -1.0, 1.0)))
        if tilt > limit:
            axis = np.cross(up, normal)
            if np.linalg.norm(axis) > 1e-9:
                axis = axis / np.linalg.norm(axis)
                correction = Rotation.from_rotvec(axis * (tilt - limit)).as_matrix()
                centre = rotation @ centroid + translation
                rotation = correction @ rotation
                translation = correction @ (translation - centre) + centre

        lowest = ((vertices @ rotation.T + translation) @ normal + offset).min()
        # Never allow penetration; pull down to the surface only as the object settles.
        drop = lowest if lowest < 0.0 else lowest * settled[index]
        translation = translation - drop * normal

        rotations[index], translations[index] = rotation, translation

    order = np.flatnonzero(valid)
    window = min(smooth_window, len(order) - (1 - len(order) % 2))
    if window >= 5:
        # Snapping resting frames introduces steps at the rest/held boundaries.
        # This is the last smoothing the trajectory sees -- it runs after the
        # anchoring step rewrites translations -- so it has to be the manifold one.
        smoothed_translations, smoothed_rotations = smooth_on_manifold(
            translations[order], Rotation.from_matrix(rotations[order]), window
        )
        translations[order] = smoothed_translations
        rotations[order] = smoothed_rotations.as_matrix()

        for index in order:
            lowest = ((vertices @ rotations[index].T + translations[index]) @ normal + offset).min()
            if lowest < 0.0:
                translations[index] = translations[index] - lowest * normal

    return rotations, translations, resting


def main():
    args = parse_args()

    import cv2
    import mediapy as media
    from tapnet.torch import tapir_model
    from tapnet.utils import transforms

    with open(args.intrinsics) as handle:
        calibration = yaml.safe_load(handle)
    fu, fv, pu, pv = (float(calibration[k]) for k in ("fu", "fv", "pu", "pv"))

    names = sorted(f for f in os.listdir(args.frames_dir) if f.lower().endswith((".png", ".jpg")))
    video = np.stack([
        cv2.cvtColor(cv2.imread(os.path.join(args.frames_dir, n)), cv2.COLOR_BGR2RGB) for n in names
    ])
    n_frames, height, width = video.shape[:3]

    def mask_path(frame_idx):
        return os.path.join(args.masks_root, f"frame_{frame_idx:06d}_masks", f"{args.object}.png")

    ref_mask = cv2.imread(mask_path(args.ref_frame), 0) > 128
    ys, xs = np.nonzero(ref_mask)
    if len(ys) == 0:
        raise SystemExit(f"object '{args.object}' has an empty mask at frame {args.ref_frame}")

    # Spread queries over the mask area rather than along its principal axis: a rigid
    # fit needs points distributed in 3D, not collinear ones.
    rng = np.random.default_rng(0)
    pick = rng.choice(len(ys), min(args.num_points, len(ys)), replace=False)
    queries = np.stack(
        [np.full(len(pick), args.ref_frame, dtype=np.float32), ys[pick], xs[pick]], axis=1
    ).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = tapir_model.TAPIR(pyramid_level=1)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device).eval()

    resize_h, resize_w = args.resize
    queries_resized = transforms.convert_grid_coordinates(
        queries, (1, height, width), (1, resize_h, resize_w), coordinate_format="tyx"
    )
    frames = torch.tensor(media.resize_video(video, (resize_h, resize_w))).to(device)

    tracks = np.zeros((len(queries), n_frames, 2), dtype=np.float32)
    visible = np.zeros((len(queries), n_frames), dtype=bool)

    def sweep(order):
        """Track outwards from the reference frame, chaining chunk to chunk.

        Each chunk re-queries at its first frame using the previous chunk's final
        positions, which preserves point identity across the whole sequence.
        """
        points = queries_resized[:, 1:]
        position = 0
        while position < len(order) - 1:
            segment = order[position : position + args.chunk]
            local_queries = np.concatenate(
                [np.zeros((len(points), 1), dtype=np.float32), points], axis=1
            )
            with torch.no_grad():
                outputs = model(
                    (frames[segment].float() / 255 * 2 - 1)[None],
                    torch.tensor(local_queries).float().to(device)[None],
                )
                segment_tracks = outputs["tracks"][0].cpu().numpy()
                segment_visible = (
                    (1 - F.sigmoid(outputs["occlusion"][0]))
                    * (1 - F.sigmoid(outputs["expected_dist"][0]))
                    > args.vis_threshold
                ).cpu().numpy()
            tracks[:, segment] = segment_tracks
            visible[:, segment] = segment_visible
            points = segment_tracks[:, -1][:, ::-1].astype(np.float32)
            position += len(segment) - 1
            torch.cuda.empty_cache()

    print(f"[track] TAPIR over {n_frames} frames, {len(queries)} query points", flush=True)
    sweep(np.arange(args.ref_frame, n_frames))
    sweep(np.arange(args.ref_frame, -1, -1))

    tracks = transforms.convert_grid_coordinates(tracks, (resize_w, resize_h), (width, height))

    depth_names = sorted(f for f in os.listdir(args.depth_dir) if f.endswith(".png"))
    depths = [os.path.join(args.depth_dir, n) for n in depth_names]

    ref_depth = cv2.imread(depths[args.ref_frame], cv2.IMREAD_UNCHANGED).astype(np.float64) / 1000.0
    ref_points, ref_valid = lift(tracks[:, args.ref_frame], ref_depth, fu, fv, pu, pv)

    rotations = np.zeros((n_frames, 3, 3))
    translations = np.zeros((n_frames, 3))
    valid = np.zeros(n_frames, dtype=bool)
    inlier_counts = np.zeros(n_frames, dtype=int)

    for t in range(n_frames):
        depth = cv2.imread(depths[t], cv2.IMREAD_UNCHANGED).astype(np.float64) / 1000.0
        points, ok = lift(tracks[:, t], depth, fu, fv, pu, pv)

        mask_file = mask_path(t)
        if os.path.exists(mask_file):
            mask = cv2.imread(mask_file, 0) > 128
            x = np.clip(np.round(tracks[:, t, 0]).astype(int), 0, width - 1)
            y = np.clip(np.round(tracks[:, t, 1]).astype(int), 0, height - 1)
            # Points that drifted off the object (onto the hand or table) would
            # otherwise drag the rigid fit with them.
            ok &= mask[y, x]

        usable = ok & ref_valid & visible[:, t]
        if usable.sum() < args.min_points:
            continue

        rotation, translation, inliers = kabsch_ransac(
            ref_points[usable], points[usable], args.ransac_iters, args.ransac_thresh, rng
        )
        if rotation is None or inliers.sum() < args.min_points:
            continue

        rotations[t] = rotation
        translations[t] = translation
        inlier_counts[t] = int(inliers.sum())
        valid[t] = True

    # v(t): how well-observed each frame's solve was, as a fraction of the points
    # tracked on the object. This is the signal that says where vision alone suffices
    # and where it collapses -- consumers should gate on it rather than on `valid`,
    # which only records that a solve succeeded at all.
    confidence = inlier_counts / float(len(queries))

    solved = int(valid.sum())
    rotations, translations, valid = clean_poses(
        rotations, translations, valid, args.max_step, args.max_turn, args.smooth_window,
        confidence=confidence,
    )

    resting = np.zeros(n_frames, dtype=bool)
    if args.ground_plane:
        import json

        with open(args.ground_plane) as handle:
            normal = np.array(json.load(handle)["normal_camera"], dtype=float)
        mesh_vertices = load_obj_vertices(args.mesh)
        rotations, translations, resting = apply_ground_constraint(
            rotations, translations, valid, mesh_vertices, normal,
            args.rest_speed, args.max_tilt, args.smooth_window,
        )
        print(f"[ground] {resting[valid].sum()}/{valid.sum()} solved frames treated as resting")

    np.savez(
        args.out,
        rotation=rotations,
        translation=translations,
        valid=valid,
        inliers=inlier_counts,
        confidence=confidence,
        resting=resting,
        ref_frame=args.ref_frame,
        # The 2D tracks the poses were solved from, in the frame-first convention
        # track_points_cotracker.py writes and extract_keyframes.py reads. Keyframe
        # identification is defined on the pixel centroid of these tracks, so emitting
        # them lets the Video2Sim2Real keyframes come from the same tracker as the
        # poses rather than from a second tracking pass that would disagree with it.
        tracks=tracks.transpose(1, 0, 2),
        visible=visible.T,
        queries=queries[:, [2, 1]].astype(np.float32),
    )
    print(f"Wrote {args.out}: {solved}/{n_frames} frames solved, {valid.sum()} after cleanup")
    if valid.any():
        print(f"  median inliers {np.median(inlier_counts[valid]):.0f} of {len(queries)} points")
        weak = int((confidence[valid] < args.low_confidence).sum())
        print(
            f"  confidence v(t): median {np.median(confidence[valid]):.2f}, "
            f"{weak} frames below {args.low_confidence:.2f}"
        )


if __name__ == "__main__":
    main()
