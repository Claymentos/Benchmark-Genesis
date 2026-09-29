#!/usr/bin/env python3
"""
Object-centric keyframes, the way Video2Sim2Real identifies them.

From "Video2Sim2Real: Full-Stack Autonomous Dexterous Skill Acquisition from a Single
Human Video" (arXiv 2606.08828), appendix A.2.1 "Keyframe Identification": three
frames are picked out of the clip -- contact, interaction and detachment -- and they
become the anchors its refinement optimises the robot's configuration at.

What matters about the criterion is that it reads the *object*, not the hand. A
grasp-window detector asks when the hand was near the surface, which is a statement
about the reconstruction of the hand; this asks when the object started moving, which
is a statement about the demonstration's effect on the world. The two disagree
whenever the hand hovers before it grips, or keeps holding after it has set the object
down, and it is the object's answer the refinement wants.

  contact      the earliest frame from which the object's tracked centroid stays
               displaced for L_c frames together:

                   c_t = (1/K) sum_k u_k_t          2D pixel centroid of the tracks
                   d_t = || c_t - c_1 ||            displacement from the first frame
                   m_t = 1(d_t > delta_c)
                   T_c = min { t : sum_{j=t}^{t+L_c-1} m_j = L_c }

               with delta_c = 4 px and L_c = 50 frames, the paper's values, fixed
               across its tasks. The streak is what makes it robust: a single frame of
               tracker jitter clears 4 px easily, fifty consecutive ones do not.

  interaction  for a two-object task, the frame from which the manipulated object
               stops moving relative to the target -- the pour, the insertion, the
               placement:

                   T_i = min { t >= T_c : sum_{j=t}^{t+L_i-1} 1(s_j < delta_i) = L_i }

               over the relative speed smoothed with a window of 33, delta_i = 1.0 and
               L_i = 3. Needs a second object's tracks (--target-points); skipped for
               a single-object clip, which is what this pipeline records today.

  detachment   where the object comes back down: its height above the supporting
               plane crosses a threshold downward, having first risen above it. The
               paper uses 33 cm and words it as "the first frame after Tc where the
               manipulated object's height falls below 33 cm"; the crossing is read in
               because at contact the object is still standing on the table, already
               below any threshold, so the literal rule returns the frame after
               contact whatever the clip contains. See detachment_frame.

The 33 cm is worth checking against your own setup rather than taking on faith. On
these ZED recordings the mug's centroid peaks 22.6 cm above the table, so it never
crosses 33 cm and there is no detachment frame at all. --detach-height auto sets the
threshold from the object's own lift instead, a quarter of the way from where it
rests to where it peaks, which on take002 puts it at 7.6 cm and finds the set-down.

Run in the `genesis` env:
    python extract_keyframes.py --points points_cotracker.npz \\
        --object-poses cup_traj_points.npz --mesh posed_mesh.obj \\
        --ground-plane ground_plane.json --frames-dir all_frames \\
        --out keyframes.json --strip keyframes.png
"""

import argparse
import json
import os

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Video2Sim2Real object-centric keyframes")
    parser.add_argument("--points", required=True, help="points_cotracker.npz (the manipulated object)")
    parser.add_argument("--target-points", default=None,
                        help="A second object's tracks, for the interaction frame")
    parser.add_argument("--object-poses", default=None, help="Object trajectory .npz, for detachment")
    parser.add_argument("--mesh", default=None, help="Object mesh, for its height")
    parser.add_argument("--ground-plane", default=None, help="ground_plane.json")
    parser.add_argument("--frames-dir", default=None, help="Frames, to draw the keyframes")
    parser.add_argument("--out", required=True, help="keyframes.json")
    parser.add_argument("--strip", default=None, help="Write a labelled image of the keyframes")
    parser.add_argument("--plot", default=None, help="Write the signals the decision was made on")

    # The paper's values, fixed across its tasks.
    parser.add_argument("--contact-displacement", type=float, default=4.0,
                        help="delta_c: centroid displacement counting as motion, pixels")
    parser.add_argument("--contact-streak", type=int, default=50,
                        help="L_c: consecutive moving frames required")
    parser.add_argument("--interaction-speed", type=float, default=1.0,
                        help="delta_i: relative speed counting as settled, pixels/frame")
    parser.add_argument("--interaction-streak", type=int, default=3, help="L_i")
    parser.add_argument("--interaction-smooth", type=int, default=33,
                        help="Window the relative speed is smoothed with")
    parser.add_argument("--detach-height", default="0.33",
                        help="Height above the table below which the object counts as set "
                             "down, metres; or 'auto' to take it from the object's own lift")
    parser.add_argument("--detach-fraction", type=float, default=0.25,
                        help="With 'auto', where between the resting and peak height to put it")
    parser.add_argument("--height-from", choices=("centroid", "lowest"), default="centroid",
                        help="Which point of the object its height is measured at")
    parser.add_argument("--visible-only", action="store_true",
                        help="Average only the tracks CoTracker calls visible. The paper "
                             "averages all of them, which is the default here")
    return parser.parse_args()


def centroids(tracks, visible, visible_only):
    """Per-frame 2D pixel centroid of the tracked points."""
    if not visible_only:
        return tracks.mean(axis=1)
    weights = visible.astype(float)
    weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-9)
    return np.einsum("tk,tkc->tc", weights, tracks)


def first_streak(flags, length):
    """Earliest index from which `flags` holds for `length` frames together."""
    if length <= 0 or len(flags) < length:
        return None
    # A run of `length` Trues is a windowed sum equal to `length`.
    window = np.convolve(flags.astype(int), np.ones(length, dtype=int), mode="valid")
    hits = np.flatnonzero(window == length)
    return int(hits[0]) if len(hits) else None


def contact_frame(tracks, visible, args):
    """T_c: the earliest frame from which the object stays displaced."""
    centre = centroids(tracks, visible, args.visible_only)
    displacement = np.linalg.norm(centre - centre[0], axis=1)
    moving = displacement > args.contact_displacement
    return first_streak(moving, args.contact_streak), displacement


def interaction_frame(tracks, visible, target_tracks, target_visible, start, args):
    """T_i: the frame from which the two objects stop moving relative to each other."""
    centre = centroids(tracks, visible, args.visible_only)
    target = centroids(target_tracks, target_visible, args.visible_only)
    # s_t = || r_t - r_{t-1} ||, over r_t = c_manip_t - c_target_t.
    offset = centre - target
    relative = np.concatenate([[0.0], np.linalg.norm(np.diff(offset, axis=0), axis=1)])
    window = max(1, args.interaction_smooth)
    # Padded with the end values rather than convolved straight: a plain 'same'
    # convolution averages the last half-window against implicit zeros, which drags
    # the smoothed speed down near the end of the clip and can report an interaction
    # the objects never had.
    pad = window // 2
    padded = np.pad(relative, pad, mode="edge")
    smoothed = np.convolve(padded, np.ones(window) / window, mode="valid")[: len(relative)]

    settled = smoothed < args.interaction_speed
    settled[:start] = False
    return first_streak(settled, args.interaction_streak), smoothed


def object_heights(args):
    """Height of the object above the supporting plane, per frame."""
    import trimesh

    with open(args.ground_plane) as handle:
        plane = json.load(handle)
    normal = np.asarray(plane["normal_camera"], dtype=float)
    norm = np.linalg.norm(normal)
    normal, offset = normal / norm, float(plane["offset"]) / norm

    track = np.load(args.object_poses)
    vertices = np.asarray(trimesh.load(args.mesh, process=False).vertices)
    if args.height_from == "centroid":
        points = vertices.mean(axis=0)[None, :]
    else:
        points = vertices[:: max(1, len(vertices) // 3000)]

    heights = np.full(len(track["rotation"]), np.nan)
    solved = track["valid"].astype(bool)
    for index in np.flatnonzero(solved):
        camera = points @ track["rotation"][index].T + track["translation"][index]
        heights[index] = (camera @ normal + offset).min()
    return heights, solved


def detachment_frame(heights, solved, start, args):
    """T_d: where the object comes back down after having been lifted.

    Read as a downward crossing rather than literally. The paper says "the first frame
    after Tc where the manipulated object's height falls below 33 cm above the table",
    but at contact the object is still standing on the table and so is already below
    any threshold worth setting -- taken word for word the rule returns the frame
    straight after contact and means nothing. It only becomes the drop-off it is
    named for once the object has first risen above the threshold, so that is what is
    required here.
    """
    usable = heights[solved]
    if args.detach_height == "auto":
        resting, peak = float(np.nanmin(usable)), float(np.nanmax(usable))
        threshold = resting + args.detach_fraction * (peak - resting)
    else:
        threshold = float(args.detach_height)

    below = np.zeros(len(heights), dtype=bool)
    above = np.zeros(len(heights), dtype=bool)
    below[solved] = heights[solved] < threshold
    above[solved] = heights[solved] >= threshold
    # Only frames that some earlier frame, after contact, has risen above.
    lifted = np.zeros(len(heights), dtype=bool)
    lifted[start + 1:] = np.maximum.accumulate(above[start:-1]) if start + 1 < len(heights) else False
    hits = np.flatnonzero(below & lifted)
    return (int(hits[0]) if len(hits) else None), threshold


def draw_strip(args, keyframes):
    """The keyframes themselves, side by side and labelled."""
    import cv2

    tiles = []
    for name, frame in keyframes.items():
        if frame is None:
            continue
        path = os.path.join(args.frames_dir, f"{frame:06d}.png")
        if not os.path.exists(path):
            continue
        image = cv2.resize(cv2.imread(path), (640, 360))
        cv2.rectangle(image, (0, 0), (640, 34), (0, 0, 0), -1)
        cv2.putText(image, f"{name}  frame {frame}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        tiles.append(image)
    if tiles:
        cv2.imwrite(args.strip, np.hstack(tiles))
        print(f"Wrote {args.strip}")


def draw_plot(args, displacement, heights, keyframes, threshold):
    """The signals the decision was made on, so a wrong keyframe is diagnosable."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(displacement, color="#3b7dd8")
    axes[0].axhline(args.contact_displacement, color="#888", linestyle="--",
                    label=f"delta_c = {args.contact_displacement:g} px")
    axes[0].set_ylabel("centroid displacement (px)")
    axes[0].legend(loc="upper left")

    if heights is not None:
        axes[1].plot(100 * heights, color="#d87a3b")
        axes[1].axhline(100 * threshold, color="#888", linestyle="--",
                        label=f"detach at {100 * threshold:.1f} cm")
        axes[1].legend(loc="upper left")
    axes[1].set_ylabel("height above table (cm)")
    axes[1].set_xlabel("frame")

    colours = {"contact": "#2e8b57", "interaction": "#8b2e8b", "detachment": "#b22222"}
    for name, frame in keyframes.items():
        if frame is None:
            continue
        for axis in axes:
            axis.axvline(frame, color=colours.get(name, "#333"), linewidth=1.6)
        axes[0].annotate(f"{name}\n{frame}", (frame, axes[0].get_ylim()[1]),
                         color=colours.get(name, "#333"), fontsize=8,
                         ha="center", va="top")
    figure.tight_layout()
    figure.savefig(args.plot, dpi=130)
    print(f"Wrote {args.plot}")


def main():
    args = parse_args()
    data = np.load(args.points)
    tracks, visible = data["tracks"], data["visible"]
    n_frames = len(tracks)

    notes = {}
    contact, displacement = contact_frame(tracks, visible, args)
    if contact is None:
        notes["contact"] = (f"searched: the centroid never stayed beyond "
                            f"{args.contact_displacement:g} px for {args.contact_streak} "
                            f"frames (peak {displacement.max():.1f} px)")
    if contact is None:
        print(f"[keyframes] no contact frame: the centroid never stayed beyond "
              f"{args.contact_displacement:g} px for {args.contact_streak} frames "
              f"(it reached {displacement.max():.1f} px)")
    else:
        print(f"[keyframes] contact     frame {contact:4d}  "
              f"(centroid displaced past {args.contact_displacement:g} px and stayed there)")

    interaction = None
    if args.target_points:
        target = np.load(args.target_points)
        interaction, _ = interaction_frame(
            tracks, visible, target["tracks"], target["visible"], contact or 0, args
        )
        if interaction is None:
            notes["interaction"] = ("searched: the relative motion between the two objects "
                                    "never settled below the threshold")
        print(f"[keyframes] interaction frame {interaction}" if interaction is not None
              else "[keyframes] no interaction frame: the relative motion never settled")
    else:
        notes["interaction"] = ("not applicable: no --target-points, so there is no target "
                                "object to measure relative motion against")
        print("[keyframes] interaction: skipped, no --target-points (single-object clip)")

    detachment, threshold, heights = None, None, None
    if args.object_poses and args.mesh and args.ground_plane:
        heights, solved = object_heights(args)
        if contact is not None:
            detachment, threshold = detachment_frame(heights, solved, contact, args)
        usable = heights[solved]
        print(f"[keyframes] object height above the table: "
              f"{100 * np.nanmin(usable):.1f} to {100 * np.nanmax(usable):.1f} cm")
        if detachment is None:
            notes["detachment"] = (
                f"searched: the object never rose above {100 * threshold:.1f} cm and came "
                f"back down (it spans {100 * np.nanmin(usable):.1f} to "
                f"{100 * np.nanmax(usable):.1f} cm), so --detach-height is set for a "
                f"different rig; try --detach-height auto")
            print(f"[keyframes] no detachment frame below {100 * (threshold or 0):.1f} cm")
        else:
            print(f"[keyframes] detachment frame {detachment:4d}  "
                  f"(height fell below {100 * threshold:.1f} cm)")
    else:
        notes["detachment"] = ("not applicable: needs --object-poses, --mesh and "
                               "--ground-plane to measure the object's height")
        print("[keyframes] detachment: skipped, needs --object-poses, --mesh and --ground-plane")

    keyframes = {"contact": contact, "interaction": interaction, "detachment": detachment}
    with open(args.out, "w") as handle:
        json.dump({
            "keyframes": keyframes,
            "n_frames": int(n_frames),
            "parameters": {
                "contact_displacement": args.contact_displacement,
                "contact_streak": args.contact_streak,
                "interaction_speed": args.interaction_speed,
                "interaction_streak": args.interaction_streak,
                "detach_height": threshold,
                "height_from": args.height_from,
            },
            "absent": notes,
            "source": "Video2Sim2Real (arXiv 2606.08828) appendix A.2.1",
        }, handle, indent=2)
    print(f"Wrote {args.out}")

    if args.strip and args.frames_dir:
        draw_strip(args, keyframes)
    if args.plot:
        draw_plot(args, displacement, heights, keyframes, threshold or 0.0)


if __name__ == "__main__":
    main()
