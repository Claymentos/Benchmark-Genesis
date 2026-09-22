#!/usr/bin/env python3
"""
Fill in a held object's pose from the hand wherever the tracker cannot see it.

While an object is grasped it moves rigidly with the hand, so its pose relative to
the wrist is constant. The hand is also what hides the object, and hides exactly the
features -- a mug's handle, a label, a tool's tip -- that pin down the pose, so the
tracker drifts precisely where the hand knows the answer. This stage combines the
two by asking, per frame, which pose explains the image better -- the tracked one or
the hand's -- and preferring the hand on a tie:

    tracker's silhouette clearly better -> keep the tracked pose
    the two explain the mask equally    -> wrist pose * grasp offset
    hand's silhouette better            -> wrist pose * grasp offset

with a smooth blend in between, so the object never snaps. A tie is exactly what an
occluded or symmetric view produces: the image cannot say, and then the rigid grasp
is the better evidence. Where the object is visible and the tracker is right, its
silhouette wins on the details the hand cannot know, and it is kept.

Nothing here assumes a shape, an upright pose or a symmetry axis; an asymmetric
object simply keeps its tracked pose, because its silhouette stays convincing.
The assumptions are that a grasp is rigid while it lasts, and that the tracker is
right during some part of it -- usually before the fingers close or after they open,
which is where the grasp offset is measured.

Grasps are found from hand-to-surface distance, so the threshold is a finger
thickness rather than anything about the object. Several grasps in one clip each get
their own offset, and a grasp whose frames do not agree on one offset is left alone
rather than forced.

Run in the `genesis` env:
    python lock_object_to_hand.py --object-traj cup_traj_fp.npz --hand-traj wuji_traj.npz \\
        --hand-meshes all_hand_meshes.npz --side left --mesh posed_mesh.obj \\
        --out cup_traj_fp.npz
"""

import argparse
import os

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp


def parse_args():
    parser = argparse.ArgumentParser(description="Blend a grasped object's pose with the hand's")
    parser.add_argument("--object-traj", required=True, help="Object trajectory .npz from the tracker")
    parser.add_argument("--hand-traj", required=True, help="wuji_traj.npz (wrist pose, camera frame)")
    parser.add_argument("--hand-meshes", default=None, help="HaWoR all_hand_meshes.npz, for contact")
    parser.add_argument("--side", default="left", choices=["left", "right"])
    parser.add_argument("--mesh", required=True, help="Object mesh the trajectory was tracked with")
    parser.add_argument("--out", required=True, help="May be the same file as --object-traj")
    parser.add_argument(
        "--contact",
        type=float,
        default=0.03,
        help="Hand-to-object-surface distance counting as contact, metres. A finger "
        "thickness plus reconstruction error, so it does not depend on the object.",
    )
    parser.add_argument(
        "--fallback-near",
        type=float,
        default=0.25,
        help="Without hand meshes, contact falls back to this wrist-to-surface distance",
    )
    parser.add_argument("--min-grasp", type=int, default=10, help="Shortest grasp kept, frames")
    parser.add_argument("--fill-gap", type=int, default=5, help="Contact gaps up to this are bridged")
    parser.add_argument(
        "--onset",
        type=int,
        default=12,
        help="Frames at the start of a grasp used to measure its offset. The object is "
        "still where the tracker last saw it unoccluded, so these frames are the most "
        "trustworthy; a vote over the whole grasp is the fallback.",
    )
    parser.add_argument(
        "--agree",
        type=float,
        default=15.0,
        help="Degrees within which two frames count as agreeing on one grasp offset",
    )
    parser.add_argument(
        "--min-agree",
        type=float,
        default=0.3,
        help="Fraction of a grasp's trusted frames that must agree before it is used; below "
        "it the grasp is not rigid enough and the trajectory is left alone",
    )
    parser.add_argument("--masks-root", default=None, help="video_segmentation/masks")
    parser.add_argument("--object", default=None, help="Mask name for the object")
    parser.add_argument("--intrinsics", default=None, help="intrinsics.yaml, to project the mesh")
    parser.add_argument(
        "--jump",
        type=float,
        default=25.0,
        help="Rotation between consecutive frames, degrees, above which the tracker counts "
        "as having jumped. A jump, a spin correction or a re-registration inside a grasp "
        "is what marks the track as needing repair; a steady track is left alone even "
        "though the hand could replace it, because the tracker also sees texture and "
        "detail that a silhouette comparison cannot.",
    )
    parser.add_argument(
        "--tie",
        type=float,
        default=0.02,
        help="Silhouette IoU within which the two poses count as explaining the mask equally; "
        "the hand wins there",
    )
    parser.add_argument(
        "--decide",
        type=float,
        default=0.06,
        help="IoU by which the tracker must beat the hand to be kept outright",
    )
    parser.add_argument(
        "--blend",
        type=int,
        default=10,
        help="Frames over which the hand takes over and hands back, at the ends of a grasp "
        "where the two sources agree",
    )
    parser.add_argument(
        "--lock-position",
        action="store_true",
        help="Also take the object's position from the hand where the tracker is doubted. "
        "Off by default: with stereo depth the tracked position stays good even when the "
        "silhouette is ambiguous.",
    )
    return parser.parse_args()


def runs_of(flags, fill_gap, minimum):
    """Index ranges where flags is True, short gaps bridged and short runs dropped."""
    filled = flags.copy()
    gap_start = None
    for index, flag in enumerate(flags):
        if not flag and gap_start is None:
            gap_start = index
        elif flag and gap_start is not None:
            if index - gap_start <= fill_gap and gap_start > 0:
                filled[gap_start:index] = True
            gap_start = None
    spans, start = [], None
    for index, flag in enumerate(np.append(filled, False)):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            if index - start >= minimum:
                spans.append((start, index))
            start = None
    return spans


def hand_to_surface(points_by_frame, valid, rotation, translation, surface):
    """Per-frame distance from the hand to the object's surface, metres."""
    n_frames = len(rotation)
    distance = np.full(n_frames, np.inf)
    for index in range(n_frames):
        points = points_by_frame[index]
        points = points[np.isfinite(points).all(axis=1)]
        if not valid[index] or len(points) == 0:
            continue
        posed = surface @ rotation[index].T + translation[index]
        gaps, _ = cKDTree(posed).query(points)
        distance[index] = gaps.min()
    return distance


def load_masks(masks_root, object_name, index, dilate=9):
    """The object's mask and the union of the other masks, dilated."""
    import cv2

    frame_dir = os.path.join(masks_root, f"frame_{index:06d}_masks")
    if not os.path.isdir(frame_dir):
        return None, None
    target, others = None, None
    for name in os.listdir(frame_dir):
        mask = cv2.imread(os.path.join(frame_dir, name), 0)
        if mask is None:
            continue
        mask = mask > 128
        if name == f"{object_name}.png":
            target = mask
        else:
            others = mask if others is None else others | mask
    if others is not None:
        others = cv2.dilate(others.astype(np.uint8), np.ones((dilate, dilate), np.uint8)) > 0
    return target, others


def silhouette_iou(points, rotation, position, centre, calibration, target, occluders, radius=3):
    """IoU between the mask and the mesh's projected silhouette, outside other masks.

    The silhouette is splatted from surface samples rather than rasterised: the stage
    then needs no renderer, and a few pixels of slack do not change which of two poses
    explains the mask better.
    """
    import cv2

    fu, fv, pu, pv = calibration
    posed = (points - centre) @ rotation.T + position
    ahead = posed[:, 2] > 1e-6
    if not ahead.any():
        return 0.0
    u = (fu * posed[ahead, 0] / posed[ahead, 2] + pu).astype(np.int32)
    v = (fv * posed[ahead, 1] / posed[ahead, 2] + pv).astype(np.int32)
    height, width = target.shape
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    splat = np.zeros((height, width), np.uint8)
    splat[v[inside], u[inside]] = 1
    splat = cv2.dilate(splat, np.ones((radius, radius), np.uint8)) > 0
    splat = cv2.morphologyEx(splat.astype(np.uint8), cv2.MORPH_CLOSE,
                             np.ones((9, 9), np.uint8)) > 0
    considered = np.ones_like(target) if occluders is None else ~occluders
    union = ((splat | target) & considered).sum()
    return float(((splat & target & considered).sum()) / max(union, 1))


def main():
    args = parse_args()
    track = dict(np.load(args.object_traj))
    hand = np.load(args.hand_traj)
    rotation, translation = track["rotation"], track["translation"]
    n_frames = len(rotation)
    if len(hand["wrist_quat"]) != n_frames:
        raise SystemExit(f"object has {n_frames} frames, hand has {len(hand['wrist_quat'])}")

    mesh = trimesh.load(args.mesh, process=False)
    vertices = np.asarray(mesh.vertices)
    surface = vertices[:: max(1, len(vertices) // 4000)]
    centre = vertices.mean(axis=0)
    position = np.einsum("tij,j->ti", rotation, centre) + translation

    hand_valid = hand["valid"].astype(bool)
    valid = track["valid"].astype(bool) & hand_valid
    wrist = Rotation.from_quat(hand["wrist_quat"][:, [1, 2, 3, 0]])  # stored wxyz

    # Contact from the hand's own surface when HaWoR meshes are available: a finger
    # thickness is the same for every object, unlike a wrist-to-centre distance.
    if args.hand_meshes:
        meshes = np.load(args.hand_meshes)
        points = meshes[f"{args.side}_vertices"]
        if len(points) != n_frames:
            raise SystemExit(f"hand meshes have {len(points)} frames, object has {n_frames}")
        # HaWoR's own flag: frames it did not fit hold non-finite vertices.
        hawor_valid = meshes[f"{args.side}_valid"].astype(bool)
        distance = hand_to_surface(points, valid & hawor_valid, rotation, translation, surface)
        threshold = args.contact
    else:
        distance = hand_to_surface(
            hand["wrist_pos"][:, None, :], valid, rotation, translation, surface
        )
        threshold = args.fallback_near
        print("No --hand-meshes: falling back to wrist-to-surface distance")

    calibration = None
    if args.masks_root and args.object and args.intrinsics:
        import yaml

        with open(args.intrinsics) as handle:
            values = yaml.safe_load(handle)
        calibration = tuple(float(values[k]) for k in ("fu", "fv", "pu", "pv"))
    else:
        print("No --masks-root/--object/--intrinsics: the hand is used across each whole grasp")

    grasps = runs_of(valid & (distance < threshold), args.fill_gap, args.min_grasp)
    print(f"grasps found: {[(int(a), int(b - 1)) for a, b in grasps] or 'none'}")

    new_rotation, new_position = rotation.copy(), position.copy()
    hand_weight = np.zeros(n_frames)
    for start, stop in grasps:
        span = np.arange(start, stop)
        in_wrist = wrist[span].inv() * Rotation.from_matrix(rotation[span])
        # One rigid offset per grasp, from the frames that agree on it: drifted frames
        # disagree with each other too, so they lose the vote. When the tracker is wrong
        # for most of the grasp no majority forms, and the frames just after contact are
        # the fallback -- the object has not moved yet, so the tracker still holds the
        # pose it had in clear view.
        gaps = np.degrees(np.stack(
            [(in_wrist[i].inv() * in_wrist).magnitude() for i in range(len(span))]
        ))
        agreeing = gaps[int(np.argmax((gaps < args.agree).sum(axis=1)))] < args.agree
        source = "grasp-wide vote"
        if agreeing.mean() < args.min_agree:
            onset = min(args.onset, len(span))
            spread = np.degrees((in_wrist[0].inv() * in_wrist[:onset]).magnitude())
            if (spread < args.agree).mean() < 0.6:
                print(f"  {start}-{stop - 1}: no majority over the grasp "
                      f"({agreeing.mean():.0%} < {args.min_agree:.0%}) and unsteady at "
                      f"contact, left unchanged")
                continue
            agreeing = np.zeros(len(span), dtype=bool)
            agreeing[:onset] = spread < args.agree
            source = f"onset {start}-{start + onset - 1}"
        offset = in_wrist[agreeing].mean()
        offset_position = np.mean(
            wrist[span[agreeing]].inv().apply(
                position[span[agreeing]] - hand["wrist_pos"][span[agreeing]]
            ),
            axis=0,
        )
        from_hand = wrist[span] * offset
        tracked = Rotation.from_matrix(rotation[span])

        # Does the tracker look troubled here? Corrections, re-registrations and jumps
        # are its own record of the ambiguity the hand can settle.
        unsteady = []
        corrections = track.get("spin_corrections")
        if corrections is not None and np.any(np.asarray(corrections)[span] != 0):
            unsteady.append("spin corrections")
        relocalized = track.get("relocalized")
        if relocalized is not None and np.any(np.asarray(relocalized)[span]):
            unsteady.append("re-registrations")
        inner_steps = np.degrees(
            (Rotation.from_matrix(rotation[span][:-1]).inv()
             * Rotation.from_matrix(rotation[span][1:])).magnitude()
        )
        if len(inner_steps) and inner_steps.max() > args.jump:
            unsteady.append(f"a {inner_steps.max():.0f} deg jump")

        # Which source explains the mask better over the grasp as a whole. The choice is
        # per grasp, not per frame: the object is rigid in the hand for the whole of it,
        # and a per-frame vote flickers between two poses that can be 180 degrees apart,
        # which is a worse artefact than either source.
        weight = np.ones(len(span))
        lead = np.full(len(span), np.nan)
        if calibration is not None:
            for k, index in enumerate(span):
                target, occluders = load_masks(args.masks_root, args.object, index)
                if target is None or target.sum() < 50:
                    continue
                shared = dict(points=surface, centre=centre, calibration=calibration,
                              target=target, occluders=occluders)
                tracked_iou = silhouette_iou(
                    rotation=rotation[index], position=position[index], **shared
                )
                hand_iou = silhouette_iou(
                    rotation=from_hand[k].as_matrix(),
                    position=hand["wrist_pos"][index] + wrist[index].apply(offset_position)
                    if args.lock_position else position[index],
                    **shared,
                )
                lead[k] = tracked_iou - hand_iou
            typical = np.nanmedian(lead) if np.isfinite(lead).any() else 0.0
            if typical > args.decide:
                print(f"  {start}-{stop - 1}: tracker explains the mask better "
                      f"(+{typical:.3f} IoU), left unchanged")
                continue
            if typical > -args.tie and not unsteady:
                print(f"  {start}-{stop - 1}: tracker steady and explains the mask "
                      f"({typical:+.3f} IoU), left unchanged")
                continue
            reason = ", ".join(unsteady) if unsteady else "the mask favours the hand"
            print(f"  {start}-{stop - 1}: tracker leads by {typical:+.3f} IoU and shows "
                  f"{reason}: hand takes the grasp")

        # Ease in and out at the ends, where tracker and hand agree by construction.
        ramp = min(args.blend, len(span) // 2)
        if ramp > 0:
            edge = (np.arange(ramp) + 1) / (ramp + 1)
            weight[:ramp] = edge
            weight[-ramp:] = edge[::-1]

        for k, w in enumerate(weight):
            if w <= 1e-3:
                continue
            pair = Rotation.concatenate([tracked[k], from_hand[k]])
            new_rotation[span[k]] = Slerp([0.0, 1.0], pair)(w).as_matrix()
            if args.lock_position:
                from_hand_position = (
                    hand["wrist_pos"][span[k]] + wrist[span[k]].apply(offset_position)
                )
                new_position[span[k]] = (1 - w) * position[span[k]] + w * from_hand_position
        hand_weight[span] = weight
        change = np.degrees((tracked.inv() * Rotation.from_matrix(new_rotation[span])).magnitude())
        print(f"  {start}-{stop - 1}: offset from {agreeing.sum()} frames ({source}), "
              f"hand weight median {np.median(weight):.2f}, rotation changed "
              f"median {np.median(change):.1f} max {change.max():.0f} deg")

    steps = lambda r: np.degrees(
        (Rotation.from_matrix(r[:-1]).inv() * Rotation.from_matrix(r[1:])).magnitude()
    )
    before, after = steps(rotation), steps(new_rotation)
    print(f"per-frame rotation over the clip: before median {np.median(before):.1f} / "
          f"max {before.max():.0f} deg, after median {np.median(after):.1f} / max {after.max():.0f} deg")

    track["rotation"] = new_rotation
    # Rotating about the object's centre keeps the tracked position untouched.
    track["translation"] = new_position - np.einsum("tij,j->ti", new_rotation, centre)
    track["hand_weight"] = hand_weight
    np.savez(args.out, **track)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
