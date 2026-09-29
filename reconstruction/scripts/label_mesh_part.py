#!/usr/bin/env python3
"""
Carry a SAM3 part mask onto the object's mesh, over the whole clip.

A part -- a mug's handle, a tool's grip -- is not a separate body: it moves rigidly
with the object and belongs in the same mesh and the same trajectory. What it needs is
a label, so a later stage can say "make contact here" rather than treating the whole
surface as equivalent. That is what this writes: a subset of the mesh, as both a
boolean per vertex and a boolean per face.

The lift is a projection. At a frame where the part was segmented, the mesh is posed by
the object's tracked pose, projected through the camera, and a vertex is part of the
handle if it lands inside the mask and is really visible there. Visibility carries the
work: the mug's far wall projects inside the handle's outline exactly as the handle
does, and labelling it would put "handle" on the opposite side of the object. Two tests
remove it -- the vertex must face the camera, and it must be the nearest surface along
its pixel.

One frame only ever labels the half of the part that faces the camera. On take_006_v2 a
single frame reached 1.9% of the mesh area where the handle is about 4.7% of it. So
every frame with a usable mask is projected and the evidence is pooled: the object turns
as it is carried, so between them the frames see around it.

Pooled by vote, not by union. Union looks right -- a vertex inside the mask really is
part of the handle -- but it makes every mistake permanent: one frame where the mask
overhangs the silhouette, or where the tracked pose is a few millimetres out, paints
body vertices that no later frame can unpaint. On this mug union bled visibly across the
body. So each vertex is counted against the frames that actually saw it, and is kept
only if it fell inside the mask in a clear majority of them -- a genuine handle vertex
is in the mask whenever it is visible, while a body vertex caught by a bad frame is not.

A projection can only ever label what the camera saw, and for a handle that is its
outer surface alone -- the inner face, the one inside the C, is turned away in every
frame of the clip. That is fatal for grasping rather than untidy: a grasp needs
opposing contacts, and a region whose normals all point the same way admits none.
Restricted to the visible surface, Lightning Grasp returned no grasps at all; with the
handle's two sides it returned hundreds. So the voted region is thickened through the
mesh afterwards -- a vertex within --thicken of a voted one joins it when its
normal opposes that vertex's -- which reaches through a thin rib to the face the
camera could not see, without walking along the surface onto the rest of the object.

The subset's purpose is grasp synthesis. Lightning Grasp searches for contacts among
the object points it is handed, so restricting that set to the part makes it synthesise
grasps that touch the part and nothing else (keyframe_grasps.py --part ... --part-mode
restrict). The collision check keeps the whole cloud, because the rest of the object is
still there to be hit.

Run in the `genesis` env:
    python label_mesh_part.py --mesh posed_mesh.obj --masks-root video_segmentation/masks \\
        --part-id cup_handle --intrinsics intrinsics.yaml \\
        --object-poses cup_traj_points.npz --name handle --out part_handle.npz
"""

import argparse
import os

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Label a mesh subset from SAM3 part masks")
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--masks-root", required=True, help="video_segmentation/masks")
    parser.add_argument("--part-id", required=True, help="Mask name, e.g. cup_handle")
    parser.add_argument("--object-id", default=None,
                        help="The whole object's mask name, e.g. cup. A vertex only gets a "
                             "vote, for or against, at a pixel where the object is actually "
                             "the visible thing -- otherwise the hand covering the handle "
                             "reads as evidence that the handle is not there")
    parser.add_argument("--intrinsics", required=True)
    parser.add_argument("--object-poses", required=True, help="Object trajectory .npz")
    parser.add_argument("--name", default="part", help="Part name, recorded in the output")
    parser.add_argument("--out", required=True, help="part_<name>.npz")

    parser.add_argument("--stride", type=int, default=4,
                        help="Use every n-th frame. The object turns slowly, so most "
                             "frames repeat what their neighbours already showed")
    parser.add_argument("--min-mask-pixels", type=int, default=200,
                        help="Skip a frame whose part mask is smaller than this, which is "
                             "what an occluded or badly tracked frame looks like")
    parser.add_argument("--depth-tolerance", type=float, default=0.006,
                        help="How far behind the nearest surface at a pixel a vertex may "
                             "still count as visible, metres")
    parser.add_argument("--erode", type=int, default=3,
                        help="Shrink each mask by this many pixels before testing. The "
                             "segmentation boundary is looser than the geometry, and a "
                             "mask edge overhanging the silhouette would label the body "
                             "behind it")
    parser.add_argument("--min-votes", type=int, default=3,
                        help="Frames a vertex must be seen in before it can be labelled")
    parser.add_argument("--min-ratio", type=float, default=0.6,
                        help="Share of the frames that saw a vertex in which it must fall "
                             "inside the mask. Below 0.5 the union's bleed comes back")
    parser.add_argument("--thicken", type=float, default=0.006,
                        help="After voting, take in every vertex within this distance of a "
                             "voted one, metres. This is what reaches the far side of a "
                             "thin feature, which no view can show; roughly the feature's "
                             "thickness. 0 disables it and leaves a one-sided region")
    parser.add_argument("--opposing", type=float, default=0.3,
                        help="How opposed a neighbour's normal must be to count as the far "
                             "side of the feature rather than more of the same surface")
    parser.add_argument("--face-rule", choices=("all", "any", "majority"), default="majority",
                        help="When a face counts as part of the region, given its vertices")
    parser.add_argument("--preview", default=None, help="Write a check image")
    parser.add_argument("--preview-frame", type=int, default=None,
                        help="Frame the check image is drawn at (default: the largest mask)")
    return parser.parse_args()


def visible_vertices(vertices, normals, calibration, shape, tolerance):
    """Which vertices are front-facing and unoccluded, and where they land in the image."""
    fu, fv, pu, pv = calibration
    height, width = shape

    depth = vertices[:, 2]
    in_front = depth > 1e-6
    x = np.full(len(vertices), -1, dtype=int)
    y = np.full(len(vertices), -1, dtype=int)
    x[in_front] = np.round(fu * vertices[in_front, 0] / depth[in_front] + pu).astype(int)
    y[in_front] = np.round(fv * vertices[in_front, 1] / depth[in_front] + pv).astype(int)
    inside = in_front & (x >= 0) & (x < width) & (y >= 0) & (y < height)

    # Facing the camera: the surface normal points back toward the origin.
    facing = np.einsum("ij,ij->i", normals, vertices) < 0

    # A one-pixel z-buffer built from the vertices themselves. Coarse next to
    # rasterising the faces, but a mesh this dense puts several vertices in every
    # pixel it covers, so the nearest of them is a fair stand-in for the surface.
    nearest = np.full(height * width, np.inf)
    flat = np.where(inside, y * width + x, 0)
    np.minimum.at(nearest, flat[inside], depth[inside])
    unoccluded = np.zeros(len(vertices), dtype=bool)
    unoccluded[inside] = depth[inside] <= nearest[flat[inside]] + tolerance

    return inside & facing & unoccluded, x, y


def main():
    args = parse_args()
    import cv2
    import trimesh
    import yaml

    mesh = trimesh.load(args.mesh, process=False)
    base_vertices = np.asarray(mesh.vertices, dtype=float)
    base_normals = np.asarray(mesh.vertex_normals, dtype=float)
    faces = np.asarray(mesh.faces)

    with open(args.intrinsics) as handle:
        values = yaml.safe_load(handle)
    calibration = tuple(float(values[k]) for k in ("fu", "fv", "pu", "pv"))

    track = np.load(args.object_poses)
    solved = track["valid"].astype(bool)
    n_frames = len(solved)

    votes = np.zeros(len(base_vertices), dtype=np.int32)
    witnessed = np.zeros(len(base_vertices), dtype=np.int32)
    used, per_frame, best = [], [], (0, None)
    kernel = np.ones((args.erode, args.erode), np.uint8) if args.erode else None

    for frame in range(0, n_frames, max(args.stride, 1)):
        if not solved[frame]:
            continue
        path = os.path.join(args.masks_root, f"frame_{frame:06d}_masks", f"{args.part_id}.png")
        mask = cv2.imread(path, 0)
        if mask is None:
            continue
        mask = mask > 128
        if mask.sum() < args.min_mask_pixels:
            continue
        if kernel is not None:
            mask = cv2.erode(mask.astype(np.uint8), kernel) > 0
            if not mask.any():
                continue

        judged = None
        if args.object_id:
            whole = cv2.imread(os.path.join(args.masks_root, f"frame_{frame:06d}_masks",
                                            f"{args.object_id}.png"), 0)
            if whole is None:
                continue
            judged = whole > 128

        vertices = base_vertices @ track["rotation"][frame].T + track["translation"][frame]
        normals = base_normals @ track["rotation"][frame].T
        seen, x, y = visible_vertices(vertices, normals, calibration, mask.shape,
                                      args.depth_tolerance)
        hit = seen.copy()
        hit[seen] = mask[y[seen], x[seen]]

        # The z-buffer only knows about the object, so it calls a vertex visible even
        # where the hand is in front of it. Those pixels carry no evidence either way.
        countable = seen
        if judged is not None:
            countable = seen.copy()
            countable[seen] = judged[y[seen], x[seen]]
            hit &= countable

        votes += hit
        witnessed += countable
        used.append(frame)
        per_frame.append(int(hit.sum()))
        if int(mask.sum()) > best[0]:
            best = (int(mask.sum()), frame)

    labelled = (votes >= args.min_votes) & (votes >= args.min_ratio * np.maximum(witnessed, 1))
    unioned = int((votes > 0).sum())
    voted = int(labelled.sum())

    if args.thicken > 0 and labelled.any():
        from scipy.spatial import cKDTree

        # The far side of a rib, not a ball around it. A plain radius grows along the
        # surface as readily as through it, so on this mug it walked off the handle
        # onto the body wherever the two nearly touch -- 3 mm already took the region
        # past the handle's own 4.7% of the area. What distinguishes the opposite face
        # of a thin feature is that it looks back: its normal opposes the labelled
        # one. Neighbouring surface, being the same surface, does not.
        voted_index = np.flatnonzero(labelled)
        # Look from inside the material, not from the vertex. Stepping one feature
        # thickness along -normal lands near the far surface, so a small ball there
        # finds it; a ball around the vertex itself is centred on the surface the
        # vertex is already part of and picks up its neighbours instead.
        probe = base_vertices[voted_index] - base_normals[voted_index] * args.thicken
        tree_ = cKDTree(base_vertices)
        found = tree_.query_ball_point(probe, r=args.thicken * 0.5)

        opposed = np.zeros(len(base_vertices), dtype=bool)
        for source, neighbours in zip(voted_index, found):
            if not neighbours:
                continue
            candidates = np.asarray(neighbours)
            facing = base_normals[candidates] @ base_normals[source]
            opposed[candidates[facing < -args.opposing]] = True
        labelled = labelled | opposed

    if not used:
        raise SystemExit(f"no usable '{args.part_id}' mask found under {args.masks_root}")
    if not labelled.any():
        raise SystemExit("no vertex fell inside any mask; check the poses and the part id")

    # Faces, so the region can be handed on as geometry rather than as a point set.
    counts = labelled[faces].sum(axis=1)
    rule = {"all": counts == 3, "any": counts >= 1, "majority": counts >= 2}[args.face_rule]

    triangles = base_vertices[faces]
    area = 0.5 * np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                         triangles[:, 2] - triangles[:, 0]), axis=1)
    share = area[rule].sum() / area.sum()

    print(f"[part] fused {len(used)} frames ({used[0]}..{used[-1]}), "
          f"{np.median(per_frame):.0f} vertices per frame")
    print(f"[part] vote: {unioned} vertices seen in the mask at least once, "
          f"{voted} kept at >= {args.min_ratio:.0%} of their sightings "
          f"({unioned - voted} dropped as bleed)")
    if args.thicken > 0:
        print(f"[part] thickened by {100 * args.thicken:.1f} cm: {voted} -> "
              f"{int(labelled.sum())} vertices, reaching the side no view could show")
    print(f"[part] '{args.name}': {int(labelled.sum())} vertices ({100 * labelled.mean():.1f}% "
          f"of the mesh), {int(rule.sum())} faces, {100 * share:.1f}% of the surface area")

    np.savez(args.out, vertices=labelled, faces=rule, frames=np.array(used),
             area_fraction=share, name=args.name, part_id=args.part_id)
    print(f"Wrote {args.out}")

    if args.preview:
        frame = args.preview_frame if args.preview_frame is not None else best[1]
        path = os.path.join(args.masks_root, f"frame_{frame:06d}_masks", f"{args.part_id}.png")
        mask = cv2.imread(path, 0) > 128
        vertices = base_vertices @ track["rotation"][frame].T + track["translation"][frame]
        normals = base_normals @ track["rotation"][frame].T
        seen, x, y = visible_vertices(vertices, normals, calibration, mask.shape,
                                      args.depth_tolerance)
        image = np.zeros(mask.shape + (3,), np.uint8)
        image[mask] = (40, 40, 40)
        body = seen & ~labelled
        image[y[body], x[body]] = (90, 90, 200)
        part = seen & labelled
        image[y[part], x[part]] = (90, 230, 120)
        cv2.imwrite(args.preview, image)
        print(f"Wrote {args.preview}: frame {frame}, green = the fused region, "
              f"blue = the rest of the object")


if __name__ == "__main__":
    main()
