#!/usr/bin/env python3
"""
Score hand-object reconstructions of an EPIC-Contact clip, before and after contact.

Each variant is a (hand meshes, object mesh, object trajectory) triple in the camera
frame. Metrics, and what each one can and cannot show:

  contact_mm      GT contact vertices -> object surface, median. The refinement is
                  fitted to this, so it shows the fit worked, not that it is right.
  contact_5mm     share of GT contact vertices within 5 mm of the surface.
  penetration_mm  deepest hand vertex inside the object per frame, mean over frames.
  mask_iou        projected object silhouette vs the SAM3 mask, outside the SAM3 hand
                  mask (the hand hides part of the object). Independent of the
                  contact labels: it is the image evidence, and says whether contact
                  was bought by moving the object off what the camera saw.
  grasp_drift_mm  std of the object centroid in the hand's own frame (wrist, index
  grasp_drift_deg and middle MCP joints). An EPIC-Contact clip is a stable grasp, so the
                  object should not move in the hand; not directly optimised.
  pgt_chamfer_mm  against EPIC's per-frame pseudo-GT, whose hand-object relative
  pgt_axis_deg    geometry is GT quality: the GT hand is aligned onto the variant's
                  hand (similarity Procrustes), carrying the GT object with it, and
                  compared with the variant's object -- symmetric chamfer, and the angle
                  between principal axes. The central frame (where the contact labels
                  were transferred) is excluded.

Run in the `genesis` env:
    python eval_contact_reconstruction.py --epic-frames epic_frames.json \\
        --epic-clip-dir .../central_frames/<clip> --per-frame-npz .../<clip>.npz \\
        --intrinsics intrinsics.yaml --masks-root video_segmentation/masks --object bottle \\
        --side left --variant baseline all_hand_meshes.npz posed_mesh.obj traj.npz \\
        --variant contact all_hand_meshes_contact.npz posed_mesh_contact.obj traj_contact.npz
"""

import argparse
import json
import os

import numpy as np
import trimesh
import yaml

from refine_object_contact import load_epic_contact, similarity_procrustes


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate hand-object reconstructions with EPIC-Contact")
    parser.add_argument("--epic-frames", required=True)
    parser.add_argument("--epic-clip-dir", required=True)
    parser.add_argument("--per-frame-npz", default=None)
    parser.add_argument("--intrinsics", required=True)
    parser.add_argument("--masks-root", required=True)
    parser.add_argument("--object", required=True)
    parser.add_argument("--side", required=True, choices=("left", "right"))
    parser.add_argument("--variant", nargs=4, action="append", required=True,
                        metavar=("NAME", "HAND_MESHES", "OBJECT_MESH", "OBJECT_POSES"))
    parser.add_argument("--faces", type=int, default=3000, help="Decimation budget for silhouettes")
    parser.add_argument("--out", default=None, help="Write the metrics as JSON here")
    return parser.parse_args()


def hand_frame(joints):
    """Rotation/origin of a hand-local frame from MANO joints (wrist, index/middle MCP)."""
    origin = joints[0]
    x = joints[9] - origin
    x /= np.linalg.norm(x)
    z = np.cross(x, joints[5] - origin)
    z /= np.linalg.norm(z)
    y = np.cross(z, x)
    return np.stack([x, y, z], 1), origin


def principal_axis(points):
    centred = points - points.mean(0)
    return np.linalg.svd(centred, full_matrices=False)[2][0]


def silhouette(vertices, faces, fu, fv, pu, pv, shape):
    import cv2
    z = np.maximum(vertices[:, 2], 1e-4)
    uv = np.stack([fu * vertices[:, 0] / z + pu, fv * vertices[:, 1] / z + pv], 1)
    mask = np.zeros(shape, np.uint8)
    # One fill per triangle: fillPoly over many polygons fills even-odd, so overlapping
    # triangles cancel and a closed mesh comes out as a hollow wireframe (IoU ~0.15 on a
    # pose that is visibly right).
    for tri in np.round(uv[faces]).astype(np.int32):
        cv2.fillConvexPoly(mask, tri, 1)
    return mask.astype(bool)


def chamfer(a, b):
    from scipy.spatial import cKDTree
    return 0.5 * (cKDTree(b).query(a)[0].mean() + cKDTree(a).query(b)[0].mean())


def evaluate(args, name, hand_path, mesh_path, poses_path, gt, calib, meta, pgt):
    import cv2

    fu, fv, pu, pv = calib
    hands = np.load(hand_path, allow_pickle=True)
    hv = hands[f"{args.side}_vertices"].astype(np.float64)
    hj = hands[f"{args.side}_joints"].astype(np.float64)
    h_ok = hands[f"{args.side}_valid"].astype(bool)
    h_ok &= np.isfinite(hv).all((1, 2)) & np.isfinite(hj).all((1, 2))   # HaWoR can mark NaN frames valid
    poses = np.load(poses_path, allow_pickle=True)
    R, T, o_ok = poses["rotation"], poses["translation"], poses["valid"].astype(bool)
    mesh = trimesh.load(mesh_path, process=False, force="mesh")
    decimated = mesh.simplify_quadric_decimation(face_count=args.faces) if len(mesh.faces) > args.faces else mesh
    dv, df = np.asarray(decimated.vertices), np.asarray(decimated.faces)
    samples = trimesh.sample.sample_surface_even(mesh, 3000, seed=0)[0]
    n = min(len(hv), len(R))
    both = h_ok[:n] & o_ok[:n]
    # A clip of your own (annotate_contact.py) holds the grasp only inside contact_frames;
    # contact, penetration and drift are scored there. Mask IoU uses every frame.
    lo, hi = meta.get("contact_frames") or (0, n - 1)
    both[:lo] = False
    both[hi + 1:] = False

    contact = np.unique(np.concatenate([v for v in gt["hand_regions"].values() if len(v)]))
    # Distance queries run on a decimated copy: exact signed distance on a 400k-face
    # RecGen mesh takes minutes per variant, and 20k faces is ~1.5 mm edges on a
    # 15 cm object. Every variant gets the same treatment.
    query_mesh = mesh.simplify_quadric_decimation(face_count=20000) if len(mesh.faces) > 20000 else mesh
    query = trimesh.proximity.ProximityQuery(query_mesh)
    contact_d, pen, ious, local_c, local_axis = [], [], [], [], []
    axis_canon = principal_axis(samples)
    for t in range(n):
        if o_ok[t]:
            mfile = os.path.join(args.masks_root, f"frame_{t:06d}_masks", f"{args.object}.png")
            if os.path.exists(mfile):
                ref = cv2.imread(mfile, 0) > 128
                sil = silhouette(dv @ R[t].T + T[t], df, fu, fv, pu, pv, ref.shape)
                # The hand covers the grasped object, and the mask only has what is
                # visible: a complete object correctly placed behind the fingers would
                # otherwise be scored as spilling out of its mask. Pixels the hand
                # covers say nothing about the object, so they are left out.
                hfile = os.path.join(args.masks_root, f"frame_{t:06d}_masks", f"{args.side}_hand_0.png")
                if os.path.exists(hfile):
                    keep = ~(cv2.imread(hfile, 0) > 128)
                    ref, sil = ref & keep, sil & keep
                union = (ref | sil).sum()
                if union:
                    ious.append((ref & sil).sum() / union)
        if not both[t]:
            continue
        # Hand into the object's canonical frame, where the mesh's own queries apply.
        canon = (hv[t] - T[t]) @ R[t]
        sd = query.signed_distance(canon[::2])            # trimesh: positive inside
        pen.append(max(sd.max(), 0.0))
        contact_d.append(np.abs(query.signed_distance(canon[contact])))
        rot_h, org_h = hand_frame(hj[t])
        centre = samples.mean(0) @ R[t].T + T[t]
        local_c.append((centre - org_h) @ rot_h)
        local_axis.append((R[t] @ axis_canon) @ rot_h)

    contact_d = np.concatenate(contact_d)
    local_c = np.array(local_c)
    local_axis = np.array(local_axis)
    local_axis *= np.sign(local_axis @ local_axis[0])[:, None]   # axis sign is arbitrary
    mean_axis = local_axis.mean(0) / np.linalg.norm(local_axis.mean(0))
    result = {
        "frames_scored": int(both.sum()),
        "contact_mm": float(np.median(contact_d) * 1000),
        "contact_5mm": float((contact_d < 0.005).mean()),
        "penetration_mm": float(np.mean(pen) * 1000),
        "penetration_frames_gt5mm": float(np.mean(np.array(pen) > 0.005)),
        "mask_iou": float(np.mean(ious)) if ious else None,
        "grasp_drift_mm": float(np.linalg.norm(local_c - local_c.mean(0), axis=1).mean() * 1000),
        "grasp_drift_deg": float(np.degrees(np.arccos(np.clip(local_axis @ mean_axis, -1, 1))).mean()),
    }

    if pgt is not None:
        side = args.side[0]
        local = np.asarray(meta["epic_frames"])
        chamfers, axes = [], []
        for j, frame in enumerate(pgt["_frame_num"]):
            # Nearest local frame, within one 30 fps frame (the clip may be subsampled).
            t = int(np.argmin(np.abs(local - int(frame))))
            if abs(local[t] - int(frame)) > 2 or t >= n or not both[t] \
                    or abs(int(frame) - meta["keyframe_epic"]) <= 2:
                continue
            if not pgt[f"{args.side}_valid"][j]:
                continue
            gh = pgt[f"mano.v3d.cam.{side}"][j].astype(np.float64)
            go = pgt["object.v.cam"][j][: int(pgt["object.v_len"][j])].astype(np.float64)
            s, r, tr = similarity_procrustes(gh, hv[t])
            placed = s * go @ r.T + tr
            ours = samples @ R[t].T + T[t]
            chamfers.append(chamfer(ours, placed))
            a, b = principal_axis(ours), principal_axis(placed)
            axes.append(np.degrees(np.arccos(min(abs(a @ b), 1.0))))
        result["pgt_frames"] = len(chamfers)
        result["pgt_chamfer_mm"] = float(np.mean(chamfers) * 1000) if chamfers else None
        result["pgt_axis_deg"] = float(np.mean(axes)) if axes else None
    return result


def main():
    args = parse_args()
    calib = yaml.safe_load(open(args.intrinsics))
    calib = tuple(float(calib[k]) for k in ("fu", "fv", "pu", "pv"))
    meta = json.load(open(args.epic_frames))
    _, _, _, hand_regions, _ = load_epic_contact(args.epic_clip_dir)
    gt = {"hand_regions": hand_regions}
    pgt = np.load(args.per_frame_npz, allow_pickle=True) if args.per_frame_npz else None

    results = {}
    for name, hand_path, mesh_path, poses_path in args.variant:
        results[name] = evaluate(args, name, hand_path, mesh_path, poses_path, gt, calib, meta, pgt)

    keys = list(next(iter(results.values())).keys())
    width = max(len(k) for k in keys)
    print(f"{'metric':<{width}}  " + "  ".join(f"{n:>12}" for n in results))
    for key in keys:
        cells = []
        for n in results:
            v = results[n].get(key)
            cells.append(f"{v:>12.3f}" if isinstance(v, float) else f"{str(v):>12}")
        print(f"{key:<{width}}  " + "  ".join(cells))
    if args.out:
        json.dump(results, open(args.out, "w"), indent=1)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
