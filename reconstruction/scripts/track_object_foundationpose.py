#!/usr/bin/env python3
"""
Per-frame rigid object pose with FoundationPose.

Model-based alternative to track_object_v2s2r.py. That tracker infers pose from
sparse point correspondences, which a textureless near-cylindrical object barely
constrains; FoundationPose renders the mesh and matches it against RGB-D directly,
so the object's own geometry does the disambiguating.

The mesh is passed in the reference frame's camera coordinates (RecGen's
posed_mesh.obj), so FoundationPose's object-to-camera transform *is* this pipeline's
(rotation, translation) with no extra conversion, and the pose it recovers at the
reference frame should come back close to identity -- which the script checks.

Run in the `foundationpose` env:
    python track_object_foundationpose.py --mesh posed_mesh.obj --frames-dir all_frames \\
        --depth-dir depth_zed --masks-root video_segmentation/masks --object cup \\
        --ref-frame 49 --intrinsics intrinsics.yaml --out cup_traj_raw.npz
"""

import argparse
import os
import sys

import numpy as np
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="FoundationPose object tracking")
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--depth-dir", required=True)
    parser.add_argument("--masks-root", required=True)
    parser.add_argument("--object", required=True)
    parser.add_argument("--ref-frame", type=int, required=True)
    parser.add_argument("--intrinsics", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--foundationpose-dir",
        default=os.environ.get("FOUNDATIONPOSE_DIR", os.path.expanduser("~/FoundationPose")),
    )
    parser.add_argument("--est-refine-iter", type=int, default=5)
    parser.add_argument(
        "--seed-reference",
        default="identity",
        choices=["identity", "register"],
        help="At the reference frame the mesh is already in camera coordinates, so the "
        "correct pose is identity by construction. Seeding it beats re-deriving it: "
        "global registration can settle on a 180-degree flip for a near-symmetric "
        "object like a mug, where only the handle breaks the tie.",
    )
    parser.add_argument("--track-refine-iter", type=int, default=2)
    parser.add_argument("--zfar", type=float, default=3.0, help="Ignore depth beyond this, metres")
    parser.add_argument(
        "--spin-axis",
        default=None,
        help="Symmetry axis in the mesh's own coordinates, 'x,y,z'. By default it is found "
        "from the mesh, so the object needs no particular pose and need not be a surface of "
        "revolution at all: the spin check simply switches itself off when there is no axis.",
    )
    parser.add_argument(
        "--symmetry-tolerance",
        type=float,
        default=0.03,
        help="Distance within which a rotated surface point counts as landing back on the "
        "surface, as a fraction of the mesh diameter",
    )
    parser.add_argument(
        "--symmetry-support",
        type=float,
        default=0.7,
        help="Fraction of surface points that must land back on the surface, at every tested "
        "angle, for the object to count as a surface of revolution. A mug passes on its body "
        "while its handle does not, which is exactly the ambiguity the spin check resolves.",
    )
    parser.add_argument(
        "--spin-hypotheses",
        type=int,
        default=12,
        help="Spins about the symmetry axis tried per check (0 disables the check). A "
        "near-cylindrical object's spin is only constrained by features like a mug's "
        "handle; while those are occluded frame-to-frame refinement drifts, and it never "
        "recovers a large error on its own because it only searches locally.",
    )
    parser.add_argument("--spin-every", type=int, default=1, help="Run the spin check every N frames")
    parser.add_argument(
        "--spin-margin",
        type=float,
        default=0.03,
        help="Silhouette IoU by which a spin must beat the current pose before it replaces "
        "it, so near-ties while the handle is hidden do not make the pose hop",
    )
    parser.add_argument(
        "--spin-min-angle",
        type=float,
        default=45.0,
        help="Smallest correction the spin check applies, degrees, measured between the "
        "refined poses. Small spin errors are left to FoundationPose's refinement: on a "
        "small or distant object the silhouette cannot resolve them and the two would "
        "otherwise keep undoing each other.",
    )
    parser.add_argument(
        "--spin-persist",
        type=int,
        default=3,
        help="Consecutive checks that must agree on the same correction before it is "
        "applied. A true flip keeps winning; silhouette noise on a small or distant "
        "object wins one frame and loses the next.",
    )
    parser.add_argument(
        "--spin-strong-margin",
        type=float,
        default=0.10,
        help="IoU gain at which a correction applies at once, without waiting for "
        "--spin-persist: the current pose has plainly lost the object, and waiting lets "
        "the track drift beyond recovery",
    )
    parser.add_argument(
        "--spin-min-iou",
        type=float,
        default=0.5,
        help="Apply no correction unless the winning spin overlaps the mask at least this "
        "well; below it no hypothesis explains the object and the ranking is noise",
    )
    parser.add_argument(
        "--relocalize-iou",
        type=float,
        default=0.3,
        help="When the object is masked but no spin of the current pose overlaps it this "
        "well, the track is lost (typically after the object left the image) and the "
        "pose is re-registered from the mask. Needs the spin check; 0 disables.",
    )
    parser.add_argument(
        "--max-faces",
        type=int,
        default=20000,
        help="Decimate the mesh to at most this many faces. FoundationPose renders "
        "hundreds of pose hypotheses per frame, so a RecGen mesh at full resolution "
        "(~850k faces) exhausts GPU memory during registration.",
    )
    return parser.parse_args()


def symmetry_axis(mesh, tolerance, support, angles=(30, 60, 90, 120, 150, 180)):
    """Find an axis the object is (nearly) a surface of revolution about, or None.

    Each principal axis of the surface is a candidate. The candidate line is placed by
    the same circle fit the spin uses, because a mug's handle pulls the centroid off
    the body and rotating about a displaced line moves the body instead of leaving it
    in place. Points are sampled over the surface rather than taken from the vertices,
    whose density follows the reconstruction's detail.

    The score is the worst fraction of points landing back on the surface over several
    angles -- worst, so an object matching only at 180 degrees (a box, say) is
    rejected: its pose is not continuously ambiguous and the tracker resolves it.
    """
    import trimesh
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation

    # Seeded: the detection decides whether the spin check runs at all, so it must not
    # vary between runs of the same clip.
    try:
        points, _ = trimesh.sample.sample_surface(mesh, 20000, seed=0)
    except TypeError:  # older trimesh
        np.random.seed(0)
        points, _ = trimesh.sample.sample_surface(mesh, 20000)
    points = np.asarray(points, dtype=float)
    centred = points - points.mean(axis=0)
    diameter = np.linalg.norm(centred.max(axis=0) - centred.min(axis=0))
    tree = cKDTree(centred)

    best_axis, best_score = None, 0.0
    _, eigenvectors = np.linalg.eigh(np.cov(centred.T))
    for axis in eigenvectors.T:
        pivot = body_axis_point(centred, axis)
        about = centred - pivot
        score = 1.0
        for angle in angles:
            rotated = Rotation.from_rotvec(np.radians(angle) * axis).apply(about) + pivot
            distances, _ = tree.query(rotated)
            score = min(score, float((distances < tolerance * diameter).mean()))
        if score > best_score:
            best_axis, best_score = axis, score
    if best_score < support:
        return None, best_score
    return best_axis, best_score


def body_axis_point(vertices, axis):
    """A point on the axis of the object's cylindrical body.

    Found by a circle fit to the vertices projected along the axis, trimmed so a handle
    does not pull it. A plain vertex mean is biased toward whichever side the
    reconstruction covers best (18 mm on a mug), and spinning about an off-centre axis
    shifts the whole body.
    """
    u = np.cross(axis, [1.0, 0.0, 0.0])
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(axis, [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u)
    w = np.cross(axis, u)
    planar = np.stack([vertices @ u, vertices @ w], axis=1)
    keep = np.ones(len(planar), dtype=bool)
    for _ in range(5):
        A = np.c_[2 * planar[keep], np.ones(keep.sum())]
        b = (planar[keep] ** 2).sum(axis=1)
        cx, cy, c = np.linalg.lstsq(A, b, rcond=None)[0]
        radius = np.sqrt(c + cx**2 + cy**2)
        residual = np.abs(np.hypot(planar[:, 0] - cx, planar[:, 1] - cy) - radius)
        keep = residual <= np.percentile(residual, 70)
    return cx * u + cy * w + np.median(vertices @ axis) * axis


def main():
    args = parse_args()
    sys.path.insert(0, os.path.abspath(args.foundationpose_dir))

    import cv2
    import nvdiffrast.torch as dr
    import trimesh
    from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor

    with open(args.intrinsics) as handle:
        calibration = yaml.safe_load(handle)
    fu, fv, pu, pv = (float(calibration[k]) for k in ("fu", "fv", "pu", "pv"))
    K = np.array([[fu, 0.0, pu], [0.0, fv, pv], [0.0, 0.0, 1.0]])

    names = sorted(f for f in os.listdir(args.frames_dir) if f.lower().endswith((".png", ".jpg")))
    depth_names = sorted(f for f in os.listdir(args.depth_dir) if f.endswith(".png"))
    n_frames = len(names)

    mesh = trimesh.load(args.mesh, process=False)
    if len(mesh.faces) > args.max_faces:
        import open3d as o3d

        original = len(mesh.faces)
        # Decimation only changes tessellation, not the surface the pose is fitted to,
        # and the pose is reported in the same coordinates either way. open3d is used
        # because trimesh's own quadric decimation needs a package this env lacks.
        simplified = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.asarray(mesh.vertices)),
            o3d.utility.Vector3iVector(np.asarray(mesh.faces)),
        ).simplify_quadric_decimation(target_number_of_triangles=args.max_faces)
        mesh = trimesh.Trimesh(
            vertices=np.asarray(simplified.vertices),
            faces=np.asarray(simplified.triangles),
            process=False,
        )
        print(f"[fp] decimated mesh {original} -> {len(mesh.faces)} faces", flush=True)

    estimator = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=ScorePredictor(),
        refiner=PoseRefinePredictor(),
        glctx=dr.RasterizeCudaContext(),
        debug=0,
        debug_dir="/tmp",
    )

    import torch
    from estimater import bilateral_filter_depth, depth2xyzmap_batch, erode_depth
    from Utils import nvdiffrast_render

    spins = None
    if args.spin_hypotheses > 0:
        from scipy.spatial.transform import Rotation

        if args.spin_axis:
            axis = np.asarray([float(v) for v in args.spin_axis.split(",")], dtype=float)
            score = float("nan")
        else:
            axis, score = symmetry_axis(
                mesh, args.symmetry_tolerance, args.symmetry_support
            )
        if axis is None:
            print(
                f"[fp] no symmetry axis in the mesh (best {score:.2f} < "
                f"{args.symmetry_support:.2f}): spin check disabled",
                flush=True,
            )
        else:
            axis = axis / np.linalg.norm(axis)
            print(f"[fp] symmetry axis {np.round(axis, 3)} (support {score:.2f})", flush=True)
            pivot = body_axis_point(np.asarray(mesh.vertices), axis) - estimator.model_center
        # Spins act on FoundationPose's centred mesh, since pose_last is expressed
        # against it; hypothesis 0 is the current pose itself.
            angles = np.arange(args.spin_hypotheses) * 2 * np.pi / args.spin_hypotheses
            spins = np.tile(np.eye(4), (len(angles), 1, 1))
            spins[:, :3, :3] = Rotation.from_rotvec(angles[:, None] * axis).as_matrix()
            spins[:, :3, 3] = pivot - spins[:, :3, :3] @ pivot
            spins = torch.as_tensor(spins, dtype=torch.float, device="cuda")
            spin_degrees = np.degrees(angles)

    def load_masks(index):
        """The object's mask, and the union of every other mask (its occluders)."""
        frame_dir = os.path.join(args.masks_root, f"frame_{index:06d}_masks")
        target, others = None, None
        if not os.path.isdir(frame_dir):
            return target, others
        for name in os.listdir(frame_dir):
            mask = cv2.imread(os.path.join(frame_dir, name), 0)
            if mask is None:
                continue
            mask = mask > 128
            if name == f"{args.object}.png":
                target = mask
            else:
                others = mask if others is None else others | mask
        if others is not None:
            # Mask edges between hand and object are uncertain; keep them out of the IoU.
            others = cv2.dilate(others.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        return target, others

    def masked_ious(poses, target, occluders, chunk=32):
        """Silhouette IoU of each pose with the object's mask, outside other masks."""
        height, width = target.shape
        target_t = torch.as_tensor(target, device="cuda")
        considered = (
            torch.ones_like(target_t) if occluders is None
            else ~torch.as_tensor(occluders, device="cuda")
        )
        ious = []
        for start in range(0, len(poses), chunk):
            _, rendered, _ = nvdiffrast_render(
                K=K, H=height, W=width, ob_in_cams=poses[start:start + chunk],
                glctx=estimator.glctx, mesh_tensors=estimator.mesh_tensors,
                output_size=(height, width), get_normal=False,
            )
            silhouettes = rendered > 0
            inter = (silhouettes & target_t & considered).sum(dim=(1, 2)).float()
            union = ((silhouettes | target_t) & considered).sum(dim=(1, 2)).float()
            ious.append((inter / union.clamp(min=1)).cpu().numpy())
        return np.concatenate(ious)

    def rotation_gap(a, b):
        """Angle between two rotation matrices, degrees."""
        return np.degrees(np.arccos(np.clip((np.trace(a.T @ b) - 1) / 2, -1, 1)))

    def relocalize(colour, depth, target, occluders, last_good):
        """Re-register a lost track from the mask.

        Global registration is ambiguous for a near-symmetric object -- upside-down
        and spun poses fit the silhouette almost equally -- so among the hypotheses
        that fit nearly as well as the best, the one closest to the last orientation
        the track trusted wins. The camera is static, so that orientation is a fair
        prior across the frames the object spent out of view.
        """
        estimator.register(
            K=K, rgb=colour, depth=depth, ob_mask=target, iteration=args.est_refine_iter
        )
        poses = estimator.poses
        ious = masked_ious(poses, target, occluders)
        candidates = np.flatnonzero(ious >= ious.max() - args.spin_margin)
        rotations = poses[:, :3, :3].data.cpu().numpy()
        pick = min(candidates, key=lambda i: rotation_gap(rotations[i], last_good))
        estimator.pose_last = poses[pick]
        pose = (poses[pick] @ estimator.get_tf_to_centered_mesh()).data.cpu().numpy()
        return pose, ious[pick]

    def spin_check(colour, depth, target, occluders):
        """Try the current pose spun about the symmetry axis; keep the best silhouette.

        Each spin is refined first (a spun pose is only roughly placed), then scored by
        IoU with the object's mask, ignoring pixels any other mask covers: a handle
        hidden by the hand then neither helps nor hurts, while one drawn over empty
        table does. FoundationPose's own scorer is not used because it barely separates
        spins of a near-cylindrical object (about 1 point in 110).
        Returns the winning index (0 = keep), the refined poses and the IoUs.
        """
        depth_t = torch.as_tensor(depth, device="cuda", dtype=torch.float)
        depth_t = erode_depth(depth_t, radius=2, device="cuda")
        depth_t = bilateral_filter_depth(depth_t, radius=2, device="cuda")
        K_t = torch.as_tensor(K, dtype=torch.float, device="cuda")
        xyz_map = depth2xyzmap_batch(depth_t[None], K_t[None], zfar=np.inf)[0]
        hypotheses = (estimator.pose_last.reshape(1, 4, 4) @ spins).data.cpu().numpy()
        poses, _ = estimator.refiner.predict(
            mesh=estimator.mesh, mesh_tensors=estimator.mesh_tensors, rgb=colour,
            depth=depth_t, K=K, ob_in_cams=hypotheses, normal_map=None, xyz_map=xyz_map,
            mesh_diameter=estimator.diameter, glctx=estimator.glctx,
            iteration=args.track_refine_iter,
        )
        ious = masked_ious(poses, target, occluders)
        best = int(np.argmax(ious))
        if best == 0 or ious[best] < args.spin_min_iou or ious[best] - ious[0] <= args.spin_margin:
            return 0, poses, ious
        # Refinement can pull a spun hypothesis most of the way back, so the nominal
        # spin overstates the change; judge the correction by what it actually does.
        relative = poses[best, :3, :3] @ poses[0, :3, :3].T
        cos = ((torch.trace(relative) - 1) / 2).clamp(-1, 1)
        if np.degrees(torch.arccos(cos).item()) < args.spin_min_angle:
            return 0, poses, ious
        return best, poses, ious

    def load(index):
        colour = cv2.imread(os.path.join(args.frames_dir, names[index]))[:, :, ::-1]
        depth = cv2.imread(
            os.path.join(args.depth_dir, depth_names[index]), cv2.IMREAD_UNCHANGED
        ).astype(np.float64) / 1000.0
        depth[(depth < 0.001) | (depth >= args.zfar)] = 0
        return np.ascontiguousarray(colour), depth

    mask_path = os.path.join(
        args.masks_root, f"frame_{args.ref_frame:06d}_masks", f"{args.object}.png"
    )
    reference_mask = cv2.imread(mask_path, 0) > 128
    if reference_mask.sum() < 50:
        raise SystemExit(f"empty mask for '{args.object}' at frame {args.ref_frame}")

    rotations = np.tile(np.eye(3), (n_frames, 1, 1))
    translations = np.zeros((n_frames, 3))
    valid = np.zeros(n_frames, dtype=bool)
    # Spin applied by the check at each frame, degrees (0 where it kept the pose).
    spin_corrections = np.zeros(n_frames)
    relocalized = np.zeros(n_frames, dtype=bool)
    # How well the recorded pose's silhouette matches the mask, per frame: the later
    # stages use it to tell where the tracker is worth trusting. NaN without a mask.
    silhouette_iou = np.full(n_frames, np.nan)

    def record(index, pose):
        rotations[index] = pose[:3, :3]
        translations[index] = pose[:3, 3]
        valid[index] = True

    # Registration happens at the reference frame in both directions, rather than at
    # frame 0, because that is the frame the mesh's coordinates are defined in.
    for direction in ("forward", "backward"):
        colour, depth = load(args.ref_frame)
        if args.seed_reference == "identity":
            # pose_last is expressed against FoundationPose's internally centred mesh,
            # and register() returns pose_last @ tf_to_centered_mesh. Inverting that
            # transform therefore makes the reported reference pose exactly identity.
            estimator.pose_last = torch.linalg.inv(estimator.get_tf_to_centered_mesh())
            pose = np.eye(4)
        else:
            pose = estimator.register(
                K=K, rgb=colour, depth=depth, ob_mask=reference_mask, iteration=args.est_refine_iter
            )
        record(args.ref_frame, pose)

        if direction == "forward":
            order = range(args.ref_frame + 1, n_frames)
        else:
            order = range(args.ref_frame - 1, -1, -1)

        # The correction a check proposes, as a rotation in the object's own frame, and
        # how many consecutive checks have proposed about the same one.
        pending, streak = None, 0
        # Orientation (camera frame) of the last pose that matched the mask.
        last_good = np.eye(3)
        for index in order:
            colour, depth = load(index)
            pose = estimator.track_one(
                rgb=colour, depth=depth, K=K, iteration=args.track_refine_iter
            )
            target, occluders = load_masks(index)
            if target is None or target.sum() < 50:
                record(index, pose)
                pending, streak = None, 0
                continue

            checking = spins is not None and index % args.spin_every == 0
            if checking:
                best, poses, ious = spin_check(colour, depth, target, occluders)
                confidence = float(ious.max())
            else:
                best = 0
                confidence = float(masked_ious(
                    estimator.pose_last.reshape(1, 4, 4), target, occluders
                )[0])
            silhouette_iou[index] = float(ious[0]) if checking else confidence

            # Lost: the object is masked but no pose explains it. A global search is
            # worth it whatever the object's shape, so this runs without the spin check.
            if args.relocalize_iou > 0 and confidence < args.relocalize_iou:
                pose, iou = relocalize(colour, depth, target, occluders, last_good)
                relocalized[index] = True
                silhouette_iou[index] = iou
                print(
                    f"[fp] {direction} frame {index}: track lost "
                    f"(best IoU {confidence:.3f}), re-registered (IoU {iou:.3f})",
                    flush=True,
                )
                if iou >= args.spin_min_iou:
                    last_good = pose[:3, :3]
                # The fresh registration has the same spin ambiguity as the original
                # one, so the checks that follow start from scratch.
                record(index, pose)
                pending, streak = None, 0
                continue

            if best == 0:
                pending, streak = None, 0
            else:
                correction = (poses[0, :3, :3].T @ poses[best, :3, :3]).data.cpu().numpy()
                agrees = pending is not None and np.degrees(np.arccos(np.clip(
                    (np.trace(pending.T @ correction) - 1) / 2, -1, 1))) < 30
                pending, streak = correction, (streak + 1 if agrees else 1)
                strong = ious[best] - ious[0] >= args.spin_strong_margin
                if streak >= args.spin_persist or strong:
                    estimator.pose_last = poses[best]
                    pose = (
                        poses[best] @ estimator.get_tf_to_centered_mesh()
                    ).data.cpu().numpy()
                    spin_corrections[index] = spin_degrees[best]
                    silhouette_iou[index] = float(ious[best])
                    print(
                        f"[fp] {direction} frame {index}: spin {spin_degrees[best]:.0f} deg "
                        f"(IoU {ious[best]:.3f} vs {ious[0]:.3f})",
                        flush=True,
                    )
                    pending, streak = None, 0
            if silhouette_iou[index] >= args.spin_min_iou:
                last_good = pose[:3, :3]
            record(index, pose)
            if index % 50 == 0:
                print(f"[fp] {direction} frame {index}", flush=True)

    reference_pose = np.eye(4)
    reference_pose[:3, :3] = rotations[args.ref_frame]
    reference_pose[:3, 3] = translations[args.ref_frame]
    offset = np.linalg.norm(reference_pose[:3, 3])
    angle = np.degrees(
        np.arccos(np.clip((np.trace(reference_pose[:3, :3]) - 1) / 2, -1, 1))
    )

    np.savez(
        args.out,
        rotation=rotations,
        translation=translations,
        valid=valid,
        inliers=np.zeros(n_frames, dtype=int),
        resting=np.zeros(n_frames, dtype=bool),
        spin_corrections=spin_corrections,
        silhouette_iou=silhouette_iou,
        relocalized=relocalized,
        ref_frame=args.ref_frame,
    )
    print(f"Wrote {args.out}: {valid.sum()}/{n_frames} frames tracked")
    if np.isfinite(silhouette_iou).any():
        print(f"  silhouette IoU with the mask: median {np.nanmedian(silhouette_iou):.3f}, "
              f"p10 {np.nanpercentile(silhouette_iou, 10):.3f}")
    # The mesh already sits in the reference frame's camera coordinates, so a good
    # registration returns near-identity there; a large value means the mesh and the
    # frame it was reconstructed from have drifted apart.
    print(f"  reference-frame sanity: {offset * 1000:.1f} mm, {angle:.1f} deg from identity")


if __name__ == "__main__":
    main()
