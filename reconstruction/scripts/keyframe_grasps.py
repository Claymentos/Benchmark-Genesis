#!/usr/bin/env python3
"""
Synthesize Wuji Hand grasps with Lightning Grasp at the video's grasp keyframes.

The Video2Sim2Real half of this pipeline reconstructs the grasp the human actually
made; this asks a grasp synthesiser what grasps the object *affords*, at the moments
the human was holding it, and puts the two side by side. The reconstructed grasp is
the demonstration and stays the reference -- what Lightning Grasp adds is a set of
alternatives that provably close on the mesh, which refine_grasp.py can be seeded
with when the reconstructed one lands beside the object rather than on it.

Keyframes are the grasp windows the pipeline already finds: hand-to-object-surface
distance under a finger's thickness, short gaps bridged, short runs dropped
(lock_object_to_hand.runs_of). Each window contributes its onset, middle and release
-- the moments a grasp has to work at, and the ones where the hand's pose relative to
the object is most different.

One thing worth being straight about: Lightning Grasp synthesises against the object's
*mesh*, so the candidate set does not depend on where the object is at a given frame.
Running it per keyframe would resample the same distribution with a different seed.
What genuinely differs per keyframe is which candidate belongs there -- the object has
turned, so a grasp that was an overhand pinch at the onset is an underhand one at
release, and the supporting plane rules out a different subset each time. So the
synthesis runs once by default and the *selection* is per keyframe, scored against the
hand the human actually had at that frame. --per-keyframe re-synthesises anyway, for
when a wider spread of candidates is wanted.

The Wuji Hand itself is described in the Lightning Grasp checkout, as its own hands
are (lygra/robot/wuji.py), so `demo.py --robot wuji_left --visualize` works too.

Run in the `lygra` env:
    python keyframe_grasps.py --object-poses cup_traj_points.npz --mesh posed_mesh.obj \\
        --hand-traj wuji_traj.npz --hand-meshes all_hand_meshes.npz --side left \\
        --ground-plane ground_plane.json --out keyframe_grasps.npz
"""

import argparse
import json
import os
import sys

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

# Lightning Grasp allocates large short-lived buffers per pass and is documented not
# to release GPU memory cleanly; this keeps the allocator from fragmenting on top of
# that. Set before torch is imported anywhere below. The pipeline exports it too.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lock_object_to_hand import hand_to_surface, runs_of

DEFAULT_LYGRA_DIR = os.environ.get(
    "LIGHTNING_GRASP_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lightning-grasp"),
)


def parse_args():
    parser = argparse.ArgumentParser(description="Lightning Grasp grasps at the video's grasp keyframes")
    parser.add_argument("--object-poses", required=True, help="Object trajectory .npz")
    parser.add_argument("--mesh", required=True, help="Object mesh, metres")
    parser.add_argument("--hand-traj", required=True, help="wuji_traj.npz")
    parser.add_argument("--hand-meshes", default=None, help="HaWoR all_hand_meshes.npz, for contact")
    parser.add_argument("--side", default="left", choices=("left", "right"))
    parser.add_argument("--ground-plane", default=None, help="ground_plane.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--lygra-dir", default=DEFAULT_LYGRA_DIR)

    # Keyframe selection, matching lock_object_to_hand.py's grasp detection.
    parser.add_argument("--contact", type=float, default=0.02,
                        help="Hand-to-surface distance counting as contact, metres")
    parser.add_argument("--fallback-near", type=float, default=0.12,
                        help="Without hand meshes, the wrist-to-surface distance used instead")
    parser.add_argument("--fill-gap", type=int, default=5, help="Contact gaps up to this are bridged")
    parser.add_argument("--min-grasp", type=int, default=8, help="Shorter contact runs are dropped")
    parser.add_argument("--keyframes", default="onset,middle,release",
                        help="Which moment of each grasp window to take")
    parser.add_argument("--keyframes-json", default=None,
                        help="keyframes.json from extract_keyframes.py. Uses Video2Sim2Real's "
                             "object-centric keyframes instead of this script's hand-centric "
                             "grasp windows, which is what its refinement anchors on")

    # Lightning Grasp's own knobs; the defaults follow its demo.
    parser.add_argument("--per-keyframe", action="store_true",
                        help="Re-run synthesis at every keyframe instead of sharing one candidate set")
    parser.add_argument("--target-candidates", type=int, default=250,
                        help="Accumulate passes until this many grasps survive filtering")
    parser.add_argument("--max-passes", type=int, default=12,
                        help="Give up after this many passes even if the target is not met")
    parser.add_argument("--n-contact", type=int, default=3)
    parser.add_argument("--batch-size-outer", type=int, default=128)
    parser.add_argument("--batch-size-inner", type=int, default=256)
    parser.add_argument("--n-sample-point", type=int, default=2048)
    parser.add_argument("--ik-finetune-iter", type=int, default=5)
    parser.add_argument("--zo-lr-sigma", type=float, default=5.0)
    parser.add_argument("--support-radius", type=float, default=0.01,
                        help="Lightning Grasp's concave-point filter (paper section 5.1): a "
                             "point is dropped when material stands proud within this radius "
                             "of it, since a finger could not reach in. The default is its "
                             "demo's. It is a probe size, so it must suit the feature being "
                             "grasped: at 10 mm only 11%% of a mug handle's inner wall "
                             "survives against 45%% of its outer, which leaves the search "
                             "able to wrap the handle from outside but not to reach through "
                             "it. 5 mm doubles the inner surface")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--part", default=None,
                        help="part_<name>.npz from label_mesh_part.py: the mesh subset a "
                             "region of the object occupies, such as a handle")
    parser.add_argument("--part-mode", choices=("restrict", "stratify"), default="restrict",
                        help="restrict: Lightning Grasp may only place contacts on the "
                             "part, so every grasp it returns holds the object there. "
                             "stratify: the part is merely over-represented among the "
                             "placements, and grasps elsewhere are still allowed")
    parser.add_argument("--part-share", type=float, default=0.5,
                        help="With --part-mode stratify, the share of placements drawn "
                             "from the part")
    parser.add_argument("--seed-from-demo", action="store_true",
                        help="Sample object placements around the hand-object transform the "
                             "demonstration had at the contact keyframe, instead of "
                             "uniformly. Lightning Grasp draws the hand-side contact "
                             "direction isotropically, so without this the candidates' "
                             "wrist roll is uniform over the sphere and matching the human "
                             "is luck -- 85 degrees away at best over 132 candidates here")
    parser.add_argument("--seed-cone", type=float, default=35.0,
                        help="Half-angle the seeded contact direction is sampled within, degrees")
    parser.add_argument("--seed-radius", type=float, default=0.03,
                        help="Radius the seeded contact position is sampled within, metres")

    # Scoring a candidate against the demonstration.
    parser.add_argument("--rotation-weight", type=float, default=0.05,
                        help="Metres per radian, trading wrist orientation against position "
                             "when ranking candidates against the demonstrated grasp")
    parser.add_argument("--plane-margin", type=float, default=-0.005,
                        help="A candidate whose palm sits below the supporting plane by more "
                             "than this is rejected, metres")
    parser.add_argument("--keep", type=int, default=16,
                        help="Candidates stored per keyframe, best first")
    return parser.parse_args()


def v2s2r_keyframes(args, valid):
    """The object-centric keyframes extract_keyframes.py found, as (frame, label) pairs."""
    with open(args.keyframes_json) as handle:
        found = json.load(handle)["keyframes"]
    keyframes = []
    for name, frame in found.items():
        if frame is None:
            continue
        if not valid[frame]:
            print(f"[keyframes] {name} frame {frame} has no solved object pose, skipped")
            continue
        keyframes.append((int(frame), name))
    print(f"[keyframes] Video2Sim2Real keyframes: {keyframes}")
    return sorted(keyframes), np.zeros((0, 2), dtype=int)


def grasp_keyframes(args, rotation, translation, valid):
    """Grasp windows from hand-to-surface contact, and the keyframes inside them."""
    hand = np.load(args.hand_traj)
    mesh = trimesh.load(args.mesh, process=False)
    vertices = np.asarray(mesh.vertices)
    surface = vertices[:: max(1, len(vertices) // 4000)]

    hand_valid = hand["valid"].astype(bool)
    usable = valid & hand_valid
    if args.hand_meshes:
        meshes = np.load(args.hand_meshes)
        points = meshes[f"{args.side}_vertices"]
        hawor_valid = meshes[f"{args.side}_valid"].astype(bool)
        distance = hand_to_surface(points, usable & hawor_valid, rotation, translation, surface)
        threshold = args.contact
    else:
        distance = hand_to_surface(
            hand["wrist_pos"][:, None, :], usable, rotation, translation, surface
        )
        threshold = args.fallback_near
        print("[keyframes] no --hand-meshes: wrist-to-surface distance instead of the hand's own")

    windows = runs_of(usable & (distance < threshold), args.fill_gap, args.min_grasp)
    print(f"[keyframes] grasp windows: {[(int(a), int(b - 1)) for a, b in windows] or 'none'}")

    wanted = [name.strip() for name in args.keyframes.split(",")]
    keyframes = []
    for start, stop in windows:
        moments = {"onset": start, "middle": (start + stop - 1) // 2, "release": stop - 1}
        for name in wanted:
            if name not in moments:
                raise SystemExit(f"unknown keyframe '{name}'; pick from {sorted(moments)}")
            frame = int(moments[name])
            if valid[frame] and frame not in [k for k, _ in keyframes]:
                keyframes.append((frame, f"{name}@{start}-{stop - 1}"))
    return sorted(keyframes), windows


def cone_perturb(directions, half_angle):
    """Rotate each direction by a random angle up to `half_angle`, uniform on the cap."""
    count = len(directions)
    cosines = 1.0 - np.random.rand(count) * (1.0 - np.cos(half_angle))
    sines = np.sqrt(np.maximum(1.0 - cosines ** 2, 0.0))
    azimuth = np.random.rand(count) * 2 * np.pi
    local = np.stack([sines * np.cos(azimuth), sines * np.sin(azimuth), cosines], axis=1)

    # A frame per direction, with z along it.
    axis = np.tile(np.array([0.0, 0.0, 1.0]), (count, 1))
    axis[np.abs(directions[:, 2]) > 0.9] = np.array([1.0, 0.0, 0.0])
    x = np.cross(axis, directions)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    y = np.cross(directions, x)
    return np.einsum("nij,nj->ni", np.stack([x, y, directions], axis=2), local)


def demonstration_object_poses(n, points, normals, prior, tree, mesh_data):
    """Object placements drawn around the transform the demonstration had.

    Replaces Lightning Grasp's own sampler for this case. Its version pairs a random
    object point with an independently drawn hand contact, which is what makes the
    wrist roll uniform over the sphere; here the two are paired on purpose. For an
    object point of normal n, the hand contact direction that reproduces the
    demonstrated transform M is -(R_M n) and the contact position is M p, so each
    point carries its own target and the cone is applied per point rather than around
    an average of them. Averaging was the first attempt and it does not survive the
    handle's shape: the normals sweep round the C, so their mean points nowhere in
    particular.
    """
    import torch

    from lygra.pipeline.module.collision import batch_object_hand_collision_check
    from lygra.pipeline.module.object_placement import get_align_transform

    # The same point set the contact stages will use, so a restricted region seeds the
    # placements as well as the contacts; reading the context here instead would place
    # the whole object while only its part was allowed to touch.
    cloud_points = points.cpu().numpy()
    cloud_normals = normals.cpu().numpy()
    rotation, translation = prior["rotation"], prior["translation"]

    index = np.random.randint(0, len(cloud_points), n)
    object_points, object_normals = cloud_points[index], cloud_normals[index]

    targets = -(object_normals @ rotation.T)
    targets /= np.linalg.norm(targets, axis=1, keepdims=True)
    directions = cone_perturb(targets, np.radians(prior["cone"]))

    centres = object_points @ rotation.T + translation
    offsets = np.random.randn(n, 3)
    offsets *= (prior["radius"] * np.random.rand(n, 1) ** (1 / 3)
                / np.linalg.norm(offsets, axis=1, keepdims=True))
    positions = centres + offsets

    transforms = np.array([
        get_align_transform(object_points[i], object_normals[i], positions[i], directions[i])
        for i in range(n)
    ])
    poses = torch.from_numpy(transforms).cuda().float()

    cloud = points.unsqueeze(0).expand(len(poses), -1, -1)
    moved = torch.bmm(cloud, poses[:, :3, :3].transpose(-1, -2)) + poses[:, :3, 3].unsqueeze(1)
    free = batch_object_hand_collision_check(tree=tree, mesh=mesh_data, object_point=moved)
    keep = torch.where(free)

    zeros = torch.zeros((len(poses), 1, 3), device=poses.device)
    condition = {
        "extra_contact_pos": zeros[keep],
        "extra_contact_normal": zeros[keep],
        "extra_contact_mask": torch.zeros((len(poses), 1), device=poses.device)[keep],
    }
    return poses[keep], condition


def seed_placement(context, palm_rotation, palm_position, object_rotation,
                   object_translation, cone, radius):
    """Bias object placement toward the hand-object transform the human had.

    Lightning Grasp places the object by aligning one of its surface points to a
    sampled hand contact (position p, outward normal d), with d drawn uniformly over
    the sphere. That is the right prior with no demonstration; with one it throws away
    the thing worth keeping, because the wrist roll the human used is one direction
    among all of them.

    This records the demonstrated transform M = T_palm<-object on the context;
    demonstration_object_poses turns it into placements, one target per object point.
    """
    inverse_palm = palm_rotation.T
    context["demo_prior"] = dict(
        rotation=inverse_palm @ object_rotation,
        translation=inverse_palm @ (object_translation - palm_position),
        cone=cone,
        radius=radius,
    )
    print(f"[lygra] placements seeded from the demonstration: each contact direction "
          f"within {cone:.0f} deg of the one that reproduces the demonstrated transform, "
          f"position within {100 * radius:.0f} cm of it")


def build_context(args, robot_name):
    """Everything Lightning Grasp needs that depends only on the hand and the mesh.

    Built once and reused across passes. The IK buffer pool in particular is a GPU
    memory pool meant to be held: allocating one per pass is what exhausted a 16 GB
    card after four passes.
    """
    import torch

    from lygra.contact_set import get_link_dependency_matrix
    from lygra.kinematics import build_kinematics_tree
    from lygra.memory import IKGPUBufferPool
    from lygra.mesh import get_urdf_mesh_decomposed, get_urdf_mesh_for_projection
    from lygra.mesh_analyzer import get_support_point_mask
    from lygra.robot import build_robot
    from lygra.utils.geom_utils import MeshObject

    robot = build_robot(robot_name)
    tree = build_kinematics_tree(robot.urdf_path, active_joint_names=robot.get_active_joints())
    mesh_for_ik = get_urdf_mesh_for_projection(
        urdf_path=robot.urdf_path, tree=tree,
        config=robot.get_contact_field_config(), mesh_scale=robot.get_mesh_scale(),
    )
    decomposed_static = get_urdf_mesh_decomposed(
        urdf_path=robot.urdf_path, tree=tree,
        override_link_names=robot.get_static_links(), mesh_scale=robot.get_mesh_scale(),
    )
    decomposed = get_urdf_mesh_decomposed(
        urdf_path=robot.urdf_path, tree=tree, mesh_scale=robot.get_mesh_scale()
    )
    self_collision_pairs = torch.from_numpy(tree.get_self_collision_check_link_pairs(
        link_body_id=decomposed["link_body_id"], whitelist_link=[],
        whitelist_pairs=robot.get_white_list_pairs(),
    )).cuda().int()

    contact_field = robot.get_contact_field()
    dependency_sets = tree.get_dependency_sets([robot.get_base_link()])
    contact_parent_ids = torch.tensor(
        [tree.get_link_id(link) for link in contact_field.get_all_parent_link_names()]
    ).cuda()
    dependency_matrix = get_link_dependency_matrix(contact_field, dependency_sets).cuda()
    accel_structure = contact_field.generate_acceleration_structure(method="lbvhs2")

    target = MeshObject(args.mesh)
    zo_lr = ((target.get_area() / args.n_sample_point) ** 0.5) * args.zo_lr_sigma
    points_np, normals_np = target.sample_point_and_normal(count=args.n_sample_point)
    points_all = torch.from_numpy(points_np).cuda().float()
    normals_all = torch.from_numpy(normals_np).cuda().float()
    support = get_support_point_mask(points_all, normals_all, [args.support_radius])[0]

    # A part label lives on mesh vertices; the sampled points take the label of the
    # vertex nearest each of them.
    part = None
    if args.part:
        import trimesh
        from scipy.spatial import cKDTree

        stored = np.load(args.part, allow_pickle=True)
        labels = stored["vertices"] if hasattr(stored, "files") else stored
        vertices = np.asarray(trimesh.load(args.mesh, process=False).vertices)
        if len(labels) != len(vertices):
            raise SystemExit(f"{args.part} has {len(labels)} labels for {len(vertices)} vertices")
        # The label lives on vertices; a sampled surface point takes the label of the
        # vertex nearest it.
        _, nearest = cKDTree(vertices).query(points_np)
        part = torch.from_numpy(labels[nearest]).cuda()[torch.where(support)]
        share = float(part.float().mean())
        print(f"[lygra] part covers {int(part.sum())}/{len(part)} support points "
              f"({100 * share:.1f}%); mode={args.part_mode}")
        if args.part_mode == "restrict" and int(part.sum()) < 8:
            raise SystemExit(f"only {int(part.sum())} support points on the part -- too few "
                             f"to search contacts in; widen the region or raise "
                             f"--n-sample-point")

    return dict(
        robot=robot, tree=tree, mesh_for_ik=mesh_for_ik, decomposed=decomposed,
        decomposed_static=decomposed_static, self_collision_pairs=self_collision_pairs,
        contact_field=contact_field, contact_parent_ids=contact_parent_ids,
        dependency_matrix=dependency_matrix, accel_structure=accel_structure,
        points_all=points_all, zo_lr=zo_lr,
        points=points_all[torch.where(support)], normals=normals_all[torch.where(support)],
        part=part,
        pool=IKGPUBufferPool(
            n_dof=tree.n_dof(), n_link=tree.n_link(), n_actuated_dof=tree.n_actuated_dof(),
            max_batch=min(args.batch_size_outer * args.batch_size_inner, 65536), retry=10,
        ),
    )


def sample_once(args, context):
    """One Lightning Grasp pass against a prepared context.

    Returns (q, object_pose): joint values in the order the robot config declares, and
    the transform putting the object in the hand's palm frame -- so a grasp is read as
    "the object sits here relative to my palm, with my fingers like this".
    """
    import torch

    from lygra.pipeline.module.contact_collection import sample_pose_and_contact_from_interaction
    from lygra.pipeline.module.contact_optimization import search_contact_point
    from lygra.pipeline.module.contact_query import batch_object_all_contact_fields_interaction
    from lygra.pipeline.module.kinematics import batch_contact_adjustment, batch_ik
    from lygra.pipeline.module.object_placement import (get_object_pose_sampling_args, sample_object_pose)
    from lygra.pipeline.module.postprocess import batch_assign_free_finger_and_filter

    tree = context["tree"]
    contact_field = context["contact_field"]
    points, normals = context["points"], context["normals"]

    part = context.get("part")
    if part is not None and part.any() and not part.all():
        # These points are the object as Lightning Grasp sees it for the purpose of
        # *touching* it: object placement, the contact-field traversal and the contact
        # search all run over this set. Cutting it down to the part is therefore what
        # makes the region a contact constraint rather than a preference. The whole
        # cloud stays in context["points_all"], which is what the collision check reads,
        # so the rest of the object is still solid and still rejects grasps that hit it.
        if args.part_mode == "restrict":
            index = torch.where(part)[0]
        else:
            on, off = torch.where(part)[0], torch.where(~part)[0]
            wanted = int(round(args.part_share * len(off) / max(1 - args.part_share, 1e-6)))
            repeats = on[torch.randint(len(on), (wanted,), device=on.device)]
            index = torch.cat([off, repeats])
        points, normals = points[index], normals[index]

    with torch.no_grad():
        prior = context.get("demo_prior")
        if prior is not None:
            object_poses, condition = demonstration_object_poses(
                args.batch_size_outer, points, normals, prior, tree,
                context["decomposed_static"]
            )
        else:
            object_poses, condition = sample_object_pose(
                n=args.batch_size_outer, points=points, normals=normals,
                contact_field=contact_field, tree=tree, mesh_data=context["decomposed_static"],
                sampling_args=get_object_pose_sampling_args("canonical", context["robot"]),
            )
        if len(object_poses) == 0:
            return np.zeros((0, tree.n_actuated_dof())), np.zeros((0, 4, 4))
        interaction_idx = batch_object_all_contact_fields_interaction(
            object_pos=points, object_normal=normals,
            object_pose=object_poses, accel_structure=context["accel_structure"],
        )
        interaction = (interaction_idx >= 0).int()
        link_interaction = contact_field.reduce_link_interaction(interaction)

        (domain_pos, domain_normal, domain_idx, object_poses,
         contact_link_ids, condition, valid_outer) = sample_pose_and_contact_from_interaction(
            n_contact=args.n_contact, interaction_matrix=link_interaction,
            dependency_matrix=context["dependency_matrix"], object_points=points,
            object_normals=normals, object_poses=object_poses, condition=condition,
        )
        (target_pos, target_normal, target_idx, object_poses,
         target_link_ids, target_outer) = search_contact_point(
            contact_domain_pos=domain_pos, contact_domain_normal=domain_normal,
            contact_domain_point_idx=domain_idx, object_poses=object_poses,
            contact_ids=contact_link_ids, batch_size=args.batch_size_inner,
            return_hand_frame=True, condition=condition, zo_lr=context["zo_lr"],
        )
        contact_ids, local_ids = contact_field.sample_contact_ids(
            interaction_matrix=interaction[valid_outer],
            interaction_matrix_hand_point_idx=interaction_idx[valid_outer],
            target_batch_outer_ids=target_outer, target_contact_link_ids=target_link_ids,
            target_contact_point_idx=target_idx,
        )
        contact_pos_local, contact_normal_local = contact_field.sample_contact_geometry(
            contact_ids, local_ids
        )
        # A restricted contact region can leave a pass with nothing to solve: no
        # placement put the part within reach of a contact field. The IK cannot
        # reshape an empty batch, so the pass simply returns nothing.
        if len(target_pos) == 0 or len(contact_ids) == 0:
            return np.zeros((0, tree.n_actuated_dof())), np.zeros((0, 4, 4))

        result = batch_ik(
            tree=tree, contact_ids=contact_ids, contact_parent_ids=context["contact_parent_ids"],
            contact_pos_in_linkf=contact_pos_local.float(),
            contact_normal_in_linkf=contact_normal_local.float(),
            target_contact_pos=target_pos.float(), target_contact_normal=target_normal.float(),
            object_pose=object_poses.float(), gpu_memory_pool=context["pool"],
        )
        result = batch_contact_adjustment(
            tree=tree, mesh=context["mesh_for_ik"], q_init=result["q"], q_mask=result["q_mask"],
            contact_ids=contact_ids, contact_link_ids=result["contact_link_id"],
            contact_pos_in_linkf=result["contact_pos"],
            contact_normal_in_linkf=result["contact_normal"],
            target_contact_pos=result["target_pos"], target_contact_normal=result["target_normal"],
            object_pose=result["object_pose"], n_iter=args.ik_finetune_iter,
            gpu_memory_pool=context["pool"], ret_mesh_buffer=True,
        )
        result = batch_assign_free_finger_and_filter(
            tree=tree, result=result, object_point=context["points_all"],
            self_collision_link_pairs=context["self_collision_pairs"],
            decomposed_mesh_data=context["decomposed"],
        )

    return (result["q"].detach().cpu().numpy(),
            result["object_pose"].detach().cpu().numpy())


def synthesize_pool(args, robot_name):
    """Accumulate passes until enough grasps survive, or the pass budget runs out.

    One pass returns a handful on this hand -- 0 to 159 across runs of the same
    settings -- because almost everything it proposes is rejected for penetrating the
    object: 1 of 1014 candidates came back free of hand-object collision in a traced
    run. That is this release's documented weak point ("we will integrate position
    iterations into kinematic fine-tuning to reduce hand-object penetrations"), not
    something the hand config can fix. Raising the batch size does not help and runs
    out of memory on a 16 GB card at 256x256, so the pool is built from repeated small
    passes against one prepared context.

    Running out of memory partway through is expected rather than exceptional: the
    library is documented not to release GPU memory across repeated runs of its main
    loop, and how far it gets depends on how many candidates survive each pass. The
    pool that has been built by then is still good, so the loop keeps it and stops
    instead of losing the stage. Downgrading the env to torch 2.7.0 is upstream's own
    suggested remedy if the yield is consistently cut short.
    """
    import torch

    context = build_context(args, robot_name)
    if getattr(args, "demo_prior", None):
        seed_placement(context, **args.demo_prior)
    joints, poses, total = [], [], 0
    for index in range(args.max_passes):
        try:
            batch_q, batch_pose = sample_once(args, context)
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            print(f"[lygra] pass {index + 1}/{args.max_passes}: out of GPU memory; "
                  f"keeping the {total} grasps already found")
            break
        total += len(batch_q)
        if len(batch_q):
            joints.append(batch_q)
            poses.append(batch_pose)
        torch.cuda.empty_cache()
        print(f"[lygra] pass {index + 1}/{args.max_passes}: {len(batch_q)} grasps ({total} so far)")
        if total >= args.target_candidates:
            break
    if not joints:
        return np.zeros((0, 20)), np.zeros((0, 4, 4))
    return np.concatenate(joints), np.concatenate(poses)


def palm_poses(object_rotation, object_translation, object_in_palm):
    """Where the palm has to be, in camera coordinates, for each candidate grasp.

    A candidate says where the object sits in the palm's frame; the tracker says where
    the object is in the camera's. The palm is what is left:

        T_cam<-palm = T_cam<-object * (T_palm<-object)^-1
    """
    inverse = np.linalg.inv(object_in_palm)
    rotation = object_rotation @ inverse[:, :3, :3]
    position = np.einsum("ij,njk->nik", object_rotation, inverse[:, :3, 3:])[:, :, 0] + object_translation
    return rotation, position


def main():
    args = parse_args()
    # Lightning Grasp resolves its own asset paths relative to its checkout, so this
    # runs from there -- which means every path from the caller has to be absolute
    # before the move.
    for field in ("object_poses", "mesh", "hand_traj", "hand_meshes", "ground_plane", "out"):
        value = getattr(args, field)
        if value:
            setattr(args, field, os.path.abspath(value))
    sys.path.insert(0, args.lygra_dir)
    os.chdir(args.lygra_dir)

    from lygra.robot import build_robot

    robot_name = f"wuji_{args.side}"

    track = np.load(args.object_poses)
    rotation, translation = track["rotation"], track["translation"]
    valid = track["valid"].astype(bool)
    hand = np.load(args.hand_traj)
    demonstrated = Rotation.from_quat(hand["wrist_quat"][:, [1, 2, 3, 0]])  # stored wxyz

    if args.keyframes_json:
        keyframes, windows = v2s2r_keyframes(args, valid)
    else:
        keyframes, windows = grasp_keyframes(args, rotation, translation, valid)
    if not keyframes:
        raise SystemExit("no grasp windows found; nothing to synthesize against")
    print(f"[keyframes] {len(keyframes)} keyframes: "
          f"{[(frame, label) for frame, label in keyframes]}")

    plane_normal = plane_offset = None
    if args.ground_plane:
        with open(args.ground_plane) as handle:
            plane = json.load(handle)
        # fit_ground_plane.py writes the plane as n . x + offset = 0, so the signed
        # height above the table is n . x + offset once both are scaled to a unit
        # normal. Checked against the demonstration: the wrist runs from table level
        # at the grasp onset to 15 cm up at the top of the lift.
        plane_normal = np.asarray(plane["normal_camera"], dtype=float)
        norm = np.linalg.norm(plane_normal)
        plane_normal, plane_offset = plane_normal / norm, float(plane["offset"]) / norm

    import torch

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Seeding is per keyframe, so the pool is built around the first one -- the
    # contact frame when Video2Sim2Real keyframes are used, which is the grasp the
    # refinement anchors on.
    demo_prior = None
    if args.seed_from_demo:
        frame = keyframes[0][0]
        demo_prior = dict(
            palm_rotation=Rotation.from_quat(hand["wrist_quat"][frame][[1, 2, 3, 0]]).as_matrix(),
            palm_position=hand["wrist_pos"][frame],
            object_rotation=rotation[frame],
            object_translation=translation[frame],
            cone=args.seed_cone,
            radius=args.seed_radius,
        )
        print(f"[lygra] seeding placements from the demonstration at frame {frame} "
              f"({keyframes[0][1]})")
    args.demo_prior = demo_prior

    shared = None if args.per_keyframe else synthesize_pool(args, robot_name)
    if shared is not None:
        print(f"[lygra] {len(shared[0])} candidate grasps synthesized on the mesh")

    frames, labels, chosen_q, chosen_rot, chosen_pos, chosen_cost = [], [], [], [], [], []
    kept_q, kept_rot, kept_pos = [], [], []

    for frame, label in keyframes:
        joints, object_in_palm = shared if shared is not None else synthesize_pool(args, robot_name)
        if len(joints) == 0:
            print(f"[lygra] frame {frame} ({label}): no candidates survived filtering")
            continue

        candidate_rot, candidate_pos = palm_poses(rotation[frame], translation[frame], object_in_palm)

        keep = np.ones(len(joints), dtype=bool)
        if plane_normal is not None:
            # Signed height of the palm above the supporting plane; a grasp that puts
            # the hand through the table is not one the arm could have reached.
            height = candidate_pos @ plane_normal + plane_offset
            keep &= height > args.plane_margin
            if not keep.any():
                print(f"[lygra] frame {frame} ({label}): every candidate is below the table")
                continue

        # Rank by agreement with the hand the human actually had: the demonstration is
        # the reference, and a synthesized grasp is a correction to it, not a new task.
        gap = np.linalg.norm(candidate_pos - hand["wrist_pos"][frame], axis=1)
        turn = (Rotation.from_matrix(candidate_rot).inv() * demonstrated[frame]).magnitude()
        cost = np.where(keep, gap + args.rotation_weight * turn, np.inf)
        order = np.argsort(cost)[: args.keep]
        best = order[0]

        frames.append(frame)
        labels.append(label)
        chosen_q.append(joints[best])
        chosen_rot.append(candidate_rot[best])
        chosen_pos.append(candidate_pos[best])
        chosen_cost.append(cost[best])
        kept_q.append(joints[order])
        kept_rot.append(candidate_rot[order])
        kept_pos.append(candidate_pos[order])
        print(f"[lygra] frame {frame:4d} ({label}): {int(keep.sum())}/{len(joints)} reachable, "
              f"best is {1000 * gap[best]:.0f} mm and {np.degrees(turn[best]):.0f} deg "
              f"from the demonstrated wrist")

    if not frames:
        raise SystemExit("no keyframe produced a usable grasp")

    np.savez(
        args.out,
        frames=np.array(frames),
        labels=np.array(labels),
        qpos=np.stack(chosen_q),
        wrist_rotation=np.stack(chosen_rot),
        wrist_pos=np.stack(chosen_pos),
        cost=np.array(chosen_cost),
        candidate_qpos=np.stack(kept_q),
        candidate_rotation=np.stack(kept_rot),
        candidate_pos=np.stack(kept_pos),
        joint_names=np.array(build_robot(robot_name).get_active_joints()),
        grasp_windows=np.array(windows, dtype=int).reshape(-1, 2),
        side=args.side,
    )
    print(f"Wrote {args.out}: {len(frames)} keyframe grasps, "
          f"{len(kept_q[0])} candidates kept at each")


if __name__ == "__main__":
    main()
