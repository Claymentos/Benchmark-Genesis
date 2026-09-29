#!/usr/bin/env python3
"""
Replay a retargeted Wuji Hand trajectory in SuperDex.

A port of replay_wuji_genesis.py onto SuperDex Physics, keeping the same command line
so the two can be run against each other on one trajectory. The reconstruction side is
unchanged -- this only swaps the simulator.

What the port changes, and why:

  * The wrist is welded rather than free. Genesis needed a free base, which is what
    made the hand accumulate gravity between teleports; here the URDF's world_joint is
    set to HARD and the wrist is moved with set_root_transform, so the hand is
    kinematic by construction and that failure mode cannot recur. The cost is that
    SuperDex infers no velocity from successive root transforms, so the friction a
    moving grasp transmits has to come from the fingers rather than from the wrist
    dragging the object.
  * Fingers are driven by SuperDex's articulated pose controller instead of Genesis's
    per-DoF kp/kv. Same idea -- a target pose the joints are pulled towards rather than
    teleported to -- so a finger pressing on something yields instead of sweeping
    through it.
  * Rendering is Polyscope offscreen rather than a Genesis camera. It returns RGBA;
    the alpha is dropped before the mp4 is written.

Run in the `superdex` env:
    python replay_wuji_superdex.py --traj wuji_traj.npz --record replay.mp4
"""

import argparse
import os

import numpy as np
from scipy.spatial.transform import Rotation

DEFAULT_WUJI_DIR = os.environ.get(
    "WUJI_RETARGETING_DIR",
    os.path.expanduser("~/wuji-ego-mint/eval/simulate/wuji-retargeting"),
)


def camera_to_world(up_cam):
    """Rotation taking OpenCV camera coordinates to a z-up world.

    Identical to the Genesis replay's, and deliberately so: both simulators are z-up
    here, so a trajectory placed by one lands in the same place in the other and the
    two recordings can be compared frame for frame.
    """
    z_axis = np.asarray(up_cam, dtype=float)
    z_axis /= np.linalg.norm(z_axis)
    forward = np.array([0.0, 0.0, 1.0])
    y_axis = forward - forward.dot(z_axis) * z_axis
    y_axis /= np.linalg.norm(y_axis)
    return np.stack([np.cross(y_axis, z_axis), y_axis, z_axis])


def parse_args():
    parser = argparse.ArgumentParser(description="Replay a Wuji Hand trajectory in SuperDex")
    parser.add_argument("--traj", required=True, help="Trajectory .npz from hawor_to_wuji_qpos.py")
    parser.add_argument("--urdf", default=None, help="Wuji Hand URDF (default: bundled for the side)")
    parser.add_argument("--wuji-dir", default=DEFAULT_WUJI_DIR)
    parser.add_argument("--object-mesh", default=None, help="RecGen posed_mesh.obj")
    parser.add_argument("--object-poses", default=None, help="Per-frame object poses .npz")
    parser.add_argument("--up", default="0,-1,0", help="Real up in camera coordinates, 'x,y,z'")
    parser.add_argument("--ground-plane", default=None, help="ground_plane.json; overrides --up")
    parser.add_argument(
        "--physics",
        action="store_true",
        help="Simulate the object instead of replaying its tracked pose: gravity on, real "
        "contacts, fingers driven by the pose controller.",
    )
    parser.add_argument(
        "--no-object-physics",
        action="store_true",
        help="Exempt the object from the physics: it keeps its tracked pose every frame "
        "while the hand still runs under --physics.",
    )
    parser.add_argument("--physics-substeps", type=int, default=4,
                        help="Physics steps per video frame. SuperDex integrates implicitly and "
                             "takes far larger stable steps than Genesis, so this is lower.")
    parser.add_argument("--release-frame", type=int, default=None,
                        help="Frame at which --physics hands the object to the simulation")
    parser.add_argument("--frame-range", type=int, nargs=2, default=None, metavar=("START", "STOP"))
    parser.add_argument("--object-density", type=float, default=500.0, help="kg/m3")
    parser.add_argument("--object-friction", type=float, default=1.0)
    parser.add_argument("--finger-stiffness", type=float, default=50.0,
                        help="Pose-controller stiffness for the finger joints")
    parser.add_argument("--finger-damping", type=float, default=5.0)
    parser.add_argument("--sim-out", default=None, help="Write the simulated object trajectory here")
    parser.add_argument("--record", default=None, help="Write an mp4 to this path")
    parser.add_argument("--res", type=int, nargs=2, default=(960, 720))
    parser.add_argument("--threads", type=int, default=-1,
                        help="SuperDex worker threads; -1 auto-selects, 0 is single-threaded")
    return parser.parse_args()


def object_transform(physics, rotation, translation, centre):
    """Place a centred object mesh: its own centroid offset folded into the transform."""
    return to_transform(physics, rotation, rotation @ centre + translation)


def to_transform(physics, rotation, translation):
    """A SuperDex TransformRT from a rotation matrix and a translation."""
    transform = physics.TransformRT()
    quat = Rotation.from_matrix(rotation).as_quat()  # xyzw
    transform.rotation = [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]
    transform.translation = [float(value) for value in translation]
    return transform


def dof_index_map(prefab):
    """Map each joint name to the indices of the DoFs it owns.

    SuperDex keeps the URDF's fixed tip joints as zero-DoF entries and prepends a free
    world_joint, so a joint's position in `prefab.joints` is not its DoF offset. The
    map is built by walking the joints and accumulating their DoF counts, which is the
    only ordering the actor guarantees.
    """
    mapping, offset = {}, 0
    for joint in prefab.joints:
        count = _joint_dof_count(joint)
        mapping[joint.name] = list(range(offset, offset + count))
        offset += count
    return mapping, offset


def _joint_dof_count(joint):
    import superdex.physics as physics

    kind = joint.type
    if kind == physics.ArticulatedJointType.HARD:
        return 0
    if kind == physics.ArticulatedJointType.FREE:
        return 6
    # Revolute and prismatic joints are the single-DoF cases the Wuji hand is made of.
    return 1


def main():
    args = parse_args()
    traj = np.load(args.traj)
    qpos = traj["qpos"]
    joint_names = [str(name) for name in traj["joint_names"]]
    side = str(traj["side"])
    fps = float(traj["fps"])
    valid = traj["valid"].astype(bool)

    urdf = args.urdf or os.path.join(
        args.wuji_dir, "wuji_retargeting", "wuji-description", "hand", "body", "urdf", f"{side}.urdf"
    )

    if args.ground_plane:
        import json

        with open(args.ground_plane) as handle:
            up = json.load(handle)["normal_camera"]
    else:
        up = [float(value) for value in args.up.split(",")]
    cam_to_world = camera_to_world(up)

    positions = traj["wrist_pos"] @ cam_to_world.T
    rotations = cam_to_world @ Rotation.from_quat(
        np.c_[traj["wrist_quat"][:, 1:], traj["wrist_quat"][:, 0]]
    ).as_matrix()

    object_vertices = None
    if args.object_mesh:
        import trimesh

        mesh = trimesh.load(args.object_mesh, process=False)
        object_vertices = np.asarray(mesh.vertices) @ cam_to_world.T

    object_poses = None
    release_frame = 0
    if args.object_poses:
        tracked = np.load(args.object_poses)
        solved = tracked["valid"].astype(bool)
        object_rotations = np.einsum("ij,tjk->tik", cam_to_world, tracked["rotation"])
        object_poses = (object_rotations, tracked["translation"] @ cam_to_world.T, solved)
        release_frame = args.release_frame
        if release_frame is None:
            release_frame = int(tracked["ref_frame"]) if "ref_frame" in tracked else 0
        if not solved[release_frame]:
            later = np.flatnonzero(solved[release_frame:])
            if later.size:
                release_frame += int(later[0])

    # Same recentring as the Genesis replay, so a trajectory lands in the same place in
    # both and the recordings line up.
    anchor = positions[valid] if object_vertices is None else object_vertices
    shift = np.zeros(3)
    shift[:2] = -anchor[:, :2].mean(axis=0)
    shift[2] = -min(positions[valid][:, 2].min(), anchor[:, 2].min())
    positions = positions + shift

    frame_start, frame_stop = 0, len(qpos)
    if args.frame_range:
        frame_start = max(0, args.frame_range[0])
        frame_stop = min(len(qpos), args.frame_range[1])
        if frame_stop <= frame_start:
            raise SystemExit(f"empty --frame-range {args.frame_range}")
        release_frame = max(release_frame, frame_start)

    object_simulated = args.physics and not args.no_object_physics
    if args.physics and not object_simulated:
        print("[superdex] object exempt from the physics: holding its tracked pose")

    # ────────────────────────────── scene ──────────────────────────────
    import superdex.physics as physics
    import superdex.robotics as robotics

    physics.initialize(num_worker_threads=args.threads)
    scene = physics.create_scene("wuji replay")
    scene.set_gravity([0.0, 0.0, -9.81 if args.physics else 0.0])

    ground = physics.create_plane_shape(normal=[0.0, 0.0, 1.0], distance=0.0)
    ground_params = physics.RigidActorParams()
    ground_params.name = "ground"
    ground_params.shape = ground
    ground_params.is_static = True
    scene.create_rigid_actor(ground_params)

    prefab = robotics.load_bot_prefab_from_urdf_file(urdf)
    # The URDF loader prepends a free world_joint. Welding it makes the wrist kinematic:
    # it goes exactly where the reconstruction says and contact cannot push it around,
    # which is the behaviour the replay wants and Genesis had to be coerced into.
    prefab.joints[0].type = physics.ArticulatedJointType.HARD
    mapping, total_dofs = dof_index_map(prefab)
    context = robotics.create_context()
    bot = robotics.create_bot(scene, prefab, context)
    hand = bot.get_articulated_actor()
    print(f"[superdex] hand: {len(prefab.links)} links, {hand.get_num_dofs()} dofs "
          f"({total_dofs} from the prefab)")

    missing = [name for name in joint_names if not mapping.get(name)]
    if missing:
        raise SystemExit(f"joints missing from the URDF: {missing[:5]}")
    finger_dofs = np.array([mapping[name][0] for name in joint_names])

    if args.physics:
        # The pose controller is the counterpart of Genesis's per-DoF kp/kv: a target the
        # joints are pulled towards, so a finger that meets the object stops there rather
        # than driving through it. SuperDex wants one tracking entry per DoF.
        controller = physics.PoseControllerParams()
        # One entry per joint, not per DoF: the welded world_joint and the five fixed
        # tip joints carry no DoFs but still need a slot, or the actor rejects the
        # controller with "Invalid number of tracking params".
        for _ in range(len(prefab.joints)):
            tracking = physics.PoseTrackingParams()
            tracking.stiffness = args.finger_stiffness
            tracking.damping = args.finger_damping
            controller.joint_tracking.append(tracking)
        hand.add_articulated_pose_controller(controller)

    object_actor = None
    object_centre = np.zeros(3)
    if args.object_mesh:
        # SuperDex bakes a grid SDF per collider, and its resolution follows the mesh's
        # smallest features. A raw reconstruction mesh -- 1.2M faces, many of them
        # slivers -- asks for a grid of over a billion cells and the bake fails, so the
        # mesh is decimated first, exactly as the Genesis replay decimates before CoACD.
        collider = mesh.copy()
        try:
            collider = collider.simplify_quadric_decimation(face_count=5000)
        except Exception as error:  # noqa: BLE001 - decimation is an optimisation, not a requirement
            print(f"[superdex] could not decimate ({error}); using the mesh as it is")
        # The mesh sits in camera coordinates, so its own origin is the camera, a metre
        # away. That offset would be baked into the SDF's bounds for no reason, so the
        # vertices are centred and the offset moved into the per-frame transform.
        object_centre = np.asarray(collider.vertices).mean(axis=0)
        vertices = np.asarray(collider.vertices, dtype=float) - object_centre
        shape = physics.create_tri_mesh_shape(
            vertices.ravel().tolist(),
            np.asarray(collider.faces, dtype=int).ravel().tolist(),
        )
        print(f"[superdex] object collider: {len(collider.faces)} faces")
        params = physics.RigidActorParams()
        params.name = "object"
        params.shape = shape
        params.is_static = not object_simulated
        if hasattr(params, "density"):
            params.density = args.object_density
        object_actor = scene.create_rigid_actor(params)
        if object_simulated:
            # Contact points are not computed unless asked for: the query is registered
            # once and its results become available after each step. Reading it without
            # registering raises rather than returning nothing.
            # Contact *points* are every contact the object has, the table included, so
            # a resting cup reads as touching on every frame. The force from the hand
            # specifically is what the Genesis replay counted, so that is what is
            # registered and compared.
            object_actor.register_query(physics.QueryType.TOTAL_CONTACT_FORCE)

    # ────────────────────────────── viewer ──────────────────────────────
    viewer = None
    if args.record:
        from superdex.physics.viewer import Viewer, ViewerCfg

        # SuperDex is X-forward, Y-left, Z-up, which is the "ros" preset. Left at
        # Polyscope's own Y-up default the ground plane renders as a diagonal wall.
        viewer = Viewer(ViewerCfg(size=tuple(args.res), offscreen=True, coordinate_system="ros"))
        viewer.set_scene(scene)
        framed = positions[valid]
        if object_vertices is not None:
            framed = np.vstack([framed, object_vertices + shift])
        if object_poses is not None:
            _, translations_w, solved = object_poses
            framed = np.vstack([framed, translations_w[solved] + shift])
        centre = framed.mean(axis=0)
        distance = 2.0 * max(np.ptp(framed, axis=0).max(), 0.2)
        viewer.set_camera_view(
            [float(v) for v in centre + distance * np.array([0.7, -0.7, 0.45])],
            [float(v) for v in centre],
        )

    # ────────────────────────────── replay ──────────────────────────────
    step = 1.0 / (fps * args.physics_substeps)
    pose = np.zeros(hand.get_num_dofs())
    simulated_pos, contact_frames, finger_error = [], [], []
    placed = False

    # Frames are streamed to the file rather than collected. Polyscope holds its own
    # render structures alongside them, and a clip's worth of 960x720 RGBA on top of
    # that exhausted memory partway through -- the run died in the viewer's teardown
    # with std::bad_alloc and wrote nothing at all.
    writer = None
    written = 0
    if args.record:
        import imageio.v2 as imageio

        writer = imageio.get_writer(args.record, fps=fps, macro_block_size=1)

    for t in range(frame_start, frame_stop):
        hand.set_root_transform(to_transform(physics, rotations[t], positions[t]))
        pose[finger_dofs] = qpos[t]
        if args.physics:
            hand.set_articulated_target_pose(pose)
        else:
            hand.set_articulated_pose_from_joints(pose)

        if object_poses is not None and object_actor is not None:
            object_rotations, translations_w, solved = object_poses
            if object_simulated:
                if not placed and t >= release_frame and solved[t]:
                    object_actor.set_root_transform(object_transform(
                        physics, object_rotations[t], translations_w[t] + shift, object_centre))
                    placed = True
                    print(f"[superdex] object released to the simulation at frame {t}")
                elif not placed and solved[t]:
                    object_actor.set_root_transform(object_transform(
                        physics, object_rotations[t], translations_w[t] + shift, object_centre))
            elif solved[t]:
                object_actor.set_root_transform(object_transform(
                    physics, object_rotations[t], translations_w[t] + shift, object_centre))

        for _ in range(args.physics_substeps):
            scene.step(step)

        if args.physics:
            # Mochi writes through single-precision spans only; a float64 buffer is
            # rejected outright rather than converted.
            reached = np.zeros(len(finger_dofs), dtype=np.float32)
            hand.get_dof_values(finger_dofs.tolist(), reached)
            finger_error.append(np.abs(reached - qpos[t]))
        if object_simulated and object_actor is not None:
            transform = object_actor.get_root_transform()
            # The actor's origin is the centred mesh's centroid, so the stored position
            # is comparable with the tracked translation only after taking that back out.
            origin = np.asarray(list(transform.translation), dtype=float)
            rotation = Rotation.from_quat(list(transform.rotation)).as_matrix()
            simulated_pos.append(origin - rotation @ object_centre)
            force = np.asarray(
                list(object_actor.get_contact_force_from_actor_world(hand)), dtype=float
            )
            contact_frames.append(int(np.linalg.norm(force) > 1e-6))
        if viewer is not None and writer is not None:
            frame = viewer.render()
            if frame is not None:
                writer.append_data(np.asarray(frame)[..., :3])
                written += 1

    # ────────────────────────────── reporting ──────────────────────────────
    if args.physics and finger_error:
        report = np.degrees(np.stack(finger_error))
        print(f"[superdex] finger tracking error vs the retargeted pose: "
              f"median {np.median(report):.1f} deg, 90th pct {np.percentile(report, 90):.1f} deg")
    if object_simulated and simulated_pos:
        simulated_pos = np.stack(simulated_pos)
        contact_frames = np.asarray(contact_frames)
        touching = contact_frames > 0
        print(f"[superdex] hand-object contact on {touching.sum()}/{len(touching)} frames"
              + (f", first at {frame_start + int(np.argmax(touching))}" if touching.any()
                 else " (never touched)"))
        if object_poses is not None:
            _, translations_w, solved = object_poses
            free = solved.copy()
            free[:release_frame] = False
            free = free[frame_start:frame_stop][: len(simulated_pos)]
            gap = np.linalg.norm(
                simulated_pos - (translations_w + shift)[frame_start:frame_stop][: len(simulated_pos)],
                axis=1,
            )[free]
            if gap.size:
                print(f"[superdex] simulated vs tracked object position, from the release at "
                      f"frame {release_frame}: median {np.median(gap) * 100:.1f} cm, "
                      f"final {gap[-1] * 100:.1f} cm")
        if args.sim_out:
            np.savez(args.sim_out, position=simulated_pos, shift=shift,
                     contacts=contact_frames,
                     frame_range=np.array([frame_start, frame_stop]),
                     release_frame=release_frame)
            print(f"Wrote {args.sim_out}")

    if writer is not None:
        writer.close()
        print(f"Wrote {args.record}: {written} frames @ {fps} fps")

    physics.shutdown()


if __name__ == "__main__":
    main()
