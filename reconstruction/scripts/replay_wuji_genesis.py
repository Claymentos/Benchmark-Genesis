#!/usr/bin/env python3
"""
Replay a retargeted Wuji Hand trajectory in Genesis.

Consumes the output of hawor_to_wuji_qpos.py and drives the Wuji Hand URDF
kinematically: each frame sets the wrist pose and the finger joint angles, with
gravity disabled so the pose is held exactly as reconstructed.

Usage:
    python replay_wuji_genesis.py --traj wuji_traj.npz --record replay.mp4
    python replay_wuji_genesis.py --traj wuji_traj.npz --viewer
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
    """Rotation taking OpenCV camera coordinates to a z-up Genesis world.

    `up_cam` is the real up direction expressed in the camera frame. It defaults to
    the camera's own up axis, which is only correct for a level camera -- for the
    tabletop recordings it is tens of degrees off, and the scene comes out tilted.
    Estimate it from the ground plane, or from the reconstruction pipeline's
    GeoCalib gravity stage.
    """
    z_axis = np.asarray(up_cam, dtype=float)
    z_axis /= np.linalg.norm(z_axis)
    forward = np.array([0.0, 0.0, 1.0])
    y_axis = forward - forward.dot(z_axis) * z_axis
    y_axis /= np.linalg.norm(y_axis)
    return np.stack([np.cross(y_axis, z_axis), y_axis, z_axis])


def to_quat_wxyz(matrix):
    quat_xyzw = Rotation.from_matrix(matrix).as_quat()
    return np.concatenate([quat_xyzw[..., 3:4], quat_xyzw[..., :3]], axis=-1)


def parse_args():
    parser = argparse.ArgumentParser(description="Replay a Wuji Hand trajectory in Genesis")
    parser.add_argument("--traj", required=True, help="Trajectory .npz from hawor_to_wuji_qpos.py")
    parser.add_argument("--urdf", default=None, help="Wuji Hand URDF (default: bundled for the side)")
    parser.add_argument("--wuji-dir", default=DEFAULT_WUJI_DIR)
    parser.add_argument(
        "--object-mesh",
        default=None,
        help="RecGen posed_mesh.obj to place in the scene (camera-frame, static)",
    )
    parser.add_argument(
        "--object-poses",
        default=None,
        help="Per-frame object poses from track_object_v2s2r.py; without it the object is static",
    )
    parser.add_argument(
        "--up",
        default="0,-1,0",
        help="Real up direction in camera coordinates, 'x,y,z' (default: the camera's own up axis)",
    )
    parser.add_argument(
        "--ground-plane",
        default=None,
        help="ground_plane.json from fit_ground_plane.py; overrides --up",
    )
    parser.add_argument(
        "--physics",
        action="store_true",
        help="Simulate the object instead of replaying its tracked pose: gravity on, real "
        "contacts, fingers driven by position control. The object is placed once at the "
        "first tracked frame and is on its own from there, so how far it drifts from the "
        "tracked trajectory measures how physically consistent the reconstruction is.",
    )
    parser.add_argument(
        "--no-object-physics",
        action="store_true",
        help="Exempt the object from the physics: it keeps its tracked pose every frame "
        "instead of being released, with gravity and contact off for it alone. The hand "
        "still runs under --physics, so this shows the reconstructed motion with the "
        "grasp failure taken out -- useful for judging the trajectory when the object "
        "would otherwise be dropped in the first second and dominate the video.",
    )
    parser.add_argument(
        "--physics-substeps",
        type=int,
        default=8,
        help="Physics steps per video frame. Contact needs a far smaller step than 1/30 s, "
        "or a finger crosses the object's wall within one step.",
    )
    parser.add_argument(
        "--release-frame",
        type=int,
        default=None,
        help="Frame at which --physics hands the object to the simulation (default: the "
        "tracker's reference frame). Before it the object is held on its tracked pose.",
    )
    parser.add_argument(
        "--frame-range",
        type=int,
        nargs=2,
        default=None,
        metavar=("START", "STOP"),
        help="Simulate only frames [START, STOP). Refinement scores a grasp by replaying "
        "the window around it, and a window is a fraction of the cost of the clip.",
    )
    parser.add_argument("--object-density", type=float, default=500.0, help="kg/m3")
    parser.add_argument("--object-friction", type=float, default=1.0)
    parser.add_argument(
        "--decompose-threshold",
        type=float,
        default=0.05,
        help="CoACD concavity threshold for the collision shape. It has to stay small "
        "enough to keep openings a finger passes through, such as a mug's handle.",
    )
    parser.add_argument("--finger-kp", type=float, default=50.0)
    parser.add_argument("--finger-kv", type=float, default=5.0)
    parser.add_argument(
        "--sim-out", default=None, help="Write the simulated object trajectory here (.npz)"
    )
    parser.add_argument("--record", default=None, help="Write an mp4 to this path")
    parser.add_argument("--viewer", action="store_true", help="Open the interactive viewer")
    parser.add_argument("--cpu", action="store_true", help="Run Genesis on CPU")
    parser.add_argument("--res", type=int, nargs=2, default=(960, 720), help="Recording resolution")
    parser.add_argument("--camera-dir", default="0.7,-0.7,0.45",
                        help="Direction from the scene centre to the camera, 'x,y,z' in the "
                             "z-up world. Vary the first two to orbit")
    parser.add_argument("--camera-distance", type=float, default=None,
                        help="Camera distance in metres. The default scales with the "
                             "scene's extent, which frames a whole trajectory but leaves a "
                             "single pose small; set it to look closely at a grasp")
    return parser.parse_args()


def main():
    args = parse_args()
    traj = np.load(args.traj)

    qpos = traj["qpos"]
    joint_names = [str(n) for n in traj["joint_names"]]
    side = str(traj["side"])
    fps = float(traj["fps"])

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

    valid = traj["valid"].astype(bool)
    positions = traj["wrist_pos"] @ cam_to_world.T
    quats = to_quat_wxyz(
        cam_to_world
        @ Rotation.from_quat(np.c_[traj["wrist_quat"][:, 1:], traj["wrist_quat"][:, 0]]).as_matrix()
    )

    object_vertices = None
    if args.object_mesh:
        import trimesh

        object_vertices = trimesh.load(args.object_mesh, process=False).vertices @ cam_to_world.T

    object_poses = None
    release_frame = 0
    if args.object_poses:
        tracked = np.load(args.object_poses)
        solved = tracked["valid"].astype(bool)
        object_rotations = np.einsum("ij,tjk->tik", cam_to_world, tracked["rotation"])
        object_quats = np.tile(to_quat_wxyz(cam_to_world), (len(solved), 1))
        object_quats[solved] = to_quat_wxyz(object_rotations[solved])
        object_poses = (object_quats, tracked["translation"] @ cam_to_world.T, solved)
        # Under --physics the object is placed once and is an outcome from there, so the
        # frame it is placed at sets up everything after it. The first solved frame is
        # usually frame 0, which leaves the object settling under gravity for the whole
        # approach and no longer where the hand expects it by the time the grasp closes.
        # The tracker's own reference frame is the pose that was actually fitted, so that
        # is the default.
        release_frame = args.release_frame
        if release_frame is None:
            release_frame = int(tracked["ref_frame"]) if "ref_frame" in tracked else 0
        # A frame with no pose cannot place anything; take the next one that has one.
        if not solved[release_frame]:
            later = np.flatnonzero(solved[release_frame:])
            if later.size:
                release_frame += int(later[0])

    frame_start, frame_stop = 0, len(qpos)
    if args.frame_range:
        frame_start = max(0, args.frame_range[0])
        frame_stop = min(len(qpos), args.frame_range[1])
        if frame_stop <= frame_start:
            raise SystemExit(f"empty --frame-range {args.frame_range}")
        # A release before the window would never fire inside it, leaving the object an
        # input for the whole run and measuring nothing.
        release_frame = max(release_frame, frame_start)

    # Recentre on the origin and lift clear of the ground plane so the camera framing
    # does not depend on where the scene happened to sit in front of the camera. Hand
    # and object share one shift, which is what keeps them registered to each other.
    anchor = positions[valid] if object_vertices is None else object_vertices
    shift = np.zeros(3)
    shift[:2] = -anchor[:, :2].mean(axis=0)
    shift[2] = -min(positions[valid][:, 2].min(), anchor[:, 2].min())
    positions += shift

    # Wrist velocity, for driving the base under --physics. Central differences where
    # both neighbours are reconstructed, so the velocity matches the motion the poses
    # describe; zero wherever the trajectory has a gap, since a jump across one is
    # travel the hand never made. A constant shift does not affect a difference, so
    # this is equally valid before or after the recentring above.
    base_velocity = np.zeros((len(positions), 6))
    rotations_w = Rotation.from_quat(np.c_[quats[:, 1:], quats[:, 0]])
    for t in range(len(positions)):
        previous, following = max(t - 1, 0), min(t + 1, len(positions) - 1)
        if previous == following or not (valid[previous] and valid[following]):
            continue
        span = (following - previous) / fps
        base_velocity[t, :3] = (positions[following] - positions[previous]) / span
        # World-frame angular velocity: the rotation carrying one neighbour onto the
        # other, as a rotation vector, per second.
        delta = rotations_w[following] * rotations_w[previous].inv()
        base_velocity[t, 3:] = delta.as_rotvec() / span
    if args.physics:
        print(f"[physics] wrist speed: median {np.median(np.linalg.norm(base_velocity[valid, :3], axis=1)) * 100:.1f} cm/s, "
              f"peak {np.linalg.norm(base_velocity[:, :3], axis=1).max() * 100:.1f} cm/s")

    # The hand and the object are simulated independently: --physics drives the hand
    # either way, and --no-object-physics takes only the object back out of it.
    object_simulated = args.physics and not args.no_object_physics
    if args.physics and not object_simulated:
        print("[physics] object exempt from the physics: holding its tracked pose")

    import genesis as gs

    gs.init(backend=gs.cpu if args.cpu else gs.gpu)
    substeps = args.physics_substeps if args.physics else 1
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1.0 / (fps * substeps),
            # Playback teleports both bodies every frame, so gravity would only fight the
            # poses being set; under --physics the object is free and needs it.
            gravity=(0.0, 0.0, -9.81) if args.physics else (0.0, 0.0, 0.0),
        ),
        show_viewer=args.viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    hand = scene.add_entity(gs.morphs.URDF(file=urdf, fixed=False))
    print(f"type of hand: {type(hand)}")

    object_entity = None
    if args.object_mesh:
        if object_simulated:
            # A reconstruction mesh is far too dense to decompose into collision shapes,
            # and one convex hull would seal the handle the fingers pass through, so it
            # is decimated and split into convex pieces instead.
            morph = gs.morphs.Mesh(
                file=os.path.abspath(args.object_mesh),
                fixed=False,
                collision=True,
                convexify=True,
                decompose_object_error_threshold=args.decompose_threshold,
                decimate=True,
                decimate_face_num=5000,
                pos=tuple(shift),
                quat=tuple(to_quat_wxyz(cam_to_world)),
            )
            object_entity = scene.add_entity(
                morph,
                material=gs.materials.Rigid(
                    rho=args.object_density, friction=args.object_friction
                ),
            )
        else:
            object_entity = scene.add_entity(
                gs.morphs.Mesh(
                    file=os.path.abspath(args.object_mesh),
                    fixed=object_poses is None,
                    collision=False,
                    pos=tuple(shift),
                    quat=tuple(to_quat_wxyz(cam_to_world)),
                )
            )

    camera = None
    if args.record:
        framed = positions[valid]
        if object_vertices is not None:
            framed = np.vstack([framed, object_vertices + shift])
        if object_poses is not None:
            # The object leaves the reference-frame footprint once it is lifted, so
            # frame on where it actually travels.
            _, translations_w, solved = object_poses
            framed = np.vstack([framed, translations_w[solved] + shift])
        centre = framed.mean(axis=0)
        distance = args.camera_distance
        if distance is None:
            distance = 2.0 * max(np.ptp(framed, axis=0).max(), 0.2)
        direction = np.array([float(v) for v in args.camera_dir.split(",")])
        direction /= np.linalg.norm(direction)
        camera = scene.add_camera(
            res=tuple(args.res),
            pos=tuple(centre + distance * direction),
            lookat=tuple(centre),
            fov=45,
        )

    scene.build()

    dofs_idx = [hand.get_joint(name).dof_idx_local for name in joint_names]
    # The URDF root is floating, so the first six local dofs are the base: 0-2 linear and
    # 3-5 angular, both about world axes (genesis/utils/urdf.py, "floating").
    base_dofs_idx = list(range(6))

    if args.physics:
        # Position control rather than set_dofs_position: a joint that is *set* holds its
        # angle whatever it is pressing on, so the fingers would sweep through the object
        # instead of gripping it. Without these calls the joints fall back to the URDF
        # defaults (kp 100, kv 10), which are stiffer than intended and push harder
        # through a contact rather than holding it.
        hand.set_dofs_kp(np.full(len(dofs_idx), args.finger_kp), dofs_idx)
        hand.set_dofs_kv(np.full(len(dofs_idx), args.finger_kv), dofs_idx)
        # Mass and friction only mean something for an object the solver is integrating;
        # an exempt one is a rendered pose and its inertia is never consulted.
        if object_entity is not None and object_simulated:
            print(f"[physics] object mass {object_entity.get_mass() * 1000:.0f} g, "
                  f"friction {args.object_friction}, {substeps} steps per frame")

    simulated_pos, simulated_quat, contact_frames = [], [], []
    # Under position control the fingers are asked for a pose, not set to it, so what
    # they actually reach is a separate question from what was reconstructed. Tracked
    # here so a hand that looks stiff can be told apart from one that was.
    finger_error = []
    placed = False
    frames = []
    for t in range(frame_start, frame_stop):
        # zero_velocity has to stay False here, because it zeroes *every* dof of the
        # entity, fingers included -- which restarts the finger controller from rest
        # every frame and leaves it unable to track, so the hand articulates far less
        # than it was reconstructed doing. Only the base needs correcting, and under
        # --physics it is corrected explicitly below.
        hand.set_pos(positions[t], zero_velocity=not args.physics)
        hand.set_quat(quats[t], zero_velocity=not args.physics)
        if args.physics:
            # The base is free and unactuated, so left alone it accumulates the gravity
            # it picks up between teleports without bound: the pose still renders
            # correctly, being re-set every frame, but in the solver the hand is a body
            # in free fall. Overwriting the six base dofs each frame kills that, and
            # giving them the trajectory's real velocity is also what lets the grasp
            # transmit friction -- a hand that reports no velocity carries nothing.
            hand.set_dofs_velocity(base_velocity[t], base_dofs_idx)
            hand.control_dofs_position(qpos[t], dofs_idx)
        else:
            hand.set_dofs_position(qpos[t], dofs_idx)

        if object_poses is not None:
            object_quats, translations_w, solved = object_poses
            if object_simulated:
                # Placed once, then left to the simulation: after this the object's pose
                # is an outcome, not an input.
                if not placed and t >= release_frame and solved[t]:
                    object_entity.set_pos(translations_w[t] + shift, zero_velocity=True)
                    object_entity.set_quat(object_quats[t], zero_velocity=True)
                    placed = True
                    print(f"[physics] object released to the simulation at frame {t}")
                elif not placed and solved[t]:
                    # Before the release the object is still an input, so hold it on its
                    # tracked pose rather than letting it fall from the spawn pose. This
                    # keeps the approach readable and makes every later divergence
                    # attributable to the simulation rather than to the wait.
                    object_entity.set_pos(translations_w[t] + shift, zero_velocity=True)
                    object_entity.set_quat(object_quats[t], zero_velocity=True)
            elif solved[t]:
                # Mesh vertices sit in the reference frame's camera coordinates, so the
                # tracked transform composes as (C R_t) v + (C T_t + shift). At the
                # reference frame R_t = I and T_t = 0, recovering the static placement.
                object_entity.set_pos(translations_w[t] + shift, zero_velocity=True)
                object_entity.set_quat(object_quats[t], zero_velocity=True)

        for _ in range(substeps):
            scene.step()
        if args.physics:
            reached = np.asarray(hand.get_dofs_position(dofs_idx).cpu())
            finger_error.append(np.abs(reached - qpos[t]))
        if object_simulated and object_entity is not None:
            contacts = object_entity.get_contacts(with_entity=hand)
            contact_frames.append(len(contacts.get("position", [])))
            simulated_pos.append(np.asarray(object_entity.get_pos().cpu()))
            simulated_quat.append(np.asarray(object_entity.get_quat().cpu()))
        elif args.physics and object_entity is not None and object_poses is not None:
            # Gravity is global, so an exempt object still falls during the substeps and
            # is rendered half a centimetre low, jittering by that much every frame. It
            # is placed again after the stepping, which is what gets drawn.
            object_quats, translations_w, solved = object_poses
            if solved[t]:
                object_entity.set_pos(translations_w[t] + shift, zero_velocity=True)
                object_entity.set_quat(object_quats[t], zero_velocity=True)
        if camera is not None:
            frames.append(camera.render()[0])

    if args.physics and finger_error:
        report = np.degrees(np.stack(finger_error))
        print(f"[physics] finger tracking error vs the retargeted pose: "
              f"median {np.median(report):.1f} deg, "
              f"90th pct {np.percentile(report, 90):.1f} deg, "
              f"worst joint {report.mean(axis=0).max():.1f} deg mean")

    if object_simulated and simulated_pos:
        simulated_pos = np.stack(simulated_pos)
        simulated_quat = np.stack(simulated_quat)
        if object_poses is not None:
            _, translations_w, solved = object_poses
            tracked = (translations_w + shift)[frame_start:frame_stop]
            # Only from the release onwards: before it the object is pinned to the very
            # poses being compared against, so including those frames reports a zero the
            # simulation did not earn.
            free = solved.copy()
            free[:release_frame] = False
            free = free[frame_start:frame_stop]
            gap = np.linalg.norm(simulated_pos - tracked, axis=1)[free]
            print(f"[physics] simulated vs tracked object position, from the release at "
                  f"frame {release_frame}: median {np.median(gap) * 100:.1f} cm, "
                  f"final {gap[-1] * 100:.1f} cm")
            # get_pos returns the mesh's own origin, which for a mesh posed in camera
            # coordinates is nowhere near the object; the AABB is what tells you whether
            # the object actually left the table.
            aabb = np.asarray(object_entity.get_AABB().cpu())
            print("[physics] object at the end: resting on the table"
                  if aabb[0, 2] < 0.01 else
                  f"[physics] object at the end: {aabb[0, 2] * 100:.1f} cm above the table")
        contact_frames = np.asarray(contact_frames)
        touching = contact_frames > 0
        print(f"[physics] hand-object contact on {touching.sum()}/{len(touching)} frames"
              + (f", first at {int(np.argmax(touching))}" if touching.any() else " (never touched)"))
        if args.sim_out:
            np.savez(args.sim_out, position=simulated_pos, quat=simulated_quat,
                     shift=shift, contacts=contact_frames,
                     frame_range=np.array([frame_start, frame_stop]),
                     release_frame=release_frame)
            print(f"Wrote {args.sim_out}")

    if camera is not None:
        import imageio.v2 as imageio

        imageio.mimsave(args.record, frames, fps=fps, macro_block_size=1)
        print(f"Wrote {args.record}: {len(frames)} frames @ {fps} fps")


if __name__ == "__main__":
    main()
