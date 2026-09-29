#!/usr/bin/env python3
"""
Solve Franka Panda arm joint angles that carry the Wuji Hand along the reconstructed
wrist trajectory, and replay the result in Genesis.

The retargeting gives finger joint angles plus a free-floating wrist pose. Mounting
the hand on an arm makes that wrist pose a constraint rather than a given: the arm
has 7 joints to place a 6-DoF frame, so IK is solved per frame, warm-started from the
previous solution to keep the arm in one continuous branch instead of flipping
between elbow configurations.

Run in the `genesis` env:
    python solve_franka_ik.py --traj wuji_traj.npz --urdf panda_wuji_left.urdf \\
        --ground-plane ground_plane.json --record arm_replay.mp4
"""

import argparse
import json
import os

import numpy as np
from scipy.spatial.transform import Rotation

DEFAULT_URDF = os.path.expanduser(
    "~/genesis-world/robots/panda_wuji/urdf/panda_wuji_left.urdf"
)


def camera_to_world(up_cam):
    """Rotation taking OpenCV camera coordinates to a z-up Genesis world."""
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
    parser = argparse.ArgumentParser(description="Franka IK for the reconstructed wrist trajectory")
    parser.add_argument("--traj", required=True, help="wuji_traj.npz from hawor_to_wuji_qpos.py")
    parser.add_argument("--urdf", default=DEFAULT_URDF, help="Combined Panda + Wuji Hand URDF")
    parser.add_argument("--palm-link", default="palm_link", help="Link the wrist pose targets")
    parser.add_argument("--object-mesh", default=None)
    parser.add_argument("--object-poses", default=None)
    parser.add_argument("--ground-plane", default=None)
    parser.add_argument("--up", default="0,-1,0")
    parser.add_argument(
        "--base-pos",
        default=None,
        help="Arm base position 'x,y,z'; default places it just outside the trajectory",
    )
    parser.add_argument("--reach", type=float, default=0.55, help="Default stand-off from the trajectory, metres")
    parser.add_argument("--out", default=None, help="Write the solved joint trajectory here")
    parser.add_argument("--record", default=None)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--res", type=int, nargs=2, default=(960, 720))
    return parser.parse_args()


def main():
    args = parse_args()
    traj = np.load(args.traj)

    qpos_hand = traj["qpos"]
    hand_joint_names = [str(n) for n in traj["joint_names"]]
    side = str(traj["side"])
    fps = float(traj["fps"])
    valid = traj["valid"].astype(bool)

    if args.ground_plane:
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

        object_vertices = trimesh.load(args.object_mesh, process=False).vertices @ cam_to_world.T

    anchor = positions[valid] if object_vertices is None else object_vertices
    shift = np.zeros(3)
    shift[:2] = -anchor[:, :2].mean(axis=0)
    shift[2] = -min(positions[valid][:, 2].min(), anchor[:, 2].min())
    positions += shift

    object_poses = None
    if args.object_poses:
        tracked = np.load(args.object_poses)
        solved = tracked["valid"].astype(bool)
        object_rotations = np.einsum("ij,tjk->tik", cam_to_world, tracked["rotation"])
        object_quats = np.tile(to_quat_wxyz(cam_to_world), (len(solved), 1))
        object_quats[solved] = to_quat_wxyz(object_rotations[solved])
        object_poses = (object_quats, tracked["translation"] @ cam_to_world.T, solved)

    if args.base_pos:
        base_pos = np.array([float(v) for v in args.base_pos.split(",")])
    else:
        # Stand the arm off to one side of the reachable trajectory rather than under
        # it, so the Panda's own workspace, not the scene origin, decides placement.
        centre = positions[valid].mean(axis=0)
        base_pos = np.array([centre[0], centre[1] - args.reach, 0.0])

    import genesis as gs

    gs.init(backend=gs.cpu if args.cpu else gs.gpu)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1.0 / fps, gravity=(0.0, 0.0, 0.0)),
        show_viewer=args.viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(
        gs.morphs.URDF(
            file=os.path.abspath(args.urdf),
            fixed=True,
            pos=tuple(base_pos),
            # The palm is bolted to the flange through fixed joints, which Genesis
            # merges away by default; IK needs it to survive as a nameable frame.
            links_to_keep=[args.palm_link],
        )
    )

    object_entity = None
    if args.object_mesh:
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
        framed = np.vstack([positions[valid], base_pos[None]])
        if object_vertices is not None:
            framed = np.vstack([framed, object_vertices + shift])
        centre = framed.mean(axis=0)
        distance = 1.6 * max(np.ptp(framed, axis=0).max(), 0.4)
        camera = scene.add_camera(
            res=tuple(args.res),
            pos=tuple(centre + distance * np.array([0.8, -0.8, 0.5])),
            lookat=tuple(centre),
            fov=45,
        )

    scene.build()

    palm = robot.get_link(args.palm_link)
    arm_dofs = [robot.get_joint(f"panda_joint{i}").dof_idx_local for i in range(1, 8)]
    # The combined URDF drops the side prefix the standalone hand URDF carries.
    finger_dofs = [
        robot.get_joint(name.replace(f"{side}_", "", 1)).dof_idx_local
        for name in hand_joint_names
    ]

    quats = to_quat_wxyz(rotations)
    arm_qpos = np.zeros((len(qpos_hand), 7))
    solved_ik = np.zeros(len(qpos_hand), dtype=bool)
    errors = np.full(len(qpos_hand), np.nan)

    previous = None
    frames = []
    for t in range(len(qpos_hand)):
        if valid[t]:
            qpos, error = robot.inverse_kinematics(
                link=palm,
                pos=positions[t],
                quat=quats[t],
                init_qpos=previous,
                return_error=True,
            )
            arm_qpos[t] = qpos.cpu().numpy()[arm_dofs] if hasattr(qpos, "cpu") else np.asarray(qpos)[arm_dofs]
            residual = error.cpu().numpy() if hasattr(error, "cpu") else np.asarray(error)
            errors[t] = np.linalg.norm(residual[:3])
            solved_ik[t] = True
            previous = qpos
        elif previous is not None:
            arm_qpos[t] = arm_qpos[t - 1]

        robot.set_dofs_position(arm_qpos[t], arm_dofs)
        robot.set_dofs_position(qpos_hand[t], finger_dofs)

        if object_poses is not None and object_entity is not None:
            object_quats, translations_w, object_solved = object_poses
            if object_solved[t]:
        # zero_velocity=True is essential: Genesis defaults it to False, so the pose
        # we set is treated as a teleport that leaves a huge implied velocity behind,
        # which the following scene.step() integrates -- rendering the object away
        # from the pose we asked for. That, not the trajectory, was the visible jitter.
                object_entity.set_pos(translations_w[t] + shift, zero_velocity=True)
                object_entity.set_quat(object_quats[t], zero_velocity=True)

        scene.step()
        if camera is not None:
            frames.append(camera.render()[0])

    if args.out:
        np.savez(
            args.out,
            arm_qpos=arm_qpos,
            hand_qpos=qpos_hand,
            solved=solved_ik,
            position_error=errors,
            base_pos=base_pos,
            arm_joint_names=np.array([f"panda_joint{i}" for i in range(1, 8)]),
            fps=fps,
        )
        print(f"Wrote {args.out}")

    reached = errors[np.isfinite(errors)]
    print(f"IK solved {solved_ik.sum()}/{len(qpos_hand)} frames")
    if len(reached):
        print(
            f"  position error: median {1000 * np.median(reached):.1f} mm, "
            f"p90 {1000 * np.percentile(reached, 90):.1f} mm, max {1000 * reached.max():.1f} mm"
        )

    if camera is not None:
        import imageio.v2 as imageio

        imageio.mimsave(args.record, frames, fps=fps, macro_block_size=1)
        print(f"Wrote {args.record}: {len(frames)} frames @ {fps} fps")


if __name__ == "__main__":
    main()
