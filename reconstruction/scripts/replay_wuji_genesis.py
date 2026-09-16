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
    "/home/cl-ment-prigent/wuji-ego-mint/eval/simulate/wuji-retargeting",
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
    parser.add_argument("--record", default=None, help="Write an mp4 to this path")
    parser.add_argument("--viewer", action="store_true", help="Open the interactive viewer")
    parser.add_argument("--cpu", action="store_true", help="Run Genesis on CPU")
    parser.add_argument("--res", type=int, nargs=2, default=(960, 720), help="Recording resolution")
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
    if args.object_poses:
        tracked = np.load(args.object_poses)
        solved = tracked["valid"].astype(bool)
        object_rotations = np.einsum("ij,tjk->tik", cam_to_world, tracked["rotation"])
        object_quats = np.tile(to_quat_wxyz(cam_to_world), (len(solved), 1))
        object_quats[solved] = to_quat_wxyz(object_rotations[solved])
        object_poses = (object_quats, tracked["translation"] @ cam_to_world.T, solved)

    # Recentre on the origin and lift clear of the ground plane so the camera framing
    # does not depend on where the scene happened to sit in front of the camera. Hand
    # and object share one shift, which is what keeps them registered to each other.
    anchor = positions[valid] if object_vertices is None else object_vertices
    shift = np.zeros(3)
    shift[:2] = -anchor[:, :2].mean(axis=0)
    shift[2] = -min(positions[valid][:, 2].min(), anchor[:, 2].min())
    positions += shift

    import genesis as gs

    gs.init(backend=gs.cpu if args.cpu else gs.gpu)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1.0 / fps, gravity=(0.0, 0.0, 0.0)),
        show_viewer=args.viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    hand = scene.add_entity(gs.morphs.URDF(file=urdf, fixed=False))

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
        framed = positions[valid]
        if object_vertices is not None:
            framed = np.vstack([framed, object_vertices + shift])
        if object_poses is not None:
            # The object leaves the reference-frame footprint once it is lifted, so
            # frame on where it actually travels.
            _, translations_w, solved = object_poses
            framed = np.vstack([framed, translations_w[solved] + shift])
        centre = framed.mean(axis=0)
        distance = 2.0 * max(np.ptp(framed, axis=0).max(), 0.2)
        camera = scene.add_camera(
            res=tuple(args.res),
            pos=tuple(centre + distance * np.array([0.7, -0.7, 0.45])),
            lookat=tuple(centre),
            fov=45,
        )

    scene.build()

    dofs_idx = [hand.get_joint(name).dof_idx_local for name in joint_names]

    frames = []
    for t in range(len(qpos)):
        hand.set_pos(positions[t])
        hand.set_quat(quats[t])
        hand.set_dofs_position(qpos[t], dofs_idx)

        if object_poses is not None:
            object_quats, translations_w, solved = object_poses
            if solved[t]:
                # Mesh vertices sit in the reference frame's camera coordinates, so the
                # tracked transform composes as (C R_t) v + (C T_t + shift). At the
                # reference frame R_t = I and T_t = 0, recovering the static placement.
                object_entity.set_pos(translations_w[t] + shift)
                object_entity.set_quat(object_quats[t])

        scene.step()
        if camera is not None:
            frames.append(camera.render()[0])

    if camera is not None:
        import imageio.v2 as imageio

        imageio.mimsave(args.record, frames, fps=fps, macro_block_size=1)
        print(f"Wrote {args.record}: {len(frames)} frames @ {fps} fps")


if __name__ == "__main__":
    main()
