#!/usr/bin/env python3
"""
Replay both reconstructed hands (MANO meshes) and the object in Genesis.

The hands are the MANO fits from mano_from_mocap.py, drawn two ways:

  * skinned (default): the true skinned MANO mesh of each frame, Loop-subdivided once
    and drawn as a visual-only mesh. This is the reconstruction as it is.
  * parts: each hand as its 16 rigid skinning parts, every part a mesh entity carried
    by its joint's transform -- the form a rigid-body simulator can hold and collide.
    Rigid parts cannot blend at a joint, so the mesh cracks open there (median 6 mm,
    up to 5 cm at the thumb base) and the dark inside shows through: use it for
    physics, not to look at.

The object is the RecGen mesh carried by its tracked trajectory. Everything is kinematic -- set every frame,
gravity off, collisions off -- so the scene shows the reconstruction exactly, not a
simulation's reading of it.

The world is z-up with the fitted table at z = 0 (the Genesis ground plane), from
the reference camera through camera_to_world(). Three views are written side by side:

  * the input frame;
  * the scene rendered from the real, moving camera (camera_traj.npz), which should
    overlay the input frame -- the direct check of the whole reconstruction;
  * a fixed orbit view of the scene.

Run in the `genesis` env:
    python replay_hands_genesis.py --mano mano_hands.npz --parts-dir mano_parts \\
        --object-mesh posed_mesh.obj --object-poses obj_traj.npz --camera-traj camera_traj.npz \\
        --ground-plane ground_plane.json --intrinsics intrinsics.yaml --frames-dir all_frames \\
        --record replay_hands.mp4
"""

import argparse
import json
import os

import cv2
import numpy as np
import yaml

from replay_wuji_genesis import camera_to_world, to_quat_wxyz

COLOURS = {"left": (0.93, 0.72, 0.60), "right": (0.80, 0.62, 0.52)}


def parse_args():
    parser = argparse.ArgumentParser(description="Replay both MANO hands and the object in Genesis")
    parser.add_argument("--mano", required=True, help="mano_from_mocap.py output")
    parser.add_argument("--parts-dir", required=True, help="Its per-part meshes")
    parser.add_argument("--object-mesh", default=None, help="posed_mesh.obj (reference camera)")
    parser.add_argument("--object-poses", default=None, help="Object trajectory, reference camera")
    parser.add_argument("--camera-traj", default=None,
                        help="T_ct<-cref; renders from the real camera next to the input")
    parser.add_argument("--ground-plane", required=True)
    parser.add_argument("--intrinsics", required=True)
    parser.add_argument("--frames-dir", default=None, help="Input frames, shown alongside")
    parser.add_argument("--record", required=True, help="Output .mp4")
    parser.add_argument("--sides", nargs="+", default=["left", "right"])
    parser.add_argument("--hand-render", choices=("skinned", "parts"), default="skinned")
    parser.add_argument("--subdivide", type=int, default=1,
                        help="Loop subdivisions of the skinned mesh; MANO's 778 vertices facet visibly")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--camera-dir", default="0.6,-0.8,0.7",
                        help="Orbit view: direction from the scene centre, z-up world")
    parser.add_argument("--camera-distance", type=float, default=0.9)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.ground_plane) as handle:
        plane = json.load(handle)
    normal = np.asarray(plane["normal_camera"], dtype=float)
    offset = float(plane["offset"]) / np.linalg.norm(normal)
    normal /= np.linalg.norm(normal)
    with open(args.intrinsics) as handle:
        calib = yaml.safe_load(handle)
    fv = float(calib["fv"])
    names = sorted(f for f in os.listdir(args.frames_dir) if f.endswith(".png")) if args.frames_dir else []
    if names:
        height, width = cv2.imread(os.path.join(args.frames_dir, names[0])).shape[:2]
    else:
        width, height = int(round(2 * float(calib["pu"]))), int(round(2 * float(calib["pv"])))

    # p_world = C p_cam + shift. C's z axis is the table normal, so a point's world z is
    # its height above the table plus `offset`; shifting by it puts the table at z = 0.
    C = camera_to_world(normal)
    mano = np.load(args.mano)
    n_frames = len(mano[f"{args.sides[0]}_part_transforms"])

    object_poses = None
    anchor = np.concatenate([mano[f"{side}_vertices"][0] for side in args.sides])
    if args.object_poses and args.object_mesh:
        import trimesh

        vertices = np.asarray(trimesh.load(args.object_mesh, process=False).vertices)
        track = np.load(args.object_poses)
        object_poses = (track["rotation"], track["translation"], track["valid"].astype(bool))
        anchor = vertices
    shift = np.r_[-(anchor @ C.T)[:, :2].mean(axis=0), offset]

    def to_world(rotation, translation):
        return to_quat_wxyz(C @ rotation), C @ translation + shift

    import genesis as gs

    gs.init(backend=gs.cpu if args.cpu else gs.gpu)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1.0 / args.fps, gravity=(0.0, 0.0, 0.0)),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())

    parts = {}
    for side in args.sides if args.hand_render == "parts" else []:
        parts[side] = [
            scene.add_entity(
                gs.morphs.Mesh(file=os.path.abspath(os.path.join(args.parts_dir, f"{side}_part_{k:02d}.obj")),
                               fixed=False, collision=False),
                surface=gs.surfaces.Default(color=COLOURS[side]),
            )
            for k in range(16)
        ]
    object_entity = None
    if object_poses is not None:
        object_entity = scene.add_entity(
            gs.morphs.Mesh(file=os.path.abspath(args.object_mesh), fixed=False, collision=False))

    fov = float(np.degrees(2 * np.arctan(height / (2 * fv))))
    real_camera = None
    if args.camera_traj:
        cam = np.load(args.camera_traj)
        real_camera = scene.add_camera(res=(width, height), pos=(0, 0, 1), lookat=(0, 0, 0), fov=fov,
                                       debug=True)
    centre = np.r_[(anchor @ C.T)[:, :2].mean(axis=0) + shift[:2], 0.1]
    direction = np.array([float(v) for v in args.camera_dir.split(",")])
    direction /= np.linalg.norm(direction)
    orbit = scene.add_camera(res=(width, height), pos=tuple(centre + args.camera_distance * direction),
                             lookat=tuple(centre), fov=45, debug=True)
    scene.build()

    import trimesh

    skin = {side: (np.array(COLOURS[side] + (1.0,)) * 255).astype(np.uint8) for side in args.sides}
    drawn = []
    writer = None
    for t in range(n_frames):
        for side in parts:
            transforms = mano[f"{side}_part_transforms"][t]
            for k, entity in enumerate(parts[side]):
                quat, pos = to_world(transforms[k, :3, :3], transforms[k, :3, 3])
                entity.set_pos(pos, zero_velocity=True)
                entity.set_quat(quat, zero_velocity=True)
        if object_poses is not None and object_poses[2][t]:
            quat, pos = to_world(object_poses[0][t], object_poses[1][t])
            object_entity.set_pos(pos, zero_velocity=True)
            object_entity.set_quat(quat, zero_velocity=True)
        scene.step()
        # The skinned hands are debug meshes, which a camera only draws when built with
        # debug=True (genesis/vis/rasterizer.py skips markers otherwise).
        if args.hand_render == "skinned":
            for node in drawn:
                scene.clear_debug_object(node)
            drawn = []
            for side in args.sides:
                if not mano[f"{side}_valid"][t]:
                    continue
                mesh = trimesh.Trimesh(mano[f"{side}_vertices"][t] @ C.T + shift,
                                       mano[f"{side}_faces"], process=False)
                for _ in range(args.subdivide):
                    mesh = mesh.subdivide_loop()
                mesh.visual.vertex_colors = np.tile(skin[side], (len(mesh.vertices), 1))
                drawn.append(scene.draw_debug_mesh(mesh))

        views = []
        if names:
            views.append(cv2.imread(os.path.join(args.frames_dir, names[t])))
        if real_camera is not None:
            # The camera of frame t in the reference frame is T_ct<-cref inverted: centre
            # -R^T t, looking down R^T z, with R^T (-y) up (OpenCV y points down).
            R, tr = cam["rotation"][t], cam["translation"][t]
            centre_t = -R.T @ tr
            real_camera.set_pose(
                pos=C @ centre_t + shift,
                lookat=C @ (centre_t + R.T @ np.array([0.0, 0.0, 1.0])) + shift,
                up=C @ (R.T @ np.array([0.0, -1.0, 0.0])),
            )
            views.append(real_camera.render()[0][..., ::-1])
        views.append(orbit.render()[0][..., ::-1])
        views = [cv2.resize(np.ascontiguousarray(v), (width, height)) for v in views]
        row = np.hstack(views)
        if writer is None:
            writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                                     (row.shape[1], row.shape[0]))
        writer.write(row)
    writer.release()
    print(f"Wrote {args.record}: {n_frames} frames, views: "
          + " | ".join((["input"] if names else []) + (["real camera"] if real_camera else []) + ["orbit"]))


if __name__ == "__main__":
    main()
