#!/usr/bin/env python3
"""
Look at the grasps Lightning Grasp proposed at each keyframe.

keyframe_grasps.npz stores a ranked set of candidate hand configurations per keyframe,
and refine_keyframes_v2s2r.py then picks one of them by simulation. Both are hard to
judge as numbers: "84 mm and 59 degrees from the demonstrated wrist" says a candidate
is different without saying whether it is a sensible way to hold a mug.

So each candidate is posed around the object at that keyframe's tracked pose and
rendered, with the human's own reconstructed grasp rendered first for comparison. The
trick is that a trajectory whose successive frames are *different candidates* replays
as a flip-book of grasps seen from one fixed camera, which the existing Genesis replay
already knows how to render -- nothing here has to build a scene.

Run in the `genesis` env:
    python show_keyframe_grasps.py --candidates keyframe_grasps.npz \\
        --traj wuji_traj.npz --object-poses cup_traj_points.npz \\
        --mesh posed_mesh.obj --ground-plane ground_plane.json \\
        --out grasps.png
"""

import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description="Render the candidate grasps at each keyframe")
    parser.add_argument("--candidates", required=True, help="keyframe_grasps.npz")
    parser.add_argument("--traj", required=True, help="The trajectory the demonstration is read from")
    parser.add_argument("--object-poses", required=True)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--ground-plane", default=None)
    parser.add_argument("--out", required=True, help="Output image (one row per keyframe)")
    parser.add_argument("--keyframe", type=int, default=None,
                        help="Only this frame; by default every keyframe in the file")
    parser.add_argument("--show", type=int, default=5, help="Candidates per keyframe, best first")
    parser.add_argument("--refined", default=None,
                        help="A refined trajectory; its chosen candidate is marked")
    parser.add_argument("--part", default=None,
                        help="part_<name>.npz from label_mesh_part.py. The region is "
                             "coloured in the render, so whether the fingers are actually "
                             "on it can be seen rather than inferred")
    parser.add_argument("--part-colour", default="60,220,90",
                        help="RGB for the region, 0-255")
    parser.add_argument("--wuji-dir", default=os.environ.get(
        "WUJI_RETARGETING_DIR",
        os.path.expanduser("~/wuji-ego-mint/eval/simulate/wuji-retargeting")))
    parser.add_argument("--urdf", default=None)
    parser.add_argument("--res", type=int, nargs=2, default=(560, 480))
    parser.add_argument("--orbit", type=int, default=0,
                        help="Render one candidate from this many viewpoints around it "
                             "instead of a row of different candidates. A finger through a "
                             "2 cm handle and a finger wrapped round it look the same from "
                             "one angle, which is what this is for")
    parser.add_argument("--candidate", type=int, default=0,
                        help="With --orbit, which candidate to circle; -1 for the demonstration")
    parser.add_argument("--camera-distance", type=float, default=None,
                        help="Passed to the replay; small values look closely at the grasp")
    parser.add_argument("--elevation", type=float, default=0.45,
                        help="Camera height component while orbiting")
    return parser.parse_args()


def coloured_mesh(mesh_path, part_path, colour, out_dir):
    """A copy of the mesh with the part's faces painted, for the render only.

    Genesis loads the object from a file, so the colour has to travel in the geometry.
    Face colours rather than vertex colours: a face is either in the region or not,
    and interpolating across the boundary would blur exactly the edge worth seeing.
    """
    import trimesh

    mesh = trimesh.load(mesh_path, process=False)
    # label_mesh_part.py writes an .npz with the vertex and face masks side by side;
    # a bare .npy is just the vertex mask. Both are handed around, so take either.
    stored = np.load(part_path, allow_pickle=True)
    region = stored["vertices"] if hasattr(stored, "files") else stored
    if len(region) != len(mesh.vertices):
        raise SystemExit(f"{part_path} labels {len(region)} vertices, mesh has {len(mesh.vertices)}")

    colours = np.tile(np.array([185, 185, 190, 255], np.uint8), (len(mesh.vertices), 1))
    colours[region] = np.array(list(colour) + [255], np.uint8)
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=colours)

    # OBJ, not PLY: Genesis rejects PLY outright ("File type not supported"). Vertex
    # colours ride along in the "v x y z r g b" extension that trimesh writes.
    path = os.path.join(out_dir, "part_coloured.obj")
    mesh.export(path, include_color=True)
    print(f"[grasps] {int(region.sum())}/{len(region)} vertices painted as the region")
    return path


def render_poses(args, source, rotations, positions, qpos, object_index, camera_dir=None):
    """Render one image per pose, all from the same camera, by replaying them in order."""
    n_poses = len(rotations)
    quaternions = Rotation.from_matrix(rotations).as_quat()

    fields = {key: source[key] for key in ("joint_names", "side")}
    fields["qpos"] = np.asarray(qpos)
    fields["wrist_pos"] = np.asarray(positions)
    fields["wrist_quat"] = np.c_[quaternions[:, 3], quaternions[:, :3]]
    fields["valid"] = np.ones(n_poses, dtype=bool)
    fields["fps"] = 1.0

    poses = np.load(args.object_poses)
    held = dict(
        rotation=np.tile(poses["rotation"][object_index], (n_poses, 1, 1)),
        translation=np.tile(poses["translation"][object_index], (n_poses, 1)),
        valid=np.ones(n_poses, dtype=bool),
        ref_frame=0,
    )

    paths = []
    for _ in range(3):
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as handle:
            paths.append(handle.name)
    traj_path, object_path, _ = paths
    video_path = traj_path.replace(".npz", ".mp4")
    np.savez(traj_path, **fields)
    np.savez(object_path, **held)

    command = [
        sys.executable, os.path.join(HERE, "replay_wuji_genesis.py"),
        "--traj", traj_path, "--object-mesh", args.render_mesh, "--object-poses", object_path,
        "--wuji-dir", args.wuji_dir, "--record", video_path,
        "--res", str(args.res[0]), str(args.res[1]),
    ]
    if camera_dir is not None:
        # "=" form: an orbit direction can start with a minus sign, which argparse
        # would otherwise read as the next option.
        command += [f"--camera-dir={camera_dir}"]
    if args.camera_distance is not None:
        command += ["--camera-distance", str(args.camera_distance)]
    if args.ground_plane:
        command += ["--ground-plane", args.ground_plane]
    if args.urdf:
        command += ["--urdf", args.urdf]
    result = subprocess.run(command, capture_output=True, text=True)

    images = []
    if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
        import cv2

        capture = cv2.VideoCapture(video_path)
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            images.append(frame)
        capture.release()
    else:
        tail = "\n".join(result.stderr.strip().splitlines()[-3:])
        print(f"[grasps] replay failed: {tail}")

    for path in paths + [video_path]:
        if os.path.exists(path):
            os.unlink(path)
    return images


def label(image, text, colour=(255, 255, 255)):
    import cv2

    cv2.rectangle(image, (0, 0), (image.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(image, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.52, colour, 1, cv2.LINE_AA)
    return image


def main():
    args = parse_args()
    import cv2

    args.render_mesh = args.mesh
    if args.part:
        colour = [int(v) for v in args.part_colour.split(",")]
        args.render_mesh = coloured_mesh(args.mesh, args.part, colour,
                                         os.path.dirname(os.path.abspath(args.out)))

    data = np.load(args.candidates, allow_pickle=True)
    source = np.load(args.traj, allow_pickle=True)
    frames = [int(f) for f in data["frames"]]
    labels = [str(l) for l in data["labels"]]

    chosen_index = None
    if args.refined:
        refined = np.load(args.refined, allow_pickle=True)
        if "v2s2r_candidate" in refined.files:
            chosen_index = int(refined["v2s2r_candidate"])
            print(f"[grasps] refinement chose candidate {chosen_index}")

    wanted = [i for i, f in enumerate(frames) if args.keyframe in (None, f)]
    if not wanted:
        raise SystemExit(f"keyframe {args.keyframe} not in {frames}")

    if args.orbit:
        slot = wanted[0]
        frame, name = frames[slot], labels[slot]
        if args.candidate < 0:
            rotation = Rotation.from_quat(source["wrist_quat"][frame][[1, 2, 3, 0]]).as_matrix()
            position, angles = source["wrist_pos"][frame], source["qpos"][frame]
            title = "demonstrated"
        else:
            rotation = data["candidate_rotation"][slot][args.candidate]
            position = data["candidate_pos"][slot][args.candidate]
            angles = data["candidate_qpos"][slot][args.candidate]
            title = f"candidate {args.candidate}"
        print(f"[grasps] orbiting {title} at {name} frame {frame}, {args.orbit} viewpoints")

        tiles = []
        for step in range(args.orbit):
            azimuth = 2 * np.pi * step / args.orbit
            direction = f"{np.cos(azimuth):.4f},{np.sin(azimuth):.4f},{args.elevation}"
            images = render_poses(args, source, rotation[None], position[None], angles[None],
                                  frame, camera_dir=direction)
            if images:
                tiles.append(label(images[0], f"{title}  {np.degrees(azimuth):.0f} deg"))
        if not tiles:
            raise SystemExit("nothing rendered")
        half = (len(tiles) + 1) // 2
        top, bottom = tiles[:half], tiles[half:]
        while len(bottom) < len(top):
            bottom.append(np.zeros_like(top[0]))
        cv2.imwrite(args.out, np.vstack([np.hstack(top), np.hstack(bottom)]))
        print(f"Wrote {args.out}")
        return

    rows = []
    for slot in wanted:
        frame, name = frames[slot], labels[slot]
        count = min(args.show, len(data["candidate_rotation"][slot]))

        # The human's own grasp first, then the candidates in rank order.
        rotations = [Rotation.from_quat(
            source["wrist_quat"][frame][[1, 2, 3, 0]]).as_matrix()]
        positions = [source["wrist_pos"][frame]]
        qpos = [source["qpos"][frame]]
        for rank in range(count):
            rotations.append(data["candidate_rotation"][slot][rank])
            positions.append(data["candidate_pos"][slot][rank])
            qpos.append(data["candidate_qpos"][slot][rank])

        print(f"[grasps] {name} frame {frame}: rendering the demonstration + {count} candidates")
        images = render_poses(args, source, np.array(rotations), np.array(positions),
                              np.array(qpos), frame)
        if len(images) < len(rotations):
            print(f"[grasps]   only {len(images)} of {len(rotations)} frames came back")
        if not images:
            continue

        tiles = [label(images[0], f"{name} {frame}: demonstrated", (120, 220, 255))]
        for rank, image in enumerate(images[1:]):
            mark = "  <- chosen" if rank == chosen_index else ""
            gap = np.linalg.norm(data["candidate_pos"][slot][rank] - source["wrist_pos"][frame])
            tiles.append(label(image, f"candidate {rank}  {100 * gap:.0f} cm{mark}",
                               (120, 255, 160) if mark else (255, 255, 255)))
        rows.append(np.hstack(tiles))

    if not rows:
        raise SystemExit("nothing rendered")
    width = max(row.shape[1] for row in rows)
    rows = [np.pad(row, ((0, 0), (0, width - row.shape[1]), (0, 0))) for row in rows]
    cv2.imwrite(args.out, np.vstack(rows))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
