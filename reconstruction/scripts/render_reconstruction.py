#!/usr/bin/env python3
"""
The reconstruction itself, as a video: HaWoR's hand mesh and the tracked object mesh
drawn together through the real camera.

Every other view in this pipeline shows something derived from the reconstruction --
the retargeted Wuji hand, the Franka arm carrying it, a mask outline. This shows the
two things the reconstruction actually produces, in the frame they were reconstructed
in: the per-frame MANO hand mesh from HaWoR, and the object mesh carried by the tracked
trajectory. If the hand and the object disagree about where the grasp is, it is visible
here rather than inferred from a contact statistic.

Both meshes already live in camera coordinates and in metres, so they are projected
through the recording's own intrinsics with no fitting of any kind.

Rendering is a depth-sorted triangle fill rather than a z-buffer: with two compact,
roughly convex bodies, painter's order is correct almost everywhere and costs one
cv2.fillConvexPoly per triangle instead of a per-pixel loop in Python. The object mesh
is decimated first -- a reconstruction mesh runs to hundreds of thousands of faces and
none of that detail survives at video resolution.

Run in the `genesis` env:
    python render_reconstruction.py --hand-meshes all_hand_meshes.npz --side right \\
        --object-mesh posed_mesh.obj --object-poses object_traj.npz \\
        --frames-dir all_frames --intrinsics intrinsics.yaml --out reconstruction.mp4
"""

import argparse
import os

import numpy as np
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Render the reconstructed hand and object")
    parser.add_argument("--hand-meshes", required=True, help="HaWoR all_hand_meshes.npz")
    parser.add_argument("--side", default="right", choices=("left", "right"))
    parser.add_argument("--object-mesh", required=True, help="posed_mesh.obj, camera frame")
    parser.add_argument("--object-poses", required=True, help="Object trajectory .npz")
    parser.add_argument("--intrinsics", required=True, help="YAML with fu/fv/pu/pv")
    parser.add_argument("--out", required=True)
    parser.add_argument("--frames-dir", default=None,
                        help="Source frames, drawn behind the meshes for context")
    parser.add_argument("--background", choices=("frame", "dark"), default="frame",
                        help="'frame' dims the source image behind the render; 'dark' "
                             "shows the reconstruction on its own")
    parser.add_argument("--backdrop", type=float, default=0.45,
                        help="How much of the source frame shows through, 0-1")
    parser.add_argument("--scale", type=float, default=0.5, help="Output scale factor")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--faces", type=int, default=4000,
                        help="Face budget the object mesh is decimated to")
    parser.add_argument("--hand-colour", default="235,170,140", help="RGB, 0-255")
    parser.add_argument("--object-colour", default="120,180,245", help="RGB, 0-255")
    parser.add_argument("--ambient", type=float, default=0.35,
                        help="Floor on the Lambert term, so faces turned away stay readable")
    return parser.parse_args()


def decimate(mesh, budget):
    """Reduce the mesh to roughly `budget` faces, whatever trimesh offers to do it with."""
    if len(mesh.faces) <= budget:
        return mesh
    for name, kwargs in (
        ("simplify_quadric_decimation", {"face_count": budget}),
        ("simplify_quadric_decimation", {"faces": budget}),
    ):
        method = getattr(mesh, name, None)
        if method is None:
            continue
        try:
            return method(**kwargs)
        except TypeError:
            continue
        except Exception:
            break
    # No decimator available: keep a random face subset. The silhouette suffers, but a
    # readable render beats refusing to draw one.
    keep = np.random.default_rng(0).choice(len(mesh.faces), budget, replace=False)
    import trimesh
    return trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[keep], process=False)


def shade(vertices, faces, colour, ambient):
    """Per-face depth, projected-area sign and a Lambert term against a headlight."""
    tri = vertices[faces]
    centre = tri.mean(axis=1)
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    length = np.linalg.norm(normal, axis=1, keepdims=True)
    normal = normal / np.maximum(length, 1e-12)
    # The camera sits at the origin looking down +z, so the view direction to a face is
    # just its centre, normalised.
    view = centre / np.maximum(np.linalg.norm(centre, axis=1, keepdims=True), 1e-12)
    lambert = np.abs((normal * view).sum(axis=1))
    intensity = ambient + (1.0 - ambient) * lambert
    return centre[:, 2], (np.asarray(colour, float)[None, :] * intensity[:, None])


def main():
    args = parse_args()
    import cv2
    import trimesh

    calibration = yaml.safe_load(open(args.intrinsics))
    fu, fv = calibration["fu"], calibration["fv"]
    pu, pv = calibration["pu"], calibration["pv"]

    hand = np.load(args.hand_meshes, allow_pickle=True)
    hand_vertices = hand[f"{args.side}_vertices"]
    hand_faces = hand[f"{args.side}_faces"].astype(np.int32)
    hand_valid = hand[f"{args.side}_valid"].astype(bool)

    poses = np.load(args.object_poses)
    rotation, translation = poses["rotation"], poses["translation"]
    object_valid = poses["valid"].astype(bool)

    mesh = decimate(trimesh.load(args.object_mesh, process=False), args.faces)
    object_vertices = np.asarray(mesh.vertices, dtype=np.float64)
    object_faces = np.asarray(mesh.faces, dtype=np.int32)

    n_frames = min(len(hand_vertices), len(rotation))
    names = []
    if args.frames_dir:
        names = sorted(f for f in os.listdir(args.frames_dir) if f.endswith(".png"))

    # Resolution follows the source frames when there are any, else the principal point.
    if names:
        first = cv2.imread(os.path.join(args.frames_dir, names[0]))
        height, width = first.shape[:2]
    else:
        width, height = int(round(2 * pu)), int(round(2 * pv))
    out_w, out_h = int(round(width * args.scale)), int(round(height * args.scale))

    hand_colour = [float(v) for v in args.hand_colour.split(",")][::-1]      # to BGR
    object_colour = [float(v) for v in args.object_colour.split(",")][::-1]

    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (out_w, out_h))
    if not writer.isOpened():
        raise SystemExit(f"could not open {args.out} for writing")

    drawn_hand = drawn_object = 0
    for index in range(n_frames):
        # The canvas is built at output size, because the projection below already
        # carries args.scale -- drawing scaled triangles onto a full-size canvas would
        # put the meshes in the wrong corner.
        if args.background == "frame" and index < len(names):
            frame = cv2.imread(os.path.join(args.frames_dir, names[index]))
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            canvas = (frame.astype(np.float32) * args.backdrop).astype(np.uint8)
        else:
            canvas = np.full((out_h, out_w, 3), 18, np.uint8)

        depths, colours, polygons = [], [], []

        def collect(vertices, faces, colour):
            """Project one mesh and queue its front-facing triangles."""
            z = vertices[:, 2]
            if not np.all(z > 1e-6):
                # A vertex behind the camera makes the projection meaningless; with the
                # whole object well in front this never fires, so drop the frame's mesh
                # rather than draw a smeared one.
                if np.mean(z > 1e-6) < 0.999:
                    return 0
            u = fu * vertices[:, 0] / np.maximum(z, 1e-6) + pu
            v = fv * vertices[:, 1] / np.maximum(z, 1e-6) + pv
            pixels = np.stack([u, v], axis=1) * args.scale
            face_depth, face_colour = shade(vertices, faces, colour, args.ambient)
            tri = pixels[faces]
            # Drop triangles entirely outside the frame before sorting them.
            inside = ~((tri[:, :, 0] < 0).all(1) | (tri[:, :, 0] > out_w).all(1) |
                       (tri[:, :, 1] < 0).all(1) | (tri[:, :, 1] > out_h).all(1))
            depths.append(face_depth[inside])
            colours.append(face_colour[inside])
            polygons.append(np.round(tri[inside]).astype(np.int32))
            return int(inside.sum())

        if hand_valid[index]:
            drawn_hand += collect(hand_vertices[index].astype(np.float64),
                                  hand_faces, hand_colour) > 0
        if object_valid[index]:
            posed = object_vertices @ rotation[index].T + translation[index]
            drawn_object += collect(posed, object_faces, object_colour) > 0

        if depths:
            depth = np.concatenate(depths)
            colour = np.concatenate(colours)
            polygon = np.concatenate(polygons)
            # Painter's order: furthest first, so nearer triangles overwrite them.
            for face in np.argsort(-depth):
                cv2.fillConvexPoly(canvas, polygon[face],
                                   tuple(float(c) for c in colour[face]),
                                   lineType=cv2.LINE_AA)

        label = f"frame {index}"
        if not hand_valid[index]:
            label += "   hand: not reconstructed"
        if not object_valid[index]:
            label += "   object: no pose"
        cv2.putText(canvas, label, (12, out_h - 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(canvas)

    writer.release()
    print(f"Wrote {args.out}: {n_frames} frames @ {args.fps:g} fps")
    print(f"  hand drawn on {drawn_hand}/{n_frames} frames, "
          f"object on {drawn_object}/{n_frames}")
    print(f"  object mesh decimated to {len(object_faces)} faces")


if __name__ == "__main__":
    main()
