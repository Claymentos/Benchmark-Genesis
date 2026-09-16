"""Convert a RecGen result into do-as-i-do's sam-3d-objects mesh + layout.json format.

Lets RecGen stand in for do-as-i-do/reconstruction stage 2
(`modules/sam-3d-objects/generate_mesh_sam3d.py`, which builds the canonical
per-object mesh at the reference frame) while leaving stage 3-4
(`modules/Fast-SAM3D/track_object.py` and friends) untouched: this script
just needs to reproduce the exact two files those stages read for the
reference frame:

    <masks_root>/frame_NNNNNN_masks/<object_name>/<object_name>.obj
    <masks_root>/frame_NNNNNN_masks/layout.json

`layout.json`'s per-object "local_to_scene" transform (translation, scale,
quat_wxyz) places the *local-frame* mesh into what sam-3d-objects calls
"scene" space, which turns out to be real camera ("R3") space with X and Y
negated and NO extra translation -- confirmed two ways: (a) their own
`compute_pointmap()` (in generate_mesh_sam3d.py) only ever applies
`camera_to_pytorch3d_camera().rotation`, never `.translation`, when moving
R3-space points into this same "scene" space; (b) track_object.py's renderer
builds `PerspectiveCameras(focal_length, principal_point)` with default
(identity) extrinsics, i.e. the render camera sits at the scene-frame origin.
The mesh's local frame must additionally be Y-up (a fixed `_R_YUP_TO_ZUP`
rotation is applied to every loaded .obj's vertices before placing it, by
both generate_mesh_sam3d.py and track_object.py) -- this script pre-applies
the inverse so the exported .obj round-trips through that fixed rotation
back to RecGen's own local mesh exactly.

In "same-frame" mode (`--recgen-out-dir`), the whole conversion is validated
against RecGen's own `posed_mesh.obj` (see the round-trip block in `main()`):
it re-derives the camera-frame vertices from the computed local_to_scene
transform using sam3d_objects' own `compose_transform`, and checks the max
vertex error is near machine epsilon before writing anything.

In "reused-mesh" mode (`--mesh-path` + `--fitted-rotation/translation/scale`),
there is no such ground truth -- the mesh comes from a *different*
video/frame (e.g. reusing an already-reconstructed object's shape across
multiple demos of the same physical object) and the pose was instead fit
externally (e.g. multi-hypothesis ICP against this frame's own masked depth
point cloud). The script still runs the same compose_transform math, but
validates by re-projecting into `--verify-rgb`/`--verify-mask`/`--verify-intrinsics`
and reporting IoU -- inspect the saved overlay yourself, this is a much
weaker guarantee than the same-frame round-trip.

Must run in an environment with `sam3d_objects` + `pytorch3d` importable
(the do-as-i-do `sam3d` conda env), pointed at via `--sam3d-repo` if it is
not already on the path. Uses `LIDRA_SKIP_INIT=1` internally to avoid
sam3d_objects' heavy repo-wide init for this lightweight use.

Examples:
    # same-frame (mesh + pose both from this RecGen run)
    conda run -n sam3d env LIDRA_SKIP_INIT=1 python scripts/recgen_mesh_to_layout.py \
        --recgen-out-dir outputs/inference_outputs/grasping_cup_take001_f000_moge \
        --object-name cup \
        --out-dir /path/to/video_dir/video_segmentation/masks/frame_000000_masks \
        --sam3d-repo /home/cl-ment-prigent/do-as-i-do/reconstruction/modules/sam-3d-objects

    # reused mesh (shape from one video, pose fit against another video's frame)
    conda run -n sam3d env LIDRA_SKIP_INIT=1 python scripts/recgen_mesh_to_layout.py \
        --mesh-path outputs/inference_outputs/grasping_cup_take001_f000_moge/mesh.obj \
        --fitted-rotation R.npy --fitted-translation t.npy --fitted-scale s.npy \
        --object-name cup \
        --out-dir /path/to/other_video_dir/video_segmentation/masks/frame_000240_masks \
        --verify-rgb frame_000240_rgb.png --verify-mask frame_000240_masks/cup.png \
        --verify-fx 530.28 --verify-fy 530.28 --verify-cx 641.34 --verify-cy 354.53 \
        --sam3d-repo /home/cl-ment-prigent/do-as-i-do/reconstruction/modules/sam-3d-objects
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh

# Exact constant from do-as-i-do/reconstruction/modules/sam-3d-objects/generate_mesh_sam3d.py
# (named "_R_YUP_TO_ZUP" there; "_R_YUP_TO_ZUP_RENDER" in Fast-SAM3D/track_object.py is the same matrix).
_R_ZUP_TO_YUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
RY = _R_ZUP_TO_YUP.T


def decimate_mesh(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    """Reduce face count via open3d quadric decimation (~SAM-3D-Objects' own mesh scale).

    RecGen ships full-resolution meshes (often 1M+ faces); track_object.py renders many
    pose candidates per frame, which is impractical at that scale. open3d's decimator
    doesn't preserve UV texture coordinates, so any texture is first baked into per-vertex
    colors (trimesh's TextureVisuals.to_color()) which DOES survive decimation.
    """
    if not target_faces or len(mesh.faces) <= target_faces:
        return mesh

    import open3d as o3d

    m = mesh.copy()
    try:
        m.visual = m.visual.to_color()
    except Exception as e:
        print(f"[recgen_mesh_to_layout] WARNING: could not bake texture to vertex colors ({e}); "
              f"decimated mesh will be untextured.")

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(m.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(m.faces)
    vertex_colors = getattr(m.visual, "vertex_colors", None)
    if vertex_colors is not None:
        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(vertex_colors)[:, :3].astype(np.float64) / 255.0)

    simplified = o3d_mesh.simplify_quadric_decimation(target_number_of_triangles=target_faces)
    simplified.remove_unreferenced_vertices()

    new_mesh = trimesh.Trimesh(
        vertices=np.asarray(simplified.vertices), faces=np.asarray(simplified.triangles), process=False,
    )
    if simplified.has_vertex_colors():
        colors = np.clip(np.asarray(simplified.vertex_colors) * 255, 0, 255).astype(np.uint8)
        new_mesh.visual.vertex_colors = colors
    print(f"[recgen_mesh_to_layout] decimated mesh: {len(mesh.faces)} -> {len(new_mesh.faces)} faces")
    return new_mesh


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert a RecGen result into do-as-i-do's mesh+layout.json format")
    p.add_argument("--recgen-out-dir", default=None, help="Same-frame mode: a directory produced by RecGenResult.save(...)")
    p.add_argument("--mesh-path", default=None, help="Reused-mesh mode: path to a RecGen mesh.obj (local frame)")
    p.add_argument("--fitted-rotation", default=None, help="Reused-mesh mode: .npy, (3,3) local->R3 rotation")
    p.add_argument("--fitted-translation", default=None, help="Reused-mesh mode: .npy, (3,) local->R3 translation")
    p.add_argument("--fitted-scale", default=None, help="Reused-mesh mode: .npy, scalar (or (1,)) local->R3 scale")
    p.add_argument("--verify-rgb", default=None, help="Reused-mesh mode: RGB image to render the verification overlay onto")
    p.add_argument("--verify-mask", default=None, help="Reused-mesh mode: object mask, for the reported IoU")
    p.add_argument("--verify-fx", type=float, default=None)
    p.add_argument("--verify-fy", type=float, default=None)
    p.add_argument("--verify-cx", type=float, default=None)
    p.add_argument("--verify-cy", type=float, default=None)
    p.add_argument("--verify-out", default=None, help="Where to save the verification overlay PNG (reused-mesh mode)")
    p.add_argument("--object-name", required=True, help="Object id, matching the SAM3 mask filename (without .png)")
    p.add_argument("--out-dir", required=True, help="Target frame_NNNNNN_masks/ directory")
    p.add_argument(
        "--sam3d-repo", default=None,
        help="Path to modules/sam-3d-objects, if sam3d_objects isn't already importable",
    )
    p.add_argument(
        "--max-round-trip-error", type=float, default=1e-3,
        help="Same-frame mode only: abort instead of writing files if the round-trip check exceeds this (metres)",
    )
    p.add_argument(
        "--target-faces", type=int, default=16000,
        help="Decimate the exported mesh to about this many faces (~SAM-3D-Objects' own scale); "
             "0 to disable and export at full RecGen resolution. Validation/verification always "
             "runs against the full-resolution mesh regardless of this setting.",
    )
    args = p.parse_args()
    if bool(args.recgen_out_dir) == bool(args.mesh_path):
        p.error("pass exactly one of --recgen-out-dir (same-frame mode) or --mesh-path (reused-mesh mode)")
    if args.mesh_path and not (args.fitted_rotation and args.fitted_translation and args.fitted_scale):
        p.error("--mesh-path requires --fitted-rotation/--fitted-translation/--fitted-scale")
    return args


def main() -> None:
    args = parse_args()

    if args.sam3d_repo:
        sys.path.insert(0, args.sam3d_repo)

    import torch
    from pytorch3d.transforms import Transform3d, matrix_to_quaternion
    from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform, decompose_transform
    from sam3d_objects.pipeline.inference_pipeline_pointmap import camera_to_pytorch3d_camera

    same_frame_mode = args.recgen_out_dir is not None
    if same_frame_mode:
        recgen_out = Path(args.recgen_out_dir)
        raw_mesh = trimesh.load(str(recgen_out / "mesh.obj"), force="mesh")
        with open(recgen_out / "metadata.json") as f:
            meta = json.load(f)

        pose_matrix = np.array(meta["pose_matrix"])   # local -> normalized-cam (ncam), column-vector convention
        cam2ncam = np.array(meta["cam2ncam"])          # real-cam (R3) -> ncam, column-vector convention

        s_ncam = cam2ncam[0, 0]
        A = pose_matrix[:3, :3] / s_ncam                # local -> R3, still rotation * scale
        t_total = (pose_matrix[:3, 3] - cam2ncam[:3, 3]) / s_ncam

        row_norms = np.linalg.norm(A, axis=1)
        s_total = float(row_norms.mean())
        if not np.allclose(row_norms, s_total, rtol=1e-3):
            print(f"[recgen_mesh_to_layout] WARNING: non-uniform scale detected ({row_norms}); "
                  f"using mean. local_to_scene assumes a similarity transform.")
        R_total = A / s_total
    else:
        raw_mesh = trimesh.load(args.mesh_path, force="mesh")
        R_total = np.load(args.fitted_rotation)
        t_total = np.load(args.fitted_translation).reshape(3)
        s_total = float(np.load(args.fitted_scale).reshape(-1)[0])
        print(f"[recgen_mesh_to_layout] reused-mesh mode: mesh={args.mesh_path}  scale={s_total:.4f}")

    # ---- Build the do-as-i-do "local_to_scene" transform with sam3d_objects' own functions ----
    T_R3 = compose_transform(
        scale=torch.tensor([[s_total] * 3], dtype=torch.float64),
        rotation=torch.tensor(R_total.T, dtype=torch.float64).unsqueeze(0),
        translation=torch.tensor(t_total, dtype=torch.float64).unsqueeze(0),
    )
    c2p = camera_to_pytorch3d_camera(device=torch.device("cpu"))
    # Deliberately rotation-only -- see module docstring.
    T_scene = T_R3.compose(Transform3d(dtype=torch.float64).rotate(c2p.rotation.double()))

    scale_out, rotation_out, translation_out = decompose_transform(T_scene)
    quat_wxyz = matrix_to_quaternion(rotation_out).squeeze(0).numpy()
    quat_xyzw = quat_wxyz[[1, 2, 3, 0]]

    # ---- Round-trip / verification before writing anything ----
    obj_vertices = raw_mesh.vertices @ RY.T
    v_reloaded = obj_vertices @ RY
    v_reloaded_t = torch.tensor(v_reloaded, dtype=torch.float64).unsqueeze(0)
    v_scene = compose_transform(scale_out, rotation_out, translation_out).transform_points(v_reloaded_t)
    v_r3_recon = (v_scene @ c2p.rotation.double().transpose(-1, -2)).squeeze(0).numpy()

    if same_frame_mode:
        posed_expected = (pose_matrix[:3, :3] @ raw_mesh.vertices.T).T + pose_matrix[:3, 3]
        posed_expected = (posed_expected - cam2ncam[:3, 3]) / cam2ncam[0, 0]
        max_err = float(np.abs(v_r3_recon - posed_expected).max())
        print(f"[recgen_mesh_to_layout] coordinate round-trip max error: {max_err:.3e} m")
        if max_err > args.max_round_trip_error:
            raise RuntimeError(
                f"Round-trip error {max_err:.3e} m exceeds --max-round-trip-error "
                f"{args.max_round_trip_error:.3e} m -- refusing to write a possibly-wrong layout.json."
            )
    elif args.verify_rgb and args.verify_mask:
        # v_r3_recon must equal raw_mesh.vertices placed by (R_total, t_total, s_total) directly
        # (same identity as the same-frame branch, just with no independent ground truth to check
        # it against) -- reuse it directly to render a check overlay instead.
        direct = s_total * (R_total @ raw_mesh.vertices.T).T + t_total
        assert np.abs(v_r3_recon - direct).max() < 1e-6, "internal consistency check failed"

        rgb = cv2.imread(args.verify_rgb)
        mask = cv2.imread(args.verify_mask, cv2.IMREAD_UNCHANGED) > 0
        fx, fy, cx, cy = args.verify_fx, args.verify_fy, args.verify_cx, args.verify_cy
        z = direct[:, 2]
        u = (direct[:, 0] / z * fx + cx)
        v = (direct[:, 1] / z * fy + cy)
        H, W = mask.shape
        proj = np.zeros((H, W), dtype=bool)
        inb = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z > 1e-3)
        proj[v[inb].astype(int), u[inb].astype(int)] = True
        iou = (proj & mask).sum() / max((proj | mask).sum(), 1)
        print(f"[recgen_mesh_to_layout] reused-mesh verification: projected-silhouette IoU = {iou:.3f} "
              f"(coarse metric -- inspect the overlay, don't trust this number alone)")

        overlay = rgb.copy()
        for face in raw_mesh.faces[::5]:
            pts2d = np.stack([u[face], v[face]], axis=1).astype(np.int32)
            cv2.polylines(overlay, [pts2d], True, (255, 0, 255), 1, cv2.LINE_AA)
        verify_out = args.verify_out or str(Path(args.out_dir) / f"{args.object_name}_fit_overlay.png")
        cv2.imwrite(verify_out, overlay)
        print(f"[recgen_mesh_to_layout] saved verification overlay: {verify_out}")
    else:
        print("[recgen_mesh_to_layout] WARNING: reused-mesh mode with no --verify-rgb/--verify-mask given -- "
              "writing files with NO visual check at all.")

    # ---- Write mesh.obj (local/Y-up frame), decimated for the tracker's sake ----
    export_mesh = decimate_mesh(raw_mesh, args.target_faces)
    export_mesh.vertices = export_mesh.vertices @ RY.T

    out_dir = Path(args.out_dir)
    object_folder = out_dir / args.object_name
    object_folder.mkdir(parents=True, exist_ok=True)
    mesh_out_path = object_folder / f"{args.object_name}.obj"
    export_mesh.export(str(mesh_out_path))
    print(f"[recgen_mesh_to_layout] saved: {mesh_out_path}")

    # ---- Merge into layout.json (preserving any other objects already recorded for this frame) ----
    layout_path = out_dir / "layout.json"
    if layout_path.exists():
        with open(layout_path) as f:
            payload = json.load(f)
        payload["objects"] = [o for o in payload.get("objects", []) if o.get("mesh_obj") != f"{args.object_name}.obj"]
    else:
        payload = {
            "frame": "sam3d_scene",
            "note": (
                "Transforms are local->scene as used by SAM3D make_scene/_export_scene_glb. "
                "Scale is typically uniform (same xyz)."
            ),
            "objects": [],
        }

    M = T_scene.get_matrix().squeeze(0).numpy()  # (4, 4), row-vector convention (p' = p @ M)
    payload["objects"].append({
        "index": None,  # placeholder, renumbered below
        "mesh_obj": f"{args.object_name}.obj",
        "source": "recgen" if same_frame_mode else "recgen-reused-mesh-icp-fit",
        "local_to_scene": {
            "translation": translation_out.squeeze(0).tolist(),
            "scale": scale_out.squeeze(0).tolist(),
            "quat_wxyz": quat_wxyz.tolist(),
            "quat_xyzw": quat_xyzw.tolist(),
            "matrix_4x4_row_major": M.reshape(-1).tolist(),
        },
    })
    for i, obj in enumerate(payload["objects"]):
        obj["index"] = i

    with open(layout_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[recgen_mesh_to_layout] saved: {layout_path}")


if __name__ == "__main__":
    main()
