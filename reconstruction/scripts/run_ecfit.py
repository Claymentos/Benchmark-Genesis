#!/usr/bin/env python3
"""
Pose the annotated hand and object with EC-fit, EPIC-Contact's own fitting code.

EC-fit (Bansal et al., ECCV 2026, Sec. 3.3; github.com/ZJHTerry18/ec-fit) turns bijective
contact into posed meshes in three phases:
    1  contact alignment: object rotation + translation, mean squared distance of the
       paired hand/object points, from several initial rotations;
    2  image refinement: occlusion-aware object-mask IoU (pixels under the hand are left
       out), penetration (object inside the hand's SDF), contact, and object scale;
    3  hand refinement: MANO pose (+ translation) against the hand mask, contact,
       penetration (hand inside the object's SDF) and a prior to the initial pose.
The initialisation with the lowest phase-2 loss is kept, as the repository's
pick_best_init.py does.

This script only adapts inputs and outputs; the fitting is EC-fit's code, unchanged:

  inputs  (<out-dir>/ecfit/input/<sample>/, EC-fit's layout)
    <side>_hand_posed_mesh.ply  HaWoR's hand at the frame, re-expressed with the MANO
                                parameters annotate_contact.py fitted in EC-fit's
                                convention (hand_fit.npz), so phase 3 starts from it
    wilor_output.pkl            the camera and MANO parameters, in WiLoR's format:
                                EC-fit assumes the principal point at the image centre
                                and renders at twice img_size, so the frame is padded to
                                centre the principal point, scaled to EPIC's 456 px width,
                                and the focal set to 2 x the scaled focal
    hand_mask.png, object_mask.png   SAM3's masks at the frame, same padding and scale
    object.obj                  annotation_mesh.obj (the mesh the contact indexes into)
    corresponding_contacts.json from annotate_contact.py
  outputs (<out-dir>/central_frame/, completing EPIC-Contact's per-clip folder)
    hand_mesh.obj  obj_mesh.obj  hoi_mesh.obj  hand_pose.npz  transform.json  frame_<k>.jpg
  and <out-dir>/epic_frames.json, fit_summary.json, overlay.png.

Object pose initialisation: EPIC uses "multi-mixed" (their category pose priors + four
fixed rotations); the priors are per EPIC category and are not used here, so the default
is "multi-random" (the four fixed rotations).

Run in the `ecfit` env with the same clip arguments as annotate_contact.py:
    python run_ecfit.py --video-dir ../human_videos/grasping_cup/take_006v3 \\
        --object cup --side left --frame 145 --contact-frames 114 233
"""

import argparse
import glob
import json
import os
import pickle
import shutil
import subprocess
import sys

import cv2
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from annotate_contact import (add_clip_args, load_intrinsics, load_obj, read_frame,  # noqa: E402
                              resolve_inputs, write_obj)
from contact_transfer import MANO_DIR, mano_faces  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EPIC_WIDTH = 456          # EPIC-Contact's central frames are 456 x 256


def parse_args():
    parser = argparse.ArgumentParser(description="Run EC-fit on an annotate_contact.py annotation")
    add_clip_args(parser)
    parser.add_argument("--ecfit-dir", default=os.path.join(HERE, "..", "ec-fit"))
    parser.add_argument("--mano-dir", default=MANO_DIR, help="folder with MANO_LEFT.pkl / MANO_RIGHT.pkl")
    parser.add_argument("--init", default="multi-random", choices=("single", "multi-random"),
                        help="EC-fit object_pose_init")
    parser.add_argument("--reuse", action="store_true", help="reuse EC-fit's outputs if present")
    parser.add_argument("opts", nargs=argparse.REMAINDER,
                        help="EC-fit config overrides passed through, e.g. nr_phase_2_steps=300")
    return parser.parse_args()


def pad_and_scale(image, pu, pv, scale):
    """Pad so (pu, pv) is the exact centre, then resize by `scale`."""
    h, w = image.shape[:2]
    half_w, half_h = int(np.ceil(max(pu, w - pu))), int(np.ceil(max(pv, h - pv)))
    left, top = int(round(half_w - pu)), int(round(half_h - pv))
    canvas = np.zeros((2 * half_h, 2 * half_w) + image.shape[2:], image.dtype)
    canvas[top:top + h, left:left + w] = image
    size = (int(round(2 * half_w * scale)), int(round(2 * half_h * scale)))
    return cv2.resize(canvas, size, interpolation=cv2.INTER_AREA), (left, top)


def mask_at(args, k, name):
    path = os.path.join(args.video_dir, "video_segmentation", "masks", f"frame_{k:06d}_masks", f"{name}.png")
    if not os.path.exists(path):
        raise SystemExit(f"no mask {path}")
    return ((cv2.imread(path, 0) > 128) * 255).astype(np.uint8)


def prepare(args, work, sample):
    k, side = args.frame, args.side
    out = args.out_dir
    fit = np.load(os.path.join(out, "hand_fit.npz"))
    if int(fit["frame"]) != k or str(fit["side"]) != side:
        raise SystemExit("hand_fit.npz is for another frame/side; rerun annotate_contact.py")
    contacts = os.path.join(out, "central_frame", "corresponding_contacts.json")
    if not os.path.exists(contacts):
        raise SystemExit(f"no {contacts}; save an annotation with annotate_contact.py first")
    if not json.load(open(contacts))["data"]:
        raise SystemExit("corresponding_contacts.json has no transferred region")

    fu, fv, pu, pv = load_intrinsics(args)
    if abs(fu - fv) > 1e-3 * fu:
        print(f"[warn] fu {fu:.1f} != fv {fv:.1f}; EC-fit takes one focal, using their mean")
    focal = 0.5 * (fu + fv)
    image = read_frame(args, k)
    h, w = image.shape[:2]
    half_w = int(np.ceil(max(pu, w - pu)))
    scale = min(1.0, EPIC_WIDTH / (2 * half_w))

    d = os.path.join(work, "input", sample)
    os.makedirs(d, exist_ok=True)
    frame, offset = pad_and_scale(image, pu, pv, scale)
    cv2.imwrite(os.path.join(d, "frame.jpg"), frame)
    obj_name = args.object.replace(" ", "_")
    cv2.imwrite(os.path.join(d, "object_mask.png"), pad_and_scale(mask_at(args, k, obj_name), pu, pv, scale)[0])
    hand_mask = pad_and_scale(mask_at(args, k, f"{side}_hand_0"), pu, pv, scale)[0]
    cv2.imwrite(os.path.join(d, "hand_mask.png"), hand_mask)

    # The hand, re-expressed by the fitted MANO parameters (what phase 3 starts from).
    faces = mano_faces(side, args.mano_dir)
    trimesh.Trimesh(fit["vertices"], faces, process=False).export(os.path.join(d, f"{side}_hand_posed_mesh.ply"))
    shutil.copy(os.path.join(out, "annotation_mesh.obj"), os.path.join(d, "object.obj"))
    shutil.copy(contacts, os.path.join(d, "corresponding_contacts.json"))

    # WiLoR's record. EC-fit: K = [[focal, 0, W], [0, focal, H]] rendered at 2H x 2W, with
    # (W, H) = img_size -- so the render is the frame at twice img_size, focal doubled.
    img_w, img_h = frame.shape[1], frame.shape[0]
    cam_f = 2 * focal * scale
    go = np.asarray(fit["global_orient"], np.float64)
    if side == "left":
        go = go * np.array([1.0, -1.0, -1.0])     # EC-fit flips it back for the left hand
    x = fit["vertices"]
    u = cam_f * x[:, 0] / x[:, 2] + img_w
    v = cam_f * x[:, 1] / x[:, 2] + img_h
    # EC-fit crops the hand mask to a square of half box_size around box_center, in img_size
    # pixels: centre it on the projected hand and make it cover it.
    centre = np.array([u.mean(), v.mean()]) / 2
    extent = max(np.ptp(u), np.ptp(v)) / 2
    record = {side: {
        "focal_length": np.float64(cam_f), "img_size": np.array([img_w, img_h], np.float64),
        "rot": Rotation.from_rotvec(go).as_matrix(), "pred_cam_t": np.asarray(fit["transl"], np.float64),
        "betas": np.asarray(fit["betas"], np.float64), "hand_params_original": np.asarray(fit["hand_pose"], np.float64),
        "hand_params": np.asarray(fit["hand_pose"], np.float64), "vertices": x,
        "box_center": centre, "box_size": np.float64(2 * extent * 1.3),
    }}
    with open(os.path.join(d, "wilor_output.pkl"), "wb") as fh:
        pickle.dump(record, fh)
    return {"scale": scale, "offset": offset, "focal": focal, "pu": pu, "pv": pv, "faces": faces}


def run_ecfit(args, work, sample):
    ecfit = os.path.abspath(args.ecfit_dir)
    if not os.path.isdir(ecfit):
        raise SystemExit(f"no EC-fit at {ecfit}; git clone https://github.com/ZJHTerry18/ec-fit.git there")
    # EC-fit reads MANO from static/MANO/mano_v1_2/models (src/constants.py).
    models = os.path.join(ecfit, "static", "MANO", "mano_v1_2", "models")
    if not os.path.exists(models):
        os.makedirs(os.path.dirname(models), exist_ok=True)
        os.symlink(os.path.abspath(args.mano_dir), models)
    # Overrides are set on EC-fit's config object with their declared types, then its
    # demo_run.py runs unchanged. (Its own key=value parser tries float() first, so an
    # integer such as nr_phase_2_steps=300 becomes 300.0 and breaks range().)
    overrides = {"object_pose_init": args.init}
    for opt in args.opts:
        if "=" in opt:
            key, value = opt.split("=", 1)
            overrides[key] = value
    launcher = (
        "import runpy, sys\n"
        "from src.config_packs import epic_config as cfg\n"
        f"for key, value in {overrides!r}.items():\n"
        "    if key in cfg.loss_weights: cfg.loss_weights[key] = float(value)\n"
        "    elif hasattr(cfg, key):\n"
        "        old = getattr(cfg, key)\n"
        "        setattr(cfg, key, value.lower() == 'true' if isinstance(old, bool) else type(old)(value))\n"
        "    else: raise SystemExit(f'unknown EC-fit option {key}')\n"
        f"sys.argv = ['demo_run.py', '-i', {os.path.join(work, 'input')!r}, '-o', {os.path.join(work, 'output')!r},"
        f" '-s', {sample!r}, '-r']\n"
        "runpy.run_path('demo_run.py', run_name='__main__')\n")
    cmd = [sys.executable, "-u", "-c", launcher]
    print(f"[ecfit] demo_run.py on {sample} with {overrides}")
    log = open(os.path.join(work, "ecfit.log"), "w")
    code = subprocess.call(cmd, cwd=ecfit, stdout=log, stderr=subprocess.STDOUT)
    if code:
        raise SystemExit(f"EC-fit failed (exit {code}); see {log.name}")


def pick_best(work, sample):
    """The initialisation with the lowest total phase-2 loss (EC-fit's pick_best_init.py)."""
    root = os.path.join(work, "output", sample)
    runs = sorted(glob.glob(os.path.join(root, "init*"))) or [root]
    scored = []
    for run in runs:
        path = os.path.join(run, "pred_phase2_noeval.json")
        if os.path.exists(path) and os.path.exists(os.path.join(run, "pred_obj_mesh_phase3.obj")):
            loss = json.load(open(path))["loss"]
            scored.append((sum(loss.values()), run, loss))
    if not scored:
        raise SystemExit(f"EC-fit wrote no complete result under {root}; see {work}/ecfit.log")
    scored.sort(key=lambda s: s[0])
    for total, run, _ in scored:
        print(f"[ecfit] {os.path.basename(run)}: phase-2 loss {total:.4f}")
    return scored[0]


def similarity(source, target):
    mu_s, mu_t = source.mean(0), target.mean(0)
    xs, xt = source - mu_s, target - mu_t
    u, sig, vt = np.linalg.svd(xt.T @ xs)
    d = np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))])
    rotation = u @ d @ vt
    scale = float((sig * np.diag(d)).sum() / (xs ** 2).sum())
    return scale, rotation, mu_t - scale * rotation @ mu_s


def export(args, run, cam):
    k, side, out = args.frame, args.side, args.out_dir
    cf = os.path.join(out, "central_frame")
    hand = trimesh.load(os.path.join(run, "pred_hand_mesh_phase3.obj"), process=False, force="mesh")
    placed = trimesh.load(os.path.join(run, "pred_obj_mesh_phase3.obj"), process=False, force="mesh")
    mesh_v, mesh_f = load_obj(os.path.join(out, "annotation_mesh.obj"))
    placed_v = np.asarray(placed.vertices)
    if len(placed_v) != len(mesh_v):
        raise SystemExit("EC-fit's object has a different vertex count from annotation_mesh.obj")
    # annotation_mesh.obj -> camera: a similarity (phase 2 also fits the object's scale).
    s, r, t = similarity(mesh_v, placed_v)
    residual = np.linalg.norm(s * mesh_v @ r.T + t - placed_v, axis=1).max()
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = s * r, t
    obj_v = mesh_v @ transform[:3, :3].T + transform[:3, 3]
    hand_v = np.asarray(hand.vertices)
    faces = cam["faces"]

    for old in glob.glob(os.path.join(cf, "frame_*.jpg")):
        os.remove(old)
    image = read_frame(args, k)
    cv2.imwrite(os.path.join(cf, f"frame_{k:010d}.jpg"), image)
    write_obj(os.path.join(cf, "hand_mesh.obj"), hand_v, faces)
    write_obj(os.path.join(cf, "obj_mesh.obj"), obj_v, mesh_f)
    write_obj(os.path.join(cf, "hoi_mesh.obj"), np.concatenate([hand_v, obj_v]),
              np.concatenate([faces, mesh_f + len(hand_v)]))
    json.dump({"transform": transform.tolist()}, open(os.path.join(cf, "transform.json"), "w"))
    pose = pickle.load(open(os.path.join(run, "pred_hand_pose_phase3.pkl"), "rb"))
    np.savez(os.path.join(cf, "hand_pose.npz"), **{key: np.asarray(pose[key]).reshape(-1) for key in
                                                   ("betas", "global_orient", "hand_pose", "transl")})
    n_frames = len(np.load(args.hand_meshes, allow_pickle=True)[f"{side}_vertices"])
    json.dump({"clip": os.path.basename(os.path.abspath(args.video_dir)), "side": side,
               "object_class": args.object, "source": "annotate_contact.py + EC-fit",
               "epic_frames": list(range(n_frames)), "keyframe_epic": k, "keyframe_local": k,
               "contact_frames": args.contact_frames},
              open(os.path.join(out, "epic_frames.json"), "w"), indent=1)
    return hand_v, obj_v, mesh_f, s, residual


def diagnostics(args, hand_v, obj_v, obj_f, cam):
    """Contact, penetration and mask agreement of the fitted meshes at the frame."""
    cf = os.path.join(args.out_dir, "central_frame")
    data = json.load(open(os.path.join(cf, "corresponding_contacts.json")))["data"]
    mapping = json.load(open(os.path.join(args.epic_root, "contact_assets", "sparse_dense_mapping.json")))
    d2s = np.asarray(mapping[f"m{args.side[0]}_dense2sparse"])
    entries = {e["name"]: e["contactPoints"] for e in data}
    pair, touch, sparse = [], [], set()
    for name, points in entries.items():
        if not name.startswith("objShape"):
            continue
        human = entries["humanShape" + name[len("objShape"):]]
        for o, hv in zip(points, human):
            tok = o.split()
            p = np.asarray([float(x) for x in tok[2:5]]) @ obj_v[obj_f[int(tok[1])]]
            s = d2s[int(hv.split()[1])]
            pair.append(np.linalg.norm(hand_v[s] - p))
            sparse.add(s)
    obj_mesh = trimesh.Trimesh(obj_v, obj_f, process=False)
    query = trimesh.proximity.ProximityQuery(obj_mesh)
    sd = query.signed_distance(hand_v)                       # positive inside
    touch = np.abs(sd[sorted(sparse)])

    fu, fv, pu, pv = load_intrinsics(args)
    image = read_frame(args, args.frame)
    proj = lambda x: np.stack([fu * x[:, 0] / x[:, 2] + pu, fv * x[:, 1] / x[:, 2] + pv], 1)
    sil = np.zeros(image.shape[:2], np.uint8)
    for tri in np.round(proj(obj_v)[obj_f]).astype(np.int32):
        cv2.fillConvexPoly(sil, tri, 1)
    obj_mask = mask_at(args, args.frame, args.object.replace(" ", "_")) > 0
    hand_mask = mask_at(args, args.frame, f"{args.side}_hand_0") > 0
    keep = ~hand_mask
    a, b = (sil > 0) & keep, obj_mask & keep
    iou = float((a & b).sum() / max((a | b).sum(), 1))

    overlay = image.copy()
    overlay[sil > 0] = (0.5 * overlay[sil > 0] + [0, 0, 127]).astype(np.uint8)
    for x, y in np.round(proj(hand_v)).astype(int):
        cv2.circle(overlay, (x, y), 2, (80, 220, 80), -1)
    cv2.putText(overlay, f"EC-fit, frame {args.frame}: object (red), hand (green); object-mask IoU {iou:.2f}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imwrite(os.path.join(args.out_dir, "overlay.png"), overlay)
    return {"pair_distance_mm": {"median": float(np.median(pair) * 1000), "p90": float(np.percentile(pair, 90) * 1000)},
            "contact_vertex_to_surface_mm": {"median": float(np.median(touch) * 1000),
                                             "within_5mm": float((touch < 0.005).mean())},
            "penetration_max_mm": float(max(sd.max(), 0) * 1000), "object_mask_iou": iou}


def main():
    args = parse_args()
    resolve_inputs(args)
    k, side = args.frame, args.side
    cat = args.object.replace(" ", "").replace("_", "")
    # EC-fit parses the folder name: [:-4].split("_")[5] is the category, "left" in it the side.
    sample = f"own_clip_{k}_{k}_{side}_{cat}_run"
    work = os.path.join(args.out_dir, "ecfit")
    cam = prepare(args, work, sample)
    if not (args.reuse and glob.glob(os.path.join(work, "output", sample, "**", "pred_obj_mesh_phase3.obj"),
                                     recursive=True)):
        run_ecfit(args, work, sample)
    total, run, loss = pick_best(work, sample)
    hand_v, obj_v, obj_f, scale, residual = export(args, run, cam)
    summary = {"frame": k, "side": side, "init": args.init, "best_run": os.path.basename(run),
               "phase2_loss": loss, "object_scale_from_fit": scale, "transform_residual_mm": residual * 1000,
               **diagnostics(args, hand_v, obj_v, obj_f, cam)}
    json.dump(summary, open(os.path.join(args.out_dir, "fit_summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))
    print(f"[done] {os.path.join(args.out_dir, 'central_frame')} is a complete EPIC-Contact clip folder; "
          f"check {os.path.join(args.out_dir, 'overlay.png')}")


if __name__ == "__main__":
    main()
