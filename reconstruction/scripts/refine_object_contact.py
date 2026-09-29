#!/usr/bin/env python3
"""
Refine the hand-object reconstruction with EPIC-Contact's annotated contact.

The pipeline reconstructs the hand (HaWoR) and the object (RecGen mesh + tracked pose)
independently, each from a single view, so they only agree up to the depth each one
guessed: MoGe's metric depth sets the object's size and distance, HaWoR's MANO prior
sets the hand's, and nothing makes the two meet. The result floats the object a few
centimetres off the fingers, or sinks it through the palm, while projecting perfectly
in the image. EPIC-Contact annotates exactly the missing piece -- which hand vertices
touch the object, and where on the object each hand region lands -- and every
EPIC-Contact clip is a stable grasp, so that contact holds for the whole clip.

What the labels are used for, and what they are not:

  * hand side (contact_vertices_sparse.json): the painted MANO vertices, per region
    (fingers / thumb / palm). HaWoR's hand is MANO too, so these indices apply to it
    directly. They sit a median 4.5 mm off the GT object surface, so "these vertices
    touch the surface" is a trustworthy constraint.
  * object side (corresponding_contacts.json): the object patch each hand region
    lands on. The bijection is NOT used point-to-point -- the i-th hand/object pair of
    a region is a median 28 mm apart in the GT meshes themselves -- only region to
    region: each hand region must lie on its own patch. That is what pins rotation
    about a bottle's axis, which silhouettes and point tracks leave nearly free.

The object-side patches live on the EPIC template, not the reconstructed mesh, so they
are transferred once, at the true-GT central frame: the GT hand is aligned onto
HaWoR's hand there (same 778-vertex topology, similarity Procrustes), which places the
GT object in the reconstruction's camera frame; the RecGen mesh is then aligned to that
placed object (similarity ICP) and each patch point is snapped onto its surface.

Then one joint optimisation over the clip, with
    s          one object scale about the camera centre -- this leaves the object's
               image projection exactly unchanged and moves only its depth, i.e. it
               is precisely the monocular ambiguity contact can resolve
    w_t, u_t   a per-frame object rotation / translation correction
    m_t        a per-frame hand shift along the ray through its wrist (HaWoR's depth
               jitters from frame to frame; its projection is kept)
and energy
    touch      GT contact vertices on the object surface          (SDF = 0)
    penetrate  no hand vertex inside the object                   (SDF >= 0)
    region     each hand region on its own object patch
    image      object's 2D position and 2D shape stay with the tracker's (pixels);
               its apparent size is left free, since it came from MoGe's drifting depth
    prior      hand depth shift stays small
    smooth     acceleration of object and hand
The SDF is precomputed on a grid over the canonical mesh and sampled trilinearly, so
every term is differentiable.

Writes a mesh scaled by s (the pipeline's meshes live in the reference camera frame,
so scaling about the origin keeps them consistent), a trajectory in the tracker's
.npz format for that mesh, the depth-corrected hand meshes, and the contact labels
transferred onto the mesh.

Run in the `genesis` env:
    python refine_object_contact.py --hand-meshes all_hand_meshes.npz --side left \\
        --object-mesh posed_mesh.obj --object-poses bottle_traj_v2s2r_moge.npz \\
        --epic-clip-dir epic-contact/central_frames/<clip> --epic-frames epic_frames.json \\
        --intrinsics intrinsics.yaml --out-dir contact_refined
"""

import argparse
import json
import os

import numpy as np
import torch
import trimesh
import yaml
from scipy.spatial import cKDTree

REGIONS = ("fingers", "thumb", "palm")


def parse_args():
    parser = argparse.ArgumentParser(description="Contact-constrained hand-object refinement")
    parser.add_argument("--hand-meshes", required=True, help="HaWoR all_hand_meshes.npz")
    parser.add_argument("--side", required=True, choices=("left", "right"))
    parser.add_argument("--object-mesh", required=True, help="posed_mesh.obj, reference camera frame")
    parser.add_argument("--object-poses", required=True, help="Tracked object trajectory .npz")
    parser.add_argument("--epic-clip-dir", required=True, help="epic-contact/central_frames/<clip>")
    parser.add_argument("--epic-frames", required=True, help="epic_frames.json from prepare_epic_contact_clip.py")
    parser.add_argument("--intrinsics", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--shape", choices=("recgen", "epic"), default="recgen",
                        help="Object shape: the pipeline's RecGen mesh (contact labels are "
                        "transferred onto it), or EPIC-Contact's annotated object mesh")
    parser.add_argument("--init", choices=("tracker", "grasp"), default="tracker",
                        help="Starting trajectory: the tracker's poses, or the object carried "
                        "rigidly by the hand from its pose at the central frame (stable grasp)")
    parser.add_argument("--iters", type=int, default=1500)
    parser.add_argument("--grid", type=int, default=128, help="SDF grid resolution per axis")
    parser.add_argument("--w-touch", type=float, default=1.0)
    parser.add_argument("--w-penetrate", type=float, default=10.0)
    parser.add_argument("--w-region", type=float, default=0.3)
    parser.add_argument("--w-image", type=float, default=1.0)
    parser.add_argument("--w-prior", type=float, default=0.1)
    parser.add_argument("--w-smooth", type=float, default=30.0)
    parser.add_argument("--touch-sigma", type=float, default=0.01,
                        help="Geman-McClure scale for the touch term, metres. Vertices "
                        "further than a few sigma off the surface stop pulling, so a "
                        "finger that has actually lifted off does not drag the object")
    parser.add_argument("--touch-sigma-start", type=float, default=0.1,
                        help="Robust scale the touch term starts at, annealed down to "
                        "--touch-sigma over the first 60%% of iterations, metres")
    parser.add_argument("--hand-sigma", type=float, default=0.05,
                        help="Hand depth shift that costs 1 in the prior, metres. HaWoR's "
                        "depth is a guess from MANO's size and the focal, so it is loose")
    parser.add_argument("--image-sigma", type=float, default=15.0,
                        help="Reprojection deviation from the tracker that costs 1, pixels")
    return parser.parse_args()


# ───────────────────────────── geometry helpers ─────────────────────────────

def load_obj(path):
    mesh = trimesh.load(path, process=False, force="mesh")
    return np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces, np.int64)


def similarity_procrustes(source, target):
    """(scale, R, t) minimising |s R source + t - target|^2 (Umeyama)."""
    mu_s, mu_t = source.mean(0), target.mean(0)
    xs, xt = source - mu_s, target - mu_t
    u, sig, vt = np.linalg.svd(xt.T @ xs)
    d = np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))])
    rotation = u @ d @ vt
    scale = float((sig * np.diag(d)).sum() / (xs ** 2).sum())
    return scale, rotation, mu_t - scale * rotation @ mu_s


def rotvec_to_matrix(w):
    """Batched Rodrigues, differentiable at 0."""
    theta = torch.sqrt((w ** 2).sum(-1, keepdim=True) + 1e-12)
    k = w / theta
    kx, ky, kz = k.unbind(-1)
    zero = torch.zeros_like(kx)
    skew = torch.stack([zero, -kz, ky, kz, zero, -kx, -ky, kx, zero], -1).reshape(*w.shape[:-1], 3, 3)
    eye = torch.eye(3, device=w.device, dtype=w.dtype).expand_as(skew)
    s, c = torch.sin(theta)[..., None], torch.cos(theta)[..., None]
    return eye + s * skew + (1 - c) * skew @ skew


def build_sdf_grid(vertices, faces, resolution, margin=0.04, samples=200000):
    """Signed distance on a regular grid; negative inside.

    Sign comes from the nearest surface sample's normal rather than an inside test, so
    it needs no watertight mesh -- RecGen's usually is, but a single hole would flip
    an inside test for half the grid, while the normal sign is only local.
    """
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    points, face_idx = trimesh.sample.sample_surface_even(mesh, samples, seed=0)
    normals = mesh.face_normals[face_idx]
    lo = vertices.min(0) - margin
    hi = vertices.max(0) + margin
    axes = [np.linspace(lo[i], hi[i], resolution) for i in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    dist, idx = cKDTree(points).query(grid, workers=-1)
    sign = np.sign(((grid - points[idx]) * normals[idx]).sum(1))
    sign[sign == 0] = 1.0
    sdf = (dist * sign).reshape(resolution, resolution, resolution)
    return sdf.astype(np.float32), lo, hi


class GridSDF:
    """Trilinear lookup of a precomputed SDF; outside the grid, distance to the box."""

    def __init__(self, sdf, lo, hi, device):
        # grid_sample wants (N, C, D, H, W) indexed (z, y, x); the grid is (x, y, z).
        self.volume = torch.tensor(sdf, device=device).permute(2, 1, 0)[None, None]
        self.lo = torch.tensor(lo, device=device, dtype=torch.float32)
        self.hi = torch.tensor(hi, device=device, dtype=torch.float32)

    def __call__(self, points):
        shape = points.shape[:-1]
        flat = points.reshape(-1, 3)
        normalised = 2 * (flat - self.lo) / (self.hi - self.lo) - 1
        inside = torch.nn.functional.grid_sample(
            self.volume, normalised.clamp(-1, 1)[None, :, None, None, :],
            align_corners=True, padding_mode="border",
        ).reshape(-1)
        outside = torch.linalg.norm(
            torch.relu(flat - self.hi) + torch.relu(self.lo - flat), dim=-1
        )
        return (inside + outside).reshape(shape)


# ───────────────────────────── EPIC-Contact labels ─────────────────────────────

def load_epic_contact(clip_dir):
    """GT hand/object meshes and per-region contact, central frame, camera frame."""
    hand_v, _ = load_obj(os.path.join(clip_dir, "hand_mesh.obj"))
    obj_v, obj_f = load_obj(os.path.join(clip_dir, "obj_mesh.obj"))
    sparse = json.load(open(os.path.join(clip_dir, "contact_vertices_sparse.json")))
    data = json.load(open(os.path.join(clip_dir, "corresponding_contacts.json")))
    data = data["data"] if isinstance(data, dict) else data

    hand_regions = {r: np.array(sorted(set(sparse.get(r, []))), int) for r in REGIONS}
    obj_regions = {}
    for entry in data:
        name = entry.get("name", "")
        if not name.startswith("objShape"):
            continue
        region = name[len("objShape"):]
        pts = []
        for item in entry["contactPoints"]:
            tok = item.split()
            tri = obj_v[obj_f[int(tok[1])]]
            b = np.array([float(x) for x in tok[2:5]])
            pts.append(b @ tri)
        obj_regions[region] = np.array(pts)
    return hand_v, obj_v, obj_f, hand_regions, obj_regions


def register_mesh_to_template(rec_points, template_points, iters=300, device="cpu"):
    """Similarity (s, R, t) taking the reconstructed object onto the placed template.

    Symmetric chamfer with a robust cap, from identity: the hand bridge has already put
    the template within a few centimetres, and a local fit is what is wanted -- a
    global one would happily flip a bottle end over end to match a symmetric shape.
    """
    src = torch.tensor(rec_points, dtype=torch.float32, device=device)
    dst = torch.tensor(template_points, dtype=torch.float32, device=device)
    centre = src.mean(0)
    log_s = torch.zeros(1, device=device, requires_grad=True)
    w = torch.zeros(3, device=device, requires_grad=True)
    t = (dst.mean(0) - centre).clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([log_s, w, t], lr=5e-3)
    for _ in range(iters):
        moved = torch.exp(log_s) * (src - centre) @ rotvec_to_matrix(w).T + centre + t
        d = torch.cdist(moved, dst)
        cap = 0.03
        loss = (torch.clamp(d.min(1).values, max=cap) ** 2).mean() + \
               (torch.clamp(d.min(0).values, max=cap) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        s = float(torch.exp(log_s))
        rotation = rotvec_to_matrix(w).cpu().numpy().astype(np.float64)
        translation = (centre + t - s * centre @ rotvec_to_matrix(w).T).cpu().numpy().astype(np.float64)
    return s, rotation, translation, float(loss.detach())


# ───────────────────────────── main ─────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    calib = yaml.safe_load(open(args.intrinsics))
    fu, fv, pu, pv = (float(calib[k]) for k in ("fu", "fv", "pu", "pv"))
    meta = json.load(open(args.epic_frames))

    hands = np.load(args.hand_meshes, allow_pickle=True)
    hand_v = hands[f"{args.side}_vertices"].astype(np.float64)
    hand_valid = hands[f"{args.side}_valid"].astype(bool)
    # HaWoR can flag a frame valid and still return NaN vertices (grasping_cup take006,
    # frames 286-294). One NaN poisons every term through the batched sums, so such
    # frames are invalid, and hold the nearest good hand to keep the arrays finite.
    finite = np.isfinite(hand_v).all((1, 2))
    if (hand_valid & ~finite).any():
        print(f"[data] hand NaN on frames {np.flatnonzero(hand_valid & ~finite).tolist()}; marked invalid")
    hand_valid &= finite
    good = np.flatnonzero(finite)
    hand_v = hand_v[good[np.abs(good[None] - np.arange(len(hand_v))[:, None]).argmin(1)]]

    poses = np.load(args.object_poses, allow_pickle=True)
    rot, trans = poses["rotation"].astype(np.float64), poses["translation"].astype(np.float64)
    obj_valid = poses["valid"].astype(bool)
    confidence = poses["confidence"] if "confidence" in poses.files else np.ones(len(rot))

    mesh_v, mesh_f = load_obj(args.object_mesh)
    n = min(len(hand_v), len(rot))
    hand_v, hand_valid, rot, trans, obj_valid = hand_v[:n], hand_valid[:n], rot[:n], trans[:n], obj_valid[:n]
    confidence = np.asarray(confidence[:n], float)
    both = hand_valid & obj_valid
    # An EPIC-Contact clip is one stable grasp, so contact holds on every frame. A clip
    # of your own (annotate_contact.py) can reach, grasp and release: contact then
    # holds only inside "contact_frames"; penetration is kept everywhere.
    window = np.zeros(n, bool)
    lo, hi = meta.get("contact_frames") or (0, n - 1)
    window[lo:hi + 1] = True
    grasp = both & window
    print(f"[data] {n} frames; hand valid {hand_valid.sum()}, object valid {obj_valid.sum()}, both {both.sum()}, "
          f"in contact window {lo}-{hi}: {grasp.sum()}")
    if grasp.sum() < 3:
        raise SystemExit("too few frames with both a hand and an object pose")

    # ── 1. bridge: GT hand -> HaWoR hand at the true-GT central frame ──
    gt_hand, gt_obj_v, gt_obj_f, hand_regions, obj_regions = load_epic_contact(args.epic_clip_dir)
    k = int(meta["keyframe_local"])
    if not grasp[k]:
        k = int(np.flatnonzero(grasp)[np.argmin(np.abs(np.flatnonzero(grasp) - k))])
        print(f"[bridge] central frame has no hand/object; using nearest frame {k}")
    s_h, r_h, t_h = similarity_procrustes(gt_hand, hand_v[k])
    bridge_err = np.linalg.norm(s_h * gt_hand @ r_h.T + t_h - hand_v[k], axis=1).mean()
    print(f"[bridge] frame {k}: GT hand -> HaWoR hand, scale {s_h:.3f}, residual {bridge_err*1000:.1f} mm")
    # Only the rigid part is used to place the object: HaWoR's hand size is the metric
    # anchor, and the GT relative geometry is metric too, so a scale far from 1 would
    # be a bridge failure rather than information.
    place = lambda x: x @ r_h.T * s_h + t_h
    gt_obj_placed = place(gt_obj_v)
    obj_regions_placed = {r: place(p) for r, p in obj_regions.items()}

    # ── 2. transfer the object-side patches onto the reconstructed mesh ──
    rec_mesh = trimesh.Trimesh(mesh_v, mesh_f, process=False)
    rec_samples = trimesh.sample.sample_surface_even(rec_mesh, 3000, seed=0)[0]
    gt_samples = trimesh.sample.sample_surface_even(
        trimesh.Trimesh(gt_obj_placed, gt_obj_f, process=False), 3000, seed=0)[0]
    rec_k = rec_samples @ rot[k].T + trans[k]
    # MoGe and HaWoR disagree about depth by tens of centimetres, far outside the
    # registration's capture range. Scaling about the camera centre moves the object
    # along its viewing rays without changing its image, so it is the one correction
    # that is free to make before any fitting: bring the mesh to the depth of the
    # placed GT object, then let the local fit do the rest.
    c_rec, c_gt = rec_k.mean(0), gt_samples.mean(0)
    s0 = float(c_gt @ c_rec / (c_rec @ c_rec))
    print(f"[register] depth pre-scale about the camera: {s0:.3f} "
          f"(object centre {np.linalg.norm(c_rec):.3f} m -> {np.linalg.norm(s0 * c_rec):.3f} m)")
    region_points = {}
    if args.shape == "recgen":
        s_a, r_a, t_a, reg_loss = register_mesh_to_template(s0 * rec_k, gt_samples, device=device)
        print(f"[register] RecGen mesh -> placed GT object: size ratio {s_a:.3f}, "
              f"chamfer {np.sqrt(reg_loss/2)*1000:.1f} mm rms (capped)")
        if abs(np.log(s_a)) > np.log(1.3):
            print("[register] WARNING: the RecGen mesh and the GT object differ in size by "
                  f"{s_a:.2f}x at the hand's depth -- usually a partial mesh from an occluded "
                  "reference frame. Consider --shape epic.")
        # A placed-GT point p maps back into the canonical (reference-camera) mesh frame
        # by undoing the registration, the depth pre-scale, and the tracked pose at k.
        unregister = lambda p: ((p - t_a) @ r_a / (s_a * s0) - trans[k]) @ rot[k]
        for r, p in obj_regions_placed.items():
            if len(p) == 0:
                continue
            closest, _, _ = trimesh.proximity.closest_point(rec_mesh, unregister(p))
            region_points[r] = closest
        # Only the depth pre-scale seeds the global scale. The registration's size ratio
        # is a scale about the object's own centre; used as a scale about the camera it
        # would push the object back off the hand by the same factor.
        init_scale = s0
    else:
        # EPIC-Contact's own object: the full template, at its annotated real-world size
        # and in its annotated pose relative to the hand, carried into the camera frame by
        # the hand bridge. It replaces a RecGen mesh built from whatever part of the object
        # the reference frame showed. The tracker's motion is kept, with its translations
        # brought to the hand's depth scale by the same pre-scale.
        trans = trans * s0
        to_canonical_np = lambda p: (p - trans[k]) @ rot[k]
        mesh_v, mesh_f = to_canonical_np(gt_obj_placed), gt_obj_f
        rec_mesh = trimesh.Trimesh(mesh_v, mesh_f, process=False)
        rec_samples = trimesh.sample.sample_surface_even(rec_mesh, 3000, seed=0)[0]
        region_points = {r: to_canonical_np(p) for r, p in obj_regions_placed.items() if len(p)}
        s_a = 1.0
        init_scale = 1.0
        print(f"[shape] using the EPIC-Contact object: extents {np.round(rec_mesh.extents, 3)} m")
    active = [r for r in REGIONS if len(hand_regions[r]) and r in region_points]
    print("[labels] " + ", ".join(f"{r}: {len(hand_regions[r])} hand v / {len(region_points[r])} obj pts"
                                   for r in active))

    # ── 3. joint optimisation ──
    sdf, lo, hi = build_sdf_grid(mesh_v, mesh_f, args.grid)
    field = GridSDF(sdf, lo, hi, device)
    f32 = lambda a: torch.tensor(a, dtype=torch.float32, device=device)

    # The tracker's poses are what the image term compares against, whatever the start.
    R_track, T_track = f32(rot), f32(trans)
    if args.init == "grasp":
        # Stable-grasp start, the assumption EPIC-Contact's own per-frame labels are built
        # on: the object rides rigidly with the hand from its pose at frame k. The hand's
        # frame-to-frame rigid motion is a Kabsch fit over all its vertices; finger
        # articulation during a held grasp is small next to the wrist's motion.
        base_rot, base_trans = rot.copy(), trans.copy()
        for t in np.flatnonzero(hand_valid & window):
            _, a_rot, a_trans = similarity_procrustes(hand_v[k], hand_v[t])
            # similarity_procrustes also returns a scale; a rigid motion is wanted, so
            # refit the translation with unit scale.
            a_trans = hand_v[t].mean(0) - a_rot @ hand_v[k].mean(0)
            base_rot[t] = a_rot @ rot[k]
            base_trans[t] = a_rot @ trans[k] + a_trans / init_scale
        R0, T0 = f32(base_rot), f32(base_trans)
    else:
        R0, T0 = R_track, T_track
    H0 = f32(hand_v)
    both_t = f32(both.astype(float))
    grasp_t = f32(grasp.astype(float))
    obj_t = f32(obj_valid.astype(float))
    hand_t = f32(hand_valid.astype(float))
    conf_t = f32(np.clip(confidence / max(confidence[obj_valid].max(), 1e-6), 0.2, 1.0)) * obj_t

    centroid = f32(mesh_v.mean(0))
    probe = f32(rec_samples[np.random.default_rng(0).choice(len(rec_samples), 200, replace=False)])
    corners = f32(np.array(np.meshgrid(*[[mesh_v[:, i].min(), mesh_v[:, i].max()] for i in range(3)],
                                        indexing="ij")).reshape(3, -1).T)
    contact_idx = np.unique(np.concatenate([hand_regions[r] for r in active]))
    contact_t = torch.tensor(contact_idx, device=device)
    region_idx = {r: torch.tensor(hand_regions[r], device=device) for r in active}
    region_pts = {r: f32(region_points[r]) for r in active}
    # Penetration is checked on every other vertex: MANO's 778 are dense enough that
    # half of them bound the hand's volume, and it halves the cost.
    pen_idx = torch.arange(0, 778, 2, device=device)
    wrist = H0[:, :, :].mean(1)
    ray = wrist / torch.linalg.norm(wrist, dim=-1, keepdim=True)

    log_s = torch.tensor([np.log(init_scale)], dtype=torch.float32, device=device, requires_grad=True)
    w = torch.zeros(n, 3, device=device, requires_grad=True)
    # Per-frame depth start. One global scale only matches the hand at frame k: MoGe's
    # depth scale drifts, so elsewhere the object starts in front of or behind the hand,
    # and reaching the grasp from in front means passing through the fingers, which the
    # penetration term forbids. The clip is a stable grasp, so start every frame with
    # the object at the depth offset from the hand it has at frame k, slid along its
    # own viewing ray -- which leaves its image position untouched.
    with torch.no_grad():
        c_track = centroid @ R0.transpose(1, 2) + T0                 # (n, 3), before scale
        z_obj = c_track[:, 2] * init_scale
        z_hand = H0[:, contact_t].mean(1)[:, 2]
        target = z_obj[k] + (z_hand - z_hand[k])
        u_init = c_track * (target / z_obj - 1.0)[:, None]
        u_init = u_init * grasp_t[:, None]
    if args.init == "grasp":
        u_init = torch.zeros_like(u_init)       # the hand already put every frame in depth
    else:
        moved = (u_init.norm(dim=1) * init_scale)[grasp_t > 0].median()
        print(f"[init] per-frame depth start: moved the object by median "
              f"{float(moved) * 1000:.0f} mm along its viewing ray")
    u = u_init.clone().requires_grad_(True)
    m = torch.zeros(n, 1, device=device, requires_grad=True)

    def project(x):
        return torch.stack([fu * x[..., 0] / x[..., 2].clamp(min=1e-4) + pu,
                            fv * x[..., 1] / x[..., 2].clamp(min=1e-4) + pv], -1)

    track_px = project(probe @ R_track.transpose(1, 2) + T_track[:, None])
    track_spread = ((track_px - track_px.mean(1, keepdim=True)) ** 2).sum(-1).mean(1, keepdim=True).sqrt()[..., None]

    def object_transform():
        """Canonical point q -> camera: s * (dR (R q + T - c) + c + u)."""
        s = torch.exp(log_s)
        dR = rotvec_to_matrix(w)
        c = centroid @ R0.transpose(1, 2) + T0          # (n, 3) tracked centroid
        rotation = dR @ R0
        translation = (T0 - c)[:, None, :] @ dR.transpose(1, 2)
        translation = translation[:, 0] + c + u
        return s, rotation, translation                 # x = s * (rotation q + translation)

    def to_canonical(x, s, rotation, translation):
        return ((x / s - translation[:, None]) @ rotation)

    def energies(sigma):
        s, rotation, translation = object_transform()
        hands_now = H0 + m[:, None] * ray[:, None]
        # Distances are measured in the camera frame: the canonical SDF times s.
        q_contact = to_canonical(hands_now[:, contact_t], s, rotation, translation)
        d_contact = field(q_contact) * s
        touch = (d_contact ** 2 / (d_contact ** 2 + sigma ** 2)).mean(1)
        q_pen = to_canonical(hands_now[:, pen_idx], s, rotation, translation)
        penetrate = (torch.relu(-field(q_pen) * s) ** 2).sum(1) / 1e-4
        region = 0.0
        for r in active:
            q = to_canonical(hands_now[:, region_idx[r]], s, rotation, translation)
            d = torch.cdist(q, region_pts[r][None].expand(n, -1, -1)).min(-1).values * s
            region = region + (d ** 2 / (d ** 2 + max(sigma, 0.02) ** 2)).mean(1)
        region = region / max(len(active), 1)
        x_probe = s * (probe @ rotation.transpose(1, 2) + translation[:, None])
        # The tracker's 2D position and 2D shape are image evidence; its apparent SIZE is
        # not -- it follows from the per-frame MoGe depth the points were lifted with, and
        # that depth's scale drifts from frame to frame. Comparing size-normalised
        # projections leaves each frame's depth to the contact.
        px = project(x_probe)
        centre_new, centre_track = px.mean(1, keepdim=True), track_px.mean(1, keepdim=True)
        spread_new = ((px - centre_new) ** 2).sum(-1).mean(1, keepdim=True).sqrt()[..., None]
        shape_new = (px - centre_new) / spread_new.clamp(min=1e-6) * track_spread
        shape_track = track_px - centre_track
        image = (((centre_new - centre_track) ** 2).sum(-1)[:, 0]
                 + ((shape_new - shape_track) ** 2).sum(-1).mean(1)) / args.image_sigma ** 2
        anchors = s * (corners @ rotation.transpose(1, 2) + translation[:, None])
        acc_obj = ((anchors[2:] - 2 * anchors[1:-1] + anchors[:-2]) ** 2).sum(-1).mean(1)
        acc_hand = ((m[2:] - 2 * m[1:-1] + m[:-2]) ** 2).sum(-1)
        return dict(touch=touch, penetrate=penetrate, region=region, image=image,
                    prior=(m[:, 0] / args.hand_sigma) ** 2, acc_obj=acc_obj, acc_hand=acc_hand,
                    d_contact=d_contact, s=s)

    def total(e):
        per_frame = both_t * args.w_penetrate * e["penetrate"] + \
            grasp_t * (args.w_touch * e["touch"] + args.w_region * e["region"])
        per_frame = per_frame + args.w_image * conf_t * e["image"] + args.w_prior * hand_t * e["prior"]
        smooth = args.w_smooth * (e["acc_obj"].mean() / 1e-6 * 1e-2 + e["acc_hand"].mean() / 1e-6 * 1e-2)
        return per_frame.sum() / both_t.sum() + smooth

    with torch.no_grad():
        e0 = energies(args.touch_sigma)

    # Scale and rotation live on very different numeric scales from translation.
    opt = torch.optim.Adam([
        {"params": [log_s], "lr": 5e-3},
        {"params": [w], "lr": 5e-3},
        {"params": [u, m], "lr": 1e-3},
    ])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.iters)
    for it in range(args.iters):
        # Graduated non-convexity: a Geman-McClure loss at the final 1 cm scale has no
        # gradient for a vertex that starts 10 cm off the surface, and the frames where
        # MoGe's depth drifted start exactly that far. Start wide and shrink.
        progress = min(it / (0.6 * args.iters), 1.0)
        sigma = args.touch_sigma_start * (args.touch_sigma / args.touch_sigma_start) ** progress
        e = energies(sigma)
        loss = total(e)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if it % 250 == 0 or it == args.iters - 1:
            print(f"[opt {it:4d}] loss {loss.item():.4f}  scale {e['s'].item():.3f}  "
                  f"touch {e['touch'][grasp].mean().item():.3f}  "
                  f"pen {e['penetrate'][grasp].mean().item():.3f}  "
                  f"region {float(e['region'][grasp].mean()):.3f}  "
                  f"image {e['image'][obj_valid].mean().item():.3f}", flush=True)

    # ── 4. write ──
    with torch.no_grad():
        s, rotation, translation = object_transform()
        s = float(s)
        hands_final = (H0 + m[:, None] * ray[:, None]).cpu().numpy()
        rotation = rotation.cpu().numpy().astype(np.float64)
        translation = translation.cpu().numpy().astype(np.float64)

    # x = s (R q + t) = R (s q) + s t: the scaled mesh carries s, the trajectory carries s t.
    mesh_out = os.path.join(args.out_dir, "posed_mesh_contact.obj")
    trimesh.Trimesh(mesh_v * s, mesh_f, process=False).export(mesh_out)
    new_rot = np.where(obj_valid[:, None, None], rotation, rot)
    new_trans = np.where(obj_valid[:, None], translation * s, trans * s)

    traj_out = os.path.join(args.out_dir, "object_traj_contact.npz")
    extra = {k: poses[k] for k in poses.files if k not in ("rotation", "translation")}
    np.savez(traj_out, **{**extra, "rotation": new_rot, "translation": new_trans,
                          "contact_scale": s})

    hands_out = os.path.join(args.out_dir, "all_hand_meshes_contact.npz")
    hand_data = {k: hands[k] for k in hands.files}
    hand_data[f"{args.side}_vertices"] = np.where(
        hand_valid[:, None, None], hands_final, hand_data[f"{args.side}_vertices"][:n]).astype(np.float32)
    shift = (m.detach().cpu().numpy() * ray.cpu().numpy())
    for key in (f"{args.side}_joints", f"{args.side}_trans"):
        if key in hand_data:
            arr = hand_data[key][:n].astype(np.float64)
            hand_data[key] = (arr + (shift[:, None] if arr.ndim == 3 else shift)).astype(np.float32)
    np.savez(hands_out, **hand_data)

    labels_out = os.path.join(args.out_dir, "contact_labels.npz")
    np.savez(labels_out,
             **{f"hand_{r}": hand_regions[r] for r in active},
             **{f"object_{r}": region_points[r] * s for r in active},
             bridge_frame=k, bridge_scale=s_h, bridge_residual=bridge_err,
             depth_prescale=s0, register_scale=s_a, object_scale=s)

    summary = {
        "frames": int(n), "frames_hand_and_object": int(both.sum()),
        "contact_frames": [int(lo), int(hi)], "frames_in_contact": int(grasp.sum()),
        "bridge_frame": k, "bridge_hand_scale": s_h, "bridge_residual_mm": bridge_err * 1000,
        "depth_prescale": s0, "register_scale": s_a, "object_scale": s,
        "hand_depth_shift_mm": {"median_abs": float(np.median(np.abs(m.detach().cpu().numpy())) * 1000),
                                "max_abs": float(np.abs(m.detach().cpu().numpy()).max() * 1000)},
        "energy_before": {k: float(e0[k][grasp].mean()) for k in ("touch", "penetrate", "region")},
        "energy_after": {k: float(e[k][grasp].mean()) for k in ("touch", "penetrate", "region")},
    }
    json.dump(summary, open(os.path.join(args.out_dir, "refine_summary.json"), "w"), indent=1)
    print(f"Wrote {mesh_out}\n      {traj_out}\n      {hands_out}\n      {labels_out}")
    print(f"  object scale {s:.3f}; hand depth shift median {summary['hand_depth_shift_mm']['median_abs']:.1f} mm")


if __name__ == "__main__":
    main()
