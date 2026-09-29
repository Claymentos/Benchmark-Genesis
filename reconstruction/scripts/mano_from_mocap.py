#!/usr/bin/env python3
"""
MANO hand meshes, both hands, fitted to mocap joints -- and split into rigid parts a
simulator can carry.

TouchAnything measures 21 joints per hand (Rokoko gloves on Vive trackers), which
touchanything_hands.py places in the reference camera. Joints are not a surface, so
this fits MANO to them: per frame the global orientation, translation and 45 pose
parameters, and one shape shared by the whole clip (a hand does not change size),
with a light pose prior and temporal smoothness. The fitted joints are MANO's 16
plus the five fingertips read off the mesh (HaMeR's vertex ids), in the
MediaPipe/OpenPose order the mocap uses.

A simulator moves rigid bodies, not skinned meshes, so the hand is also cut into the
16 parts MANO skins it with (each vertex to its heaviest joint) and each part gets
the rigid transform of its joint per frame. Carried by those transforms, the parts
reproduce the skinned mesh up to the blending at the joints -- a few millimetres of
seam, reported below.

Writes `--out` (.npz), per side:
  {side}_vertices          (T, 778, 3)   skinned mesh, reference camera frame
  {side}_joints            (T, 21, 3)    fitted joints, same order as the mocap
  {side}_faces             (F, 3)
  {side}_rest_vertices     (778, 3)      the shaped hand in MANO's rest frame
  {side}_part              (778,)        joint each vertex follows
  {side}_part_transforms   (T, 16, 4, 4) rest frame -> reference camera, per part
  {side}_valid             (T,)
and one mesh per part, `{side}_part_{k:02d}.obj`, in --parts-dir (rest frame).

Run in the `ecfit` env (smplx + CUDA torch):
    python mano_from_mocap.py --joints mocap_joints.npz --out mano_hands.npz --parts-dir mano_parts
"""

import argparse
import os

import numpy as np
import torch

MANO_DIR = os.path.expanduser("~/.cache/mano_models")
# MANO's joints in smplx order are wrist, index 1-3, middle 1-3, pinky 1-3, ring 1-3,
# thumb 1-3; the tips are vertices. Index 16.. below refers to the tips appended.
TIPS = [745, 317, 445, 556, 673]  # thumb, index, middle, ring, pinky
TO_OPENPOSE = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]


def parse_args():
    parser = argparse.ArgumentParser(description="Fit MANO to mocap joints, both hands")
    parser.add_argument("--joints", required=True,
                        help="npz with {side}_joints (T, 21, 3) and {side}_valid, e.g. from "
                        "touchanything_hands.py")
    parser.add_argument("--out", required=True)
    parser.add_argument("--parts-dir", required=True, help="Where the per-part .obj meshes go")
    parser.add_argument("--sides", nargs="+", default=["left", "right"])
    parser.add_argument("--iters", type=int, default=1500)
    # The joint term is in cm^2, so these weigh a squared radian (or shape unit) against
    # a squared centimetre of joint error.
    parser.add_argument("--w-pose", type=float, default=1e-3, help="Pose prior (towards the flat hand)")
    parser.add_argument("--w-shape", type=float, default=1e-2)
    parser.add_argument("--w-smooth", type=float, default=1.0,
                        help="Frame-to-frame change in pose and orientation")
    parser.add_argument("--mano-dir", default=MANO_DIR)
    return parser.parse_args()


def load_model(side, mano_dir, batch, device):
    import smplx

    model = smplx.MANO(mano_dir, is_rhand=(side == "right"), use_pca=False,
                       flat_hand_mean=True, batch_size=batch).to(device)
    if side == "left":
        # smplx's MANO_LEFT ships the right hand's first shape direction; the usual fix
        # (HaMeR, WiLoR) negates its x component, or the left hand widens mirrored.
        right = smplx.MANO(mano_dir, is_rhand=True, use_pca=False).to(device)
        if torch.sum(torch.abs(model.shapedirs[:, 0, :] - right.shapedirs[:, 0, :])) < 1:
            model.shapedirs[:, 0, :] *= -1
    return model


def kabsch(source, target):
    cs, ct = source.mean(0), target.mean(0)
    u, _, vt = np.linalg.svd((source - cs).T @ (target - ct))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1, 1, d]) @ u.T
    return rotation, ct - rotation @ cs


def fit_side(side, targets, valid, args, device):
    from scipy.spatial.transform import Rotation
    from smplx.lbs import batch_rigid_transform, batch_rodrigues, vertices2joints

    n_frames = len(targets)
    model = load_model(side, args.mano_dir, n_frames, device)

    def forward(orient, pose, betas, transl):
        out = model(global_orient=orient, hand_pose=pose, betas=betas.expand(n_frames, -1),
                    transl=transl)
        joints = torch.cat([out.joints, out.vertices[:, TIPS]], dim=1)[:, TO_OPENPOSE]
        return out.vertices, joints

    # Initialise orientation and translation by aligning the flat hand's palm (wrist and
    # the four finger bases) onto the target's, per frame.
    zeros = torch.zeros(n_frames, 3, device=device)
    _, rest = forward(zeros, torch.zeros(n_frames, 45, device=device),
                      torch.zeros(1, 10, device=device), zeros)
    rest = rest[0].detach().cpu().numpy()
    palm = [0, 5, 9, 13, 17]
    orient0, transl0 = np.zeros((n_frames, 3)), np.zeros((n_frames, 3))
    for t in range(n_frames):
        rotation, _ = kabsch(rest[palm] - rest[0], targets[t, palm] - targets[t, 0])
        orient0[t] = Rotation.from_matrix(rotation).as_rotvec()
        # MANO rotates about its root joint, so the wrist lands at rest[0] + transl
        # whatever the orientation.
        transl0[t] = targets[t, 0] - rest[0]

    orient = torch.tensor(orient0, dtype=torch.float32, device=device, requires_grad=True)
    transl = torch.tensor(transl0, dtype=torch.float32, device=device, requires_grad=True)
    pose = torch.zeros(n_frames, 45, device=device, requires_grad=True)
    betas = torch.zeros(1, 10, device=device, requires_grad=True)
    target = torch.tensor(targets, dtype=torch.float32, device=device)
    weight = torch.tensor(valid, dtype=torch.float32, device=device)[:, None]

    optimiser = torch.optim.Adam([orient, transl, pose, betas], lr=0.02)
    schedule = torch.optim.lr_scheduler.MultiStepLR(
        optimiser, [int(args.iters * 0.6), int(args.iters * 0.85)], 0.3)
    for step in range(args.iters):
        optimiser.zero_grad()
        _, joints = forward(orient, pose, betas, transl)
        data = 1e4 * (((joints - target) ** 2).sum(-1) * weight).sum() / (weight.sum() * 21)
        smooth = ((pose[1:] - pose[:-1]) ** 2).sum(-1).mean() + ((orient[1:] - orient[:-1]) ** 2).sum(-1).mean()
        loss = (data + args.w_pose * (pose ** 2).sum(-1).mean() + args.w_shape * (betas ** 2).sum()
                + args.w_smooth * smooth)
        loss.backward()
        optimiser.step()
        schedule.step()

    with torch.no_grad():
        vertices, joints = forward(orient, pose, betas, transl)
        error = torch.linalg.norm(joints - target, dim=-1).cpu().numpy()[valid]
        print(f"[mano] {side}: joint error median {np.median(error) * 1000:.1f} mm, "
              f"p90 {np.percentile(error, 90) * 1000:.1f} mm, per joint (mm) "
              f"{np.round(np.median(error, axis=0) * 1000).astype(int).tolist()}")
        print(f"[mano] {side}: betas {np.round(betas[0].cpu().numpy(), 2).tolist()}")

        # The rigid transform of every joint, the same one the skinning blends. smplx
        # rotates about the root and then adds transl, so transl goes on top.
        shaped = model.v_template + torch.einsum("vcb,b->vc", model.shapedirs, betas[0])
        rest_joints = vertices2joints(model.J_regressor, shaped[None]).expand(n_frames, -1, -1)
        full_pose = torch.cat([orient, pose], dim=1).reshape(-1, 3)
        rotations = batch_rodrigues(full_pose).reshape(n_frames, 16, 3, 3)
        _, transforms = batch_rigid_transform(rotations, rest_joints.contiguous(), model.parents)
        transforms = transforms.clone()
        transforms[:, :, :3, 3] += transl[:, None]

        part = model.lbs_weights.argmax(dim=1)
        homogeneous = torch.cat([shaped, torch.ones(778, 1, device=device)], dim=1)
        rigid = torch.einsum("tvij,vj->tvi", transforms[:, part], homogeneous)[..., :3]
        seam = torch.linalg.norm(rigid - vertices, dim=-1).cpu().numpy()[valid]
        print(f"[mano] {side}: rigid parts vs skinned mesh, median {np.median(seam) * 1000:.1f} mm, "
              f"p99 {np.percentile(seam, 99) * 1000:.1f} mm")

    return dict(
        vertices=vertices.cpu().numpy(),
        joints=joints.cpu().numpy(),
        faces=np.asarray(model.faces, dtype=np.int64),
        rest_vertices=shaped.detach().cpu().numpy(),
        part=part.cpu().numpy(),
        part_transforms=transforms.cpu().numpy(),
        valid=valid,
    )


def write_parts(side, fit, parts_dir):
    """One closed-enough mesh per part: the faces whose vertices mostly follow it."""
    import trimesh

    os.makedirs(parts_dir, exist_ok=True)
    faces, part = fit["faces"], fit["part"]
    face_part = np.array([np.bincount(part[f], minlength=16).argmax() for f in faces])
    for k in range(16):
        chosen = faces[face_part == k]
        mesh = trimesh.Trimesh(fit["rest_vertices"], chosen, process=False)
        mesh.remove_unreferenced_vertices()
        mesh.export(os.path.join(parts_dir, f"{side}_part_{k:02d}.obj"))


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(args.joints)
    out = {}
    for side in args.sides:
        joints = data[f"{side}_joints"].astype(np.float64)
        valid = data[f"{side}_valid"].astype(bool) & np.isfinite(joints).all(axis=(1, 2))
        fit = fit_side(side, np.nan_to_num(joints), valid, args, device)
        write_parts(side, fit, args.parts_dir)
        out.update({f"{side}_{key}": value for key, value in fit.items()})
    np.savez(args.out, **out)
    print(f"Wrote {args.out} and the part meshes in {args.parts_dir}")


if __name__ == "__main__":
    main()
