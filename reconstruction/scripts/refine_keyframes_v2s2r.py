#!/usr/bin/env python3
"""
Keyframe refinement, the way Video2Sim2Real does it.

From "Video2Sim2Real: Full-Stack Autonomous Dexterous Skill Acquisition from a Single
Human Video" (arXiv 2606.08828), appendix A.2.2. The retargeted trajectory is not
corrected everywhere; it is corrected at the object-centric keyframes
(extract_keyframes.py) and the corrections are then blended back into the rest of it.
The premise is that those frames are where the demonstration's effect on the world is
decided, and everywhere else the human's motion is already the thing worth copying.

For a grasping task the contact keyframe is refined by:

  1 the coarse hand-to-object transform the demonstration itself gives at that frame;
  2 a geometry-based clearance correction, the paper's adaptive top-grasp offset
    d_top = clip(rho_top * h_obj, d_min, d_max), applied along the table normal when
    the grasp comes from above;
  3 grasp candidates from Lightning Grasp (keyframe_grasps.py), which is the same
    generator the paper uses;
  4 filtering on contact-point continuity -- how far a candidate sits from the
    demonstrated hand -- and on height above the table;
  5 simulation-based selection: each surviving candidate is replayed as the hand
    approaches, adopts its pre-grasp, closes, lifts, and is shaken along the three
    Cartesian axes, and is accepted if the object stays lifted by more than
    tau_lift = 6 cm.

The accepted configuration is then written into the trajectory at the keyframe and
faded out with the paper's quintic smoothstep

    alpha(s) = 6 s^5 - 15 s^4 + 10 s^3,   alpha'(0) = alpha'(1) = 0,

so the correction arrives and leaves with zero velocity and the demonstration is
untouched outside the blend.

Two departures from the paper, both because it does not specify them and inventing
would be worse than saying so:

  * the detachment keyframe is anchored, not corrected. The paper lists "hand pose for
    object release" without giving a rule, so the demonstrated pose is pinned there and
    the blend is anchored to it rather than to a pose this script made up.
  * the interaction keyframe needs a target object, which these single-object clips do
    not have, so it never fires here.

The paper also solves exact IK at each keyframe and applies a damped-least-squares
task-space correction so an arm can hold the palm on target. This pipeline replays a
free-floating hand, so the palm is set directly and there is no arm residual to solve.

Run in the `genesis` env:
    python refine_keyframes_v2s2r.py --traj wuji_traj_optimized.npz \\
        --keyframes keyframes.json --candidates keyframe_grasps.npz \\
        --object-poses cup_traj_points.npz --mesh posed_mesh.obj \\
        --ground-plane ground_plane.json --out wuji_traj_v2s2r.npz
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description="Video2Sim2Real keyframe refinement")
    parser.add_argument("--traj", required=True, help="Retargeted (or optimized) trajectory .npz")
    parser.add_argument("--keyframes", required=True, help="keyframes.json")
    parser.add_argument("--candidates", required=True, help="keyframe_grasps.npz")
    parser.add_argument("--object-poses", required=True)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--ground-plane", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--wuji-dir", default=os.environ.get(
        "WUJI_RETARGETING_DIR",
        os.path.expanduser("~/wuji-ego-mint/eval/simulate/wuji-retargeting")))
    parser.add_argument("--urdf", default=None)

    # The paper's fixed parameters.
    parser.add_argument("--lift-threshold", type=float, default=0.06,
                        help="tau_lift: the object must stay this far up, metres")
    parser.add_argument("--top-ratio", type=float, default=0.15,
                        help="rho_top in d_top = clip(rho_top * h_obj, d_min, d_max)")
    parser.add_argument("--top-min", type=float, default=0.005, help="d_min, metres")
    parser.add_argument("--top-max", type=float, default=0.03, help="d_max, metres")

    parser.add_argument("--max-candidates", type=int, default=8,
                        help="Candidates carried into the simulation test, best first. "
                             "Each one costs a physics replay")
    parser.add_argument("--continuity", type=float, default=0.25,
                        help="Reject a candidate whose palm is further than this from the "
                             "demonstrated one at the keyframe, metres")
    parser.add_argument("--blend", type=int, default=30,
                        help="Frames the keyframe correction is faded in and out over")
    parser.add_argument("--shake-amplitude", type=float, default=0.02, help="metres")
    parser.add_argument("--shake-frames", type=int, default=18, help="Frames per axis")
    parser.add_argument("--lift-height", type=float, default=0.10,
                        help="How far the validation lifts the object, metres")
    parser.add_argument("--keep-failures", action="store_true",
                        help="Anchor the best-ranked candidate even if none passes the "
                             "shake test, instead of leaving the keyframe as demonstrated")
    return parser.parse_args()


def smoothstep(s):
    """The paper's quintic: zero velocity at both ends."""
    s = np.clip(s, 0.0, 1.0)
    return 6 * s ** 5 - 15 * s ** 4 + 10 * s ** 3


def blend_weights(n_frames, keyframe, blend):
    """1 at the keyframe, fading to 0 over `blend` frames each side."""
    index = np.arange(n_frames)
    distance = np.abs(index - keyframe).astype(float)
    return smoothstep(1.0 - np.clip(distance / max(blend, 1), 0.0, 1.0))


def plane_from(path):
    with open(path) as handle:
        plane = json.load(handle)
    normal = np.asarray(plane["normal_camera"], dtype=float)
    norm = np.linalg.norm(normal)
    return normal / norm, float(plane["offset"]) / norm


def top_grasp_clearance(args, normal, palm_rotation, object_height):
    """d_top = clip(rho_top * h_obj, d_min, d_max), applied only to a grasp from above.

    "From above" is decided by the palm's own closing direction: the Wuji palm faces
    +X in its link frame, so a grasp whose palm normal points down the table normal is
    a top grasp and gets lifted off the object by the clearance; a side grasp does not.
    """
    facing = palm_rotation @ np.array([1.0, 0.0, 0.0])
    downward = float(facing @ -normal)
    if downward <= 0.5:
        return 0.0, downward
    return float(np.clip(args.top_ratio * object_height, args.top_min, args.top_max)), downward


def shake_trajectory(args, source, palm_rotation, palm_position, grasp_qpos, normal, fps):
    """Approach, pre-grasp, close, lift, then shake on three axes.

    The paper's validation: "each grasp is replayed as hand approaches, adopts its
    pre-grasp, closes, lifts object, and shaken along three Cartesian axes". Written
    out as an ordinary trajectory so the existing Genesis replay can run it.
    """
    approach, close, lift = 15, 12, 20
    axes = np.eye(3)
    n_frames = approach + close + lift + 3 * args.shake_frames

    # Open hand for the approach: the grasp's own closure, backed off.
    open_qpos = grasp_qpos * 0.25
    # Backed off along the palm's approach direction.
    retreat = palm_rotation @ np.array([1.0, 0.0, 0.0]) * -0.09

    positions = np.zeros((n_frames, 3))
    qpos = np.zeros((n_frames, len(grasp_qpos)))
    cursor = 0
    for step in range(approach):
        weight = smoothstep(step / max(approach - 1, 1))
        positions[cursor] = palm_position + retreat * (1.0 - weight)
        qpos[cursor] = open_qpos
        cursor += 1
    for step in range(close):
        weight = smoothstep(step / max(close - 1, 1))
        positions[cursor] = palm_position
        qpos[cursor] = open_qpos + weight * (grasp_qpos - open_qpos)
        cursor += 1
    for step in range(lift):
        weight = smoothstep(step / max(lift - 1, 1))
        positions[cursor] = palm_position + normal * args.lift_height * weight
        qpos[cursor] = grasp_qpos
        cursor += 1
    top = palm_position + normal * args.lift_height
    for axis in axes:
        for step in range(args.shake_frames):
            phase = 2 * np.pi * step / max(args.shake_frames - 1, 1)
            positions[cursor] = top + axis * args.shake_amplitude * np.sin(phase)
            qpos[cursor] = grasp_qpos
            cursor += 1

    quaternion = Rotation.from_matrix(palm_rotation).as_quat()
    fields = {key: source[key] for key in ("joint_names", "side")}
    fields["qpos"] = qpos
    fields["wrist_pos"] = positions
    fields["wrist_quat"] = np.tile(np.r_[quaternion[3], quaternion[:3]], (n_frames, 1))
    fields["valid"] = np.ones(n_frames, dtype=bool)
    fields["fps"] = fps
    # The object is handed to physics as the fingers start to close.
    return fields, approach


def run_shake(args, fields, release, object_index):
    """Replay the shake in Genesis and report how far the object stayed up."""
    replay = os.path.join(HERE, "replay_wuji_genesis.py")
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as handle:
        traj_path = handle.name
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as handle:
        sim_path = handle.name
    np.savez(traj_path, **fields)

    # The object starts the test at the pose it had at the keyframe.
    poses = np.load(args.object_poses)
    n_frames = len(fields["wrist_pos"])
    held = dict(
        rotation=np.tile(poses["rotation"][object_index], (n_frames, 1, 1)),
        translation=np.tile(poses["translation"][object_index], (n_frames, 1)),
        valid=np.ones(n_frames, dtype=bool),
        ref_frame=0,
    )
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as handle:
        object_path = handle.name
    np.savez(object_path, **held)

    command = [
        sys.executable, replay, "--traj", traj_path, "--object-mesh", args.mesh,
        "--object-poses", object_path, "--ground-plane", args.ground_plane,
        "--wuji-dir", args.wuji_dir, "--physics", "--sim-out", sim_path,
        "--release-frame", str(release),
    ]
    if args.urdf:
        command += ["--urdf", args.urdf]
    result = subprocess.run(command, capture_output=True, text=True)

    lifted, contacts = -np.inf, 0
    if os.path.exists(sim_path) and os.path.getsize(sim_path) > 0:
        simulated = np.load(sim_path)
        position = simulated["position"]
        # Positions are recorded from frame_range[0], not from frame 0, so the release
        # frame has to be shifted into that indexing before it is used as the baseline.
        start = int(simulated["frame_range"][0]) if "frame_range" in simulated else 0
        baseline = min(max(release - start, 0), len(position) - 1)
        # The replay works in a z-up world, so height is simply z, measured against
        # where the object started the test.
        lifted = float(position[-1, 2] - position[baseline, 2])
        contacts = int(np.sum(simulated["contacts"] > 0)) if "contacts" in simulated else 0
    else:
        tail = "\n".join(result.stderr.strip().splitlines()[-2:])
        print(f"[v2s2r]     replay failed: {tail}")

    for path in (traj_path, sim_path, object_path):
        if os.path.exists(path):
            os.unlink(path)
    return lifted, contacts


def main():
    args = parse_args()
    source = np.load(args.traj, allow_pickle=True)
    fields = {key: source[key] for key in source.files}
    n_frames = len(fields["qpos"])
    fps = float(fields["fps"]) if "fps" in fields else 30.0

    with open(args.keyframes) as handle:
        keyframes = json.load(handle)["keyframes"]
    contact = keyframes.get("contact")
    if contact is None:
        raise SystemExit("no contact keyframe; nothing to refine against")

    normal, offset = plane_from(args.ground_plane)
    import trimesh

    vertices = np.asarray(trimesh.load(args.mesh, process=False).vertices)
    object_height = float(np.ptp(vertices @ normal))

    data = np.load(args.candidates, allow_pickle=True)
    frames = data["frames"].tolist()
    if contact not in frames:
        raise SystemExit(f"the candidates were not generated at the contact keyframe "
                         f"{contact} (they cover {frames})")
    slot = frames.index(contact)
    rotations = data["candidate_rotation"][slot]
    positions = data["candidate_pos"][slot]
    qpos = data["candidate_qpos"][slot]

    demonstrated = fields["wrist_pos"][contact]
    print(f"[v2s2r] contact keyframe {contact}, {len(rotations)} candidates, "
          f"object {100 * object_height:.1f} cm tall")

    # ── filters: contact-point continuity, then height above the table ──
    gap = np.linalg.norm(positions - demonstrated, axis=1)
    height = positions @ normal + offset
    keep = (gap <= args.continuity) & (height > 0)
    print(f"[v2s2r] continuity (<= {100 * args.continuity:.0f} cm from the demonstrated "
          f"palm) and height: {int(keep.sum())}/{len(keep)} survive")
    if not keep.any():
        raise SystemExit("no candidate survived filtering; loosen --continuity")

    order = np.flatnonzero(keep)[np.argsort(gap[keep])][: args.max_candidates]

    # ── simulation-based selection: lift and shake ──
    chosen, report = None, []
    for rank, index in enumerate(order):
        clearance, downward = top_grasp_clearance(args, normal, rotations[index], object_height)
        position = positions[index] + normal * clearance
        test, release = shake_trajectory(
            args, fields, rotations[index], position, qpos[index], normal, fps
        )
        lifted, contacts = run_shake(args, test, release, contact)
        passed = lifted > args.lift_threshold
        report.append((int(index), float(gap[index]), clearance, lifted, contacts, passed))
        print(f"[v2s2r]   candidate {rank + 1}/{len(order)}: {100 * gap[index]:5.1f} cm from "
              f"the demonstration, {'top' if clearance else 'side'} grasp "
              f"(clearance {100 * clearance:.1f} cm), held {100 * lifted:6.1f} cm, "
              f"{contacts} contact frames -> {'PASS' if passed else 'fail'}")
        if passed:
            chosen = (index, position, rotations[index], qpos[index])
            break

    if chosen is None:
        print(f"[v2s2r] no candidate kept the object above {100 * args.lift_threshold:.0f} cm")
        if not args.keep_failures:
            print("[v2s2r] leaving the contact keyframe as demonstrated "
                  "(pass --keep-failures to anchor the best-ranked one anyway)")
            np.savez(args.out, **fields, v2s2r_report=np.array(report, dtype=object),
                     v2s2r_applied=False)
            print(f"Wrote {args.out}")
            return
        index = order[0]
        clearance, _ = top_grasp_clearance(args, normal, rotations[index], object_height)
        chosen = (index, positions[index] + normal * clearance, rotations[index], qpos[index])

    index, position, rotation, grasp = chosen

    # ── anchor the correction with the quintic smoothstep ──
    weights = blend_weights(n_frames, contact, args.blend)
    detachment = keyframes.get("detachment")
    if detachment is not None:
        # The demonstration is pinned at detachment rather than corrected, so the blend
        # is not allowed to reach past it.
        weights[detachment:] = 0.0

    current = Rotation.from_quat(fields["wrist_quat"][:, [1, 2, 3, 0]])
    target = Rotation.from_matrix(rotation)
    turn = (current.inv() * target).as_rotvec()
    blended = current * Rotation.from_rotvec(turn * weights[:, None])
    quaternion = blended.as_quat()

    fields["wrist_pos"] = fields["wrist_pos"] + weights[:, None] * (position - fields["wrist_pos"])
    fields["wrist_quat"] = np.c_[quaternion[:, 3], quaternion[:, :3]]
    fields["qpos"] = fields["qpos"] + weights[:, None] * (grasp - fields["qpos"])

    moved = np.linalg.norm(fields["wrist_pos"] - source["wrist_pos"], axis=1)
    joints = np.abs(fields["qpos"] - source["qpos"])
    print(f"[v2s2r] anchored candidate {index} at frame {contact}, blended over "
          f"{args.blend} frames either side")
    print(f"[v2s2r] trajectory changed: wrist by up to {100 * moved.max():.2f} cm, "
          f"joints by up to {np.degrees(joints.max()):.1f} deg, over "
          f"{int((weights > 1e-3).sum())} frames")

    fields["v2s2r_keyframe"] = contact
    fields["v2s2r_candidate"] = index
    fields["v2s2r_weights"] = weights
    fields["v2s2r_report"] = np.array(report, dtype=object)
    fields["v2s2r_applied"] = True
    np.savez(args.out, **fields)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
