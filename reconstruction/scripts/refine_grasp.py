#!/usr/bin/env python3
"""
Correct a reconstructed grasp until it actually holds the object.

A retargeted hand lands close to the object but rarely on it: a centimetre of wrist
error, or fingers that stop a few degrees short of closing, is invisible in a
kinematic replay and fatal in a physical one -- the object is nudged instead of
gripped, and the hand finishes the trajectory empty. This refines the grasp until the
fingers make contact, and leaves the rest of the trajectory alone.

Two stages, because they fail differently:

  1 contact fit.  Geometry only, no simulation. A constant wrist offset in the hand's
    own frame, plus a closure angle per finger, is fitted so the engaged fingertips
    sit on the object's surface rather than short of it or through it. Seconds to run,
    deterministic, and it fixes the common case where the whole hand is simply offset.

  2 physics polish.  The contact fit does not know about gravity, friction or mass, so
    a grasp that touches can still slip. The grasp window is replayed under --physics
    and scored by how far the object moves relative to the wrist -- a held object moves
    with it -- and a short coordinate descent walks the residual downhill. Costly, so
    it runs on the window rather than the clip, and can be skipped.

Both stages are anchored to the demonstration: deviation from the reconstructed pose
is penalised, the correction applies only inside the grasp window, and it is blended
in and out at the edges so the approach and the retreat stay as reconstructed. The aim
is the human's grasp, corrected enough to hold -- not the most stable grasp available.

Run in the `superdex` env:
    python refine_grasp.py --traj wuji_traj.npz --object-mesh posed_mesh.obj \\
        --object-poses cup_traj.npz --out wuji_traj_refined.npz
"""

import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from lock_object_to_hand import runs_of
from replay_wuji_superdex import camera_to_world, dof_index_map, to_transform

DEFAULT_WUJI_DIR = os.environ.get(
    "WUJI_RETARGETING_DIR",
    os.path.expanduser("~/wuji-ego-mint/eval/simulate/wuji-retargeting"),
)
# SuperDex keeps the URDF's fixed tip joints as real links, unlike Genesis, which
# merged them into their parents and forced the offsets to be hardcoded. The tips are
# addressed by name and the table is gone.
FINGERS = (1, 2, 3, 4, 5)


def urdf_joint_limits(urdf, joint_names):
    """(lower, upper) per joint, in radians, straight from the URDF."""
    import xml.etree.ElementTree as ElementTree

    root = ElementTree.parse(urdf).getroot()
    found = {}
    for joint in root.iter("joint"):
        limit = joint.find("limit")
        if limit is not None and limit.get("lower") is not None:
            found[joint.get("name")] = (float(limit.get("lower")), float(limit.get("upper")))
    missing = [name for name in joint_names if name not in found]
    if missing:
        raise SystemExit(f"no limits in {urdf} for: {missing[:5]}")
    return np.array([found[name] for name in joint_names], dtype=float)


def parse_args():
    parser = argparse.ArgumentParser(description="Refine a grasp until it holds")
    parser.add_argument("--traj", required=True, help="wuji_traj.npz from hawor_to_wuji_qpos.py")
    parser.add_argument("--object-mesh", required=True, help="RecGen posed_mesh.obj")
    parser.add_argument("--object-poses", required=True, help="Object trajectory .npz")
    parser.add_argument("--out", required=True, help="Refined trajectory .npz")
    parser.add_argument("--urdf", default=None)
    parser.add_argument("--wuji-dir", default=DEFAULT_WUJI_DIR)

    parser.add_argument(
        "--grasp-window",
        type=int,
        nargs=2,
        default=None,
        metavar=("START", "STOP"),
        help="Frames the grasp spans. Detected from fingertip-to-surface distance if unset.",
    )
    parser.add_argument(
        "--contact",
        type=float,
        default=0.055,
        help="Fingertip-to-surface distance counting as a grasp attempt, metres. The "
        "detector looks for a run of frames where a finger gets this close.",
    )
    parser.add_argument("--min-grasp", type=int, default=12, help="Shortest grasp kept, frames")
    parser.add_argument("--fill-gap", type=int, default=8, help="Gap bridged inside a grasp, frames")
    parser.add_argument(
        "--engage",
        type=float,
        default=0.06,
        help="A finger closer than this to the surface at some point in the window is one "
        "the demonstration grips with, and is fitted onto the object. The others are only "
        "kept out of it, so a three-finger grasp is not forced into a five-finger one.",
    )
    parser.add_argument(
        "--target",
        type=float,
        default=0.003,
        help="Distance from the surface an engaged fingertip is fitted to, metres. Zero "
        "puts the tip exactly on the surface, where the contact solver has nothing to "
        "push on; a few mm leaves the controller room to close the last of it.",
    )
    parser.add_argument(
        "--samples", type=int, default=6, help="Frames sampled across the window for the fit"
    )
    parser.add_argument(
        "--restarts",
        type=int,
        default=8,
        help="Seeded restarts of the contact fit. Wrapping a hand around an object is not "
        "convex, and one descent from the reconstruction lands in whatever minimum is "
        "nearest -- usually the hand resting against the object rather than around it.",
    )
    parser.add_argument(
        "--deviation",
        type=float,
        default=1.0,
        help="Weight on staying near the reconstructed grasp. Raise it to keep the "
        "demonstration's pose at the cost of contact, lower it to prioritise holding.",
    )
    parser.add_argument(
        "--palm",
        type=float,
        default=0.045,
        help="Palm-to-surface distance the fit aims for, metres -- roughly a grasp's "
        "standoff. This is what puts the object inside the hand rather than beside it.",
    )
    parser.add_argument("--palm-weight", type=float, default=2.0, help="Weight on the palm term")
    parser.add_argument(
        "--opposition",
        type=float,
        default=6.0,
        help="Weight on the contact normals cancelling out. This is what separates a grip "
        "from a hand resting against the object; lower it if the fit over-curls.",
    )
    parser.add_argument("--max-shift", type=float, default=0.09, help="Wrist offset bound, metres")
    parser.add_argument("--max-turn", type=float, default=25.0, help="Wrist offset bound, degrees")
    parser.add_argument("--max-close", type=float, default=40.0, help="Closure bound, degrees")
    parser.add_argument(
        "--blend",
        type=int,
        default=10,
        help="Frames over which the correction is ramped in and out at the window's edges, "
        "so the approach and the retreat stay as reconstructed",
    )

    parser.add_argument("--skip-polish", action="store_true", help="Contact fit only")
    parser.add_argument(
        "--polish-iters",
        type=int,
        default=2,
        help="Coordinate-descent sweeps in stage 2. Each sweep replays the window once per "
        "candidate, so this is the stage's whole cost.",
    )
    parser.add_argument(
        "--polish-step",
        type=float,
        default=0.008,
        help="Wrist step for the polish search, metres (the closure step is 5 degrees)",
    )
    parser.add_argument("--ground-plane", default=None, help="ground_plane.json, for the polish")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


# ───────────────────────────── geometry helpers ─────────────────────────────


def surface_sampler(mesh_path, count=60000):
    """Points and outward normals sampled over the mesh, with a tree to query them.

    A reconstruction mesh has hundreds of thousands of faces, and an exact signed
    distance over it is far too slow to sit inside an optimiser. Nearest-sample
    distance, signed by that sample's normal, is accurate to the sample spacing --
    well under a millimetre here -- for a small fraction of the cost.
    """
    import trimesh

    mesh = trimesh.load(mesh_path, process=False)
    points, face_index = trimesh.sample.sample_surface(mesh, count)
    normals = mesh.face_normals[face_index]
    return np.asarray(points), np.asarray(normals), cKDTree(np.asarray(points))


def nearest_surface(query, points, normals, tree):
    """Signed distance to the surface (positive outside) and the normal there."""
    _, index = tree.query(query)
    offset = query - points[index]
    sign = np.sign(np.einsum("ij,ij->i", offset, normals[index]))
    sign[sign == 0] = 1.0
    return np.linalg.norm(offset, axis=1) * sign, normals[index]


def signed_distance(query, points, normals, tree):
    """Distance from each query point to the surface, positive outside."""
    return nearest_surface(query, points, normals, tree)[0]


def compose(rotation, translation, delta_rotation, delta_translation):
    """Apply a wrist-local offset: T_new = T ∘ ΔT."""
    return (
        rotation @ delta_rotation,
        translation + rotation @ delta_translation,
    )


def smoothstep(weight):
    return weight * weight * (3.0 - 2.0 * weight)


def blend_weights(n_frames, start, stop, blend):
    """1 inside the window, ramped to 0 over `blend` frames at each edge."""
    weights = np.zeros(n_frames)
    weights[start:stop] = 1.0
    if blend > 0:
        ramp = smoothstep(np.linspace(0.0, 1.0, blend + 2)[1:-1])
        lead = slice(max(0, start - blend), start)
        weights[lead] = ramp[-(lead.stop - lead.start):] if lead.stop > lead.start else weights[lead]
        trail = slice(stop, min(n_frames, stop + blend))
        if trail.stop > trail.start:
            weights[trail] = ramp[::-1][: trail.stop - trail.start]
    return weights


# ───────────────────────────── forward kinematics ─────────────────────────────


class HandModel:
    """Fingertip positions for a wrist pose and a finger configuration.

    SuperDex rather than an independent FK implementation, so the poses the fit places
    are the same ones the simulator will resolve contacts for. A mismatch there would
    move the grasp by exactly the amount the fit is trying to remove.
    """

    def __init__(self, urdf, joint_names, cpu=False):
        import superdex.physics as physics
        import superdex.robotics as robotics

        self.physics = physics
        physics.initialize(num_worker_threads=0 if cpu else -1)
        self.scene = physics.create_scene("refine")
        # No gravity and no ground: this scene is only ever asked where the links are
        # for a given pose, never stepped, so anything that could move them is a way
        # for the fit to be answered with something other than what it asked.
        self.scene.set_gravity([0.0, 0.0, 0.0])
        prefab = robotics.load_bot_prefab_from_urdf_file(urdf)
        prefab.joints[0].type = physics.ArticulatedJointType.HARD
        self.mapping, _ = dof_index_map(prefab)
        self.bot = robotics.create_bot(self.scene, prefab, robotics.create_context())
        self.hand = self.bot.get_articulated_actor()
        self.joint_names = joint_names
        self.dofs_idx = [self.mapping[name][0] for name in joint_names]
        self.n_dofs = self.hand.get_num_dofs()
        self._pose = np.zeros(self.n_dofs)

        names = [link.name for link in prefab.links]
        self.link_names = names
        self.tip_links = []
        for finger in FINGERS:
            wanted = [i for i, name in enumerate(names) if name.endswith(f"finger{finger}_tip_link")]
            if not wanted:
                # Older descriptions without a tip link: the distal link is the fallback,
                # which is a centimetre or so short of the fingertip but still monotonic
                # in the quantity the fit minimises.
                wanted = [i for i, name in enumerate(names) if name.endswith(f"finger{finger}_link4")]
            if not wanted:
                raise SystemExit(f"no tip or distal link for finger {finger}")
            self.tip_links.append(wanted[0])
        self.palm_link = next(i for i, name in enumerate(names) if name.endswith("palm_link"))
        # A plain list of TransformRT binds as an empty span and is silently left
        # unfilled -- every link reads as the origin. The pybind array type is what the
        # accessor actually writes through.
        self._transforms = physics.DynamicArrayTransformRT()
        self._transforms.resize(len(names))

        # get_articulated_dof_limits returns all zeros for a URDF-loaded bot here, and a
        # zero range clips every finger flat, so the limits are read from the URDF that
        # was loaded rather than from the actor built out of it.
        self.dof_limits = urdf_joint_limits(urdf, joint_names)

    def palm_and_tips(self, position, rotation, qpos):
        """Palm origin and the five fingertips, in the frame the base pose is given in."""
        self.hand.set_root_transform(to_transform(self.physics, rotation, position))
        self._pose[self.dofs_idx] = qpos
        self.hand.set_articulated_pose_from_joints(self._pose)
        self.hand.get_articulated_link_transforms(self._transforms)
        tips = np.array(
            [list(self._transforms[link].translation) for link in self.tip_links], dtype=float
        )
        palm = np.asarray(list(self._transforms[self.palm_link].translation), dtype=float)
        return palm, tips

    def fingertips(self, position, rotation, qpos):
        return self.palm_and_tips(position, rotation, qpos)[1]


# ───────────────────────────── stage 1: contact fit ─────────────────────────────


def closure_delta(closures, joint_names):
    """Extra flexion per finger, spread over the two distal joints of each.

    The base joints carry the finger's direction, which the retargeting got from the
    demonstration and which a closure correction has no business rewriting; the distal
    joints are what curls a finger around something.
    """
    delta = np.zeros(len(joint_names))
    for index, name in enumerate(joint_names):
        if not name.endswith(("joint3", "joint4")):
            continue
        for finger in FINGERS:
            if f"finger{finger}_" in name:
                delta[index] = closures[finger - 1]
    return delta


def fit_contact(args, model, traj, frames, object_rotation, object_translation, surface, limits):
    """Fit a wrist offset and per-finger closure so the engaged fingertips touch."""
    qpos, wrist_rotation, wrist_position = traj
    points, normals, tree = surface

    def tips_for(parameters, frame_index, frame):
        shift, turn, closures = parameters[:3], parameters[3:6], parameters[6:]
        delta_rotation = Rotation.from_rotvec(turn).as_matrix()
        rotation, position = compose(
            wrist_rotation[frame], wrist_position[frame], delta_rotation, shift
        )
        angles = np.clip(
            qpos[frame] + closure_delta(closures, model_joint_names),
            limits[:, 0],
            limits[:, 1],
        )
        palm, tips = model.palm_and_tips(position, rotation, angles)
        # Into the object's own frame, where its surface samples live.
        into = lambda p: (p - object_translation[frame]) @ object_rotation[frame]
        return into(tips), into(palm[None])

    model_joint_names = model.joint_names

    # Which fingers the demonstration grips with: the ones that get near the object at
    # some point in the window. The rest are held out of it but not pulled onto it.
    closest = np.full(5, np.inf)
    for frame_index, frame in enumerate(frames):
        local, _ = tips_for(np.zeros(11), frame_index, frame)
        closest = np.minimum(closest, np.abs(signed_distance(local, points, normals, tree)))
    engaged = closest < args.engage
    if not engaged.any():
        engaged = closest <= closest.min() + 1e-9
    print(f"[refine] engaged fingers: {[int(i) + 1 for i in np.flatnonzero(engaged)]} "
          f"(closest approach {np.round(closest * 100, 1)} cm)")

    scale = max(args.max_shift, 1e-6)

    def objective(parameters):
        total = 0.0
        for frame_index, frame in enumerate(frames):
            local, palm = tips_for(parameters, frame_index, frame)
            distance, tip_normals = nearest_surface(local, points, normals, tree)
            # Fingertips near the surface are not yet a grip: five fingers resting on
            # one side of a mug hold nothing. A grasp closes when the contact normals
            # oppose, so their sum is driven towards zero -- which is what makes the
            # optimiser curl fingers around the object instead of sliding the whole
            # hand until the tips happen to touch.
            balance = tip_normals[engaged].sum(axis=0) / max(int(engaged.sum()), 1)
            total += args.opposition * 1e-4 * float(balance @ balance)
            # Fingertips alone do not say which side of the object the hand is on, so a
            # fit can satisfy them by grazing it from outside -- which is the failure
            # being corrected. Holding the palm at a grasping standoff puts the object
            # between the palm and the fingers, where closing them does something.
            palm_gap = signed_distance(palm, points, normals, tree)[0]
            total += args.palm_weight * (palm_gap - args.palm) ** 2
            # Engaged fingers are pulled to a hair outside the surface; everything is
            # pushed out of it, and penetration costs far more than a gap because a
            # finger inside the mesh is a contact the solver will explode on.
            gap = distance[engaged] - args.target
            total += float(np.sum(gap ** 2))
            inside = np.minimum(distance, 0.0)
            total += 20.0 * float(np.sum(inside ** 2))
        total /= max(len(frames), 1)
        deviation = (
            np.sum((parameters[:3] / scale) ** 2)
            + np.sum((parameters[3:6] / np.radians(args.max_turn)) ** 2)
            + np.sum((parameters[6:] / np.radians(args.max_close)) ** 2)
        )
        return total + args.deviation * 1e-4 * deviation

    bounds = (
        [(-args.max_shift, args.max_shift)] * 3
        + [(-np.radians(args.max_turn), np.radians(args.max_turn))] * 3
        # Opening is allowed as well as closing: a finger fitted through the object has
        # to be able to come back out.
        + [(-np.radians(args.max_close) * 0.25, np.radians(args.max_close))] * 5
    )
    # Wrapping a hand around an object is not convex: from a hand resting against the
    # side of a mug, every small step away from that pose is worse, so a descent from
    # the reconstruction alone settles there and never finds the grip around the
    # handle. Seeded restarts give it somewhere else to descend from. The first seed is
    # always the reconstruction itself, so the fit can never do worse than not trying.
    rng = np.random.default_rng(0)
    seeds = [np.zeros(11)]
    for _ in range(max(args.restarts - 1, 0)):
        seed = np.empty(11)
        seed[:3] = rng.uniform(-1, 1, 3) * args.max_shift * 0.6
        seed[3:6] = rng.uniform(-1, 1, 3) * np.radians(args.max_turn) * 0.8
        seed[6:] = rng.uniform(0.0, 0.7, 5) * np.radians(args.max_close)
        seeds.append(np.clip(seed, [b[0] for b in bounds], [b[1] for b in bounds]))

    before = objective(seeds[0])
    best, evaluations = None, 0
    for index, seed in enumerate(seeds):
        result = minimize(objective, seed, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": 120, "eps": 1e-4})
        evaluations += result.nfev
        if best is None or result.fun < best.fun:
            best = result
            print(f"[refine]   seed {index}: objective {result.fun:.5f} (best so far)")
    print(f"[refine] contact fit: objective {before:.5f} -> {best.fun:.5f} "
          f"over {len(seeds)} seeds, {evaluations} evaluations")
    return best.x, engaged


def report_contact(model, traj, frames, object_rotation, object_translation, surface,
                   limits, parameters, engaged, label):
    qpos, wrist_rotation, wrist_position = traj
    points, normals, tree = surface
    shift, turn, closures = parameters[:3], parameters[3:6], parameters[6:]
    delta_rotation = Rotation.from_rotvec(turn).as_matrix()
    gaps = []
    for frame in frames:
        rotation, position = compose(
            wrist_rotation[frame], wrist_position[frame], delta_rotation, shift
        )
        angles = np.clip(qpos[frame] + closure_delta(closures, model.joint_names),
                         limits[:, 0], limits[:, 1])
        tips = model.fingertips(position, rotation, angles)
        local = (tips - object_translation[frame]) @ object_rotation[frame]
        gaps.append(signed_distance(local, points, normals, tree)[engaged])
    gaps = np.concatenate(gaps)
    print(f"[refine] {label}: engaged fingertip gap median {np.median(gaps) * 100:+.2f} cm, "
          f"worst {gaps.max() * 100:+.2f} cm, deepest {gaps.min() * 100:+.2f} cm")
    return float(np.median(np.abs(gaps)))


# ───────────────────────────── applying the correction ─────────────────────────────


def apply_correction(traj, limits, joint_names, parameters, weights):
    """Blend the offset and closure into the trajectory, weighted per frame."""
    qpos, wrist_rotation, wrist_position = traj
    shift, turn, closures = parameters[:3], parameters[3:6], parameters[6:]
    new_rotation = wrist_rotation.copy()
    new_position = wrist_position.copy()
    new_qpos = qpos.copy()
    delta = closure_delta(closures, joint_names)
    for frame, weight in enumerate(weights):
        if weight <= 0.0:
            continue
        # Scaling the rotation vector interpolates the offset from identity, so the
        # correction grows and fades smoothly rather than switching on.
        step = Rotation.from_rotvec(turn * weight).as_matrix()
        new_rotation[frame], new_position[frame] = compose(
            wrist_rotation[frame], wrist_position[frame], step, shift * weight
        )
        new_qpos[frame] = np.clip(qpos[frame] + delta * weight, limits[:, 0], limits[:, 1])
    return new_qpos, new_rotation, new_position


def write_traj(source, path, qpos, wrist_rotation, wrist_position, extra):
    fields = {key: source[key] for key in source.files}
    fields["qpos"] = qpos
    quat = Rotation.from_matrix(wrist_rotation).as_quat()
    fields["wrist_quat"] = np.c_[quat[:, 3], quat[:, :3]]
    fields["wrist_pos"] = wrist_position
    fields.update(extra)
    np.savez(path, **fields)


# ───────────────────────────── stage 2: physics polish ─────────────────────────────


def held_score(args, traj_path, window, replay_script):
    """Replay the window and measure how well the object stays with the wrist.

    Not distance to the tracked object trajectory: that trajectory is a reconstruction
    too, and on an occluded grasp it is the less reliable of the two. An object that is
    held keeps its offset from the wrist whatever the wrist does, which needs nothing
    from the tracker beyond the pose it was released at.
    """
    start, stop = window
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as handle:
        sim_out = handle.name
    command = [
        sys.executable, replay_script,
        "--traj", traj_path,
        "--object-mesh", args.object_mesh,
        "--object-poses", args.object_poses,
        "--wuji-dir", args.wuji_dir,
        "--physics", "--sim-out", sim_out,
        "--frame-range", str(start), str(stop),
        "--release-frame", str(start),
    ]
    if args.ground_plane:
        command += ["--ground-plane", args.ground_plane]
    if args.urdf:
        command += ["--urdf", args.urdf]
    if args.cpu:
        command += ["--cpu"]
    result = subprocess.run(command, capture_output=True, text=True)
    if not os.path.exists(sim_out) or os.path.getsize(sim_out) == 0:
        tail = "\n".join(result.stderr.strip().splitlines()[-3:])
        print(f"[refine]   replay failed: {tail}")
        return np.inf, 0
    simulated = np.load(sim_out)
    os.unlink(sim_out)

    traj = np.load(traj_path)
    # The replay works in a z-up world and recentres the scene; the trajectory is in
    # camera coordinates. Comparing the two directly would measure the transform rather
    # than the grasp, so the wrist is carried into the frame the object was simulated in.
    if args.ground_plane:
        import json

        with open(args.ground_plane) as handle:
            up = json.load(handle)["normal_camera"]
    else:
        up = [0.0, -1.0, 0.0]
    cam_to_world = camera_to_world(up)
    rotations = np.einsum(
        "ij,tjk->tik",
        cam_to_world,
        Rotation.from_quat(
            np.c_[traj["wrist_quat"][start:stop, 1:], traj["wrist_quat"][start:stop, 0]]
        ).as_matrix(),
    )
    positions = traj["wrist_pos"][start:stop] @ cam_to_world.T + simulated["shift"]
    object_pos = simulated["position"]
    span = min(len(object_pos), len(positions))
    if span < 2:
        return np.inf, 0
    # The object's position in the wrist's frame: constant while it is held, whatever
    # the wrist does, so its drift from the released value is the slip.
    in_wrist = np.einsum("tij,tj->ti", rotations[:span].transpose(0, 2, 1),
                         object_pos[:span] - positions[:span])
    slip = np.linalg.norm(in_wrist - in_wrist[0], axis=1)
    contacts = int((simulated["contacts"] > 0).sum())
    return float(np.mean(slip)), contacts


def polish(args, source, base_parameters, traj, limits, joint_names, weights, window):
    """Coordinate descent on the parameters the simulation is most sensitive to."""
    replay_script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "replay_wuji_superdex.py")
    scratch = tempfile.mkdtemp(prefix="refine_grasp_")

    def score_for(parameters):
        path = os.path.join(scratch, "candidate.npz")
        qpos, rotation, position = apply_correction(traj, limits, joint_names, parameters, weights)
        write_traj(source, path, qpos, rotation, position, {})
        return held_score(args, path, window, replay_script)

    best = base_parameters.copy()
    best_slip, best_contacts = score_for(best)
    print(f"[refine] polish start: slip {best_slip * 100:.2f} cm, {best_contacts} contact frames")

    # Only the directions the contact fit cannot see: where the hand sits relative to
    # the object, and how hard it closes. The wrist's orientation is left to stage 1,
    # which has the geometry to judge it and does not cost a simulation per guess.
    axes = [(0, args.polish_step), (1, args.polish_step), (2, args.polish_step)]
    axes += [(6 + finger, np.radians(5.0)) for finger in range(5)]
    for sweep in range(args.polish_iters):
        improved = False
        for index, step in axes:
            for direction in (+1, -1):
                candidate = best.copy()
                candidate[index] += direction * step
                slip, contacts = score_for(candidate)
                if slip < best_slip - 1e-5:
                    best, best_slip, best_contacts = candidate, slip, contacts
                    improved = True
                    print(f"[refine]   sweep {sweep + 1}: parameter {index} "
                          f"{direction:+d} -> slip {slip * 100:.2f} cm, {contacts} contacts")
                    break
        if not improved:
            print(f"[refine]   sweep {sweep + 1}: no improvement, stopping")
            break
    print(f"[refine] polish end: slip {best_slip * 100:.2f} cm, {best_contacts} contact frames")
    return best


# ───────────────────────────── main ─────────────────────────────


def main():
    args = parse_args()
    source = np.load(args.traj)
    qpos = source["qpos"]
    joint_names = [str(name) for name in source["joint_names"]]
    side = str(source["side"])
    valid = source["valid"].astype(bool)
    n_frames = len(qpos)

    urdf = args.urdf or os.path.join(
        args.wuji_dir, "wuji_retargeting", "wuji-description", "hand", "body", "urdf", f"{side}.urdf"
    )

    wrist_rotation = Rotation.from_quat(
        np.c_[source["wrist_quat"][:, 1:], source["wrist_quat"][:, 0]]
    ).as_matrix()
    wrist_position = source["wrist_pos"]

    tracked = np.load(args.object_poses)
    object_rotation = tracked["rotation"]
    object_translation = tracked["translation"]
    solved = tracked["valid"].astype(bool)
    if len(object_rotation) != n_frames:
        raise SystemExit(
            f"hand has {n_frames} frames, object has {len(object_rotation)}"
        )

    points, normals, tree = surface_sampler(args.object_mesh)
    surface = (points, normals, tree)
    print(f"[refine] object surface sampled with {len(points)} points")

    model = HandModel(urdf, joint_names, cpu=args.cpu)
    model.joint_names = joint_names
    limits = model.dof_limits

    # ── grasp windows ──
    # A clip rarely holds one grasp. This take puts the mug down and picks it up again,
    # and one offset averaged over both is wrong for each, so every window is fitted
    # separately and written into its own frames.
    if args.grasp_window:
        spans = [tuple(args.grasp_window)]
    else:
        usable = valid & solved
        reach = np.full(n_frames, np.inf)
        for frame in range(n_frames):
            if not usable[frame]:
                continue
            tips = model.fingertips(wrist_position[frame], wrist_rotation[frame], qpos[frame])
            local = (tips - object_translation[frame]) @ object_rotation[frame]
            reach[frame] = np.abs(signed_distance(local, points, normals, tree)).min()
        spans = runs_of(usable & (reach < args.contact), args.fill_gap, args.min_grasp)
        if not spans:
            raise SystemExit(
                f"no grasp found: the closest a fingertip comes to the object is "
                f"{np.nanmin(reach) * 100:.1f} cm, against a --contact of {args.contact * 100:.1f} cm"
            )
        print(f"[refine] grasp windows: {[(int(a), int(b - 1)) for a, b in spans]}")

    current = (qpos.copy(), wrist_rotation.copy(), wrist_position.copy())
    all_weights = np.zeros(n_frames)
    fitted = []

    for span_index, (start, stop) in enumerate(spans):
        print(f"\n[refine] === window {span_index + 1}/{len(spans)}: "
              f"frames {start}..{stop - 1} ({stop - start} frames) ===")
        frames = np.unique(np.linspace(start, stop - 1, args.samples).astype(int))
        frames = [frame for frame in frames if valid[frame] and solved[frame]]
        if not frames:
            print("[refine] no reconstructed frames in this window, left alone")
            continue

        parameters, engaged = fit_contact(
            args, model, current, frames, object_rotation, object_translation, surface, limits
        )
        report_contact(model, current, frames, object_rotation, object_translation, surface,
                       limits, np.zeros(11), engaged, "before")
        report_contact(model, current, frames, object_rotation, object_translation, surface,
                       limits, parameters, engaged, "after ")
        print(f"[refine] wrist offset {np.round(parameters[:3] * 100, 2)} cm, "
              f"turn {np.round(np.degrees(parameters[3:6]), 1)} deg, "
              f"closure {np.round(np.degrees(parameters[6:]), 1)} deg")

        weights = blend_weights(n_frames, start, stop, args.blend)

        if not args.skip_polish:
            parameters = polish(args, source, parameters, current, limits, joint_names,
                                weights, (start, stop))
            print(f"[refine] polished offset {np.round(parameters[:3] * 100, 2)} cm, "
                  f"turn {np.round(np.degrees(parameters[3:6]), 1)} deg, "
                  f"closure {np.round(np.degrees(parameters[6:]), 1)} deg")
            report_contact(model, current, frames, object_rotation, object_translation,
                           surface, limits, parameters, engaged, "final ")

        corrected = apply_correction(current, limits, joint_names, parameters, weights)
        touched = weights > 0
        for array, replacement in zip(current, corrected):
            array[touched] = replacement[touched]
        all_weights = np.maximum(all_weights, weights)
        fitted.append((start, stop, parameters, engaged))

    new_qpos, new_rotation, new_position = current
    moved = np.linalg.norm(new_position - wrist_position, axis=1)
    turned = np.degrees(np.abs(new_qpos - qpos))
    print(f"\n[refine] trajectory changed: wrist by up to {moved.max() * 100:.2f} cm, "
          f"joints by up to {turned.max():.1f} deg, over {int((all_weights > 0).sum())} frames")

    write_traj(
        source, args.out, new_qpos, new_rotation, new_position,
        {
            "refine_windows": np.array([[a, b] for a, b, _, _ in fitted], dtype=int).reshape(-1, 2),
            "refine_parameters": np.array([p for _, _, p, _ in fitted]).reshape(-1, 11),
            "refine_engaged": np.array([e for _, _, _, e in fitted]).reshape(-1, 5),
            "refine_weights": all_weights,
        },
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
