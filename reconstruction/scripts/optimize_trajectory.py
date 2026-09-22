#!/usr/bin/env python3
"""
Smooth a retargeted hand trajectory by trajectory optimization.

This is the post-retargeting step from "One Demonstration, Many Objects:
Generalizing Manipulation via Local Contact Geometry" (arXiv 2609.01938,
Appendix A.1): "After retargeting, we smooth the resulting trajectory using
trajectory optimization, incorporating regularization costs on velocity,
acceleration, and jerk."

Retargeting solves each frame on its own -- HaWoR's per-frame hand, then IK onto the
Wuji Hand -- so nothing in it knows that the previous frame existed. The result
tracks the demonstration but arrives with the per-frame solver's noise on top, and a
joint trajectory that is merely noisy kinematically is undriveable physically: the
velocity it implies reverses every few frames and the acceleration is unbounded.

So each channel is refitted as a whole rather than filtered frame by frame:

    min_x  sum_t v_t (x_t - r_t)^2
           + lambda_v ||D1 x||^2 + lambda_a ||D2 x||^2 + lambda_j ||D3 x||^2

against the retargeted reference r, with D1, D2, D3 the first, second and third
finite differences -- velocity, acceleration, jerk. It is a quadratic in x, so the
optimum is one sparse linear solve rather than an iteration, and the same
factorisation serves every joint and every axis.

Two things fall out of the weights v_t rather than needing separate handling:

  * a frame retargeting could not solve gets v_t = 0, so the optimiser interpolates
    it with the jerk-minimal curve instead of holding the last pose, which is what
    hawor_to_wuji_qpos.py leaves behind and what makes a replay stutter at dropouts.
  * raising lambda relative to v trades fidelity to the demonstration for smoothness,
    uniformly over the clip, instead of the sliding window's implicit trade.

Interpolating a gap this way is sound; continuing past the last retargeted frame is
not. Outside the retargeted span there is nothing to anchor the curve, and the
jerk-minimal continuation is a cubic that leaves at whatever speed the last frames
implied -- on grasping_cup/take_006, whose last seven frames are a dropout, that came
out at 49 mm/frame against a normal 3.8. So the optimisation runs between the first
and last retargeted frame, and the frames outside it keep the pose retargeting left.

The wrist's orientation is optimised on the manifold: the same quadratic is solved in
the tangent space around the current estimate and folded back in with a rotation
increment, repeated a few times (Gauss-Newton). Smoothing quaternion components and
renormalising would pull the wrist off the geodesic between two keyframes -- the same
reason kalman_poses.py filters the object's rotation multiplicatively.

Of the three terms, acceleration and jerk do the work; penalising velocity pulls the
whole trajectory toward standing still, because the demonstration's real motion is
exactly what a velocity cost charges for. On grasping_cup/take_006, raising it from
0.5 to 8 took 20 mm off the wrist's 1.09 m path and bought 7% of jerk. It is kept on,
as the paper specifies, but low.

Joint limits are clipped afterwards when a URDF is given; a channel driven past its
limit by smoothing is reported rather than silently saturated.

Run in the `genesis` env:
    python optimize_trajectory.py --traj wuji_traj.npz --out wuji_traj_smooth.npz \\
        --urdf right.urdf
"""

import argparse
import math
import xml.etree.ElementTree as ElementTree

import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import splu
from scipy.spatial.transform import Rotation


def parse_args():
    parser = argparse.ArgumentParser(description="Trajectory optimization over a retargeted trajectory")
    parser.add_argument("--traj", required=True, help="wuji_traj.npz from hawor_to_wuji_qpos.py")
    parser.add_argument("--out", required=True)
    parser.add_argument("--urdf", default=None, help="Hand URDF, for joint limits")
    # One weight per derivative, shared by every channel. The reference term has
    # weight 1, so these are read relative to it: 1.0 means a unit of jerk costs as
    # much as a radian (or metre) of departure from the demonstration.
    parser.add_argument("--velocity", type=float, default=0.5,
                        help="Velocity regularization weight")
    parser.add_argument("--acceleration", type=float, default=2.0,
                        help="Acceleration regularization weight")
    parser.add_argument("--jerk", type=float, default=8.0,
                        help="Jerk regularization weight")
    # The wrist travels centimetres where a joint turns radians, so a weight that
    # barely touches the joints would freeze the wrist. These scale the three weights
    # above for the wrist channels.
    parser.add_argument("--wrist-scale", type=float, default=0.02,
                        help="Multiplies the weights for the wrist's position, in metres")
    parser.add_argument("--wrist-rotation-scale", type=float, default=1.0,
                        help="Multiplies the weights for the wrist's orientation, in radians")
    parser.add_argument("--iterations", type=int, default=4,
                        help="Gauss-Newton iterations for the orientation channel")
    return parser.parse_args()


def difference_operator(n_frames, order):
    """Sparse finite-difference matrix of `order`, shape (n_frames - order, n_frames)."""
    # Binomial coefficients with alternating signs: [1,-1], [1,-2,1], [1,-3,3,-1].
    stencil = np.array([(-1) ** k * math.comb(order, k) for k in range(order + 1)][::-1], dtype=float)
    rows = n_frames - order
    if rows <= 0:
        return sparse.csr_matrix((0, n_frames))
    return sparse.diags(stencil, offsets=np.arange(order + 1), shape=(rows, n_frames), format="csr")


def build_system(valid, weights, n_frames):
    """The normal-equation matrix and the difference operators behind it.

    Returns (factorised A, [(weight, D)]), with

        A = diag(valid) + sum_k weight_k * D_k^T D_k

    which is symmetric positive definite whenever some frame is observed, and is the
    same for every channel sharing a validity mask -- so it is factorised once.
    """
    operators = [(weight, difference_operator(n_frames, order))
                 for order, weight in zip((1, 2, 3), weights) if weight > 0]
    matrix = sparse.diags(valid.astype(float), format="csr")
    for weight, operator in operators:
        matrix = matrix + weight * (operator.T @ operator)
    return splu(matrix.tocsc()), operators


def smooth_step(solver, operators, valid, reference, estimate):
    """One Gauss-Newton step, shared by the linear and the rotational channels.

    Works in increments rather than absolutes: given the current `estimate` and the
    residual `reference` still to be explained, it returns the increment that
    minimises the objective linearised around the estimate. For a linear channel one
    step is already exact; for rotations the linearisation is refreshed each time.
    """
    rhs = valid[:, None] * reference
    for weight, operator in operators:
        rhs = rhs - weight * (operator.T @ (operator @ estimate))
    return np.column_stack([solver.solve(rhs[:, column]) for column in range(rhs.shape[1])])


def smooth_linear(series, valid, weights):
    """Optimise position-like channels (joint angles, wrist position)."""
    solver, operators = build_system(valid, weights, len(series))
    # The reference residual is zero at the starting estimate, so a single step is
    # the exact minimiser; see the docstring in smooth_step.
    return series + smooth_step(solver, operators, valid, np.zeros_like(series), series)


def smooth_rotations(rotations, valid, weights, iterations):
    """Optimise the wrist's orientation in the tangent space, iterated.

    The objective is the same one, with the finite differences taken over the
    rotation increments Log(R_t^T R_t+1) rather than over coordinates, and the
    solution applied as R_t <- R_t Exp(delta_t).
    """
    n_frames = len(rotations)
    solver, operators = build_system(valid, weights, n_frames)
    estimate = Rotation.from_matrix(rotations)
    reference = Rotation.from_matrix(rotations)

    for _ in range(iterations):
        # Tangent coordinates whose first difference is the per-frame rotation, so
        # the difference operators act on the orientation curve as they do on a
        # linear one. Valid to first order, which is all a Gauss-Newton step needs.
        increments = (estimate[:-1].inv() * estimate[1:]).as_rotvec()
        curve = np.vstack([np.zeros(3), np.cumsum(increments, axis=0)])
        residual = (estimate.inv() * reference).as_rotvec()
        delta = smooth_step(solver, operators, valid, residual, curve)
        if np.abs(delta).max() < 1e-9:
            break
        estimate = estimate * Rotation.from_rotvec(delta)
    return estimate


def urdf_joint_limits(path, joint_names):
    """(lower, upper) per joint in radians. Duplicated from refine_grasp.py, which
    lives behind a SuperDex import chain this stage has no other reason to pull in."""
    tree = ElementTree.parse(path)
    limits = {}
    for joint in tree.getroot().iter("joint"):
        limit = joint.find("limit")
        if limit is not None:
            limits[joint.get("name")] = (float(limit.get("lower", -np.pi)),
                                         float(limit.get("upper", np.pi)))
    lower = np.array([limits.get(name, (-np.pi, np.pi))[0] for name in joint_names])
    upper = np.array([limits.get(name, (-np.pi, np.pi))[1] for name in joint_names])
    return lower, upper


def report(label, before, after, valid, unit, scale=1.0):
    """How much the optimiser moved the trajectory, and what it bought."""
    def derivatives(series):
        return [scale * np.abs(np.diff(series, n=order, axis=0)).mean() for order in (1, 2, 3)]

    moved = scale * np.linalg.norm((after - before)[valid], axis=-1).mean()
    velocity, acceleration, jerk = derivatives(before)
    velocity_after, acceleration_after, jerk_after = derivatives(after)
    print(f"[optimize] {label}: moved {moved:.4g} {unit} from the retargeted pose")
    print(f"[optimize]   velocity {velocity:.4g} -> {velocity_after:.4g}, "
          f"acceleration {acceleration:.4g} -> {acceleration_after:.4g}, "
          f"jerk {jerk:.4g} -> {jerk_after:.4g}  ({unit}/frame^k)")


def main():
    args = parse_args()
    source = np.load(args.traj, allow_pickle=True)
    fields = {key: source[key] for key in source.files}

    qpos = np.asarray(fields["qpos"], dtype=float)
    wrist_pos = np.asarray(fields["wrist_pos"], dtype=float)
    wrist_quat = np.asarray(fields["wrist_quat"], dtype=float)
    valid = np.asarray(fields["valid"], dtype=bool)
    n_frames = len(qpos)

    if valid.sum() < 8:
        raise SystemExit(f"only {valid.sum()} valid frames; nothing to optimise")

    # The optimised span: never extrapolated past the outermost retargeted frame.
    order = np.flatnonzero(valid)
    span = slice(order[0], order[-1] + 1)
    inside = valid[span]
    gaps = int((~inside).sum())
    print(f"[optimize] {n_frames} frames, {int(valid.sum())} retargeted; optimising "
          f"{order[0]}-{order[-1]}, filling {gaps} interior gap frames, "
          f"leaving {n_frames - len(inside)} outside the span as retargeted")

    weights = (args.velocity, args.acceleration, args.jerk)
    wrist_weights = tuple(w * args.wrist_scale for w in weights)
    rotation_weights = tuple(w * args.wrist_rotation_scale for w in weights)

    # Stored wxyz, as every other script in this pipeline reads it; scipy wants xyzw.
    rotations = Rotation.from_quat(np.c_[wrist_quat[:, 1:], wrist_quat[:, 0]]).as_matrix()

    smoothed_qpos = qpos.copy()
    smoothed_pos = wrist_pos.copy()
    smoothed_qpos[span] = smooth_linear(qpos[span], inside, weights)
    smoothed_pos[span] = smooth_linear(wrist_pos[span], inside, wrist_weights)
    smoothed_matrices = rotations.copy()
    smoothed_matrices[span] = smooth_rotations(
        rotations[span], inside, rotation_weights, args.iterations
    ).as_matrix()
    smoothed_rotation = Rotation.from_matrix(smoothed_matrices)

    report("hand joints", qpos[span], smoothed_qpos[span], inside, "rad")
    report("wrist position", wrist_pos[span], smoothed_pos[span], inside, "mm", scale=1000.0)
    turn = np.degrees((Rotation.from_matrix(rotations).inv() * smoothed_rotation).magnitude())
    print(f"[optimize] wrist orientation: moved {turn[valid].mean():.3g} deg from the retargeted pose")

    if args.urdf:
        joint_names = [str(name) for name in fields["joint_names"]]
        lower, upper = urdf_joint_limits(args.urdf, joint_names)
        clipped = (smoothed_qpos < lower) | (smoothed_qpos > upper)
        smoothed_qpos = np.clip(smoothed_qpos, lower, upper)
        if clipped.any():
            print(f"[optimize] clipped {int(clipped.sum())} joint samples to their URDF limits "
                  f"({100 * clipped.mean():.1f}% of the trajectory)")

    quaternion = smoothed_rotation.as_quat()
    fields["qpos"] = smoothed_qpos
    fields["wrist_pos"] = smoothed_pos
    fields["wrist_quat"] = np.c_[quaternion[:, 3], quaternion[:, :3]]
    # Every frame in the span now carries an optimised pose, including the gaps
    # retargeting dropped; the original mask is kept alongside so later stages can
    # still tell which frames were measured and which were reconstructed here.
    optimised = np.zeros(n_frames, dtype=bool)
    optimised[span] = True
    fields["valid"] = valid | optimised
    fields["retargeted"] = valid

    np.savez(args.out, **fields)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
