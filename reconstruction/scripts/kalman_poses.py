#!/usr/bin/env python3
"""
Constant-velocity Kalman smoothing for a rigid object trajectory.

The per-frame Kabsch/RANSAC solve treats every frame as independent, so its output
carries the depth map's frame-to-frame jitter straight into the pose. The sliding
window in `track_object_v2s2r.clean_poses` removes some of that, but it weights
frames only by how far they sit from the centre of the window: a frame solved from
six inliers counts as much as its well-observed neighbours, gaps have to be filled by
interpolation before smoothing can run at all, and the result has no notion of how
fast the object was moving.

A Kalman filter states those assumptions instead of implying them. The object is
modelled as moving at a constant velocity disturbed by white acceleration; each
solved pose is a noisy measurement of position and orientation. Frames the solve was
unsure about are given a larger measurement noise and pull the estimate less, frames
with no solve at all are simply prediction steps, and a backward RTS pass turns the
causal filter into a smoother so every frame uses the whole clip rather than its past.

Orientation is handled on the manifold (a multiplicative EKF): the state holds a
rotation and a body-frame angular velocity, and the covariance describes a small
rotation-vector error around that nominal rotation, which is reset into the nominal
after each update. Filtering quaternion components independently would distort
orientation over large turns in the same way averaging them does.

Used by pose_from_points.py (--smoother kalman); the noise parameters are exposed
there.
"""

import numpy as np
from scipy.spatial.transform import Rotation


def _constant_velocity(dt, accel_sigma):
    """Transition and process-noise matrices for a 6-state constant-velocity model.

    Q is the discrete white-noise-acceleration covariance: the process noise is an
    unknown acceleration of scale `accel_sigma` held over the step, which is what
    lets the filter follow a real change of motion instead of fighting it.
    """
    transition = np.eye(6)
    transition[:3, 3:] = np.eye(3) * dt
    variance = accel_sigma ** 2
    process = np.zeros((6, 6))
    process[:3, :3] = np.eye(3) * (dt ** 4 / 4.0) * variance
    process[:3, 3:] = np.eye(3) * (dt ** 3 / 2.0) * variance
    process[3:, :3] = process[:3, 3:]
    process[3:, 3:] = np.eye(3) * (dt ** 2) * variance
    return transition, process


def _gain(covariance, noise, innovation, gate):
    """Kalman gain for a direct measurement of the first three states.

    Returns (gain, accepted). The measurement is rejected when the Mahalanobis
    distance of its innovation exceeds `gate`, and a rejected frame is then carried
    by prediction like a frame with no solve at all.

    A consistent measurement sits near sqrt(3), which would suggest gating at 3 or so,
    but the innovations on real takes are heavily tailed by the object's own motion
    rather than by bad solves: a hand starting a lift accelerates far past anything a
    constant-velocity prediction expects. Gating at 3 on grasping_cup/take_006
    rejected 44 of 286 frames -- almost all of them real acceleration -- and the
    filter lagged the lift badly enough to take the silhouette IoU from 0.65 to 0.58.
    So the gate is set loose, to catch a solve that has gone somewhere the motion
    cannot explain at all, and the frame-to-frame jump gate upstream stays the real
    outlier test.
    """
    innovation_covariance = covariance[:3, :3] + np.eye(3) * noise
    distance = innovation @ np.linalg.solve(innovation_covariance, innovation)
    if distance > gate ** 2:
        return None, False
    gain = np.linalg.solve(innovation_covariance, covariance[:3, :]).T
    return gain, True


def _rts_correction(filtered_covariance, predicted_covariance, transition):
    """Backward smoothing gain C = P_filt F^T P_pred^-1, both covariances symmetric."""
    return np.linalg.solve(predicted_covariance, transition @ filtered_covariance).T


def smooth_translations(translations, measured, noise, accel_sigma, gate, dt=1.0):
    """Kalman filter + RTS smoother over positions. Returns (positions, updated)."""
    n_frames = len(translations)
    transition, process = _constant_velocity(dt, accel_sigma)

    states = np.zeros((n_frames, 6))
    covariances = np.zeros((n_frames, 6, 6))
    predicted_states = np.zeros((n_frames, 6))
    predicted_covariances = np.zeros((n_frames, 6, 6))
    updated = np.zeros(n_frames, dtype=bool)

    first = int(np.flatnonzero(measured)[0])
    state = np.concatenate([translations[first], np.zeros(3)])
    covariance = np.diag(np.concatenate([np.full(3, noise[first]), np.full(3, (10.0 * accel_sigma) ** 2)]))
    predicted_states[first] = state
    predicted_covariances[first] = covariance
    states[first] = state
    covariances[first] = covariance
    updated[first] = True

    for index in range(first + 1, n_frames):
        state = transition @ state
        covariance = transition @ covariance @ transition.T + process
        predicted_states[index] = state
        predicted_covariances[index] = covariance

        if measured[index]:
            innovation = translations[index] - state[:3]
            gain, accepted = _gain(covariance, noise[index], innovation, gate)
            if accepted:
                state = state + gain @ innovation
                covariance = covariance - gain @ covariance[:3, :]
                covariance = 0.5 * (covariance + covariance.T)
                updated[index] = True

        states[index] = state
        covariances[index] = covariance

    smoothed = states.copy()
    smoothed_covariances = covariances.copy()
    for index in range(n_frames - 2, first - 1, -1):
        correction = _rts_correction(covariances[index], predicted_covariances[index + 1], transition)
        smoothed[index] = states[index] + correction @ (smoothed[index + 1] - predicted_states[index + 1])
        smoothed_covariances[index] = covariances[index] + correction @ (
            smoothed_covariances[index + 1] - predicted_covariances[index + 1]
        ) @ correction.T

    positions = translations.copy()
    positions[first:] = smoothed[first:, :3]
    return positions, updated


def smooth_rotations(rotations, measured, noise, accel_sigma, gate, dt=1.0):
    """Multiplicative Kalman filter + RTS smoother over orientations.

    The state is a nominal rotation with a body-frame angular velocity; the 6x6
    covariance is over the error [rotation vector, angular velocity] around it, and
    the error is folded back into the nominal after every update so it stays small
    enough for the linearisation to hold.
    """
    n_frames = len(rotations)
    _, process = _constant_velocity(dt, accel_sigma)

    nominal = [Rotation.identity()] * n_frames
    predicted_nominal = [Rotation.identity()] * n_frames
    rates = np.zeros((n_frames, 3))
    predicted_rates = np.zeros((n_frames, 3))
    covariances = np.zeros((n_frames, 6, 6))
    predicted_covariances = np.zeros((n_frames, 6, 6))
    transitions = np.zeros((n_frames, 6, 6))
    updated = np.zeros(n_frames, dtype=bool)

    first = int(np.flatnonzero(measured)[0])
    orientation = Rotation.from_matrix(rotations[first])
    rate = np.zeros(3)
    covariance = np.diag(np.concatenate([np.full(3, noise[first]), np.full(3, (10.0 * accel_sigma) ** 2)]))
    nominal[first] = predicted_nominal[first] = orientation
    covariances[first] = predicted_covariances[first] = covariance
    transitions[first] = np.eye(6)
    updated[first] = True

    for index in range(first + 1, n_frames):
        step = Rotation.from_rotvec(rate * dt)
        # Rotating the error into the propagated frame is the adjoint of the step;
        # the right Jacobian coupling the rate in is taken as identity, which holds
        # for the small per-frame turns left after the jump gate.
        transition = np.eye(6)
        transition[:3, :3] = step.inv().as_matrix()
        transition[:3, 3:] = np.eye(3) * dt
        transitions[index] = transition

        orientation = orientation * step
        covariance = transition @ covariance @ transition.T + process
        predicted_nominal[index] = orientation
        predicted_rates[index] = rate
        predicted_covariances[index] = covariance

        if measured[index]:
            innovation = (orientation.inv() * Rotation.from_matrix(rotations[index])).as_rotvec()
            gain, accepted = _gain(covariance, noise[index], innovation, gate)
            if accepted:
                correction = gain @ innovation
                orientation = orientation * Rotation.from_rotvec(correction[:3])
                rate = rate + correction[3:]
                covariance = covariance - gain @ covariance[:3, :]
                covariance = 0.5 * (covariance + covariance.T)
                updated[index] = True

        nominal[index] = orientation
        rates[index] = rate
        covariances[index] = covariance

    smoothed = list(nominal)
    smoothed_rates = rates.copy()
    smoothed_covariances = covariances.copy()
    for index in range(n_frames - 2, first - 1, -1):
        correction_gain = _rts_correction(covariances[index], predicted_covariances[index + 1], transitions[index + 1])
        error = np.concatenate([
            (predicted_nominal[index + 1].inv() * smoothed[index + 1]).as_rotvec(),
            smoothed_rates[index + 1] - predicted_rates[index + 1],
        ])
        correction = correction_gain @ error
        smoothed[index] = nominal[index] * Rotation.from_rotvec(correction[:3])
        smoothed_rates[index] = rates[index] + correction[3:]
        smoothed_covariances[index] = covariances[index] + correction_gain @ (
            smoothed_covariances[index + 1] - predicted_covariances[index + 1]
        ) @ correction_gain.T

    matrices = rotations.copy()
    matrices[first:] = Rotation.concatenate(smoothed[first:]).as_matrix()
    return matrices, updated


def kalman_smooth_poses(
    rotations,
    translations,
    valid,
    position_sigma,
    accel_sigma,
    rotation_sigma,
    angular_accel_sigma,
    gate,
    confidence=None,
):
    """Smooth a gated pose sequence with a constant-velocity Kalman filter.

    `valid` marks the frames that were solved; gaps inside the solved span are filled
    by the filter's own prediction rather than by interpolation. `confidence` (the
    RANSAC inlier count) scales each frame's measurement noise, so a pose resting on
    a handful of inliers is trusted less than a well-observed one.

    Returns (rotations, translations, valid), with `valid` covering the solved span
    exactly as clean_poses reports it -- the filter is never extrapolated past the
    last measurement, where a constant-velocity model drifts without bound.
    """
    valid = np.asarray(valid, dtype=bool)
    order = np.flatnonzero(valid)
    if len(order) < 3:
        return rotations, translations, valid

    n_frames = len(translations)
    scale = np.ones(n_frames)
    if confidence is not None:
        confidence = np.asarray(confidence, dtype=float)
        reference = np.median(confidence[valid])
        if reference > 0:
            scale = np.clip(reference / np.maximum(confidence, 1e-6), 0.25, 4.0)

    position_noise = (position_sigma ** 2) * scale
    rotation_noise = (np.radians(rotation_sigma) ** 2) * scale

    positions, position_updates = smooth_translations(
        translations, valid, position_noise, accel_sigma, gate
    )
    matrices, rotation_updates = smooth_rotations(
        rotations, valid, rotation_noise, np.radians(angular_accel_sigma), gate
    )

    span = np.arange(order[0], order[-1] + 1)
    smoothed_rotations = rotations.copy()
    smoothed_translations = translations.copy()
    smoothed_rotations[span] = matrices[span]
    smoothed_translations[span] = positions[span]

    rejected = int(valid[span].sum() - np.minimum(position_updates, rotation_updates)[span].sum())
    shift = np.linalg.norm(smoothed_translations[span] - translations[span], axis=1)
    print(f"[kalman] smoothed {len(span)} frames, {int(valid[span].sum())} with a measurement, "
          f"{rejected} rejected by the innovation gate, median correction {1000 * np.median(shift):.1f} mm")

    cleaned = np.zeros_like(valid)
    cleaned[span] = True
    return smoothed_rotations, smoothed_translations, cleaned
