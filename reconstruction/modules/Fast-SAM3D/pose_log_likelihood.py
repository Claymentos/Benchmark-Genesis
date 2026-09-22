"""
Pose-only log-likelihood scoring for SAM3D flow-matching candidates.

Computes log p(pose | shape, image) by running the reverse ODE from x1→x0
and integrating the exact 13D Jacobian trace over pose dimensions only.

The shape still participates in the forward pass (so the model "sees" the
shape context when predicting pose velocity), but the divergence and prior
terms only cover the 13 pose dimensions.

Usage:
    from pose_log_likelihood import compute_pose_only_log_likelihoods

    log_liks = compute_pose_only_log_likelihoods(
        pipeline, ss_input_dict, all_samples, steps=25,
    )
"""

import math as _math

import torch
from loguru import logger


# Pose latent keys in a fixed order for consistent concatenation
# dims: 6 + 3 + 3 + 1 = 13 total
_POSE_KEYS_ORDERED = ["6drotation_normalized", "scale", "translation", "translation_scale"]


def _exact_pose_divergence(velocity, x_t):
    """Exact tr(dv_pose / dx_pose) via 13 backward passes (zero noise)."""
    div = torch.zeros(1, device=next(iter(x_t.values())).device)
    for k in _POSE_KEYS_ORDERED:
        v_k = velocity[k].flatten(1)  # (B, d_k)
        for j in range(v_k.shape[1]):
            grad_j = torch.autograd.grad(
                v_k[:, j].sum(), x_t[k],
                create_graph=False, retain_graph=True,
            )[0]
            div = div + grad_j.flatten(1)[:, j]
    return div


def _log_p0_pose_only(x0):
    """Standard Gaussian prior log-density restricted to pose dims."""
    parts = [x0[k].flatten(1) for k in _POSE_KEYS_ORDERED]
    flat = torch.cat(parts, dim=1)  # (B, 13)
    D = flat.shape[1]
    return -(flat ** 2).sum(dim=1) / 2 - _math.log(2 * _math.pi) / 2 * D


def _run_pose_only_ode(ss_generator, x1, cond_args, cond_kwargs, steps, device):
    """Euler ODE integration with pose-only divergence tracking."""
    t_seq = ss_generator._prepare_t(steps).to(device)
    t_seq = 1 - t_seq  # reverse: t=1 (data) -> t=0 (noise)

    x_t = {k: v.clone() for k, v in x1.items()}
    log_p_accum = torch.tensor(0.0, device=device)

    for i in range(len(t_seq) - 1):
        t_val = t_seq[i].item()
        dt = (t_seq[i + 1] - t_seq[i]).item()
        t_scaled = torch.tensor(
            [t_val * ss_generator.time_scale],
            device=device, dtype=torch.float32,
        )

        with torch.set_grad_enabled(True):
            for v in x_t.values():
                v.requires_grad_(True)
            velocity = ss_generator.reverse_fn(
                x_t, t_scaled, *cond_args, **cond_kwargs,
            )
            div = _exact_pose_divergence(velocity, x_t)

        for v in x_t.values():
            v.requires_grad_(False)
        x_t = {
            k: x_t[k].detach() + velocity[k].detach() * dt
            for k in x_t
        }
        log_p_accum = log_p_accum + div.detach() * dt

    log_p0 = _log_p0_pose_only(x_t)
    return (log_p_accum + log_p0).item()


def _setup_cfg_off(ss_generator):
    """Disable CFG and shortcut; return state to restore later."""
    state = {
        "strength": ss_generator.reverse_fn.strength,
        "strength_pm": getattr(ss_generator.reverse_fn, 'strength_pm', 0.0),
        "no_shortcut": getattr(ss_generator, 'no_shortcut', False),
    }
    ss_generator.reverse_fn.strength = 0.0
    if hasattr(ss_generator.reverse_fn, 'strength_pm'):
        ss_generator.reverse_fn.strength_pm = 0.0
    ss_generator.no_shortcut = True
    ss_generator.eval()
    return state


def _restore_cfg(ss_generator, state):
    """Restore CFG and shortcut settings."""
    ss_generator.reverse_fn.strength = state["strength"]
    if hasattr(ss_generator.reverse_fn, 'strength_pm'):
        ss_generator.reverse_fn.strength_pm = state["strength_pm"]
    ss_generator.no_shortcut = state["no_shortcut"]


def _get_cond(pipeline, ss_input_dict):
    """Compute and prepare condition args/kwargs (batch=1)."""
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=pipeline.shape_model_dtype):
            condition_args, condition_kwargs = pipeline.get_condition_input(
                pipeline.condition_embedders["ss_condition_embedder"],
                ss_input_dict,
                pipeline.ss_condition_input_mapping,
            )
    device = ss_input_dict["image"].device
    cond_args = tuple(
        a[:1] if isinstance(a, torch.Tensor) and a.shape[0] > 1 else a
        for a in condition_args
    )
    cond_kwargs = {
        k_: v[:1] if isinstance(v, torch.Tensor) and v.shape[0] > 1 else v
        for k_, v in condition_kwargs.items()
    }
    cond_kwargs["d"] = torch.tensor([0.0], device=device, dtype=torch.float32)
    return cond_args, cond_kwargs


def compute_pose_only_log_likelihoods(pipeline, ss_input_dict, samples, steps=25):
    """
    Pose-only log p(pose | shape, image) for each candidate sample.

    Uses exact 13D Jacobian trace (no Hutchinson noise).

    Args:
        pipeline:      loaded SAM3D inference pipeline
        ss_input_dict: preprocessed image dict (from pipeline.preprocess_image)
        samples:       list of dicts, each with "x1_latent" containing the raw
                       flow-matching latent dict captured before pose_decoder
        steps:         number of ODE steps for likelihood integration

    Returns:
        list of log-likelihood floats, one per candidate
    """
    ss_generator = pipeline.models["ss_generator"]
    device = ss_input_dict["image"].device
    latent_keys = list(ss_generator.reverse_fn.backbone.latent_mapping.keys())

    cond_args, cond_kwargs = _get_cond(pipeline, ss_input_dict)
    saved = _setup_cfg_off(ss_generator)

    log_likelihoods = []
    try:
        for idx, sample in enumerate(samples):
            x1 = {
                key: sample["x1_latent"][key].to(device, dtype=torch.float32)
                for key in latent_keys
            }
            with torch.autocast(device_type="cuda", dtype=pipeline.shape_model_dtype):
                ll = _run_pose_only_ode(ss_generator, x1, cond_args, cond_kwargs, steps, device)
            log_likelihoods.append(ll)
            seed = sample.get("sample_seed", sample.get("seed", "?"))
            logger.info(f"    sample {idx} (seed={seed}): pose log p = {ll:.4f}")
    finally:
        _restore_cfg(ss_generator, saved)

    return log_likelihoods


def compute_denoising_consistency_scores(
    pipeline, ss_input_dict, samples,
    t_noise=0.5, num_trials=3, denoise_steps=25,
    eval_seed=99999,
):
    """
    Denoising consistency score for each candidate's pose channels.

    For each candidate x1:
      - Add noise at time t_noise: x_t = (1-t)*x0 + t*x1
      - Run forward ODE from x_t to t=1 with image conditioning
      - Measure pose reconstruction error: -||x1_denoised[pose] - x1[pose]||^2

    Higher score (less negative) = model better preserves this pose = more consistent.
    Unlike log-likelihood, this is unbiased towards model-generated poses.

    No gradients needed -- purely forward passes.

    Args:
        pipeline:      loaded SAM3D inference pipeline
        ss_input_dict: preprocessed image dict
        samples:       list of dicts with 'x1_latent'
        t_noise:       noise level (0=pure noise, 1=no noise). Default 0.5.
        num_trials:    number of different noise samples to average over
        denoise_steps: number of Euler steps for the forward ODE
        eval_seed:     seed for reproducible noise generation

    Returns:
        list of scores (higher = more consistent)
    """
    ss_generator = pipeline.models["ss_generator"]
    device = ss_input_dict["image"].device
    latent_keys = list(ss_generator.reverse_fn.backbone.latent_mapping.keys())

    # Get condition embeddings
    cond_args, cond_kwargs = _get_cond(pipeline, ss_input_dict)
    # Strip d from kwargs -- _generate_dynamics takes it as positional arg
    cond_kwargs_clean = {k: v for k, v in cond_kwargs.items() if k != "d"}

    # Use generation-time CFG settings (NOT disabled -- we want the model's
    # best conditional prediction, which uses CFG during normal generation)
    ss_generator.no_shortcut = True
    ss_generator.eval()

    # Prepare forward time schedule: t_noise -> 1.0
    t_seq = torch.linspace(t_noise, 1.0, denoise_steps + 1).to(device)

    # Fixed RNG for reproducible noise
    rng = torch.Generator(device=device).manual_seed(eval_seed)

    d_zero = torch.tensor([0.0], device=device)

    scores = []

    with torch.no_grad():
        for idx, sample in enumerate(samples):
            x1 = {
                key: sample["x1_latent"][key].to(device, dtype=torch.float32)
                for key in latent_keys
            }

            trial_errors = []

            for trial in range(num_trials):
                # Sample noise x0
                x0 = {k: torch.randn(v.shape, device=device, generator=rng)
                      for k, v in x1.items()}

                # Noisy interpolation: x_t = (1-t)*x0 + t*x1
                x_t = {}
                for k in x1:
                    x_t[k] = (1.0 - t_noise) * x0[k] + t_noise * x1[k]

                # Forward ODE: denoise from t_noise to t=1
                with torch.autocast(device_type="cuda", dtype=pipeline.shape_model_dtype):
                    for i in range(len(t_seq) - 1):
                        t_val = t_seq[i]
                        dt = t_seq[i + 1] - t_seq[i]

                        velocity = ss_generator._generate_dynamics(
                            x_t, t_val, d_zero,
                            *cond_args, **cond_kwargs_clean,
                        )

                        x_t = {k: x_t[k] + velocity[k] * dt for k in x_t}

                # Measure pose reconstruction error
                error = 0.0
                for k in _POSE_KEYS_ORDERED:
                    diff = x_t[k].float() - x1[k].float()
                    error += (diff ** 2).sum().item()
                trial_errors.append(error)

            avg_error = sum(trial_errors) / len(trial_errors)
            score = -avg_error  # higher = more consistent
            scores.append(score)
            seed = sample.get("sample_seed", sample.get("seed", "?"))
            logger.info(
                f"    sample {idx} (seed={seed}): denoising consistency = {score:.6f} "
                f"(avg pose error = {avg_error:.6f})"
            )

    return scores
