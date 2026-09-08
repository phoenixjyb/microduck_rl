"""First-attempt numerical scoring; not a checkpoint/provenance admission CLI."""

import torch

from mjlab_microduck.stance_transition import EPISODE_STEPS, physical_failures


@torch.no_grad()
def score_attempt(state, positions, soft_exposure, steps, *, rejected_proposed_torque=None):
    """Score unreset, every-physics-boundary evidence, including the initial state.

    Full five-second evidence has indices0..2500. A short prefix must remain
    incomplete or end exactly on its first failure. No post-terminal recovery,
    frame skipping, smoothing or checkpoint selection is allowed here. The caller
    still has to bind hashes, checkpoint, seed, environment ID and runtime source.
    """
    if steps.ndim != 1 or steps.dtype != torch.long or not 1 <= len(steps) <= EPISODE_STEPS+1:
        raise ValueError('bounded integer first-attempt steps required')
    n = len(steps); device = steps.device
    if not torch.equal(steps, torch.arange(n, device=device)):
        raise ValueError('continuous first-attempt physics boundaries required')
    state.validate(n, device)
    if (positions.shape != (n, 3) or positions.device != device
            or not positions.is_floating_point() or not torch.isfinite(positions).all()):
        raise ValueError('finite world root positions required')
    if soft_exposure.shape != (n, 14) or soft_exposure.dtype != torch.bool or soft_exposure.device != device:
        raise ValueError('named joint-time soft-limit mask required')
    failed = physical_failures(state, steps)
    indices = failed.nonzero(as_tuple=False).flatten()
    if len(indices) and int(indices[0]) != n-1:
        raise ValueError('post-terminal frames cannot enter first-attempt scoring')
    proposed_failure = rejected_proposed_torque is not None
    if proposed_failure:
        torque = rejected_proposed_torque
        if (torque.shape != (14,) or not torque.is_floating_point()
                or not torch.isfinite(torque).all() or not bool((torque.abs() > .36).any())):
            raise ValueError('actual excessive proposed torque evidence required')
    full_duration = n == EPISODE_STEPS+1
    hard_failure = bool(len(indices)) or proposed_failure
    complete = full_duration or hard_failure
    displacement = torch.linalg.vector_norm(positions[:, :2]-positions[0, :2], dim=1)
    if not torch.isfinite(displacement).all():
        raise ValueError('nonfinite displacement arithmetic')
    last_second = steps >= 2000  # Inclusive boundary window4.000..5.000 seconds.
    support_window = steps > 100  # Strictly after0.2 seconds.
    metrics = dict(maximum_displacement_m=float(displacement.max()),
        soft_limit_exposure_fraction=float(soft_exposure[1:].double().mean()) if n > 1 else None,
        final_second_tilt_p95_rad=None, final_second_planar_speed_p95_mps=None,
        final_second_minimum_height_m=None, both_feet_support_fraction=None)
    if full_duration:
        metrics.update(
            final_second_tilt_p95_rad=float(torch.quantile(state.tilt[last_second].double(), .95, interpolation='linear')),
            final_second_planar_speed_p95_mps=float(torch.quantile(
                torch.linalg.vector_norm(state.root_velocity[last_second, :2].double(), dim=1), .95, interpolation='linear')),
            final_second_minimum_height_m=float(state.height[last_second].min()),
            both_feet_support_fraction=float((state.support[support_window] > .01).all(dim=1).double().mean()))
    gates = dict(complete_first_attempt=complete, full_duration=full_duration,
        no_hard_failure=not hard_failure, displacement=metrics['maximum_displacement_m'] <= .02,
        soft_limit=metrics['soft_limit_exposure_fraction'] is not None and metrics['soft_limit_exposure_fraction'] <= .01,
        final_tilt=full_duration and metrics['final_second_tilt_p95_rad'] <= .0873,
        final_speed=full_duration and metrics['final_second_planar_speed_p95_mps'] <= .03,
        final_height=full_duration and metrics['final_second_minimum_height_m'] >= .105,
        foot_support=full_duration and metrics['both_feet_support_fraction'] >= .99)
    return dict(candidate_pass=all(gates.values()), gates=gates, metrics=metrics,
        last_physics_step=n-1, hard_failure=hard_failure, complete_first_attempt=complete,
        checkpoint_admitted=False, provenance_validated=False, learned_stance_accepted=False)
