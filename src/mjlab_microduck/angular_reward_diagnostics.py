"""Measure the existing angular reward, never replace it or its gates."""

import math

import torch

from mjlab_microduck import integrated_turn_foundation as base

TERM = "track_angular_velocity"
PROTOCOL = "post-reward-angular-gaussian-decomposition-v1"


def capture(env):
    """Metrics hook: raw post-reward state and actual reward before auto-reset."""
    term = env.reward_manager.get_term_cfg(TERM)
    base.require(getattr(env, "_microduck_angular_reward_snapshot", None) is None,
                 "previous angular snapshot consumed")
    std, weight = term.params["std"], term.weight
    base.require(math.isfinite(std) and std > 0 and weight == 2., "known angular Gaussian parameters")
    # Stock cfg omits asset_cfg and uses the function's robot default.
    asset_cfg = term.params.get("asset_cfg")
    name = "robot" if asset_cfg is None else asset_cfg.name
    base.require(name == "robot", "known stock angular reward robot")
    angular = env.scene[name].data.root_link_ang_vel_b
    command = env.command_manager.get_command(term.params["command_name"])
    actual = env.reward_manager._step_reward[:, env.reward_manager.active_terms.index(TERM)]
    base.require(angular.shape == command.shape == (env.num_envs, 3)
                 and actual.shape == (env.num_envs,) and base.finite_tree((angular, command, actual)),
                 "finite raw angular reward state")
    z_error = command[:, 2]-angular[:, 2]
    xy_error_sq = angular[:, :2].square().sum(-1)
    expected = weight*torch.exp(-(z_error.square()+xy_error_sq)/std**2)
    base.require(torch.allclose(actual, expected, atol=1e-6, rtol=1e-5),
                 "actual stock angular reward matches full XYZ Gaussian")
    env._microduck_angular_reward_snapshot = torch.cat((angular, command[:, 2:3],
        actual[:, None]), -1).detach().clone()
    return torch.zeros(env.num_envs, device=env.device)


def summarize(samples, *, std=math.sqrt(.5), weight=2., startup=60):
    base.require(isinstance(samples, torch.Tensor) and samples.ndim == 3 and samples.shape[-1] == 5 and samples.shape[0] > 0
                 and samples.shape[1] > 0 and base.finite_tree(samples)
                 and type(startup) is int and startup >= 0 and std == math.sqrt(.5) and weight == 2.,
                 "finite known angular diagnostic trajectory")
    settled = samples[startup:].double()
    result = dict(protocol=PROTOCOL, steps=len(samples), startup_steps=startup,
        settled_steps=len(settled), std=std, weight=weight,
        timing="post-reward pre-reset body angular XYZ and current command; not admission timing",
        counterfactual="yaw-only Gaussian on same states; not a trained policy or causal benefit",
        policy_acceptance=False, physical_motion_authorized=False)
    if not len(settled):
        return {**result, "per_world": None}
    angular, yaw, actual = settled[..., :3], settled[..., 3], settled[..., 4]
    z = yaw-angular[..., 2]
    xy_sq = angular[..., :2].square().sum(-1)
    full = torch.exp(-(z.square()+xy_sq)/std**2)
    yaw_only = torch.exp(-z.square()/std**2)
    base.require(torch.allclose(actual, weight*full, atol=1e-6, rtol=1e-5), "retained actual angular reward")
    slope = weight*2*z.abs()/std**2
    result["per_world"] = dict(
        mean_body_angular_xyz=angular.mean(0).tolist(),
        rms_roll_pitch_rate=xy_sq.mean(0).sqrt().tolist(),
        yaw_error_rms=z.square().mean(0).sqrt().tolist(),
        yaw_bias_abs=z.mean(0).abs().tolist(),
        yaw_mae=z.abs().mean(0).tolist(),
        actual_weighted_reward_mean=actual.mean(0).tolist(),
        yaw_only_weighted_reward_mean=(weight*yaw_only).mean(0).tolist(),
        roll_pitch_multiplicative_factor_mean=torch.exp(-xy_sq/std**2).mean(0).tolist(),
        full_gaussian_below_point1_fraction=(full < .1).double().mean(0).tolist(),
        yaw_only_above_point8_full_below_point1_fraction=((yaw_only > .8) & (full < .1)).double().mean(0).tolist(),
        full_abs_yaw_gradient_mean=(slope*full).mean(0).tolist(),
        yaw_only_abs_yaw_gradient_mean=(slope*yaw_only).mean(0).tolist())
    base.require(base.finite_tree(result), "finite decomposition report")
    return result
