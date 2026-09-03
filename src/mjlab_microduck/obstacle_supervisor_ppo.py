"""HC3 PPO fine-tuning of the bounded obstacle supervisor over a frozen gait.

Only the 17D-to-2D supervisor and its value function are optimized. The loaded
61D locomotion actor remains in inference mode and never enters the optimizer.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from mjlab_microduck.hierarchical_obstacle import (
    SUPERVISOR_OBSERVATION_DIM,
    ObstaclePhase,
    ObstacleTeacherCfg,
    advance_obstacle_state,
    apply_bounded_supervisor_command,
    make_teacher_state,
    reset_teacher_state,
    supervisor_observation,
)
from mjlab_microduck.hierarchical_obstacle_rollout import (
    BASE_TASK_ID,
    _route_state,
    load_learned_supervisor,
    prepare_rollout_configs,
)
from mjlab_microduck.obstacle_supervisor_bc import ObstacleSupervisor, SupervisorBcCfg


@dataclass(frozen=True)
class Hc3RewardCfg:
    progress_scale: float = 2.0
    time_penalty: float = 0.01
    speed_error_scale: float = 0.25
    recovery_lateral_scale: float = 0.20
    recovery_heading_scale: float = 0.05
    close_clearance_m: float = 0.08
    close_clearance_scale: float = 2.0
    clean_pass_reward: float = 8.0
    collision_penalty: float = 12.0
    timeout_penalty: float = 4.0
    fall_penalty: float = 12.0
    nonfinite_penalty: float = 20.0

    def __post_init__(self) -> None:
        values = asdict(self)
        if any(value < 0.0 for value in values.values()):
            raise ValueError("HC3 reward coefficients must be non-negative")


@dataclass(frozen=True)
class Hc3PpoCfg:
    iterations: int = 40
    rollout_steps: int = 64
    low_level_steps_per_action: int = 5
    update_epochs: int = 4
    minibatches: int = 4
    learning_rate: float = 1.0e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.20
    value_loss_scale: float = 0.50
    entropy_scale: float = 0.002
    anchor_scale: float = 0.05
    max_grad_norm: float = 0.50
    initial_log_std: tuple[float, float] = (-1.5, -1.5)

    def __post_init__(self) -> None:
        integer_values = (
            self.iterations,
            self.rollout_steps,
            self.low_level_steps_per_action,
            self.update_epochs,
            self.minibatches,
        )
        if any(value <= 0 for value in integer_values):
            raise ValueError("HC3 iteration and batch values must be positive")
        if self.rollout_steps % self.minibatches:
            raise ValueError("rollout_steps must be divisible by minibatches")
        if self.learning_rate <= 0.0 or self.max_grad_norm <= 0.0:
            raise ValueError("HC3 optimizer values must be positive")
        if not 0.0 < self.gamma <= 1.0 or not 0.0 < self.gae_lambda <= 1.0:
            raise ValueError("HC3 discount values must be in (0, 1]")
        if not 0.0 < self.clip_ratio < 1.0:
            raise ValueError("HC3 clip_ratio must be in (0, 1)")
        if any(value < 0.0 for value in (
            self.value_loss_scale,
            self.entropy_scale,
            self.anchor_scale,
        )):
            raise ValueError("HC3 loss coefficients must be non-negative")
        if len(self.initial_log_std) != 2:
            raise ValueError("initial_log_std must contain speed and yaw values")


class SupervisorValue(torch.nn.Module):
    def __init__(self, hidden_dims: tuple[int, int] = (64, 64)) -> None:
        super().__init__()
        h1, h2 = hidden_dims
        self.network = torch.nn.Sequential(
            torch.nn.Linear(SUPERVISOR_OBSERVATION_DIM, h1),
            torch.nn.ELU(),
            torch.nn.Linear(h1, h2),
            torch.nn.ELU(),
            torch.nn.Linear(h2, 1),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation).squeeze(-1)


def normalized_command_from_latent(latent: torch.Tensor) -> torch.Tensor:
    if latent.ndim != 2 or latent.shape[1] != 2:
        raise ValueError("HC3 latent command must have shape (N, 2)")
    return torch.stack(
        (torch.sigmoid(latent[:, 0]), torch.tanh(latent[:, 1])), dim=-1
    )


def hc3_reward(
    progress_delta_m: torch.Tensor,
    route_lateral_error_m: torch.Tensor,
    route_heading_error_rad: torch.Tensor,
    measured_speed_mps: torch.Tensor,
    nominal_speed_mps: torch.Tensor,
    obstacle_surface_range_m: torch.Tensor,
    phase: torch.Tensor,
    collision: torch.Tensor,
    clean_pass: torch.Tensor,
    timeout: torch.Tensor,
    fell: torch.Tensor,
    nonfinite: torch.Tensor,
    *,
    cfg: Hc3RewardCfg = Hc3RewardCfg(),
) -> torch.Tensor:
    """Return supervisor reward without interaction-phase speed pressure."""
    tensors = (
        progress_delta_m,
        route_lateral_error_m,
        route_heading_error_rad,
        measured_speed_mps,
        nominal_speed_mps,
        obstacle_surface_range_m,
        phase,
        collision,
        clean_pass,
        timeout,
        fell,
        nonfinite,
    )
    if any(value.ndim != 1 for value in tensors):
        raise ValueError("HC3 reward inputs must be one-dimensional")
    if any(value.shape != progress_delta_m.shape for value in tensors[1:]):
        raise ValueError("HC3 reward inputs must share one batch shape")

    terminal = collision | clean_pass | timeout | fell | nonfinite
    shaping = cfg.progress_scale * progress_delta_m.clamp(-0.05, 0.10)
    shaping -= cfg.time_penalty

    track_speed = phase != int(ObstaclePhase.INTERACTION)
    shaping -= (
        cfg.speed_error_scale
        * (measured_speed_mps - nominal_speed_mps).abs()
        * track_speed
    )
    recovery = phase == int(ObstaclePhase.RECOVERY)
    shaping -= cfg.recovery_lateral_scale * route_lateral_error_m.abs() * recovery
    shaping -= cfg.recovery_heading_scale * route_heading_error_rad.abs() * recovery
    shaping -= cfg.close_clearance_scale * (
        cfg.close_clearance_m - obstacle_surface_range_m
    ).clamp_min(0.0)
    reward = torch.where(terminal, torch.zeros_like(shaping), shaping)
    reward += cfg.clean_pass_reward * clean_pass
    reward -= cfg.collision_penalty * collision
    reward -= cfg.timeout_penalty * timeout
    reward -= cfg.fall_penalty * fell
    reward -= cfg.nonfinite_penalty * nonfinite
    return reward


def generalized_advantage_estimate(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    bootstrap_value: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if rewards.shape != values.shape or rewards.shape != dones.shape:
        raise ValueError("reward, value, and done tensors must share shape")
    if rewards.ndim != 2 or bootstrap_value.shape != rewards.shape[1:]:
        raise ValueError("invalid HC3 rollout tensor shape")
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros_like(bootstrap_value)
    next_value = bootstrap_value
    for step in range(rewards.shape[0] - 1, -1, -1):
        not_done = (~dones[step]).to(rewards.dtype)
        delta = rewards[step] + gamma * next_value * not_done - values[step]
        gae = delta + gamma * gae_lambda * not_done * gae
        advantages[step] = gae
        next_value = values[step]
    return advantages, advantages + values


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone() for name, value in module.state_dict().items()
    }


def train_hc3_supervisor(
    locomotion_checkpoint: Path,
    hc2_checkpoint: Path,
    output_path: Path,
    *,
    num_envs: int = 128,
    nominal_speed_mps: float = 0.5,
    obstacle_forward_m: float = 1.15,
    obstacle_lateral_m: float = 0.0,
    seed: int = 73,
    ppo_cfg: Hc3PpoCfg = Hc3PpoCfg(),
    reward_cfg: Hc3RewardCfg = Hc3RewardCfg(),
) -> Path:
    """Fine-tune HC2 in simulation while keeping the locomotion actor frozen."""
    if not 1 <= num_envs <= 256:
        raise ValueError("num_envs must be in [1, 256]")
    if not 0.0 < nominal_speed_mps <= ObstacleTeacherCfg().max_forward_speed_mps:
        raise ValueError("nominal speed is outside the frozen gait envelope")
    if obstacle_forward_m <= 0.0:
        raise ValueError("obstacle forward position must be positive")
    locomotion_checkpoint = locomotion_checkpoint.resolve(strict=True)
    hc2_checkpoint = hc2_checkpoint.resolve(strict=True)
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(output_path)

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_runner_cls
    from mjlab_microduck.tasks import mdp as microduck_mdp

    device = (
        "cuda:0"
        if torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES", "")
        else "cpu"
    )
    torch.manual_seed(seed)
    env_cfg, agent_cfg = prepare_rollout_configs(
        num_envs,
        nominal_speed_mps,
        obstacle_forward_m,
        obstacle_lateral_m,
    )
    env_cfg.seed = seed
    agent_cfg.seed = seed
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    try:
        runner_cls = load_runner_cls(BASE_TASK_ID)
        assert runner_cls is not None
        runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
        runner.load(str(locomotion_checkpoint), map_location=device)
        locomotion_policy = runner.get_inference_policy(device=device)

        actor = load_learned_supervisor(
            hc2_checkpoint, locomotion_checkpoint, device
        )
        actor.train()
        anchor = copy.deepcopy(actor).eval()
        for parameter in anchor.parameters():
            parameter.requires_grad_(False)
        hc2_payload = torch.load(hc2_checkpoint, map_location="cpu", weights_only=False)
        model_cfg = dict(hc2_payload["model_config"])
        hidden_dims = tuple(model_cfg["hidden_dims"])
        value_function = SupervisorValue(hidden_dims).to(device)
        log_std = torch.nn.Parameter(
            torch.tensor(ppo_cfg.initial_log_std, device=device)
        )
        optimizer = torch.optim.Adam(
            [*actor.parameters(), *value_function.parameters(), log_std],
            lr=ppo_cfg.learning_rate,
        )

        observations = wrapped.get_observations()
        state = make_teacher_state(
            num_envs, device=device, nominal_speed_mps=nominal_speed_mps
        )
        nominal = torch.full((num_envs,), nominal_speed_mps, device=device)
        command = env.command_manager.get_command("twist")
        command[:, 0] = nominal
        command[:, 1:] = 0.0
        limits = ObstacleTeacherCfg()

        def policy_observation() -> tuple[torch.Tensor, torch.Tensor]:
            route_lateral, route_heading, route_speed = _route_state(env)
            obstacle = microduck_mdp.obstacle_geometry_observation(
                env,
                asset_name="obstacle",
                width_m=0.20,
                height_m=0.10,
                horizontal_fov_rad=2.0 * math.pi,
                max_range_m=2.0,
            )
            previous_command = state.previous_command.clone()
            advance_obstacle_state(
                obstacle, route_lateral, route_heading, state, cfg=limits
            )
            return (
                supervisor_observation(
                    obstacle,
                    nominal,
                    route_lateral,
                    route_heading,
                    route_speed,
                    state,
                    cfg=limits,
                    previous_command=previous_command,
                ),
                obstacle,
            )

        current_supervisor_observation, current_obstacle = policy_observation()
        history: list[dict] = []
        generator = torch.Generator(device=device).manual_seed(seed)

        for iteration in range(1, ppo_cfg.iterations + 1):
            rollout_observations = []
            rollout_latents = []
            rollout_log_probs = []
            rollout_values = []
            rollout_rewards = []
            rollout_dones = []
            iteration_events = {
                "clean_pass": 0,
                "collision": 0,
                "timeout": 0,
                "fall": 0,
                "nonfinite": 0,
            }

            for _ in range(ppo_cfg.rollout_steps):
                with torch.no_grad():
                    latent_mean = actor.raw_action(current_supervisor_observation)
                    distribution = torch.distributions.Normal(
                        latent_mean, log_std.exp().expand_as(latent_mean)
                    )
                    latent = distribution.sample()
                    old_log_prob = distribution.log_prob(latent).sum(dim=-1)
                    value = value_function(current_supervisor_observation)
                    normalized_command = normalized_command_from_latent(latent)

                desired_command = torch.stack(
                    (
                        normalized_command[:, 0] * limits.max_forward_speed_mps,
                        normalized_command[:, 1] * limits.max_yaw_rate_rps,
                    ),
                    dim=-1,
                )
                supervisor_command = apply_bounded_supervisor_command(
                    desired_command, current_obstacle, state, cfg=limits
                )
                command[:, 0] = supervisor_command[:, 0]
                command[:, 1] = 0.0
                command[:, 2] = supervisor_command[:, 1]

                robot_xy = env.scene["robot"].data.root_link_pos_w[:, :2]
                progress_before = (
                    (robot_xy - env._obstacle_route_origin_w)
                    * env._obstacle_path_dir_w
                ).sum(dim=-1)
                phase = state.phase.clone()
                surface_range_m = current_obstacle[:, 0] * 2.0
                done = torch.zeros(num_envs, dtype=torch.bool, device=device)
                collision = torch.zeros_like(done)
                clean_pass = torch.zeros_like(done)
                timeout = torch.zeros_like(done)
                fell = torch.zeros_like(done)
                nonfinite = torch.zeros_like(done)

                for _ in range(ppo_cfg.low_level_steps_per_action):
                    with torch.no_grad():
                        low_level_action = locomotion_policy(observations)
                        observations, low_reward, low_done, _ = wrapped.step(
                            low_level_action
                        )
                    new_done = low_done.bool() & ~done
                    collision |= (
                        env.termination_manager.get_term("obstacle_collision")
                        & new_done
                    )
                    clean_pass |= (
                        env.termination_manager.get_term("obstacle_passed") & new_done
                    )
                    timeout |= (
                        env.termination_manager.get_term("obstacle_attempt_timeout")
                        & new_done
                    )
                    fell |= env.termination_manager.get_term("fell_over") & new_done
                    nan_term = env.termination_manager.get_term("nan_state") & new_done
                    finite = torch.isfinite(low_level_action).all(dim=-1)
                    finite &= torch.isfinite(low_reward)
                    nonfinite |= nan_term | (~finite & ~done)
                    done |= low_done.bool() | nonfinite
                    if bool(new_done.any()):
                        reset_teacher_state(
                            state,
                            new_done,
                            nominal_speed_mps=nominal_speed_mps,
                        )
                        command[new_done, 0] = nominal_speed_mps
                        command[new_done, 1:] = 0.0

                robot_xy = env.scene["robot"].data.root_link_pos_w[:, :2]
                progress_after = (
                    (robot_xy - env._obstacle_route_origin_w)
                    * env._obstacle_path_dir_w
                ).sum(dim=-1)
                progress_delta = torch.where(
                    done, torch.zeros_like(progress_after), progress_after - progress_before
                )
                route_lateral, route_heading, route_speed = _route_state(env)
                reward = hc3_reward(
                    progress_delta,
                    route_lateral,
                    route_heading,
                    route_speed,
                    nominal,
                    surface_range_m,
                    phase,
                    collision,
                    clean_pass,
                    timeout,
                    fell,
                    nonfinite,
                    cfg=reward_cfg,
                )

                rollout_observations.append(current_supervisor_observation.detach())
                rollout_latents.append(latent.detach())
                rollout_log_probs.append(old_log_prob.detach())
                rollout_values.append(value.detach())
                rollout_rewards.append(reward.detach())
                rollout_dones.append(done.detach())
                for name, mask in (
                    ("clean_pass", clean_pass),
                    ("collision", collision),
                    ("timeout", timeout),
                    ("fall", fell),
                    ("nonfinite", nonfinite),
                ):
                    iteration_events[name] += int(mask.sum())
                current_supervisor_observation, current_obstacle = policy_observation()

            with torch.no_grad():
                bootstrap_value = value_function(current_supervisor_observation)
            batch_observations = torch.stack(rollout_observations)
            batch_latents = torch.stack(rollout_latents)
            batch_old_log_probs = torch.stack(rollout_log_probs)
            batch_values = torch.stack(rollout_values)
            batch_rewards = torch.stack(rollout_rewards)
            batch_dones = torch.stack(rollout_dones)
            advantages, returns = generalized_advantage_estimate(
                batch_rewards,
                batch_values,
                batch_dones,
                bootstrap_value,
                gamma=ppo_cfg.gamma,
                gae_lambda=ppo_cfg.gae_lambda,
            )

            flat_observations = batch_observations.flatten(0, 1)
            flat_latents = batch_latents.flatten(0, 1)
            flat_old_log_probs = batch_old_log_probs.flatten()
            flat_returns = returns.flatten()
            flat_advantages = advantages.flatten()
            flat_advantages = (
                flat_advantages - flat_advantages.mean()
            ) / flat_advantages.std().clamp_min(1.0e-6)
            batch_size = flat_observations.shape[0]
            minibatch_size = batch_size // ppo_cfg.minibatches
            update_metrics = []

            for _ in range(ppo_cfg.update_epochs):
                permutation = torch.randperm(
                    batch_size, generator=generator, device=device
                )
                for start in range(0, batch_size, minibatch_size):
                    indices = permutation[start : start + minibatch_size]
                    obs = flat_observations[indices]
                    latent_mean = actor.raw_action(obs)
                    distribution = torch.distributions.Normal(
                        latent_mean, log_std.exp().expand_as(latent_mean)
                    )
                    new_log_prob = distribution.log_prob(
                        flat_latents[indices]
                    ).sum(dim=-1)
                    ratio = torch.exp(new_log_prob - flat_old_log_probs[indices])
                    unclipped = ratio * flat_advantages[indices]
                    clipped = ratio.clamp(
                        1.0 - ppo_cfg.clip_ratio, 1.0 + ppo_cfg.clip_ratio
                    ) * flat_advantages[indices]
                    policy_loss = -torch.minimum(unclipped, clipped).mean()
                    value_loss = torch.nn.functional.mse_loss(
                        value_function(obs), flat_returns[indices]
                    )
                    with torch.no_grad():
                        anchor_command = anchor(obs)
                    actor_command = normalized_command_from_latent(latent_mean)
                    anchor_loss = torch.nn.functional.mse_loss(
                        actor_command, anchor_command
                    )
                    entropy = distribution.entropy().sum(dim=-1).mean()
                    loss = (
                        policy_loss
                        + ppo_cfg.value_loss_scale * value_loss
                        - ppo_cfg.entropy_scale * entropy
                        + ppo_cfg.anchor_scale * anchor_loss
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [*actor.parameters(), *value_function.parameters(), log_std],
                        ppo_cfg.max_grad_norm,
                    )
                    optimizer.step()
                    log_std.data.clamp_(-3.0, -0.3)
                    update_metrics.append(
                        (float(policy_loss), float(value_loss), float(anchor_loss))
                    )

            metrics = torch.tensor(update_metrics).mean(dim=0)
            resolved = sum(
                iteration_events[name]
                for name in ("clean_pass", "collision", "timeout")
            )
            record = {
                "iteration": iteration,
                "mean_reward": float(batch_rewards.mean()),
                "clean_pass_events": iteration_events["clean_pass"],
                "collision_events": iteration_events["collision"],
                "timeout_events": iteration_events["timeout"],
                "fall_events": iteration_events["fall"],
                "nonfinite_events": iteration_events["nonfinite"],
                "resolved_attempts": resolved,
                "clean_pass_rate": (
                    iteration_events["clean_pass"] / resolved if resolved else None
                ),
                "policy_loss": float(metrics[0]),
                "value_loss": float(metrics[1]),
                "anchor_loss": float(metrics[2]),
                "log_std": [float(value) for value in log_std.detach().cpu()],
            }
            history.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)

        checkpoint = {
            "schema_version": 1,
            "stage": "HC3-supervisor-PPO",
            "decision": "training-complete-pending-rollout",
            "rollout_acceptance_required": True,
            "model_state_dict": _cpu_state_dict(actor),
            "value_state_dict": _cpu_state_dict(value_function),
            "log_std": log_std.detach().cpu(),
            "model_config": hc2_payload["model_config"],
            "ppo_config": asdict(ppo_cfg),
            "reward_config": asdict(reward_cfg),
            "training_cell": {
                "nominal_speed_mps": nominal_speed_mps,
                "obstacle_forward_m": obstacle_forward_m,
                "obstacle_lateral_m": obstacle_lateral_m,
                "num_envs": num_envs,
            },
            "source_supervisor_checkpoint": str(hc2_checkpoint),
            "source_supervisor_checkpoint_sha256": _sha256(hc2_checkpoint),
            "source_locomotion_checkpoint": str(locomotion_checkpoint),
            "source_locomotion_checkpoint_sha256": _sha256(locomotion_checkpoint),
            "seed": seed,
            "device": device,
            "history": history,
            "physical_motion_authorized": False,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, output_path)
        manifest = {
            key: value
            for key, value in checkpoint.items()
            if key not in {"model_state_dict", "value_state_dict", "log_std"}
        }
        output_path.with_suffix(".json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        return output_path
    finally:
        env.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("locomotion_checkpoint", type=Path)
    parser.add_argument("hc2_checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--nominal-speed", type=float, default=0.5)
    parser.add_argument("--obstacle-forward", type=float, default=1.15)
    parser.add_argument("--obstacle-lateral", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--rollout-steps", type=int, default=64)
    args = parser.parse_args()
    train_hc3_supervisor(
        args.locomotion_checkpoint,
        args.hc2_checkpoint,
        args.output,
        num_envs=args.num_envs,
        nominal_speed_mps=args.nominal_speed,
        obstacle_forward_m=args.obstacle_forward,
        obstacle_lateral_m=args.obstacle_lateral,
        seed=args.seed,
        ppo_cfg=Hc3PpoCfg(
            iterations=args.iterations, rollout_steps=args.rollout_steps
        ),
    )


if __name__ == "__main__":
    main()
