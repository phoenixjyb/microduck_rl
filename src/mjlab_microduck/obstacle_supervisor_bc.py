"""Offline imitation training for bounded obstacle-supervisor stages."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from mjlab_microduck.hierarchical_obstacle import (
    SUPERVISOR_OBSERVATION_DIM,
    ObstaclePhase,
    ObstacleTeacherCfg,
)
from mjlab_microduck.tasks.obstacle_observation import (
    DEFAULT_OBSTACLE_OBSERVATION_LIMITS,
)

HC2_STAGE = "HC2-behavioral-cloning"
HC4L_STAGE = "HC4L-lateral-behavioral-cloning"
HC4R_STAGE = "HC4R-near-range-behavioral-cloning"
HC4LH_STAGE = "HC4LH-lateral-gated-supervisor"
SUPPORTED_BC_STAGES = (HC2_STAGE, HC4L_STAGE, HC4R_STAGE)


@dataclass(frozen=True)
class SupervisorBcCfg:
    hidden_dims: tuple[int, int] = (64, 64)
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-5
    validation_fraction: float = 0.20
    speed_mae_gate_mps: float = 0.025
    yaw_mae_gate_rps: float = 0.050

    def __post_init__(self) -> None:
        if len(self.hidden_dims) != 2 or any(width <= 0 for width in self.hidden_dims):
            raise ValueError("hidden_dims must contain two positive widths")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("optimizer values are invalid")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in (0, 1)")


class ObstacleSupervisor(torch.nn.Module):
    """Small policy that can command only bounded forward speed and yaw."""

    def __init__(self, cfg: SupervisorBcCfg = SupervisorBcCfg()) -> None:
        super().__init__()
        h1, h2 = cfg.hidden_dims
        self.network = torch.nn.Sequential(
            torch.nn.Linear(SUPERVISOR_OBSERVATION_DIM, h1),
            torch.nn.ELU(),
            torch.nn.Linear(h1, h2),
            torch.nn.ELU(),
            torch.nn.Linear(h2, 2),
        )

    def raw_action(self, observation: torch.Tensor) -> torch.Tensor:
        """Return the unconstrained latent command used by BC and HC3 PPO."""
        return self.network(observation)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        raw = self.raw_action(observation)
        return torch.stack((torch.sigmoid(raw[:, 0]), torch.tanh(raw[:, 1])), dim=-1)


def interaction_speed_only_command(
    observation: torch.Tensor,
    speed_latent: torch.Tensor,
    hc2_command: torch.Tensor,
    *,
    min_interaction_speed_mps: float = 0.30,
) -> torch.Tensor:
    """Compose a normalized command with speed-only interaction authority.

    The caller supplies the frozen HC2 command for yaw and for speed outside
    interaction. During interaction, the learned speed is bounded between the
    measured minimum and nominal.
    """
    if observation.ndim != 2 or observation.shape[1] != SUPERVISOR_OBSERVATION_DIM:
        raise ValueError(
            f"observation must have shape (N, {SUPERVISOR_OBSERVATION_DIM})"
        )
    if speed_latent.shape != (observation.shape[0], 1):
        raise ValueError("speed_latent must have shape (N, 1)")
    if hc2_command.shape != (observation.shape[0], 2):
        raise ValueError("hc2_command must have shape (N, 2)")
    limits = ObstacleTeacherCfg()
    if not 0.0 <= min_interaction_speed_mps <= limits.max_forward_speed_mps:
        raise ValueError("minimum interaction speed is outside the command envelope")

    nominal_normalized = observation[:, 0].clamp(0.0, 1.0)
    minimum_normalized = min_interaction_speed_mps / limits.max_forward_speed_mps
    learned_speed = torch.sigmoid(speed_latent[:, 0])
    learned_speed = torch.maximum(
        learned_speed, torch.full_like(learned_speed, minimum_normalized)
    )
    learned_speed = torch.minimum(learned_speed, nominal_normalized)
    interaction_index = -4 + int(ObstaclePhase.INTERACTION)
    interaction = observation[:, interaction_index] > 0.5
    speed = torch.where(interaction, learned_speed, hc2_command[:, 0])
    return torch.stack((speed, hc2_command[:, 1]), dim=-1)


class InteractionSpeedOnlySupervisor(torch.nn.Module):
    """Execution wrapper that fixes HC2 yaw and bounds speed authority by phase."""

    def __init__(
        self,
        supervisor: ObstacleSupervisor,
        hc2_supervisor: ObstacleSupervisor,
        *,
        min_interaction_speed_mps: float = 0.30,
    ) -> None:
        super().__init__()
        self.supervisor = supervisor
        self.hc2_supervisor = hc2_supervisor
        self.min_interaction_speed_mps = min_interaction_speed_mps

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        raw = self.supervisor.raw_action(observation)
        hc2_command = self.hc2_supervisor(observation)
        return interaction_speed_only_command(
            observation,
            raw[:, :1],
            hc2_command,
            min_interaction_speed_mps=self.min_interaction_speed_mps,
        )


def obstacle_route_lateral_from_supervisor_observation(
    observation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover obstacle route-lateral position and validity from compact input."""
    if observation.ndim != 2 or observation.shape[1] != SUPERVISOR_OBSERVATION_DIM:
        raise ValueError(
            f"observation must have shape (N, {SUPERVISOR_OBSERVATION_DIM})"
        )
    limits = DEFAULT_OBSTACLE_OBSERVATION_LIMITS
    surface_range = observation[:, 1] * limits.max_range_m
    width = observation[:, 4] * limits.max_width_m
    center_range = surface_range + width / 2.0
    relative_x = center_range * observation[:, 3]
    relative_y = center_range * observation[:, 2]
    route_lateral = observation[:, 8] * 0.75
    route_heading = observation[:, 9] * torch.pi
    obstacle_route_lateral = (
        route_lateral
        + torch.sin(route_heading) * relative_x
        + torch.cos(route_heading) * relative_y
    )
    valid = observation[:, 7] > 0.5
    return obstacle_route_lateral, valid


class LateralGatedSupervisor(torch.nn.Module):
    """Use HC2 at center and an HC4-L specialist only for shifted obstacles."""

    def __init__(
        self,
        center_supervisor: ObstacleSupervisor,
        lateral_supervisor: ObstacleSupervisor,
        *,
        lateral_gate_m: float = 0.06,
    ) -> None:
        super().__init__()
        if lateral_gate_m <= 0.0:
            raise ValueError("lateral_gate_m must be positive")
        self.center_supervisor = center_supervisor
        self.lateral_supervisor = lateral_supervisor
        self.lateral_gate_m = lateral_gate_m

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        obstacle_lateral, valid = obstacle_route_lateral_from_supervisor_observation(
            observation
        )
        use_lateral = valid & (obstacle_lateral.abs() >= self.lateral_gate_m)
        center_command = self.center_supervisor(observation)
        lateral_command = self.lateral_supervisor(observation)
        return torch.where(use_lateral.unsqueeze(-1), lateral_command, center_command)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_episode_keys(
    episode_keys: torch.Tensor, validation_fraction: float, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    unique = torch.unique(episode_keys.cpu())
    if unique.numel() < 2:
        raise ValueError("dataset needs at least two successful episodes")
    generator = torch.Generator().manual_seed(seed)
    unique = unique[torch.randperm(unique.numel(), generator=generator)]
    validation_count = max(1, round(unique.numel() * validation_fraction))
    validation_keys = unique[:validation_count]
    validation_mask = torch.isin(episode_keys.cpu(), validation_keys)
    training_mask = ~validation_mask
    if not bool(training_mask.any()) or not bool(validation_mask.any()):
        raise ValueError("episode split produced an empty partition")
    return training_mask, validation_mask


def _physical_commands(normalized: torch.Tensor) -> torch.Tensor:
    limits = ObstacleTeacherCfg()
    return torch.stack(
        (
            normalized[:, 0] * limits.max_forward_speed_mps,
            normalized[:, 1] * limits.max_yaw_rate_rps,
        ),
        dim=-1,
    )


def _error_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict:
    error = (_physical_commands(prediction) - target).abs()
    return {
        "speed_mae_mps": float(error[:, 0].mean()),
        "speed_p95_abs_error_mps": float(torch.quantile(error[:, 0], 0.95)),
        "yaw_mae_rps": float(error[:, 1].mean()),
        "yaw_p95_abs_error_rps": float(torch.quantile(error[:, 1], 0.95)),
    }


def train_supervisor(
    dataset_paths: tuple[Path, ...],
    output_path: Path,
    *,
    epochs: int = 200,
    batch_size: int = 1024,
    seed: int = 42,
    cfg: SupervisorBcCfg = SupervisorBcCfg(),
    stage: str = HC2_STAGE,
) -> Path:
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    if stage not in SUPPORTED_BC_STAGES:
        raise ValueError(f"unsupported behavioral-cloning stage: {stage}")
    if not dataset_paths:
        raise ValueError("at least one dataset is required")
    dataset_paths = tuple(path.resolve(strict=True) for path in dataset_paths)
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    payloads = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in dataset_paths
    ]
    source_hashes = {payload.get("checkpoint_sha256") for payload in payloads}
    if len(source_hashes) != 1 or None in source_hashes:
        raise ValueError("datasets do not share one frozen locomotion checkpoint")
    observations = torch.cat(
        [payload["observations"].float() for payload in payloads]
    )
    commands = torch.cat([payload["commands"].float() for payload in payloads])
    episode_keys = torch.cat(
        [
            payload["episode_keys"].long() + index * 1_000_000_000
            for index, payload in enumerate(payloads)
        ]
    )
    if observations.ndim != 2 or observations.shape[1] != SUPERVISOR_OBSERVATION_DIM:
        raise ValueError("dataset has the wrong supervisor observation shape")
    if commands.shape != (observations.shape[0], 2):
        raise ValueError("dataset commands must have shape (N, 2)")
    if episode_keys.shape != (observations.shape[0],):
        raise ValueError("dataset episode keys must have shape (N,)")
    if not torch.isfinite(observations).all() or not torch.isfinite(commands).all():
        raise ValueError("dataset contains non-finite values")

    training_mask, validation_mask = split_episode_keys(
        episode_keys, cfg.validation_fraction, seed
    )
    target_normalized = torch.stack(
        (
            commands[:, 0] / ObstacleTeacherCfg().max_forward_speed_mps,
            commands[:, 1] / ObstacleTeacherCfg().max_yaw_rate_rps,
        ),
        dim=-1,
    )
    device = "cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES", "") else "cpu"
    torch.manual_seed(seed)
    model = ObstacleSupervisor(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    train_x = observations[training_mask].to(device)
    train_y = target_normalized[training_mask].to(device)
    validation_x = observations[validation_mask].to(device)
    validation_y = target_normalized[validation_mask].to(device)
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    generator = torch.Generator(device=device).manual_seed(seed)

    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(train_x.shape[0], generator=generator, device=device)
        for start in range(0, train_x.shape[0], batch_size):
            indices = permutation[start : start + batch_size]
            loss = torch.nn.functional.mse_loss(model(train_x[indices]), train_y[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            validation_loss = float(
                torch.nn.functional.mse_loss(model(validation_x), validation_y)
            )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        training_prediction = model(observations[training_mask].to(device)).cpu()
        validation_prediction = model(observations[validation_mask].to(device)).cpu()
    training_metrics = _error_metrics(
        training_prediction, commands[training_mask]
    )
    validation_metrics = _error_metrics(
        validation_prediction, commands[validation_mask]
    )
    passed_offline_gate = (
        validation_metrics["speed_mae_mps"] <= cfg.speed_mae_gate_mps
        and validation_metrics["yaw_mae_rps"] <= cfg.yaw_mae_gate_rps
    )
    checkpoint = {
        "schema_version": 1,
        "stage": stage,
        "decision": "offline-imitation-pass" if passed_offline_gate else "rejected",
        "rollout_acceptance_required": True,
        "model_state_dict": best_state,
        "model_config": asdict(cfg),
        "teacher_config": asdict(ObstacleTeacherCfg()),
        "datasets": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in dataset_paths
        ],
        "source_locomotion_checkpoint_sha256": next(iter(source_hashes)),
        "samples": int(observations.shape[0]),
        "successful_episodes": int(torch.unique(episode_keys).numel()),
        "training_samples": int(training_mask.sum()),
        "validation_samples": int(validation_mask.sum()),
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_validation_mse": best_loss,
        "training_metrics": training_metrics,
        "validation_metrics": validation_metrics,
        "seed": seed,
        "device": device,
        "physical_motion_authorized": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    manifest_path = output_path.with_suffix(".json")
    manifest = {key: value for key, value in checkpoint.items() if key != "model_state_dict"}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage", choices=SUPPORTED_BC_STAGES, default=HC2_STAGE)
    args = parser.parse_args()
    train_supervisor(
        tuple(args.datasets),
        args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        stage=args.stage,
    )


if __name__ == "__main__":
    main()
