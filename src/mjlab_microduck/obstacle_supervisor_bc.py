"""Offline imitation training for bounded obstacle-supervisor stages."""

from __future__ import annotations

import argparse
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
)
from mjlab_microduck.tasks.obstacle_observation import (
    DEFAULT_OBSTACLE_OBSERVATION_LIMITS,
)

HC2_STAGE = "HC2-behavioral-cloning"
HC4L_STAGE = "HC4L-lateral-behavioral-cloning"
HC4R_STAGE = "HC4R-near-range-behavioral-cloning"
HC4R2_STAGE = "HC4R2-student-state-correction-BC"
HC4U1_STAGE = "HC4U1-unified-range-lateral-correction-BC"
HC4U2_STAGE = "HC4U2-far-center-student-state-correction-BC"
HC4U3_STAGE = "HC4U3-phase-separated-BC"
HC4LH_STAGE = "HC4LH-lateral-gated-supervisor"
HC4R2H_STAGE = "HC4R2H-range-speed-gated-supervisor"
HC4R2L_STAGE = "HC4R2L-episode-latched-supervisor"
SUPPORTED_BC_STAGES = (
    HC2_STAGE,
    HC4L_STAGE,
    HC4R_STAGE,
    HC4R2_STAGE,
    HC4U1_STAGE,
    HC4U2_STAGE,
    HC4U3_STAGE,
)

# HC4-U1 is a frozen one-candidate experiment. Order is part of the contract
# because it affects episode-key namespacing, the held-out split, and training.
HC4U1_REQUIRED_DATASET_SHA256 = (
    "660cdfa8b618f8af425baf0e2f9c3d7b01d59eab93fca24848c9a82a84408467",
    "18d4faf1b37c8fd9982677bd2bff7635f5d254483b8641bd9745315219cf38be",
    "0fddb6412ea39595bdceeba7bc762d3397f89af49d32811dd783e807be6314ea",
    "76da0fd8fb9efe99e332ba9d7e787f7d3c607417ee48b906bb3091a6e941f15f",
    "9bc85efa2917c16fb007fa51647e87c1115e87dd469dd7eb267c384af3a1fad9",
    "3d9d24355457e033e42448dfb1b71438443ee65186b4f6d644d20127b6264026",
    "69c8238505a4f60f8de9f17816993947597fbbf5920253898117e6667a961f06",
    "e145e34c9ac61cc3e2778151139847cbc98ede0be2dea4850066e584652bc08a",
    "3f20078db7cfc0e460ab0564de608735d5c919c50df5efe4d7dc6eb53fcfb7ca",
    "e871dda49fbe2155f0e9fbc3699abd609a0c711a757dde824c889107527fbce5",
)
HC4U2_REQUIRED_DATASET_SHA256 = (
    *HC4U1_REQUIRED_DATASET_SHA256,
    "8bbf3560faeb2758c88b5326b5d35975d57db685d6ad47bde83ad691ac55fb71",
    "efa3ceec37ea9e4b9677175b38c696a8452463a76c122ca103c1693c363142fb",
    "1fc67b945436700a1aa9a5fe711188ec80e7f9d1c26e0b15eb478279d5538bfb",
)
HC4U2_STUDENT_SHA256 = (
    "2196d2ed2dbc3e182fa0b36edf663d11187330d430cd319ceb368c8a28e9753b"
)


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


class PhaseSeparatedSupervisor(torch.nn.Module):
    """Independent approach, interaction, and recovery fits over the same 17D input."""

    def __init__(self, cfg: SupervisorBcCfg = SupervisorBcCfg()) -> None:
        super().__init__()
        self.experts = torch.nn.ModuleList(
            ObstacleSupervisor(cfg) for _ in ObstaclePhase
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.ndim != 2 or observation.shape[1] != SUPERVISOR_OBSERVATION_DIM:
            raise ValueError("phase supervisor requires an (N, 17) observation")
        phase = observation[:, 13:16]
        valid = ((phase == 0) | (phase == 1)).all(dim=-1) & (phase.sum(-1) == 1)
        # Malformed phase state must not blend contradictory controllers.
        weights = torch.where(valid[:, None], phase, torch.zeros_like(phase))
        commands = torch.stack([expert(observation) for expert in self.experts], dim=1)
        selected = (commands * weights.unsqueeze(-1)).sum(dim=1)
        return torch.where(valid[:, None], selected, torch.zeros_like(selected))


def phase_partition_counts(
    observations: torch.Tensor, mask: torch.Tensor
) -> dict[str, int]:
    """Reject missing or malformed maneuver phases before an HC4-U3 fit."""
    selected = observations[mask]
    if selected.ndim != 2 or selected.shape[1] != SUPERVISOR_OBSERVATION_DIM:
        raise ValueError("phase training requires an (N, 17) observation")
    phase = selected[:, 13:16]
    if not bool(torch.isfinite(selected).all()) or not bool(
        (((phase == 0) | (phase == 1)).all(-1) & (phase.sum(-1) == 1)).all()
    ):
        raise ValueError("phase training observations are non-finite or not one-hot")
    counts = {p.name.lower(): int((phase[:, int(p)] == 1).sum()) for p in ObstaclePhase}
    if any(count == 0 for count in counts.values()):
        raise ValueError("every phase needs samples in each episode partition")
    return counts


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


def obstacle_route_position_from_supervisor_observation(
    observation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recover obstacle route-frame position and validity from compact input."""
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
    obstacle_route_forward = (
        torch.cos(route_heading) * relative_x
        - torch.sin(route_heading) * relative_y
    )
    obstacle_route_lateral = (
        route_lateral
        + torch.sin(route_heading) * relative_x
        + torch.cos(route_heading) * relative_y
    )
    valid = observation[:, 7] > 0.5
    return obstacle_route_forward, obstacle_route_lateral, valid


def obstacle_route_lateral_from_supervisor_observation(
    observation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover obstacle route-lateral position and validity from compact input."""
    _, obstacle_route_lateral, valid = (
        obstacle_route_position_from_supervisor_observation(observation)
    )
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


class RangeSpeedGatedSupervisor(torch.nn.Module):
    """Delegate only the accepted near/slow envelope to an HC4-R2 specialist.

    Invalid geometry and every out-of-envelope observation retain the far
    supervisor's command. The execution layer remains responsible for its
    immediate stop on invalid geometry.
    """

    def __init__(
        self,
        far_supervisor: torch.nn.Module,
        near_supervisor: torch.nn.Module,
        *,
        near_range_gate_m: float = 0.95,
        max_near_nominal_speed_mps: float = 0.40,
    ) -> None:
        super().__init__()
        limits = ObstacleTeacherCfg()
        if near_range_gate_m <= 0.0:
            raise ValueError("near_range_gate_m must be positive")
        if not 0.0 < max_near_nominal_speed_mps <= limits.max_forward_speed_mps:
            raise ValueError(
                "max_near_nominal_speed_mps is outside the command envelope"
            )
        self.far_supervisor = far_supervisor
        self.near_supervisor = near_supervisor
        self.near_range_gate_m = near_range_gate_m
        self.max_near_nominal_speed_mps = max_near_nominal_speed_mps

    def select_near(self, observation: torch.Tensor) -> torch.Tensor:
        """Return the instantaneous valid range/speed eligibility mask."""
        obstacle_forward, _, valid = (
            obstacle_route_position_from_supervisor_observation(observation)
        )
        nominal_speed_mps = observation[:, 0].clamp(0.0, 1.0) * (
            ObstacleTeacherCfg().max_forward_speed_mps
        )
        return (
            valid
            & (obstacle_forward <= self.near_range_gate_m)
            & (nominal_speed_mps <= self.max_near_nominal_speed_mps + 1.0e-6)
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        use_near = self.select_near(observation)
        far_command = self.far_supervisor(observation)
        near_command = self.near_supervisor(observation)
        return torch.where(use_near.unsqueeze(-1), near_command, far_command)


class EpisodeLatchedRangeSpeedSupervisor(RangeSpeedGatedSupervisor):
    """Choose a range/speed specialist once and retain it until episode reset."""

    def __init__(
        self,
        far_supervisor: torch.nn.Module,
        near_supervisor: torch.nn.Module,
        *,
        near_range_gate_m: float = 0.95,
        max_near_nominal_speed_mps: float = 0.40,
    ) -> None:
        super().__init__(
            far_supervisor,
            near_supervisor,
            near_range_gate_m=near_range_gate_m,
            max_near_nominal_speed_mps=max_near_nominal_speed_mps,
        )
        self.register_buffer(
            "_episode_initialized",
            torch.empty(0, dtype=torch.bool),
            persistent=False,
        )
        self.register_buffer(
            "_episode_use_near",
            torch.empty(0, dtype=torch.bool),
            persistent=False,
        )

    def _ensure_episode_state(self, observation: torch.Tensor) -> None:
        batch_size = observation.shape[0]
        if (
            self._episode_initialized.shape != (batch_size,)
            or self._episode_initialized.device != observation.device
        ):
            self._episode_initialized = torch.zeros(
                batch_size, dtype=torch.bool, device=observation.device
            )
            self._episode_use_near = torch.zeros_like(self._episode_initialized)

    def reset_episodes(self, reset_mask: torch.Tensor) -> None:
        """Clear routing decisions only for environments whose episode ended."""
        if self._episode_initialized.numel() == 0:
            raise RuntimeError("cannot reset episode routing before first observation")
        reset_mask = reset_mask.to(
            device=self._episode_initialized.device, dtype=torch.bool
        )
        if reset_mask.shape != self._episode_initialized.shape:
            raise ValueError("reset mask does not match supervisor batch")
        keep = ~reset_mask
        self._episode_initialized = self._episode_initialized & keep
        self._episode_use_near = self._episode_use_near & keep

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        instantaneous_use_near = self.select_near(observation)
        self._ensure_episode_state(observation)
        new_episode = ~self._episode_initialized
        self._episode_use_near = torch.where(
            new_episode, instantaneous_use_near, self._episode_use_near
        )
        self._episode_initialized = self._episode_initialized | new_episode
        far_command = self.far_supervisor(observation)
        near_command = self.near_supervisor(observation)
        return torch.where(
            self._episode_use_near.unsqueeze(-1), near_command, far_command
        )


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


def validate_bc_dataset_contract(
    stage: str,
    payloads: list[dict],
    dataset_sha256: tuple[str, ...],
) -> None:
    """Enforce stage-specific dataset provenance before fitting a candidate."""
    if len(payloads) != len(dataset_sha256):
        raise ValueError("dataset payload and hash counts do not match")
    if stage == HC4U3_STAGE:
        # Architecture is the sole changed axis: keep HC4-U2's ordered corpus.
        validate_bc_dataset_contract(HC4U2_STAGE, payloads, dataset_sha256)
        return
    dataset_stages = {payload.get("stage") for payload in payloads}
    if stage == HC4R2_STAGE:
        required_stages = {
            "HC1-successful-teacher-trajectories",
            "HC4R2-student-state-teacher-corrections",
        }
        if not required_stages.issubset(dataset_stages):
            raise ValueError(
                "HC4R2 training requires both HC1 teacher and student-state "
                "correction datasets"
            )
    if stage == HC4U1_STAGE:
        if dataset_sha256 != HC4U1_REQUIRED_DATASET_SHA256:
            raise ValueError(
                "HC4U1 training requires the exact ordered predeclared dataset set"
            )
        if dataset_stages != {
            "HC1-successful-teacher-trajectories",
            "HC4R2-student-state-teacher-corrections",
        }:
            raise ValueError("HC4U1 dataset stages do not match the frozen contract")
    if stage == HC4U2_STAGE:
        if dataset_sha256 != HC4U2_REQUIRED_DATASET_SHA256:
            raise ValueError(
                "HC4U2 training requires the exact ordered predeclared dataset set"
            )
        if dataset_stages != {
            "HC1-successful-teacher-trajectories",
            "HC4R2-student-state-teacher-corrections",
        }:
            raise ValueError("HC4U2 dataset stages do not match the frozen contract")
        if any(
            payload.get("student_supervisor_checkpoint_sha256")
            != HC4U2_STUDENT_SHA256
            for payload in payloads[-3:]
        ):
            raise ValueError("HC4U2 correction shards name the wrong student")


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
    progress_path = output_path.with_suffix(".progress.pt")
    if stage == HC4U3_STAGE and progress_path.exists():
        raise FileExistsError(progress_path)
    payloads = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in dataset_paths
    ]
    dataset_sha256 = tuple(_sha256(path) for path in dataset_paths)
    validate_bc_dataset_contract(stage, payloads, dataset_sha256)
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
    phase_counts = None
    if stage == HC4U3_STAGE:
        phase_counts = {
            "training": phase_partition_counts(observations, training_mask),
            "validation": phase_partition_counts(observations, validation_mask),
        }
    model_type = PhaseSeparatedSupervisor if stage == HC4U3_STAGE else ObstacleSupervisor
    model = model_type(cfg).to(device)
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
            if stage == HC4U3_STAGE and not bool(torch.isfinite(loss)):
                raise ValueError("HC4-U3 training loss is non-finite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            validation_loss = float(
                torch.nn.functional.mse_loss(model(validation_x), validation_y)
            )
        if stage == HC4U3_STAGE and not math.isfinite(validation_loss):
            raise ValueError("HC4-U3 validation loss is non-finite")
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        if stage == HC4U3_STAGE and (epoch == 1 or epoch % 10 == 0):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = progress_path.with_suffix(".tmp")
            torch.save({
                "schema_version": 1,
                "stage": stage,
                "decision": "intermediate-not-eligible-for-rollout",
                "epoch": epoch,
                "seed": seed,
                "model_config": asdict(cfg),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "sampler_generator_state": generator.get_state(),
                "best_model_state_dict": best_state,
                "best_epoch": best_epoch,
                "best_validation_mse": best_loss,
                "source_locomotion_checkpoint_sha256": next(iter(source_hashes)),
                "dataset_sha256": dataset_sha256,
                "physical_motion_authorized": False,
            }, temporary)
            os.replace(temporary, progress_path)
            print(
                f"HC4U3 epoch={epoch}/{epochs} validation_mse={validation_loss:.9g} "
                f"checkpoint={progress_path}", flush=True,
            )

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
            {"path": str(path), "sha256": sha256}
            for path, sha256 in zip(dataset_paths, dataset_sha256, strict=True)
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
    if stage == HC4U3_STAGE:
        checkpoint["architecture"] = "three-independent-phase-experts-v1"
        checkpoint["phase_samples"] = phase_counts
        with torch.inference_mode():
            validation_phases = observations[validation_mask, 13:16]
            checkpoint["phase_validation_metrics"] = {
                phase.name.lower(): _error_metrics(
                    validation_prediction[validation_phases[:, int(phase)] == 1],
                    commands[validation_mask][validation_phases[:, int(phase)] == 1],
                )
                for phase in ObstaclePhase
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
