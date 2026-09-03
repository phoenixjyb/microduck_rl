"""Offline HC2 imitation training for the bounded obstacle supervisor."""

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
    ObstacleTeacherCfg,
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

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        raw = self.network(observation)
        return torch.stack((torch.sigmoid(raw[:, 0]), torch.tanh(raw[:, 1])), dim=-1)


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
    dataset_path: Path,
    output_path: Path,
    *,
    epochs: int = 200,
    batch_size: int = 1024,
    seed: int = 42,
    cfg: SupervisorBcCfg = SupervisorBcCfg(),
) -> Path:
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    dataset_path = dataset_path.resolve(strict=True)
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    payload = torch.load(dataset_path, map_location="cpu", weights_only=False)
    observations = payload["observations"].float()
    commands = payload["commands"].float()
    episode_keys = payload["episode_keys"].long()
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
        "stage": "HC2-behavioral-cloning",
        "decision": "offline-imitation-pass" if passed_offline_gate else "rejected",
        "rollout_acceptance_required": True,
        "model_state_dict": best_state,
        "model_config": asdict(cfg),
        "teacher_config": asdict(ObstacleTeacherCfg()),
        "dataset": str(dataset_path),
        "dataset_sha256": _sha256(dataset_path),
        "source_locomotion_checkpoint_sha256": payload.get("checkpoint_sha256"),
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
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_supervisor(
        args.dataset,
        args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
