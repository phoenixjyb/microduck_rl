"""Checkpoint migration for the obstacle policy's seven new input channels."""

import argparse
import hashlib
import os
from copy import deepcopy
from pathlib import Path

import torch

ACTOR_BASE_DIM = 61
CRITIC_BASE_DIM = 76
# Kept local so this offline checkpoint utility does not import/register the
# entire task package. A unit test pins it to the v1 observation contract.
OBSTACLE_INPUT_DIM = 7


def _expand_normalizer(state: dict, old_dim: int, extra_dim: int) -> None:
    for name, fill in (("_mean", 0.0), ("_var", 1.0), ("_std", 1.0)):
        key = f"obs_normalizer.{name}"
        value = state[key]
        if value.shape != (1, old_dim):
            raise ValueError(f"{key} expected shape (1, {old_dim}), got {value.shape}")
        extension = value.new_full((1, extra_dim), fill)
        state[key] = torch.cat((value, extension), dim=1)


def _expand_first_layer(state: dict, old_dim: int, extra_dim: int) -> torch.Size:
    key = "mlp.0.weight"
    weight = state[key]
    if weight.ndim != 2 or weight.shape[1] != old_dim:
        raise ValueError(
            f"{key} expected second dimension {old_dim}, got {tuple(weight.shape)}"
        )
    old_shape = weight.shape
    state[key] = torch.cat(
        (weight, weight.new_zeros((weight.shape[0], extra_dim))), dim=1
    )
    return old_shape


def _expand_optimizer_moments(
    optimizer_state: dict,
    old_shape: torch.Size,
    extra_dim: int,
) -> None:
    matches = []
    for parameter_id, state in optimizer_state["state"].items():
        exp_avg = state.get("exp_avg")
        if isinstance(exp_avg, torch.Tensor) and exp_avg.shape == old_shape:
            matches.append((parameter_id, state))
    if len(matches) != 1:
        raise ValueError(
            f"expected one optimizer state with shape {tuple(old_shape)}, "
            f"found {len(matches)}"
        )
    _, state = matches[0]
    for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
        value = state.get(name)
        if value is None:
            continue
        if value.shape != old_shape:
            raise ValueError(
                f"optimizer {name} expected shape {tuple(old_shape)}, "
                f"got {tuple(value.shape)}"
            )
        state[name] = torch.cat(
            (value, value.new_zeros((value.shape[0], extra_dim))), dim=1
        )


def migrate_obstacle_checkpoint(checkpoint: dict, source_sha256: str) -> dict:
    """Return a strict-loadable 61/76D -> 68/83D warm-start checkpoint.

    Existing actor and critic columns, normalizer statistics, optimizer moments,
    iteration, and all downstream parameters are preserved. New observation
    columns and their optimizer moments are zero; new normalizer channels use
    mean 0 and variance/std 1. Consequently, zero obstacle input produces the
    algebraically identical pre-migration policy function (subject only to
    floating-point kernel rounding).
    """
    required = {
        "actor_state_dict",
        "critic_state_dict",
        "optimizer_state_dict",
        "iter",
        "infos",
    }
    missing = required - checkpoint.keys()
    if missing:
        raise ValueError(f"checkpoint missing keys: {sorted(missing)}")

    migrated = deepcopy(checkpoint)
    actor = migrated["actor_state_dict"]
    critic = migrated["critic_state_dict"]
    extra_dim = OBSTACLE_INPUT_DIM

    actor_old_shape = _expand_first_layer(actor, ACTOR_BASE_DIM, extra_dim)
    critic_old_shape = _expand_first_layer(critic, CRITIC_BASE_DIM, extra_dim)
    _expand_normalizer(actor, ACTOR_BASE_DIM, extra_dim)
    _expand_normalizer(critic, CRITIC_BASE_DIM, extra_dim)
    _expand_optimizer_moments(
        migrated["optimizer_state_dict"], actor_old_shape, extra_dim
    )
    _expand_optimizer_moments(
        migrated["optimizer_state_dict"], critic_old_shape, extra_dim
    )

    infos = migrated.setdefault("infos", {})
    infos["obstacle_warm_start"] = {
        "source_sha256": source_sha256,
        "source_iteration": checkpoint["iter"],
        "actor_dims": [ACTOR_BASE_DIM, ACTOR_BASE_DIM + extra_dim],
        "critic_dims": [CRITIC_BASE_DIM, CRITIC_BASE_DIM + extra_dim],
        "new_input_columns": "zero",
        "new_normalizer_mean": 0.0,
        "new_normalizer_variance": 1.0,
        "optimizer_moments": "old columns preserved; new columns zero",
    }
    return migrated


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def migrate_checkpoint_file(source: Path, destination: Path) -> dict:
    """Migrate one trusted local checkpoint without overwriting either path."""
    source = source.resolve()
    destination = destination.resolve()
    if source == destination:
        raise ValueError("source and destination must differ")
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    source_sha256 = sha256_file(source)
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    migrated = migrate_obstacle_checkpoint(checkpoint, source_sha256)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        torch.save(migrated, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return migrated["infos"]["obstacle_warm_start"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    metadata = migrate_checkpoint_file(args.source, args.destination)
    print(f"wrote={args.destination.resolve()}")
    print(f"source_sha256={metadata['source_sha256']}")
    print(f"actor_dims={metadata['actor_dims']}")
    print(f"critic_dims={metadata['critic_dims']}")


if __name__ == "__main__":
    main()
