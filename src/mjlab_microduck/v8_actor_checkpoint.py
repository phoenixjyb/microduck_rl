"""Exact-byte actor-only inference from a frozen v8 PPO checkpoint.

This deliberately does not reconstruct PPO, optimizer, critic, or runner state.
The checkpoint is authenticated once as bounded immutable bytes, and its actor
state is admitted only when it carries the exact pinned v8 seed and affine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import io
import os
from pathlib import Path
import re
import stat
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import torch
from tensordict import TensorDictBase

from . import community_policy_inspection as inspection
from . import normalized_onnx_reference as reference
from . import v8_actor_seed


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _pinned_affine() -> dict[str, torch.Tensor]:
    """Read the pinned local ONNX using the same static parser as actor import."""
    from . import official_native_reference

    spec = official_native_reference.V8
    raw = inspection._read_payload(spec.path)
    _require(hashlib.sha256(raw).hexdigest() == spec.sha256 == reference.OFFICIAL_V8_SHA256,
             "pinned v8 ONNX bytes changed")
    _, tensors = v8_actor_seed._parse(raw)
    return tensors


@dataclass(frozen=True, slots=True, init=False)
class FrozenV8Checkpoint:
    """Validated exact-byte checkpoint identity and immutable CPU actor state."""

    path: Path
    sha256: str
    _payload: bytes = field(repr=False, compare=False)
    _actor_state: Any = field(repr=False, compare=False)
    _pinned_sha256: str = field(repr=False, compare=False)

    def __init__(self, path: str | Path, expected_sha256: str):
        _require(type(expected_sha256) is str
                 and re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is not None,
                 "checkpoint SHA256 must be 64 lowercase hexadecimal characters")
        original = Path(path)
        raw = inspection._read_payload(original)
        info = original.stat(follow_symlinks=False)
        _require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid(),
                 "checkpoint must be an owned regular file")
        digest = hashlib.sha256(raw).hexdigest()
        _require(digest == expected_sha256, "checkpoint SHA256 mismatch")
        try:
            value = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ValueError("invalid weights-only PPO checkpoint") from exc
        _require(isinstance(value, Mapping) and "actor_state_dict" in value,
                 "checkpoint must contain actor_state_dict")
        state = value["actor_state_dict"]
        _require(isinstance(state, Mapping), "actor_state_dict mapping required")
        expected_keys = {
            *(f"mlp.{index}.{leaf}" for index, ins, outs in
              ((0, 61, 512), (2, 512, 256), (4, 256, 128), (6, 128, 14))
              for leaf in ("weight", "bias")),
            "obs_normalizer._mean", "obs_normalizer._std", "obs_normalizer._var",
            "obs_normalizer.count", "v8_seed_digest", "distribution.std_param",
        }
        _require(set(state) == expected_keys, "exact actor-only state layout required")
        for name, tensor in state.items():
            _require(isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu",
                     f"CPU tensor required for {name}")
            if name not in ("v8_seed_digest", "obs_normalizer.count"):
                _require(tensor.dtype == torch.float32,
                         f"float32 actor tensor required for {name}")
            elif name == "obs_normalizer.count":
                _require(tensor.dtype == torch.int64,
                         "canonical integer target normalizer count required")
            _require(bool(torch.isfinite(tensor).all()), f"finite actor tensor required for {name}")
        shapes = {
            "mlp.0.weight": (512, 61), "mlp.0.bias": (512,),
            "mlp.2.weight": (256, 512), "mlp.2.bias": (256,),
            "mlp.4.weight": (128, 256), "mlp.4.bias": (128,),
            "mlp.6.weight": (14, 128), "mlp.6.bias": (14,),
            "obs_normalizer._mean": (1, 61), "obs_normalizer._std": (1, 61),
            "obs_normalizer._var": (1, 61), "obs_normalizer.count": (),
            "v8_seed_digest": (32,), "distribution.std_param": (14,),
        }
        _require(all(tuple(state[name].shape) == shape for name, shape in shapes.items()),
                 "exact 61-512-256-128-14 actor tensor shapes required")
        _require(state["v8_seed_digest"].dtype == torch.uint8
                 and bytes(state["v8_seed_digest"].tolist()).hex() == reference.OFFICIAL_V8_SHA256,
                 "actor v8 seed digest mismatch")
        _require(state["obs_normalizer.count"].item() == 0,
                 "frozen actor normalizer count must be zero")
        std_param = state["distribution.std_param"]
        _require(bool((std_param > 0).all()) and bool(torch.isfinite(std_param).all()),
                 "positive finite scalar Gaussian std required")

        pinned = _pinned_affine()
        expected_mean = pinned["obs_normalizer._mean"]
        expected_std = pinned["onnx::Div_24"]
        expected_var = expected_std.square()
        _require(torch.equal(state["obs_normalizer._mean"], expected_mean)
                 and torch.equal(state["obs_normalizer._std"], expected_std)
                 and torch.equal(state["obs_normalizer._var"], expected_var),
                 "checkpoint frozen affine does not match pinned v8 ONNX")

        # Clone authenticated tensors so callers cannot alter the validated source.
        copied = MappingProxyType({name: tensor.detach().clone() for name, tensor in state.items()})
        object.__setattr__(self, "path", original.resolve(strict=True))
        object.__setattr__(self, "sha256", digest)
        object.__setattr__(self, "_payload", raw)
        object.__setattr__(self, "_actor_state", copied)
        object.__setattr__(self, "_pinned_sha256", reference.OFFICIAL_V8_SHA256)

    def matches_positional_checkpoint(self, checkpoint: str | Path) -> bool:
        """Bind this verified payload to evaluate_case's legacy positional slot."""
        candidate = Path(checkpoint)
        if candidate.resolve(strict=True) != self.path:
            return False
        raw = inspection._read_payload(candidate)
        return hashlib.sha256(raw).hexdigest() == self.sha256

    def require_initial_seed(self) -> None:
        """A fixed pre-learning baseline must contain the original mean actor."""
        pinned = _pinned_affine()
        _require(all(torch.equal(self._actor_state[name], value)
                     for name, value in pinned.items() if name.startswith("mlp."))
                 and bool((self._actor_state["distribution.std_param"] == .3).all()),
                 "initial baseline must equal the original v8 mean and new std0.3")

    def construct_actor(self, observations: TensorDictBase, agent: Any):
        """Build the stock task's exact custom actor and load actor tensors only."""
        _require(self._pinned_sha256 == reference.OFFICIAL_V8_SHA256,
                 "checkpoint was validated for another v8 seed")
        from .v8_actor_seed import FrozenV8Actor

        actor_cfg = agent.actor
        # Keep the runner config's stock dimensions and set the documented
        # custom class/distribution override for this ephemeral inference actor.
        actor_cfg.class_name = "mjlab_microduck.v8_actor_seed:FrozenV8Actor"
        actor_cfg.distribution_cfg = {
            "class_name": "rsl_rl.modules:GaussianDistribution",
            "init_std": 0.3,
            "std_type": "scalar",
        }
        groups = dict(agent.obs_groups)
        actor = FrozenV8Actor(
            observations, groups, "actor", 14,
            hidden_dims=tuple(actor_cfg.hidden_dims),
            activation=actor_cfg.activation,
            obs_normalization=actor_cfg.obs_normalization,
            distribution_cfg=actor_cfg.distribution_cfg,
        )
        actor.load_state_dict(dict(self._actor_state), strict=True)
        actor.eval()
        _require(actor.obs_normalizer.count.item() == 0 and actor.obs_normalizer.eps == 0.0
                 and actor.obs_normalizer.until == 0,
                 "loaded actor normalizer must remain frozen and empty")
        return actor

    def provenance(self) -> dict[str, Any]:
        return dict(backend="frozen-v8-ppo-actor-only", artifact_sha256=self.sha256,
                    artifact_bytes=len(self._payload), source_artifact_sha256=self._pinned_sha256,
                    normalization="pinned-frozen-affine", normalizer_count=0,
                    normalizer_frozen=True, normalizer_history_imported=False,
                    gaussian_state_imported=True, critic_state_imported=False,
                    optimizer_state_imported=False, runner_history_imported=False,
                    ppo_resume=False, policy_acceptance=False,
                    physical_motion_authorized=False)
