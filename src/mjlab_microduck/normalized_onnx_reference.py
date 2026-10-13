"""Pinned CPU-only inference adapter for the archived official VelStand ONNX.

The caller supplies raw actor observations. The model contains its exported
normalizer; this module performs no feature remapping, external normalization,
action scaling, or filtering. It is an inference adapter, not a policy or plant
acceptance claim.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from mjlab_microduck import community_policy_inspection as policy_inspection
from mjlab_microduck import community_policy_normalizer as policy_normalizer

if TYPE_CHECKING:
    from tensordict import TensorDictBase
    from mjlab.envs import ManagerBasedRlEnvCfg


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


OFFICIAL_VELSTAND_SHA256 = "1c659be55da94bc5753b707de5c6a3e7c49931e05ca3b6991615cef1a8ba9a45"
OFFICIAL_V8_SHA256 = "2fc7f8ee6acd7d0d3f66a99927b5d5a56589aeed02ec714de329b252d5ccf492"
OBSERVATION_NAMES = (
    "base_ang_vel",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
    "actions",
    "command",
    "head_command",
    "body_command",
)
OBSERVATION_NAMES_CSV = ",".join(OBSERVATION_NAMES)
COMMAND_NAMES_CSV = "twist,head_pose,body_pose"
JOINT_NAMES = (
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
)
JOINT_NAMES_CSV = ",".join(JOINT_NAMES)
ACTOR_TERM_ORDER = OBSERVATION_NAMES
MAX_BATCH_SIZE = 8


def validate_actor_contract(cfg: ManagerBasedRlEnvCfg) -> tuple[str, ...]:
    """Validate actor term order and that cfg adds no scales or history.

    This deliberately does not import a simulator, resolve term dimensions, or
    infer semantics from a 61-wide tensor. The native observation manager owns
    dimension validation when a collector is integrated.
    """
    observations = getattr(cfg, "observations", None)
    _require(isinstance(observations, Mapping), "cfg observations mapping required")
    actor = observations.get("actor")
    _require(actor is not None, "actor observation group required")
    terms = getattr(actor, "terms", None)
    _require(isinstance(terms, Mapping), "actor terms mapping required")
    _require(tuple(terms) == ACTOR_TERM_ORDER,
            "actor terms must use exact 8-term raw observation order")
    _require(getattr(actor, "concatenate_terms", False) is True,
            "actor terms must be concatenated")
    _require(getattr(actor, "history_length", None) is None,
            "actor group history must be disabled")

    for name, term in terms.items():
        _require(getattr(term, "scale", None) is None,
                f"actor term {name} must not add an external scale")
        _require(getattr(term, "history_length", 0) == 0,
                f"actor term {name} must not add history")

    # The archived policy emits raw normalized-network actions which the
    # training-time joint-position action passes through at scale one. Bind
    # that boundary and the native BAM actuator from the cfg without importing
    # simulator/runtime modules into this adapter.
    actions = getattr(cfg, "actions", None)
    _require(isinstance(actions, Mapping) and "joint_pos" in actions,
             "native joint_pos action cfg required")
    action = actions["joint_pos"]
    _require(type(action).__name__ == "JointPositionActionCfg",
             "native JointPositionActionCfg required")
    scale = getattr(action, "scale", None)
    _require(type(scale) in (float, int) and float(scale) == 1.0,
             "native joint_pos action scale must be exactly 1.0")
    scene_entities = getattr(getattr(cfg, "scene", None), "entities", None)
    _require(isinstance(scene_entities, Mapping) and "robot" in scene_entities,
             "native robot entity cfg required")
    articulation = getattr(scene_entities["robot"], "articulation", None)
    actuators = getattr(articulation, "actuators", None)
    _require(isinstance(actuators, (tuple, list)) and len(actuators) == 1
             and type(actuators[0]).__name__ == "FrictionDRBamActuatorCfg",
             "native single FrictionDRBam actuator cfg required")
    return ACTOR_TERM_ORDER


class NormalizedOnnxReference:
    """One of two pinned API-1 models, with exact-byte CPU ONNX Runtime inference."""

    def __init__(self, path: str | Path, expected_sha256: str):
        _require(type(expected_sha256) is str
                 and re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is not None,
                 "expected SHA256 must be 64 lowercase hexadecimal characters")
        # Resolve these module constants at construction time so contract tests
        # can substitute a fixture digest for the original pinned v1 artifact.
        # Production accepts only the two explicit official policy digests.
        _require(expected_sha256 in (OFFICIAL_VELSTAND_SHA256, OFFICIAL_V8_SHA256),
                 "only the pinned official VelStand artifacts are accepted")

        payload = policy_inspection._read_payload(Path(path))
        digest = hashlib.sha256(payload).hexdigest()
        _require(digest == expected_sha256, "policy SHA256 mismatch")

        # The graph check rejects external tensors/custom domains and binds the
        # exact API-1 signature and metadata observation-term order before ORT.
        static = policy_normalizer.inspect_affine_prefix(payload, expected_sha256)
        _require(static["decision"] == "recognized-static-affine-prefix-only",
                 "pinned static normalizer prefix required")
        _require([tuple(term[key] for key in ("term", "start", "stop"))
                  for term in static["terms"]]
                 == [tuple(term[key] for key in ("term", "start", "stop"))
                     for term in self._expected_static_terms()],
                 "pinned observation feature order required")
        self._metadata = MappingProxyType(self._read_and_validate_metadata(payload))

        # Import the runtime only for explicit construction. Inference receives
        # the same verified immutable bytes, never a path reopened after hashing.
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - depends on optional runtime
            raise RuntimeError("onnxruntime is required for CPU reference inference") from exc

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        if hasattr(ort, "ExecutionMode"):
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self._session = ort.InferenceSession(
            payload,
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        _require(self._session.get_providers() == ["CPUExecutionProvider"],
                 "CPUExecutionProvider must be the sole active provider")

        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        _require(len(inputs) == 1 and len(outputs) == 1,
                 "exactly one ONNX input and output required")
        input_arg, output_arg = inputs[0], outputs[0]
        _require(input_arg.name == "obs" and input_arg.type == "tensor(float)"
                 and input_arg.shape == [1, 61],
                 "runtime input must be obs float32 [1,61]")
        _require(output_arg.name == "actions" and output_arg.type == "tensor(float)"
                 and output_arg.shape == [1, 14],
                 "runtime output must be actions float32 [1,14]")

        self._path = Path(path)
        self._artifact_sha256 = digest
        self._artifact_bytes = len(payload)
        self._input_name = input_arg.name
        self._output_name = output_arg.name

    @staticmethod
    def _expected_static_terms() -> list[dict[str, Any]]:
        return [
            dict(term="base_ang_vel", start=0, stop=3),
            dict(term="projected_gravity", start=3, stop=6),
            dict(term="joint_pos", start=6, stop=20),
            dict(term="joint_vel", start=20, stop=34),
            dict(term="actions", start=34, stop=48),
            dict(term="command", start=48, stop=51),
            dict(term="head_command", start=51, stop=55),
            dict(term="body_command", start=55, stop=61),
        ]

    @staticmethod
    def _read_and_validate_metadata(payload: bytes) -> dict[str, str]:
        import onnx

        try:
            model = onnx.load_model_from_string(payload)
        except Exception as exc:
            raise ValueError("invalid ONNX model metadata") from exc
        metadata: dict[str, str] = {}
        for item in model.metadata_props:
            _require(item.key not in metadata, "duplicate ONNX metadata key")
            metadata[item.key] = item.value
        expected = {
            "observation_names": OBSERVATION_NAMES_CSV,
            "command_names": COMMAND_NAMES_CSV,
            "joint_names": JOINT_NAMES_CSV,
            "action_scale": "1.0",
        }
        for key, value in expected.items():
            _require(metadata.get(key) == value,
                     f"official ONNX metadata mismatch: {key}")
        home_text = metadata.get("default_joint_pos", "")
        try:
            home = tuple(float(value) for value in home_text.split(","))
        except ValueError as exc:
            raise ValueError("official ONNX metadata mismatch: default_joint_pos") from exc
        _require(len(home) == 14 and all(math.isfinite(v) for v in home),
                 "official ONNX metadata mismatch: default_joint_pos")
        metadata["default_joint_pos"] = home_text
        return metadata

    @property
    def artifact_sha256(self) -> str:
        return self._artifact_sha256

    @property
    def artifact_bytes(self) -> int:
        return self._artifact_bytes

    @property
    def metadata(self) -> Mapping[str, str]:
        return self._metadata

    @property
    def provider(self) -> str:
        return "CPUExecutionProvider"

    @property
    def normalization(self) -> str:
        return "baked-in-onnx-only"

    def provenance(self) -> dict[str, Any]:
        """Describe inference identity; this artifact is not a PPO checkpoint."""
        return {
            "backend": "onnxruntime-cpu",
            "artifact_sha256": self._artifact_sha256,
            "artifact_bytes": self._artifact_bytes,
            "runtime_provider": self.provider,
            "normalization": self.normalization,
            "action_output": "raw-unfiltered",
            "ppo_checkpoint": False,
            "optimizer_or_runner_state": False,
            "source_policy_lineage_claim": False,
        }

    def validate_robot(self, robot: Any, joint_ids: tuple[int, ...],
                       names: tuple[str, ...]) -> None:
        """Bind the ONNX servo order and rounded HOME to the live robot entity."""
        import torch

        names = tuple(names)
        joint_ids = tuple(joint_ids)
        _require(names == JOINT_NAMES,
                 "robot motor stream names must match official ONNX joint order")
        _require(len(joint_ids) == 14 and all(type(i) is int and i >= 0 for i in joint_ids)
                 and len(set(joint_ids)) == 14,
                 "robot motor stream requires 14 unique joint ids")
        robot_names = tuple(getattr(robot, "joint_names", ()))
        _require(robot_names and max(joint_ids) < len(robot_names),
                 "robot joint-name layout does not contain motor ids")
        _require(tuple(robot_names[i] for i in joint_ids) == names,
                 "robot joint ids do not match named motor stream")
        defaults = getattr(getattr(robot, "data", None), "default_joint_pos", None)
        _require(isinstance(defaults, torch.Tensor) and defaults.ndim == 2
                 and defaults.shape[0] >= 1 and max(joint_ids) < defaults.shape[1],
                 "robot default_joint_pos must be a batched joint tensor")
        observed = defaults.detach().to(device="cpu", dtype=torch.float64)[:, list(joint_ids)]
        _require(bool(torch.isfinite(observed).all()),
                 "robot default_joint_pos must be finite")
        expected = torch.tensor(
            [float(value) for value in self._metadata["default_joint_pos"].split(",")],
            dtype=torch.float64,
        )
        _require(bool((observed - expected[None, :]).abs().max() <= 0.00051),
                 "robot HOME differs from rounded official ONNX metadata")

    def __call__(self, observations: TensorDictBase):
        """Run raw actor rows independently at batch 1; return raw actions."""
        import numpy as np
        import torch
        from tensordict import TensorDictBase

        _require(isinstance(observations, TensorDictBase),
                 "observations must be a TensorDict")
        actor = observations.get("actor")
        _require(isinstance(actor, torch.Tensor), "TensorDict actor tensor required")
        _require(actor.dtype == torch.float32, "actor observations must be float32")
        _require(actor.ndim == 2 and actor.shape[1] == 61,
                 "actor observations must have shape [N,61]")
        _require(1 <= actor.shape[0] <= MAX_BATCH_SIZE,
                 "actor batch size must be between 1 and 8")
        _require(bool(torch.isfinite(actor).all().item()),
                 "actor observations must be finite")

        raw = actor.detach().to(device="cpu").contiguous().numpy()
        rows = []
        for index in range(raw.shape[0]):
            row = np.asarray(raw[index:index + 1], dtype=np.float32)
            result = self._session.run([self._output_name], {self._input_name: row})
            _require(isinstance(result, (list, tuple)) and len(result) == 1,
                     "runtime must return exactly one output")
            action = result[0]
            _require(isinstance(action, np.ndarray) and action.dtype == np.float32
                     and action.shape == (1, 14),
                     "runtime action row must be float32 [1,14]")
            _require(bool(np.isfinite(action).all()), "runtime action row must be finite")
            rows.append(action[0].copy())
        output = np.stack(rows, axis=0)
        return torch.from_numpy(output).to(device=actor.device, dtype=torch.float32)


def load_normalized_onnx_reference(
    path: str | Path,
    expected_sha256: str,
) -> NormalizedOnnxReference:
    """Construct the pinned archived VelStand CPU reference adapter."""
    return NormalizedOnnxReference(path, expected_sha256)
