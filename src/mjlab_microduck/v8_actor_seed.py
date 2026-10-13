"""Pinned v8 deterministic actor import, not recovery of a PPO checkpoint.

Only the mean network and its exported affine observation transform exist in
ONNX. Gaussian exploration, critic, optimizer, counters and history are new
target state. This class keeps the imported transform frozen across train/eval
and checkpoint reconstruction, while using the stock RSL forward/export path.
No simulator, service, learner or output artifact is started by this module.
"""

from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from rsl_rl.models import MLPModel
from rsl_rl.modules import EmpiricalNormalization
from rsl_rl.modules.distribution import GaussianDistribution

from . import community_policy_inspection as inspection
from . import community_policy_normalizer as affine
from . import normalized_onnx_reference as reference

PROTOCOL = "pinned-v8-actor-only-import-v1"
HIDDEN_DIMS = (512, 256, 128)
MAX_PARITY_ERROR = 2e-5


def require(condition, message):
    if not condition:
        raise ValueError(message)


class FrozenV8Actor(MLPModel):
    """Stock 61→14 ELU actor with durable, frozen affine normalization.

Must be seeded explicitly before inference, or loaded from a checkpoint made
with this exact class. The actor class must also be recorded in the target
runner configuration; loading its state into the generic MLPModel is invalid.
"""

    def __init__(self, obs, obs_groups, obs_set, output_dim,
                 hidden_dims=HIDDEN_DIMS, activation="elu",
                 obs_normalization=True, distribution_cfg=None):
        require(obs_set == "actor" and obs_groups.get("actor") == ["actor"],
                "single raw actor observation group required")
        require(obs["actor"].ndim == 2 and obs["actor"].shape[-1] == 61,
                "raw 61D actor observations required")
        require(output_dim == 14 and tuple(hidden_dims) == HIDDEN_DIMS
                and activation == "elu" and obs_normalization is True,
                "exact normalized v8 61-512-256-128-14 ELU architecture required")
        require(isinstance(distribution_cfg, dict)
                and distribution_cfg.get("class_name") == "rsl_rl.modules:GaussianDistribution"
                and set(distribution_cfg) == {"class_name", "init_std", "std_type"}
                and distribution_cfg["std_type"] in ("scalar", "log"),
                "explicit fresh state-independent Gaussian configuration required")
        std = distribution_cfg["init_std"]
        require(type(std) in (int, float) and math.isfinite(std) and std > 0,
                "finite positive fresh Gaussian std required")
        super().__init__(obs, obs_groups, obs_set, output_dim, hidden_dims,
                         activation, True, copy.deepcopy(distribution_cfg))
        # Exported denominator already includes any original epsilon. Its
        # decomposition into historical std/epsilon cannot be recovered.
        self.obs_normalizer.eps = 0.0
        self.obs_normalizer.until = 0
        self.register_buffer("v8_seed_digest", torch.zeros(32, dtype=torch.uint8))

    def _require_seed(self):
        expected = torch.tensor(list(bytes.fromhex(reference.OFFICIAL_V8_SHA256)),
                                dtype=torch.uint8, device=self.v8_seed_digest.device)
        require(torch.equal(self.v8_seed_digest, expected), "pinned v8 actor not seeded")
        norm = self.obs_normalizer
        require(type(norm) is EmpiricalNormalization and norm.eps == 0.0
                and norm.until == 0 and norm.count.item() == 0,
                "imported affine transform must remain frozen; no historical count")

    def forward(self, *args, **kwargs):
        self._require_seed()
        return super().forward(*args, **kwargs)

    def as_onnx(self, verbose):
        self._require_seed()
        return super().as_onnx(verbose)

    def as_jit(self):
        self._require_seed()
        return super().as_jit()


def _parse(payload):
    """Recognize only the exact pinned affine + four-linear ELU graph."""
    import onnx
    from onnx import numpy_helper

    sha = hashlib.sha256(payload).hexdigest()
    require(sha == reference.OFFICIAL_V8_SHA256, "only exact pinned v8 bytes accepted")
    affine.inspect_affine_prefix(payload, sha)
    reference.NormalizedOnnxReference._read_and_validate_metadata(payload)
    graph = onnx.load_model_from_string(payload).graph
    nodes = list(graph.node)
    require([n.op_type for n in nodes] ==
            ["Sub", "Div", "Gemm", "Elu", "Gemm", "Elu", "Gemm", "Elu", "Gemm"],
            "exact affine/four-Gemm/three-ELU topology required")
    constants = {t.name: t for t in graph.initializer}
    shapes = {"obs_normalizer._mean": (1, 61), "onnx::Div_24": (1, 61)}
    for index, ins, outs in ((0, 61, 512), (2, 512, 256), (4, 256, 128), (6, 128, 14)):
        shapes[f"mlp.{index}.weight"] = (outs, ins)
        shapes[f"mlp.{index}.bias"] = (outs,)
    require(len(constants) == len(graph.initializer)
            and set(constants) == set(shapes), "exact dense initializer names required")
    tensors = {}
    for name, shape in shapes.items():
        value = numpy_helper.to_array(constants[name])
        require(value.dtype == np.float32 and value.shape == shape
                and np.isfinite(value).all(), "finite float32 initializer shape required")
        tensors[name] = torch.from_numpy(value.copy())
    require(bool((tensors["onnx::Div_24"] > 0).all()), "positive exported denominator required")
    require(bool(torch.isfinite(tensors["onnx::Div_24"].square()).all()),
            "finite derived affine variance required")
    prior = "obs"
    for position, node in enumerate(nodes):
        require(node.domain in ("", "ai.onnx") and len(node.output) == 1,
                "single standard-domain graph output required")
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        require(len(attrs) == len(node.attribute), "duplicate graph attributes rejected")
        if position == 0:
            expected_inputs = [prior, "obs_normalizer._mean"]
            require(not attrs, "Sub attributes rejected")
        elif position == 1:
            expected_inputs = [prior, "onnx::Div_24"]
            require(not attrs, "Div attributes rejected")
        elif node.op_type == "Elu":
            expected_inputs = [prior]
            require(set(attrs) <= {"alpha"} and attrs.get("alpha", 1.0) == 1.0,
                    "ELU alpha must be one")
        else:
            index = position - 2
            expected_inputs = [prior, f"mlp.{index}.weight", f"mlp.{index}.bias"]
            require(set(attrs) <= {"alpha", "beta", "transA", "transB"}
                    and attrs.get("alpha", 1.0) == attrs.get("beta", 1.0) == 1.0
                    and attrs.get("transA", 0) == 0 and attrs.get("transB", 0) == 1,
                    "exact Gemm weight orientation and coefficients required")
        require(list(node.input) == expected_inputs, "exact sequential graph wiring required")
        prior = node.output[0]
    require(prior == "actions", "final unfiltered actions output required")
    return sha, tensors


def import_v8_actor(actor, path, raw_probe):
    """Seed a new CPU actor atomically after output parity; return provenance.

No runner/critic/optimizer argument is accepted. Existing parameters are copied
in place so a freshly constructed optimizer retains correct parameter ownership.
The probe is raw float32 Nx61, not pre-normalized input. This is actor-output
consistency only: no training, simulation, policy or physical acceptance follows.
"""
    from tensordict import TensorDict
    import onnxruntime as ort

    require(type(actor) is FrozenV8Actor, "durable FrozenV8Actor class required")
    require(not bool(actor.v8_seed_digest.any()), "actor import is one-shot")
    require(all(p.device.type == "cpu" and p.dtype == torch.float32 and p.grad is None
                for p in actor.parameters()), "fresh CPU float32 actor required")
    require(type(actor.distribution) is GaussianDistribution
            and actor.distribution._distribution is None, "fresh Gaussian state required")
    require(type(actor.obs_normalizer) is EmpiricalNormalization
            and actor.obs_normalizer.count.item() == 0
            and actor.obs_normalizer.until == 0 and actor.obs_normalizer.eps == 0.0,
            "fresh frozen actor normalization required")
    require(isinstance(raw_probe, torch.Tensor) and raw_probe.device.type == "cpu"
            and raw_probe.dtype == torch.float32 and raw_probe.ndim == 2
            and raw_probe.shape[1] == 61 and 1 <= raw_probe.shape[0] <= 512
            and bool(torch.isfinite(raw_probe).all()), "bounded finite raw float32 CPU probe required")
    payload = inspection._read_payload(Path(path))
    sha, tensors = _parse(payload)
    # Validate and run a separate candidate before changing the caller's actor.
    candidate = copy.deepcopy(actor)
    with torch.no_grad():
        candidate.mlp.load_state_dict({k.removeprefix("mlp."): v
                                      for k, v in tensors.items() if k.startswith("mlp.")})
        norm = candidate.obs_normalizer
        norm._mean.copy_(tensors["obs_normalizer._mean"])
        norm._std.copy_(tensors["onnx::Div_24"])
        norm._var.copy_(tensors["onnx::Div_24"].square())
        candidate.v8_seed_digest.copy_(torch.tensor(list(bytes.fromhex(sha)), dtype=torch.uint8))
        observations = TensorDict({"actor": raw_probe}, batch_size=[len(raw_probe)])
        actual = candidate(observations)
        exported = candidate.as_onnx(False)(raw_probe)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(payload, sess_options=options, providers=["CPUExecutionProvider"])
    require(session.get_providers() == ["CPUExecutionProvider"], "sole CPU reference provider required")
    expected = torch.from_numpy(np.concatenate([
        session.run(["actions"], {"obs": row[None, :].numpy()})[0]
        for row in raw_probe], axis=0))
    require(actual.shape == expected.shape == exported.shape == (len(raw_probe), 14)
            and actual.dtype == expected.dtype == exported.dtype == torch.float32
            and bool(torch.isfinite(actual).all()) and bool(torch.isfinite(expected).all()),
            "finite raw fourteen-action parity output required")
    errors = [float((values - expected).abs().max()) for values in (actual, exported)]
    require(all(math.isfinite(e) and e <= MAX_PARITY_ERROR for e in errors),
            "actor or stock-export forward disagrees with pinned ONNX")
    # Do not replace Parameter objects or import a std/critic/optimizer state.
    with torch.no_grad():
        actor.mlp.load_state_dict(candidate.mlp.state_dict())
        actor.obs_normalizer.load_state_dict(candidate.obs_normalizer.state_dict())
        actor.v8_seed_digest.copy_(candidate.v8_seed_digest)
    return dict(protocol=PROTOCOL, artifact_sha256=sha, artifact_bytes=len(payload),
                probe_rows=len(raw_probe), maximum_actor_error=errors[0],
                maximum_export_forward_error=errors[1], parity_tolerance=MAX_PARITY_ERROR,
                normalization="frozen-exported-affine-once", normalizer_count=0,
                count_interpretation="target empty updater; not recovered source history",
                gaussian_state_imported=False, critic_state_imported=False,
                optimizer_state_imported=False, runner_history_imported=False,
                ppo_resume=False, simulation_executed=False, policy_acceptance=False,
                physical_motion_authorized=False)
