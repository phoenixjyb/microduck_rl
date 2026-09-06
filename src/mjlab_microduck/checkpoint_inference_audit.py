"""CPU-only strict actor/normalizer restoration; no simulation or admission."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
import os

import torch

from mjlab_microduck.first_attempt_smoke import ACTOR_SHA256, require, sha256


def audit_actor_state(state, actor_cfg):
    """Synthetic probes check loading only, not robot behavior or tracking."""
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict

    require(actor_cfg.class_name == "MLPModel" and actor_cfg.obs_normalization is True
            and actor_cfg.rnn_type is None and actor_cfg.cnn_cfg is None, "expected normalized nonrecurrent MLP")
    require(isinstance(state, dict) and all(isinstance(v, torch.Tensor) and v.device.type == "cpu"
            and bool(torch.isfinite(v).all()) for v in state.values()), "finite CPU state tensors")
    for name in ("_mean", "_std", "_var"):
        require(state[f"obs_normalizer.{name}"].shape == (1, 61), "61D normalizer")
    std, var, count = (state[f"obs_normalizer.{k}"] for k in ("_std", "_var", "count"))
    require(bool((std > 0).all()) and bool((var >= 0).all()) and count.shape == () and count.item() > 0,
            "nonempty trained normalization")
    require(torch.allclose(std.double().square(), var.double(), rtol=1e-5, atol=1e-10), "normalizer std/variance consistency")
    probes = TensorDict({"actor": torch.stack((torch.zeros(61), torch.linspace(-.2, .2, 61)))}, batch_size=[2])
    original = {k: v.clone() for k, v in state.items()}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = MLPModel(probes, {"actor": ["actor"]}, "actor", 14,
                         hidden_dims=actor_cfg.hidden_dims, activation=actor_cfg.activation,
                         obs_normalization=actor_cfg.obs_normalization,
                         distribution_cfg=copy.deepcopy(actor_cfg.distribution_cfg))
        loaded = model.load_state_dict(state, strict=True)
        model.eval()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        with torch.inference_mode():
            first, second = model(probes), model(probes)
        require(first.shape == (2, 14) and bool(torch.isfinite(first).all()) and torch.equal(first, second),
                "finite repeatable CPU synthetic inference")
        require(all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
                and all(torch.equal(v, original[k]) for k, v in state.items()), "inference state immutability")
    return dict(protocol="frozen-actor-cpu-inference-audit-v1", strict_load=str(loaded),
                actor_config=asdict(actor_cfg), checkpoint_tensor_shapes={k: list(v.shape) for k, v in state.items()},
                normalizer={k: dict(shape=list(v.shape), minimum=v.min().item(), maximum=v.max().item())
                            for k, v in state.items() if k.startswith("obs_normalizer.")},
                model_eval=not model.training, normalizer_eval=not model.obs_normalizer.training,
                synthetic_output_shape=list(first.shape), synthetic_output_finite=True,
                repeated_outputs_equal=True, state_unchanged=True, cpu_rng_preserved=True,
                simulation_executed=False, optimizer_step_executed=False, command_tracking_validated=False,
                complete_runtime_equivalence_validated=False, policy_acceptance=False, physical_motion_authorized=False)


def main():
    from mjlab_microduck.recovery_ab import verify_source, write_new
    from mjlab_microduck.speed_response_control import ACTOR, ROOT, prepare_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    args = parser.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU-only audit requires CUDA hidden")
    verify_source(args.source)
    require(sha256(ACTOR) == ACTOR_SHA256, "exact frozen actor")
    output = ROOT / "artifacts/evaluations/frozen-actor-input-audit-v1"
    require(not output.exists(), "exclusive audit output")
    checkpoint = torch.load(ACTOR, weights_only=True, map_location="cpu")
    _, agent = prepare_config()  # config construction only; no env or simulation
    report = audit_actor_state(checkpoint["actor_state_dict"], agent.actor)
    report.update(source=args.source, actor_sha256=ACTOR_SHA256, saved_iteration=checkpoint["iter"],
                  saved_config_sha256={name: sha256(ACTOR.parent / "params" / name)
                                       for name in ("env.yaml", "agent.yaml")})
    output.mkdir()
    write_new(output / "inference-audit.json", report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__": main()
