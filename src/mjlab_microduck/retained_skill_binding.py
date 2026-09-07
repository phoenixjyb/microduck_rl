"""Partial CPU binding for the closed F1-Y yaw artifact, never skill admission."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import yaml

from mjlab_microduck.skill_compatibility import _verify_reference


SOURCE = "2614d09a994223a025d7a0d17d5b6879c7b513dd"
CAMPAIGN = "f1y-yaw-support-paired-s499-v1"
MANIFEST_SHA = "63a9c8c5cd8a909a1f7cc1490deff60978368c22bbfd2e314985b667febfbe41"
CHECKPOINT_SHA = "1dbc54bc9780c71f6c2ff4e49ccdc7d680a9b5269f7681bab02b04a3ae2f26fc"
PROTOCOL = "f1y-retained-skill-binding-v1"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def tagged_yaml(text: str) -> dict:
    """Preserve tags, scalar spelling and mapping order without constructors.

    Python tags are inert strings, never imported or invoked. Alias expansion
    is bounded, cycles/duplicate or non-string mapping keys are rejected.
    """
    require(type(text) is str and len(text.encode("utf-8")) <= 2_000_000,
            "bounded YAML text required")
    try:
        root = yaml.compose(text, Loader=yaml.BaseLoader)
        remaining = 60_000

        def visit(node, ancestors, depth):
            nonlocal remaining
            remaining -= 1
            require(remaining >= 0 and depth <= 80, "YAML expansion limit")
            require(node is not None and id(node) not in ancestors, "empty or cyclic YAML")
            active = ancestors | {id(node)}
            if isinstance(node, yaml.ScalarNode):
                return dict(kind="scalar", tag=node.tag, value=node.value)
            if isinstance(node, yaml.SequenceNode):
                return dict(kind="sequence", tag=node.tag,
                            value=[visit(v, active, depth + 1) for v in node.value])
            require(isinstance(node, yaml.MappingNode), "supported YAML node required")
            seen, entries = set(), []
            for key, value in node.value:
                require(isinstance(key, yaml.ScalarNode) and key.tag == "tag:yaml.org,2002:str",
                        "string YAML mapping keys required")
                require(key.value not in seen, "duplicate YAML mapping key")
                seen.add(key.value)
                entries.append([key.value, visit(value, active, depth + 1)])
            return dict(kind="mapping", tag=node.tag, value=entries)

        return visit(root, set(), 0)
    except (yaml.YAMLError, RecursionError) as exc:
        raise ValueError("invalid or excessively nested YAML") from exc


def select(tree: dict, *keys: str) -> dict:
    for key in keys:
        require(tree["kind"] == "mapping", "expected mapping at " + key)
        matches = [v for k, v in tree["value"] if k == key]
        require(len(matches) == 1, "missing retained config field " + key)
        tree = matches[0]
    return tree


def declared_config(env_text: str, agent_text: str) -> dict:
    """Saved declarations only; they do not resolve regexes or compile physics."""
    env, agent = tagged_yaml(env_text), tagged_yaml(agent_text)
    terms = select(env, "observations", "actor", "terms")
    require(terms["kind"] == "mapping", "ordered actor terms required")
    decimation = select(env, "decimation")
    timestep = select(env, "sim", "mujoco", "timestep")
    require(decimation["value"] == "4" and timestep["value"] == "0.005",
            "fixed F1-Y declared timing")
    return dict(
        provenance="saved-declarations-only-not-effective-runtime",
        robot=select(env, "scene", "entities", "robot"),
        simulation=select(env, "sim"),
        actor_observations=select(env, "observations", "actor"),
        actor_term_order=[k for k, _ in terms["value"]],
        actor_model=select(agent, "actor"),
        actions=select(env, "actions"),
        runner_action_clipping=select(agent, "clip_actions"),
        commands=select(env, "commands"),
        decimation=decimation, physics_timestep=timestep,
        declared_control_period_s=0.02,
    )


def verified_f1y_inventory(root: Path) -> dict:
    """Pin the closed manifest before accepting any payload reference."""
    root = Path(root).resolve(strict=True)
    manifest_path = root / "manifest.json"
    require(not manifest_path.is_symlink(), "manifest must not be a symlink")
    raw = manifest_path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == MANIFEST_SHA, "exact closed F1-Y manifest")
    manifest = json.loads(raw)
    require(manifest["source"] == SOURCE and manifest["protocol"] == CAMPAIGN,
            "closed source and protocol")
    files = manifest["files"]
    require(len(files) == 105, "105 closed payloads")
    # Reject directory symlinks too, including ones outside the inventory.
    paths = list(root.rglob("*"))
    require(not any(p.is_symlink() for p in paths), "symlink in closed evidence")
    require(set(files) == {p.relative_to(root).as_posix() for p in paths
                           if p.is_file() and p != manifest_path}, "exact payload inventory")
    return {name: _verify_reference(root, name, dict(path=name, **row))
            for name, row in files.items()}


def checkpoint_structure(checkpoint: dict) -> dict:
    """Validate only the known fixed final's CPU tensor layout/counters."""
    import torch

    require(type(checkpoint["iter"]) is int and checkpoint["iter"] == 8998
            and type(checkpoint["infos"]["env_state"]["common_step_counter"]) is int
            and checkpoint["infos"]["env_state"] == {"common_step_counter": 216000},
            "fixed final iteration and common counter")
    actor = checkpoint["actor_state_dict"]
    shapes = {f"mlp.{i}.{suffix}": shape for i, inputs, outputs in
              ((0, 61, 512), (2, 512, 256), (4, 256, 128), (6, 128, 14))
              for suffix, shape in (("weight", (outputs, inputs)), ("bias", (outputs,)))}
    shapes.update({"obs_normalizer." + k: (1, 61) for k in ("_mean", "_var", "_std")})
    shapes.update({"obs_normalizer.count": (), "distribution.std_param": (14,)})
    require(set(actor) == set(shapes), "exact actor tensor keys")
    for key, expected in shapes.items():
        tensor = actor[key]
        dtype = torch.int64 if key == "obs_normalizer.count" else torch.float32
        require(isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu" and tensor.dtype == dtype
                and tuple(tensor.shape) == expected and bool(torch.isfinite(tensor).all()),
                "finite CPU actor layout: " + key)
    require(actor["obs_normalizer.count"].item() > 0
            and bool((actor["obs_normalizer._std"] > 0).all())
            and bool((actor["obs_normalizer._var"] >= 0).all()), "trained normalizer")
    return dict(actor_dimension=61, motor_output_dimension=14, saved_iteration=8998,
                common_step_counter=216000,
                tensors={k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in actor.items()},
                finite_cpu_actor_tensors=True, inference_executed=False,
                observation_order_bound_to_tensor_columns=False,
                motor_output_order_bound_to_joints=False)


def audit_retained(root: Path) -> dict:
    """No complete descriptor until missing effective-model bindings exist."""
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "hide CUDA for CPU-only binding")
    root = Path(root).resolve(strict=True)
    inventory = verified_f1y_inventory(root)
    name = "yaw-pilot/model_8998.pt"
    require(inventory[name]["sha256"] == CHECKPOINT_SHA, "fixed rejected yaw checkpoint")
    import torch

    checkpoint = torch.load(root / name, map_location="cpu", weights_only=True)
    declared = declared_config((root / "yaw-pilot/params/env.yaml").read_text(),
                               (root / "yaw-pilot/params/agent.yaml").read_text())
    decision = json.loads((root / "decision.json").read_text())
    report = json.loads((root / "yaw-s503.json").read_text())
    require(decision["decision"] == "numerical-gate-stop" and decision["policy_acceptance"] is False
            and report["checkpoint_sha256"] == CHECKPOINT_SHA
            and report["source"] == SOURCE and report["actor_observation_shape"] == [8, 61],
            "unchanged rejected evidence and actor shape")
    refs = (name, "yaw-pilot/params/env.yaml", "yaw-pilot/params/agent.yaml",
            "yaw-pilot/result.json", "yaw-pilot/launch.json", "yaw-pilot/fingerprint.json",
            "yaw-s503.json", "decision.json")
    return dict(
        protocol=PROTOCOL, decision="partial-artifact-binding-only",
        training_source=SOURCE, manifest_sha256=MANIFEST_SHA,
        audit_implementation_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), Path(__file__).with_name("skill_compatibility.py"))},
        payload_count=len(inventory), payload_bytes=sum(v["bytes"] for v in inventory.values()),
        references={k: inventory[k] for k in refs},
        declared_config=declared, checkpoint_structure=checkpoint_structure(checkpoint),
        recorded_evaluation=dict(protocol=report["protocol"], seed=report["seed"],
            step_dt_s=report["step_dt_s"], motor_columns=report["motor_stream"]["joint_columns"],
            command_adapter=report["command_adapter"], failed_gates=decision["failures"]),
        historical_training_backend=json.loads((root / "yaw-pilot/fingerprint.json").read_text()),
        unresolved=[
            "historical-effective-compiled-model-and-asset-inventory",
            "resolved-actor-term-column-and-preprocessing-binding",
            "resolved-action-joint-order-scale-offset-and-reset-binding",
            "complete-runtime-and-actuator-implementation-binding",
            "accepted-per-skill-and-transition-retention-evidence",
        ],
        descriptor=None, descriptor_to_runtime_binding_verified=False,
        policy_acceptance=False, behavioral_retention_verified=False,
        transition_authorized=False, physical_motion_authorized=False,
        simulation_steps_executed=0, optimizer_updates=0,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output, root = args.output.resolve(), args.root.resolve()
    require(not output.is_relative_to(root), "never write inside closed evidence")
    require(not output.exists(), "new output required")
    report = audit_retained(root)
    payload = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(output=str(output), sha256=hashlib.sha256(payload).hexdigest(),
                          decision=report["decision"], unresolved=report["unresolved"])))


if __name__ == "__main__":
    main()
