"""Actual action manager with synthetic CPU buffers; no actuator execution."""

from __future__ import annotations

import argparse
import ast
import contextlib
from copy import deepcopy
from dataclasses import asdict
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import textwrap
from types import SimpleNamespace

import torch
import yaml

from mjlab_microduck.retained_skill_binding import declared_config, require, verified_f1y_inventory
from mjlab_microduck.robot_source_reconstruction import BINDING_SHA, identity, source_inventory


PROTOCOL = "f1y-action-pipeline-synthetic-cpu-v1"


def step_order_source_check(method) -> dict:
    """Source-level ordering check, not execution of the full environment."""
    source = textwrap.dedent(inspect.getsource(method))
    tree = ast.parse(source).body[0]
    def calls(node, name):
        return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                   and ast.unparse(n.func) == name for n in ast.walk(node))
    def direct(node, name):
        return isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == name
    process = [i for i, n in enumerate(tree.body) if direct(n, "self.action_manager.process_action")]
    loops = [i for i, n in enumerate(tree.body) if isinstance(n, ast.For)
             and ast.unparse(n.iter) == "range(self.cfg.decimation)"
             and calls(n, "self.action_manager.apply_action")]
    require(len(process) == len(loops) == 1 and process[0] < loops[0],
            "fresh processing precedes application loop")
    loop = tree.body[loops[0]]
    apply = [i for i, n in enumerate(loop.body) if direct(n, "self.action_manager.apply_action")]
    physics = [i for i, n in enumerate(loop.body) if direct(n, "self.sim.step")]
    require(len(apply) == len(physics) == 1 and apply[0] < physics[0], "apply precedes physics")
    return dict(scope="installed-method-source-only", process_before_apply_loop=True,
                apply_before_physics=True, full_environment_executed=False,
                method_source_sha256=hashlib.sha256(source.encode()).hexdigest())


def probe_actions(robot, cfg, agent) -> dict:
    """Use real resolution/processing/target writes, with explicitly fake state.

    The isolated Entity is NOT initialized for simulation. Only its CPU data
    namespace is supplied. Never call write_data_to_sim, actuator.compute,
    Entity.reset, environment.step or any physics method here.
    """
    from mjlab.managers.action_manager import ActionManager
    from mjlab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg
    from mjlab.utils.string import resolve_expr

    require(torch.get_default_device() == torch.device("cpu") and torch.get_default_dtype() == torch.float32,
            "CPU float32 synthetic probe defaults required")
    require(not hasattr(robot, "_data"), "isolated uninitialized Entity required")
    require(list(cfg.actions) == ["joint_pos"] and type(cfg.actions["joint_pos"]) is JointPositionActionCfg,
            "one known joint-position action term")
    term_cfg = cfg.actions["joint_pos"]
    require(type(term_cfg.scale) is float and term_cfg.scale == 1.0
            and type(term_cfg.offset) is float and term_cfg.offset == 0.0 and term_cfg.clip is None
            and tuple(term_cfg.actuator_names) == (".*",) and term_cfg.preserve_order is False
            and term_cfg.use_default_offset is True and agent.clip_actions is None,
            "exact unmodified F1-Y action transform")
    require(len(robot.joint_names) == 14, "fourteen reconstructed joints")
    default = torch.tensor(resolve_expr(robot.cfg.init_state.joint_pos, robot.joint_names, 0.0),
                           dtype=torch.float32).repeat(2, 1)
    bias = torch.stack((torch.zeros(14), torch.linspace(-.01, .01, 14)))
    robot._data = SimpleNamespace(default_joint_pos=default.clone(), encoder_bias=bias.clone(),
                                 joint_pos_target=torch.zeros(2, 14))
    env = SimpleNamespace(scene={"robot": robot}, num_envs=2, device="cpu")
    manager = ActionManager(deepcopy(cfg.actions), env)
    term = manager.get_term("joint_pos")
    require(type(term) is JointPositionAction and manager.total_action_dim == 14
            and tuple(term.target_names) == tuple(robot.joint_names)
            and term.target_ids.tolist() == list(range(14)), "actual resolved action order")
    original_rng = torch.random.get_rng_state().clone()
    initial_offset = term.offset.clone()
    require(torch.equal(initial_offset, default), "actual default-position offset")
    records = []

    def process_and_apply(raw):
        original = raw.clone()
        manager.process_action(raw)
        manager.apply_action()
        expected = raw * term.scale + initial_offset - robot.data.encoder_bias
        require(torch.equal(robot.data.joint_pos_target, expected), "actual affine/bias target delivery")
        require(torch.equal(raw, original), "caller action remains unchanged")
        return robot.data.joint_pos_target.clone()

    zero_target = process_and_apply(torch.zeros(2, 14))
    # Orthogonal positive/negative probes: each input column reaches only its named joint.
    for column in range(14):
        raw = torch.zeros(2, 14)
        raw[:, column] = torch.tensor([.125, -.25])
        targets = process_and_apply(raw)
        records.append(dict(column=column, joint=term.target_names[column], raw=raw.tolist(),
                            targets=targets.tolist()))
    previous = manager.action.clone()
    unclipped_raw = torch.full((2, 14), 1.25)
    unclipped_target = process_and_apply(unclipped_raw)
    require(torch.equal(manager.prev_action, previous), "raw action history")
    processed_before_reset = term._processed_actions.clone()
    row1_history = [v[1].clone() for v in (manager.action, manager.prev_action, manager.prev_prev_action)]
    manager.reset(torch.tensor([0]))
    require(all(bool((v[0] == 0).all()) and torch.equal(v[1], expected)
                for v, expected in zip((manager.action, manager.prev_action, manager.prev_prev_action), row1_history))
            and bool((term.raw_action[0] == 0).all()), "selective raw/history reset")
    require(torch.equal(term._processed_actions, processed_before_reset), "documented processed cache survives reset")
    # Deliberately out-of-sequence diagnostic, targets are CPU buffers only.
    manager.apply_action()
    stale_target = robot.data.joint_pos_target.clone()
    require(torch.equal(stale_target, unclipped_target), "reset-only application retains previous processed target")
    fresh_target = process_and_apply(torch.zeros(2, 14))
    require(torch.equal(fresh_target, zero_target), "fresh zero action replaces stale target")
    robot.data.encoder_bias.add_(.007)
    manager.apply_action()
    require(torch.equal(robot.data.joint_pos_target, initial_offset - robot.data.encoder_bias),
            "encoder bias sampled at each apply")
    bias_target = robot.data.joint_pos_target.clone()
    robot.data.default_joint_pos.add_(.123)
    cached_target = process_and_apply(torch.zeros(2, 14))
    require(torch.equal(term.offset, initial_offset) and torch.equal(cached_target, bias_target),
            "default offset is a clone at construction, not a live view")
    cache = term._processed_actions.clone()
    manager.reset()
    require(all(bool((v == 0).all()) for v in (manager.action, manager.prev_action, manager.prev_prev_action,
                                             term.raw_action)) and torch.equal(term._processed_actions, cache),
            "full raw/history reset also preserves processed cache")
    require(torch.equal(original_rng, torch.random.get_rng_state()), "deterministic probes preserve torch RNG")
    return dict(
        target_names=list(term.target_names), target_ids=term.target_ids.tolist(), scale=term.scale,
        initial_default_offset=initial_offset.tolist(), initial_encoder_bias=bias.tolist(),
        zero_action_target=zero_target.tolist(), one_hot_probes=records,
        unclipped_probe_raw=unclipped_raw.tolist(), unclipped_probe_target=unclipped_target.tolist(),
        after_reset_only_target=stale_target.tolist(), after_fresh_zero_target=fresh_target.tolist(),
        after_encoder_bias_change_target=bias_target.tolist(),
        reset_clears_selected_raw_and_history=True, full_reset_clears_raw_and_history=True,
        reset_preserves_processed_cache=True,
        fresh_process_required_before_apply=True, encoder_bias_is_live_per_apply=True,
        default_offset_is_construction_clone=True, caller_actions_unchanged=True, torch_rng_preserved=True,
        synthetic_buffers_only=True, motor_control_executed=False, safe_stop_validated=False,
    )


def audit(evidence: Path, binding_path: Path) -> dict:
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_initialized(),
            "CUDA hidden and uninitialized")
    require(identity(binding_path)["sha256"] == BINDING_SHA, "exact partial binding")
    binding = json.loads(binding_path.read_text())
    repo = Path(__file__).resolve().parents[2]
    sources = source_inventory(repo)
    verified_f1y_inventory(evidence)
    with contextlib.redirect_stdout(io.StringIO()):
        from mjlab.entity import Entity
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.managers.action_manager import ActionManager
        from mjlab.envs.mdp.actions.actions import JointPositionAction
        from mjlab_microduck.foundation_yaw_experiment import prepare_config
        cfg, agent = prepare_config("pilot", arm="yaw")
        declared = declared_config(yaml.dump(asdict(cfg), sort_keys=False), yaml.dump(asdict(agent), sort_keys=False))
        require(declared == binding["declared_config"], "saved action/config declarations unchanged")
        robot = Entity(cfg.scene.entities["robot"])
        result = probe_actions(robot, cfg, agent)
        require(result["target_names"] == binding["recorded_evaluation"]["motor_columns"], "recorded column identity")
        order = step_order_source_check(ManagerBasedRlEnv.step)
        used = {name: identity(Path(inspect.getfile(cls))) for name, cls in
                (("entity", Entity), ("action_manager", ActionManager),
                 ("joint_position_action", JointPositionAction), ("environment", ManagerBasedRlEnv))}
    require(not torch.cuda.is_initialized(), "no CUDA initialization")
    return dict(
        protocol=PROTOCOL, decision="synthetic-action-pipeline-only-not-retention",
        partial_binding_sha256=BINDING_SHA, training_source=sources["training_source"],
        historical_source_files_verified=len(sources["files"]), implementation_files=used,
        audit_implementation_sha256=identity(Path(__file__))["sha256"],
        declaration_match=True, probes=result, normal_step_source_order=order,
        historical_randomized_actuator_state_verified=False, full_environment_reset_verified=False,
        closed_loop_retention_verified=False, transition_authorized=False,
        policy_acceptance=False, physical_motion_authorized=False,
        simulation_steps_executed=0, optimizer_updates=0,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("binding", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    require(not output.exists() and not output.is_relative_to(args.evidence.resolve()),
            "new output outside closed evidence")
    report = audit(args.evidence, args.binding)
    data = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(output=str(output), sha256=hashlib.sha256(data).hexdigest(), decision=report["decision"])))


if __name__ == "__main__":
    main()
