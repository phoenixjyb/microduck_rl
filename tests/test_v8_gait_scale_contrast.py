"""CPU-only contracts for matched gait-scale continuation."""
from __future__ import annotations

import copy
import importlib.metadata as metadata
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from mjlab_microduck import v8_gait_scale_contrast as contrast


def test_historical_observer_head_ranges_remain_fixed_at_every_clock():
    observer = contrast.Observer("foundation", "control")
    expected = ((-.39, .39), (-.39, .39), (-.49, .49), (-.11, .11))
    for step in (0, 80, 160, 11999, 12000, 17999, 18000, 24000):
        assert observer.expected_head_ranges(step) == expected


@pytest.mark.parametrize("mode,worlds,updates,seed", [
    ("smoke", 64, 5, 947), ("benchmark", 256, 10, 953),
    ("foundation", 256, 1000, 967),
])
def test_paired_recipe_freezes_parent_profile_and_masks_only_tau(mode, worlds, updates, seed):
    control, control_agent = contrast.prepare_config(mode, "control")
    treatment, treatment_agent = contrast.prepare_config(mode, "gait-scale")
    proof = contrast.recipe_match(mode)
    assert control.seed == control_agent.seed == treatment.seed == treatment_agent.seed == seed
    assert control.scene.num_envs == treatment.scene.num_envs == worlds
    assert control_agent.max_iterations == treatment_agent.max_iterations == updates
    assert control.rewards["yaw_bias"].params["tau_s"] == 1.0
    assert treatment.rewards["yaw_bias"].params["tau_s"] == .5
    assert proof["matched_after_declared_mask"] is True
    assert proof["full_recipe_hashes_distinct"] is True
    assert proof["control_recipe_sha256"] == proof["treatment_recipe_sha256"]
    assert {name: stage for name, term in control.curriculum.items()
            for key, value in term.params.items() if key.endswith("stages")
            for stage in value} == {
        "action_rate_weight": {"step": 0, "weight": -.4},
        "standing_envs": {"step": 0, "rel_standing_envs": .10},
        "head_pose_range": {"step": 0, "ranges": ((-.39,.39),(-.39,.39),(-.49,.49),(-.11,.11))},
        "body_pose_range": {"step": 0, "ranges": ((-.005,.005),(-.005,.005),(-.005,.005),(-.05,.05),(-.05,.05),(-.05,.05))},
        "com_range": {"step": 0, "range": .005},
        "head_com_range": {"step": 0, "range": .005},
        "head_pose_bias_weight": {"step": 0, "weight": 1.0},
    }


def test_freeze_rejects_unknown_or_missing_curriculum():
    cfg, _ = contrast.prepare_config("smoke", "control")
    del cfg.curriculum["head_pose_range"]
    with pytest.raises(ValueError, match="exact supported parent curriculum set"):
        contrast.freeze_live_profile(cfg)


def test_sealed_parent_live_profile_requires_exact_nested_snapshot():
    live = {"common_step_counter": 24_000, "action_rate_weight": -.4,
        "head_pose_bias_weight": 1.0, "rel_standing_envs": .10,
        "head_pose_ranges": [[-.39,.39],[-.39,.39],[-.49,.49],[-.11,.11]],
        "randomize_com": {"ranges": [-.005,.005], "operation": "add"},
        "randomize_head_com": {"ranges": [-.005,.005], "operation": "add"}}
    summary = {"training": {"foundation": {"final_live_curriculum": live}}}
    assert contrast.validate_parent_live_profile(summary) == live
    summary["training"]["foundation"]["final_live_curriculum"]["head_pose_bias_weight"] = 2.0
    with pytest.raises(ValueError, match="sealed parent final_live_curriculum"):
        contrast.validate_parent_live_profile(summary)


class _Normalizer(nn.Module):
    def __init__(self, count=0):
        super().__init__()
        self.register_buffer("count", torch.tensor(count, dtype=torch.int64))
        self.register_buffer("_mean", torch.zeros(1))
        self.register_buffer("_var", torch.ones(1))
        self.register_buffer("_std", torch.ones(1))
        self.eps, self.until = 0., 0


class _Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.ParameterList([nn.Parameter(torch.ones(1)*i) for i in range(9)])
        self.obs_normalizer = _Normalizer()


class _Critic(nn.Module):
    def __init__(self, count=0):
        super().__init__()
        self.weights = nn.ParameterList([nn.Parameter(torch.ones(1)*i) for i in range(8)])
        self.obs_normalizer = _Normalizer(count)


class _FakeFrozenActor(_Actor):
    pass


def _payload():
    actor, critic = _FakeFrozenActor(), _Critic(6_144_000)
    params = list(actor.parameters()) + list(critic.parameters())
    optimizer = torch.optim.Adam(params, lr=3.125e-5)
    for parameter in params:
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    for state in optimizer.state.values():
        state["step"].fill_(20_000)
    return {"actor_state_dict": copy.deepcopy(actor.state_dict()),
        "critic_state_dict": copy.deepcopy(critic.state_dict()),
        "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
        "iter": 999, "infos": {"env_state": {"common_step_counter": 24_000}}}


def _runner():
    actor, critic = _FakeFrozenActor(), _Critic()
    params = list(actor.parameters()) + list(critic.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-4)
    class Alg:
        pass
    alg = Alg()
    alg.actor, alg.critic, alg.optimizer = actor, critic, optimizer
    alg.learning_rate = 1e-4
    alg.save = lambda: {"actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(), "optimizer_state_dict": optimizer.state_dict()}
    return SimpleNamespace(alg=alg, current_learning_iteration=0), SimpleNamespace(common_step_counter=0)


def test_initializer_restores_model_and_adam_but_keeps_local_clocks(monkeypatch, tmp_path):
    import mjlab_microduck.v8_actor_seed as actor_seed
    parent = _payload()
    monkeypatch.setattr(contrast, "_read_parent", lambda: parent)
    monkeypatch.setattr(actor_seed, "FrozenV8Actor", _FakeFrozenActor)
    runner, env = _runner()
    owner = tuple(id(p) for g in runner.alg.optimizer.param_groups for p in g["params"])
    report = contrast.initialize(runner, env, tmp_path)
    assert report["inherited_learning_rate"] == pytest.approx(3.125e-5)
    assert report["adam_state_entries"] == 17
    assert report["adam_step_min"] == report["adam_step_max"] == 20_000
    assert report["critic_normalizer_count"] == 6_144_000
    assert report["local_iteration"] == report["local_common_step_counter"] == 0
    assert report["runner_load_used"] is False and report["bitwise_resume"] is False
    assert tuple(id(p) for g in runner.alg.optimizer.param_groups for p in g["params"]) == owner


@pytest.mark.parametrize("mutation,message", [
    ("iter", "exact historical parent clocks"),
    ("missing", "exact PPO checkpoint fields"),
    ("adam", "exact inherited 17-entry Adam state"),
])
def test_initializer_rejects_parent_clock_or_schema_mismatch(monkeypatch, tmp_path, mutation, message):
    value = _payload()
    if mutation == "iter": value["iter"] = 998
    elif mutation == "missing": value.pop("optimizer_state_dict")
    else: next(iter(value["optimizer_state_dict"]["state"].values()))["step"].fill_(19_999)
    monkeypatch.setattr(contrast, "_read_parent", lambda: value)
    with pytest.raises(ValueError, match=message):
        contrast.validate_parent_payload(value)


@pytest.mark.parametrize("field", ["runner", "environment"])
def test_initializer_refuses_nonzero_local_clock(monkeypatch, tmp_path, field):
    import mjlab_microduck.v8_actor_seed as actor_seed
    monkeypatch.setattr(contrast, "_read_parent", lambda: _payload())
    monkeypatch.setattr(actor_seed, "FrozenV8Actor", _FakeFrozenActor)
    runner, env = _runner()
    if field == "runner": runner.current_learning_iteration = 1
    else: env.common_step_counter = 1
    with pytest.raises(ValueError, match="fresh FrozenV8Actor runner and zero local clocks"):
        contrast.initialize(runner, env, tmp_path)


def test_capture_metric_binds_actual_reward_and_arm_tau():
    from mjlab_microduck.tasks import mdp
    cost = torch.tensor([.2, .4])
    env = SimpleNamespace(common_step_counter=8, num_envs=2, device="cpu")
    env.reward_manager = SimpleNamespace(active_terms=["yaw_bias"], _step_reward=-cost[:, None],
        get_term_cfg=lambda _name: SimpleNamespace(func=mdp.body_yaw_bias_cost, weight=-1.,
            params={"command_name": "twist", "tau_s": .5}))
    env._microduck_body_yaw_bias_state = {"step": 8,
        "signed_residual": torch.tensor([.5, -.5]), "signed_ema": torch.tensor([.2, -.4]),
        "cost": cost}
    with pytest.raises(ValueError, match="live contrast reward"):
        contrast.capture_metric(env, "control")


def test_metric_captures_cached_treatment_reward_before_terminal_reset(monkeypatch):
    from mjlab_microduck.tasks import mdp
    from mjlab.managers import RewardTermCfg
    from mjlab.managers.reward_manager import RewardManager

    cost = torch.tensor([.1, .2, .3])
    calls, advances = [], []
    def cached_cost(env, **_params):
        calls.append(env.common_step_counter)
        state = getattr(env, "_microduck_body_yaw_bias_state", None)
        if state is None or state["step"] != env.common_step_counter:
            advances.append(env.common_step_counter)
        env._microduck_body_yaw_bias_state = {"step": env.common_step_counter,
            "signed_residual": torch.tensor([1., -2., 3.]),
            "signed_ema": torch.tensor([.1, -.2, .3]), "cost": cost.clone()}
        return cost.clone()
    monkeypatch.setattr(mdp, "body_yaw_bias_cost", cached_cost)
    env = SimpleNamespace(common_step_counter=8, num_envs=3, device="cpu", max_episode_length_s=20.)
    manager = RewardManager({"yaw_bias": RewardTermCfg(func=cached_cost, weight=-1.,
        params={"command_name": "twist", "tau_s": .5})}, env, scale_by_dt=False)
    env.reward_manager = manager
    manager.compute(.02)
    metric = contrast.capture_metric(env, "gait-scale")
    snapshot = env._microduck_yaw_bias_snapshot
    manager.reset(torch.arange(3))
    assert torch.equal(metric, torch.zeros(3))
    assert calls == [8, 8] and advances == [8]
    assert torch.equal(snapshot["weighted_reward"], -cost)
    assert torch.equal(env._microduck_yaw_bias_snapshot["cost"], cost)


@pytest.mark.parametrize("arm,tau", [("control", 1.0), ("gait-scale", .5)])
def test_actual_mdp_reward_manager_metric_cache_for_both_arms(arm, tau):
    from mjlab.managers import RewardTermCfg
    from mjlab.managers.reward_manager import RewardManager
    from mjlab_microduck.tasks import mdp
    command = torch.tensor([[.3, 0., .2], [.3, 0., -.2]])
    angular = torch.zeros(2, 3)
    env = SimpleNamespace(num_envs=2, device="cpu", step_dt=.02,
        common_step_counter=1, episode_length_buf=torch.full((2,), 5, dtype=torch.long),
        command_manager=SimpleNamespace(get_command=lambda _name: command),
        scene={"robot": SimpleNamespace(data=SimpleNamespace(root_link_ang_vel_b=angular))},
        max_episode_length_s=20.)
    manager = RewardManager({"yaw_bias": RewardTermCfg(func=mdp.body_yaw_bias_cost,
        weight=-1., params={"command_name": "twist", "tau_s": tau})}, env, scale_by_dt=False)
    env.reward_manager = manager
    manager.compute(.02)
    snapshot_state = env._microduck_body_yaw_bias_state
    captured = contrast.capture_metric(env, arm)
    snapshot = env._microduck_yaw_bias_snapshot
    manager.reset(torch.arange(2))
    assert torch.equal(captured, torch.zeros(2))
    assert snapshot_state["tau_s"] == tau and snapshot_state["step"] == 1
    assert torch.equal(snapshot["cost"], snapshot_state["cost"])
    assert torch.equal(snapshot["weighted_reward"], -snapshot_state["cost"])
    assert env._microduck_yaw_bias_snapshot is snapshot


def test_final_eval_receipt_binds_ordered_raw_and_route_bytes(tmp_path, monkeypatch):
    from mjlab_microduck import gait_path_qualification as route
    from mjlab_microduck import integrated_joint_limit_refinement as limits
    from mjlab_microduck import projected_heading_retention as retention
    from mjlab_microduck import steady_steering_retention as cautious_retention
    from mjlab_microduck import official_native_reference as pinned
    monkeypatch.setattr(retention, "load_baseline", lambda: {"ref": "steady"})
    monkeypatch.setattr(retention, "compare", lambda rows, ref: {"ref": ref})
    monkeypatch.setattr(cautious_retention, "load_baseline", lambda: {"ref": "cautious"})
    monkeypatch.setattr(cautious_retention, "compare", lambda rows, ref: {"ref": ref})
    monkeypatch.setattr(route, "decision", lambda rows: {"route": len(rows)})
    monkeypatch.setattr(limits, "decision", lambda rows: {"legacy": len(rows)})
    output = tmp_path / "control" / "final-eval"
    output.mkdir(parents=True)
    digest = "b" * 64
    raw_files, route_files, route_cases = [], [], []
    for seed in contrast.base.SEEDS:
        for index, (speed, yaw) in enumerate(contrast.base.CASES):
            name = f"s{seed}-c{index}.json"
            row = {"seed": seed, "speed": speed, "yaw": yaw, "complete": True,
                "checkpoint_sha256": digest, "worlds": 8, "steps": 240,
                "terminal_worlds": [], "speed_mae": [0.] * 8, "yaw_mae": [0.] * 8,
                "response_diagnostics": {"startup_steps": 60},
                "body_yaw_trace": {"startup_steps": 60, "steps": 240,
                    "worlds": 8, "checkpoint_sha256": digest},
                "gait_yaw_trace": {"checkpoint_sha256": digest},
                "gait_path_trace": {"checkpoint_sha256": digest}}
            (output / name).write_text(json.dumps(row))
            raw_files.append({"file": name, "sha256": contrast.base.sha256(output / name)})
            route_name = name[:-5] + "-route.json"
            summary = {"seed": seed, "speed": speed, "yaw": yaw}
            (output / route_name).write_text(json.dumps(summary))
            route_files.append({"file": route_name, "sha256": contrast.base.sha256(output / route_name)})
            route_cases.append(summary)
    receipt = {"protocol": contrast.PROTOCOL, "source": "a"*40, "arm": "control",
        "mode": "final-eval", "evaluation": {"checkpoint_sha256": digest,
            "raw_case_files": raw_files, "route_case_files": route_files, "cases": route_cases,
            "legacy_gate": {"legacy": 12}, "route_gate": {"route": 12},
            "projected_heading_retention": {"ref": {"ref": "steady"}},
            "cautious_retention": {"ref": {"ref": "cautious"}},
            "policy_acceptance": False, "physical_motion_authorized": False},
        "steady_reference_summary_sha256": retention.BASELINE_SUMMARY_SHA,
        "cautious_reference_summary_sha256": cautious_retention.BASELINE_SUMMARY_SHA,
        "cautious_reference_loaded": True, "elapsed_s": 10.,
        "policy_acceptance": False, "physical_motion_authorized": False}
    runtime = {name: metadata.version(name) for name in
        (*pinned.RUNTIME, "rsl-rl-lib", "onnx", "onnxruntime")}
    launch = {"protocol": contrast.PROTOCOL, "source": "a"*40, "arm": "control",
        "mode": "final-eval", "foundation_iteration": 999,
        "cap_seconds": contrast.CAPS["final-eval"], "runtime": runtime, "host": {"ok": True}}
    (output / "launch.json").write_text(json.dumps(launch))
    (output / "health.jsonl").write_text("".join(json.dumps({"utc": i})+"\n" for i in range(36)))
    (output / "result.json").write_text(json.dumps(receipt))
    checked, rows = contrast.validate_evaluation_receipt(tmp_path, "control", "a"*40, digest)
    assert checked == receipt and len(rows) == 12
    (output / "s839-c0.json").write_text("{}")
    with pytest.raises(ValueError, match="evaluation receipt exact raw and route file hash"):
        contrast.validate_evaluation_receipt(tmp_path, "control", "a"*40, digest)


def test_final_eval_consumes_steady_evaluator_files_and_result_schema(tmp_path, monkeypatch):
    import sys
    from mjlab_microduck import projected_heading_retention as retention
    from mjlab_microduck import steady_steering_retention as cautious_retention

    source = "a" * 40
    run_id = f"{contrast.RUN_PREFIX}-{source[:8]}-20261014"
    monkeypatch.setattr(contrast.base, "ROOT", tmp_path)
    lease_path = tmp_path / "gpu.lease"
    lease_path.write_text("owned test lease\n")
    monkeypatch.setattr(contrast.base, "LOCK", lease_path)
    monkeypatch.setattr(contrast.fcntl, "flock", lambda *_args: None)
    monkeypatch.setattr(contrast.base, "shared_host", lambda: {"healthy": True})
    monkeypatch.setattr(contrast, "verify_source", lambda *_args, **_kwargs: {
        "rsl-rl-lib": "5.0.1"})
    monkeypatch.setattr(contrast, "validate_training_receipt", lambda *_args: {
        "status": "training-complete-not-accepted"})
    monkeypatch.setattr(contrast, "_read_parent", lambda: _payload())
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(contrast, "validate_evaluation_receipt", lambda *_args: ({}, []))
    monkeypatch.setattr(retention, "load_baseline", lambda: {"ref": "steady"})
    monkeypatch.setattr(retention, "compare", lambda rows, ref: {"n": len(rows), "ref": ref})
    monkeypatch.setattr(cautious_retention, "load_baseline", lambda: {"ref": "cautious"})
    monkeypatch.setattr(cautious_retention, "compare", lambda rows, ref: {"n": len(rows), "ref": ref})

    checkpoint = tmp_path / "artifacts/training" / run_id / "control/foundation/model_999.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"foundation checkpoint")
    (tmp_path / "artifacts/training" / run_id / "gait-scale/foundation").mkdir(parents=True)

    def evaluator(_checkpoint, output, health):
        raw, summaries, hashes = [], [], []
        for seed in contrast.base.SEEDS:
            for index, (speed, yaw) in enumerate(contrast.base.CASES):
                for _ in range(3):
                    health()
                name = f"s{seed}-c{index}.json"
                row = {"seed": seed, "speed": speed, "yaw": yaw, "complete": True,
                    "checkpoint_sha256": contrast.base.sha256(checkpoint), "worlds": 8,
                    "steps": 240, "terminal_worlds": [], "speed_mae": [0.] * 8,
                    "yaw_mae": [0.] * 8, "response_diagnostics": {"startup_steps": 60},
                    "body_yaw_trace": {"startup_steps": 60, "steps": 240, "worlds": 8,
                        "checkpoint_sha256": contrast.base.sha256(checkpoint)},
                    "gait_yaw_trace": {"checkpoint_sha256": contrast.base.sha256(checkpoint)},
                    "gait_path_trace": {"checkpoint_sha256": contrast.base.sha256(checkpoint)}}
                contrast.base.write_new(output / name, row)
                route_name = name[:-5] + "-route.json"
                summary = {"seed": seed, "speed": speed, "yaw": yaw, "route": index}
                contrast.base.write_new(output / route_name, summary)
                hashes.append({"file": name, "sha256": contrast.base.sha256(output / name)})
                raw.append(row)
                summaries.append(summary)
        # This mirrors steady.evaluate_matrix's real public result shape: it
        # writes raw and route files itself and returns decision inputs/gates.
        return {"checkpoint_sha256": contrast.base.sha256(checkpoint), "cases": summaries,
            "raw_case_files": hashes, "raw_decision_inputs": raw,
            "route_gate": {"route": 12}, "legacy_gate": {"legacy": 12},
            "inference_backend": {"backend": "FrozenV8Checkpoint"},
            "policy_acceptance": False, "physical_motion_authorized": False}

    monkeypatch.setattr(contrast.steady, "evaluate_matrix", evaluator)
    monkeypatch.setattr(sys, "argv", ["v8_gait_scale_contrast", "--source", source,
        "--run-id", run_id, "--arm", "control", "--mode", "final-eval"])
    result = contrast._main_impl()
    output = tmp_path / "artifacts/training" / run_id / "control/final-eval"
    report = result["evaluation"]
    assert len(report["cases"]) == 12
    assert report["legacy_gate"] == {"legacy": 12}
    assert report["route_gate"] == {"route": 12}
    assert report["inference_backend"] == {"backend": "FrozenV8Checkpoint"}
    assert "numerical_gate" not in report
    assert len(report["raw_case_files"]) == len(report["route_case_files"]) == 12
    assert len(list(output.glob("*-route.json"))) == 12
    assert len((output / "health.jsonl").read_text().splitlines()) == 36


def test_failed_first_update_kl_writes_report_and_durable_abort(tmp_path, monkeypatch):
    events = []
    def parent_update(self, _env):
        self.updates += 1
        return {"optimizer": {"optimizer_minibatches": 20,
            "kl_mean_mean": .051, "kl_mean_max": .05},
            "target_curriculum": {"action_rate_weight": -.4, "head_pose_bias_weight": 1.0,
                "rel_standing_envs": .10,
                "head_pose_ranges": [[-.39,.39],[-.39,.39],[-.49,.49],[-.11,.11]],
                "randomize_com": {"ranges": [-.005,.005]},
                "randomize_head_com": {"ranges": [-.005,.005]}}}
    monkeypatch.setattr(contrast.learning.Observer, "update", parent_update)
    observer = contrast.Observer("smoke", "gait-scale")
    observer.output, observer.source, observer.initial_lr = tmp_path, "a"*40, 3.125e-5
    observer.runner = SimpleNamespace(alg=SimpleNamespace(optimizer=None),
        save=lambda path, infos: events.append((path, infos)))
    observer.bias_costs = [0.] * 24
    observer.bias_rewards = [0.] * 24
    observer.signed_biases = [0.] * 24
    observer.raw_errors = [0.] * 24
    observer.signed_raw_errors = [0.] * 24
    observer.bucket_counts = dict.fromkeys(contrast.exposure.EXPOSURE_BINS, 0)
    observer.bucket_counts["idle"] = 24 * 8
    env = SimpleNamespace(common_step_counter=24, num_envs=8)
    with pytest.raises(ValueError, match="gait-scale-contrast first-update KL guard"):
        observer.update(env)
    report = json.loads((tmp_path / "first-update-kl.json").read_text())
    assert report["passed"] is False and report["initial_learning_rate"] == 3.125e-5
    assert len(events) == 1
    assert events[0][1]["env_state"] == {"common_step_counter": 24}
    assert events[0][1]["first_update_kl_sha256"] == contrast.base.sha256(
        tmp_path / "first-update-kl.json")
