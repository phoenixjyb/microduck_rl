"""Focused CPU contracts for the matched planar-bias retention driver."""
from __future__ import annotations

import copy
import json
import sys
from dataclasses import asdict
from types import SimpleNamespace as NS

import pytest
import torch

from mjlab_microduck import v8_motor_headroom as headroom
from mjlab_microduck import v8_planar_bias as planar
from mjlab_microduck import v8_actor_lr_profile as profiles
from mjlab_microduck import v8_staged_head_replication as replication
from mjlab_microduck import v8_steering_budget as budget
from mjlab_microduck.tasks import mdp


@pytest.mark.parametrize("mode", planar.MODES)
def test_matched_recipe_masks_only_planar_cost_weight_and_keeps_walking_refs(monkeypatch, mode):
    from mjlab_microduck import projected_heading_retention, steady_steering_retention

    proof = planar.recipe_match(mode)
    assert proof["matched_after_declared_mask"] is True
    assert proof["comparison_mask"] == ["rewards.planar_bias.weight"]
    assert proof["full_recipe_hashes_distinct"] is True
    assert proof["control_recipe_sha256"] == proof["treatment_recipe_sha256"]
    assert proof["control_full_recipe_sha256"] != proof["treatment_full_recipe_sha256"]
    assert planar.PLANAR_WEIGHTS == {"control": 0.0, "planar": -1.0}

    # The parent recipe still requires both independent, retained walking references.
    assert callable(projected_heading_retention.load_baseline)
    assert callable(steady_steering_retention.load_baseline)
    cfg0, agent0 = planar.prepare_config(mode, "control")
    cfg1, agent1 = planar.prepare_config(mode, "planar")
    a, b = json.loads(planar._canonical(mode, "control", mask=False)), json.loads(
        planar._canonical(mode, "planar", mask=False))
    assert a["environment"]["rewards"]["planar_bias"]["weight"] == 0.0
    assert b["environment"]["rewards"]["planar_bias"]["weight"] == -1.0
    a["environment"]["rewards"]["planar_bias"]["weight"] = "<declared-planar-cost-weight>"
    b["environment"]["rewards"]["planar_bias"]["weight"] = "<declared-planar-cost-weight>"
    assert a == b
    assert cfg0.commands == cfg1.commands
    assert cfg0.curriculum == cfg1.curriculum
    assert agent0.seed == agent1.seed == planar.MODES[mode][2]

    # Relative to the shared budget parent, only the planar reward and its
    # read-only metric are added. Seed and experiment labels are the declared
    # 991 planar protocol identity and are normalized before comparison.
    parent_cfg, parent_agent = budget.prepare_config(mode, "control")
    parent_env, parent_algo = asdict(parent_cfg), asdict(parent_agent)
    parent_env["seed"] = parent_algo["seed"] = planar.MODES[mode][2]
    parent_algo["experiment_name"] = planar.PROTOCOL
    parent_algo["run_name"] = mode
    planar_env, planar_algo = asdict(cfg0), asdict(agent0)
    planar_env["rewards"].pop(planar.TERM)
    planar_env["metrics"].pop("microduck_planar_bias")
    assert profiles._canonical({"environment": parent_env, "agent": parent_algo}) == profiles._canonical(
        {"environment": planar_env, "agent": planar_algo})


def test_foundation_recipe_keeps_original_ranges_terms_and_actor_contract():
    from mjlab_microduck import v8_head_goal_schedule as schedule

    cfg, agent = planar.prepare_config("foundation", "planar")
    parent_cfg, parent_agent = budget.prepare_config("foundation", "control")
    assert cfg.seed == agent.seed == 991
    assert cfg.scene.num_envs == cfg.scene.terrain.num_envs == 256
    assert agent.max_iterations == 1000
    assert cfg.commands == parent_cfg.commands
    assert cfg.curriculum == parent_cfg.curriculum
    assert tuple(tuple(pair) for pair in cfg.commands["head_pose"].ranges) == schedule.PARENT_RANGES
    head_stages = cfg.curriculum["head_pose_range"].params["range_stages"]
    assert head_stages == [{"step": 0, "ranges": schedule.PARENT_RANGES}]
    assert cfg.rewards["yaw_bias"].weight == -1.0
    assert cfg.rewards["yaw_bias"].params["tau_s"] == .5
    assert cfg.rewards["selected_motor_load"].weight == 0.0
    assert cfg.rewards["planar_bias"].weight == -1.0
    assert cfg.rewards["planar_bias"].params == {"command_name": "twist", "tau_s": .5}
    assert agent.obs_groups == parent_agent.obs_groups == {"actor": ("actor",), "critic": ("critic",)}
    assert agent.actor.obs_normalization is parent_agent.actor.obs_normalization is True
    assert agent.critic.obs_normalization is parent_agent.critic.obs_normalization is True
    assert agent.actor.class_name == parent_agent.actor.class_name
    assert agent.critic.class_name == parent_agent.critic.class_name
    assert agent.obs_groups == parent_agent.obs_groups
    # Pin both configured observation terms to the documented 61-D layout without
    # constructing the simulator or allocating accelerator state.
    actor_terms = cfg.observations["actor"].terms
    assert tuple(actor_terms) == tuple(parent_cfg.observations["actor"].terms)
    assert tuple(actor_terms) == ("base_ang_vel", "projected_gravity", "joint_pos", "joint_vel",
                                  "actions", "command", "head_command", "body_command")
    assert 3 + 3 + 14 + 14 + 14 + 3 + 4 + 6 == 61
    assert cfg.observations["critic"].terms == parent_cfg.observations["critic"].terms


def test_recipe_identity_does_not_depend_on_cli_callable_module(monkeypatch):
    before = planar.recipe_match("smoke")
    monkeypatch.setattr(planar.capture_planar_metric, "__module__", "__main__")
    assert planar.recipe_match("smoke") == before


def test_parent_reader_authenticates_fixed_json_and_rejects_corruption(monkeypatch):
    report = {"decision": "staged-head-replication-rejected", "learning_seed": 977,
        "policy_acceptance": False, "physical_motion_authorized": False,
        "groups": {"control": {"complete_moving_route_worlds": 15, "moving_speed_passes": 67},
                   "staged": {"complete_moving_route_worlds": 8, "moving_speed_passes": 67}}}
    summary = {"source": planar.REJECTED_SOURCE, "protocol": replication.PROTOCOL,
        "status": "verified-evidence-not-policy-acceptance", "learning_seed": 977,
        "comparison": report, "parent_checkpoint_sha256": planar.PARENT_SHA,
        "comparison_sha256": "6786ae7b4b4c81c154fb0a5fb24d4d6fa68a5f3bca052fbc7f245fe4b1aa8246", "policy_acceptance": False,
        "physical_motion_authorized": False}
    manifest = {"source": planar.REJECTED_SOURCE,
        "files": {"staged-head-replication-decision.json": {
            "sha256": "6786ae7b4b4c81c154fb0a5fb24d4d6fa68a5f3bca052fbc7f245fe4b1aa8246"}}}
    fixed_json = {
        planar.REJECTED_STATE / "replication-summary.json": json.dumps(summary).encode(),
        planar.REJECTED_STATE / "replication-manifest.json": json.dumps(manifest).encode(),
    }
    parent = {"sentinel": "authenticated-parent"}
    parent_reads = []
    monkeypatch.setattr(headroom, "_owned_payload", lambda path, _sha: fixed_json[path])
    monkeypatch.setattr(budget, "_read_parent", lambda: parent_reads.append(True) or parent)
    assert planar._read_parent() is parent
    assert parent_reads == [True]

    corrupt = copy.deepcopy(summary)
    corrupt["comparison"]["groups"]["planar"] = corrupt["comparison"]["groups"].pop("staged")
    fixed_json[planar.REJECTED_STATE / "replication-summary.json"] = json.dumps(corrupt).encode()
    parent_reads.clear()
    with pytest.raises(ValueError, match="authenticated rejected seed977 evidence"):
        planar._read_parent()
    assert parent_reads == []


def _reward_env(weight):
    from mjlab.managers import RewardTermCfg
    from mjlab.managers.reward_manager import RewardManager

    command = torch.tensor([[.3, .1, .2], [.1, -.2, -.4]])
    velocity = torch.tensor([[.1, .0, .0], [.0, .0, .0]])
    env = NS(common_step_counter=1, num_envs=2, device="cpu", step_dt=.02,
        episode_length_buf=torch.tensor([2, 2], dtype=torch.int64),
        command_manager=NS(get_command=lambda _name: command),
        scene={"robot": NS(data=NS(root_link_lin_vel_b=velocity))})
    env.reward_manager = RewardManager({planar.TERM: RewardTermCfg(
        func=mdp.body_planar_bias_cost, weight=weight, params=dict(planar.PARAMS))},
        env, scale_by_dt=False)
    return env


@pytest.mark.parametrize(("weight", "actual"), [(0.0, 0.0), (-1.0, -1.0)])
def test_capture_uses_actual_cost_weighted_reward_and_consumes_snapshot(monkeypatch, weight, actual):
    env = _reward_env(weight)
    env.reward_manager.compute(env.step_dt)
    reported = planar.capture_planar_metric(env)
    snapshot = env._microduck_planar_bias_snapshot
    term = env.reward_manager.get_term_cfg(planar.TERM)
    cost = term.func(env, **term.params)
    assert torch.equal(reported, torch.zeros(2))
    assert torch.all(cost >= 0)
    assert torch.equal(snapshot["weighted_reward"], env.reward_manager._step_reward[:, 0])
    assert torch.allclose(snapshot["weighted_reward"], actual * cost, atol=1e-6, rtol=1e-5)
    if weight == 0.0:
        assert torch.equal(snapshot["weighted_reward"], torch.zeros(2))
    with pytest.raises(ValueError, match="previous planar snapshot consumed"):
        planar.capture_planar_metric(env)


def test_capture_refuses_actual_weighted_reward_mismatch():
    env = _reward_env(-1.0)
    env.reward_manager.compute(env.step_dt)
    env.reward_manager._step_reward[:, 0].zero_()
    with pytest.raises(ValueError, match="actual nonpositive weighted planar reward matches configured cost"):
        planar.capture_planar_metric(env)


def test_observer_reports_configured_planar_arm(monkeypatch):
    monkeypatch.setattr(headroom.Observer, "update", lambda self, _env: {"updates": 1})
    env = NS(num_envs=1)
    for arm, weight in (("control", 0.0), ("planar", -1.0)):
        observer = planar.Observer("smoke", arm)
        observer.planar_samples = [{"cost": torch.ones(1), "weighted_reward": torch.full((1,), weight),
            "signed_residual": torch.ones(1, 2), "signed_ema": torch.ones(1, 2)}
            for _ in range(24)]
        row = observer.update(env)
        assert observer.report_arm == arm
        assert row["planar_bias_weight"] == planar.PLANAR_WEIGHTS[arm]
        assert row["planar_bias_weighted_reward_mean"] == weight


def test_observer_retains_command_before_parent_consumption(monkeypatch):
    env = _reward_env(-1.0)
    env.reward_manager.compute(env.step_dt)
    planar.capture_planar_metric(env)
    observer = planar.Observer("smoke", "planar")
    observer._bias_command = env.command_manager.get_command("twist").clone()
    monkeypatch.setattr(headroom.Observer, "step", lambda self, *_args: setattr(self, "_bias_command", None))
    observer.step(env, NS(step=0), {})
    assert observer._bias_command is None
    assert env._microduck_planar_bias_snapshot is None
    assert len(observer.planar_samples) == 1
    assert torch.equal(observer.planar_samples[0]["command"],
                       env.command_manager.get_command("twist").double())


def test_diagnostic_comparison_relabels_only_arm_labels_and_keeps_gates(monkeypatch):
    source = {"groups": {"control": {"complete_moving_route_worlds": 15,
                                      "moving_speed_passes": 67},
                         "staged": {"complete_moving_route_worlds": 8,
                                    "moving_speed_passes": 67}},
        "failures": ["fewer-than-six-additional-complete-moving-worlds",
                     ["staged", "paired-heading-endpoint-regression"],
                     ["control", "motor-range-idle-safety-regression"]],
        "thresholds": {"additional_complete_moving_worlds": 6,
                       "paired_heading_allowance_rad_s": .01,
                       "paired_xy_allowance_mps": .02},
        "decision": "staged-head-replication-rejected",
        "policy_acceptance": False, "physical_motion_authorized": False}
    monkeypatch.setattr(planar.heads, "diagnostic_comparison",
                        lambda _control, _planar: copy.deepcopy(source))
    report = planar.diagnostic_comparison([], [])
    assert report["groups"] == {"control": source["groups"]["control"],
                                "planar": source["groups"]["staged"]}
    assert report["thresholds"] == source["thresholds"]
    assert report["weights"] == {"control": -1.0, "planar": -1.0}
    assert report["planar_weights"] == planar.PLANAR_WEIGHTS
    assert report["failures"] == ["fewer-than-six-additional-complete-moving-worlds",
        ["planar", "paired-heading-endpoint-regression"],
        ["control", "motor-range-idle-safety-regression"]]
    assert report["decision"] == planar.DECISION_REJECTED
    assert report["policy_acceptance"] is False
    assert report["physical_motion_authorized"] is False


@pytest.mark.parametrize("arm,gate", [
    (arm, gate) for arm in planar.ARMS
    for gate in ("original_parent", "projected_heading_retention", "cautious_retention")
])
def test_comparison_keeps_all_three_walking_retention_gates(monkeypatch, tmp_path, arm, gate):
    from mjlab_microduck import steady_steering_retention as retention

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(planar, "ROOT", tmp_path)
    monkeypatch.setattr(planar, "verify_source", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(planar, "_read_parent", lambda: {"parent": True})
    monkeypatch.setattr(headroom, "_owned_payload", lambda *_args: json.dumps({
        "source": "parent-source", "evaluations": {"control": {"decision": "walking-idle-retained"}}
    }).encode())
    monkeypatch.setattr(planar, "validate_training_receipt", lambda *_args: {})
    monkeypatch.setattr(planar.driver, "foundation_checkpoint", lambda *_args: tmp_path / "checkpoint")
    monkeypatch.setattr(planar.base, "sha256", lambda _path: "checkpoint-sha")

    def evaluation(_root, selected_arm, _source, _sha):
        return {"evaluation": {
            "projected_heading_retention": {"decision": "walking-idle-retained"},
            "cautious_retention": {"decision": "walking-idle-retained"},
            "legacy_gate": {"decision": "pass"}, "route_gate": {"decision": "pass"},
        }}, {"arm": selected_arm}

    monkeypatch.setattr(planar, "validate_evaluation_receipt", evaluation)
    monkeypatch.setattr(retention, "_compare_sealed_reference",
        lambda *_args, **_kwargs: {"decision": "walking-idle-retained"})
    # Inject one failed retained reference and leave all others passing.
    if gate == "original_parent":
        monkeypatch.setattr(retention, "_compare_sealed_reference",
            lambda rows, _baseline, **_kwargs: {"decision": "rejected"}
                if rows["arm"] == arm and _kwargs.get("protocol", "").endswith("original-parent-retention-v1")
                else {"decision": "walking-idle-retained"})
    elif gate in ("projected_heading_retention", "cautious_retention"):
        original_evaluation = evaluation

        def evaluation_with_failure(root, selected_arm, source, sha):
            receipt, rows = original_evaluation(root, selected_arm, source, sha)
            if selected_arm == arm:
                receipt["evaluation"][gate]["decision"] = "rejected"
            return receipt, rows
        monkeypatch.setattr(planar, "validate_evaluation_receipt", evaluation_with_failure)
    monkeypatch.setattr(planar, "diagnostic_comparison", lambda *_args: {
        "failures": [], "decision": planar.DECISION_USEFUL})
    monkeypatch.setattr(planar.base, "write_new", lambda *_args: None)

    report = planar.compare_retained("a" * 40, "run-id")
    assert [arm, gate, "walking-idle-retention-failure"] in report["failures"]
    assert report["decision"] == planar.DECISION_REJECTED


def test_cli_uses_explicit_arguments_without_mutating_process_argv(monkeypatch):
    explicit = ["--source", "a" * 40, "--run-id", "planar-run",
                "--arm", "control", "--mode", "smoke"]
    monkeypatch.setattr(sys, "argv", ["unrelated-command", "--preflight"])
    calls = []
    monkeypatch.setattr(planar.driver, "main", lambda experiment, argv:
                        calls.append((experiment, argv)) or "delegated")
    before = list(sys.argv)
    assert planar.main(argv=explicit) == "delegated"
    assert calls == [(planar, explicit)]
    assert sys.argv == before
