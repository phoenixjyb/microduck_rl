"""CPU-only identity and admission contracts for staged-head replication."""
from __future__ import annotations

import json
from dataclasses import asdict, FrozenInstanceError

import pytest

from mjlab_microduck import v8_staged_head_goals as pilot
from mjlab_microduck import v8_staged_head_replication as replication
from mjlab_microduck import v8_head_goal_schedule as schedule


def test_learning_seed_binding_is_frozen_and_recipe_stages_are_fixed():
    first = replication.Experiment(977)
    second = replication.Experiment(983)
    assert first.RUN_PREFIX == "v8-staged-head-replication-s977"
    assert second.RUN_PREFIX == "v8-staged-head-replication-s983"
    assert first.MODES == {"smoke": (64, 5, 947), "benchmark": (256, 10, 953),
                           "foundation": (256, 1000, 977)}
    assert second.MODES["foundation"] == (256, 1000, 983)
    assert first.CAPS == second.CAPS == {
        "smoke": 300, "benchmark": 300, "foundation": 2100, "final-eval": 600}
    assert first.STAGE_SPECS is second.STAGE_SPECS is None
    with pytest.raises(FrozenInstanceError):
        first.learning_seed = 983


@pytest.mark.parametrize("seed", [977, 983])
def test_json_roundtrip_keeps_seed_bound_recipe_and_same_masked_arms(seed):
    experiment = replication.Experiment(seed)
    proof = experiment.recipe_match("foundation")
    assert proof["protocol"] == replication.PROTOCOL + ":recipe-v1"
    assert proof["learning_seed"] == seed
    assert proof["stage_seed"] == experiment.MODES["foundation"][2]
    assert proof["control_recipe_sha256"] == proof["treatment_recipe_sha256"]
    assert proof["matched_after_declared_mask"] is True
    assert proof == json.loads(json.dumps(proof, sort_keys=True, allow_nan=False))


def test_pair_config_changes_only_arm_and_replication_foundation_seed(monkeypatch):
    experiment = replication.Experiment(977)
    control, control_agent = experiment.prepare_config("foundation", "control")
    staged, staged_agent = experiment.prepare_config("foundation", "staged")
    assert control.seed == staged.seed == control_agent.seed == staged_agent.seed == 977
    assert control_agent.experiment_name == staged_agent.experiment_name == replication.PROTOCOL
    assert control_agent.max_iterations == staged_agent.max_iterations == 1000
    assert control.rewards == staged.rewards
    assert tuple(map(tuple, control.commands["head_pose"].ranges)) == schedule.PARENT_RANGES
    assert tuple(map(tuple, staged.commands["head_pose"].ranges)) == schedule.BEGINNER_RANGES
    assert control.curriculum["head_pose_range"].params["range_stages"] == schedule.stages_for("control")
    assert staged.curriculum["head_pose_range"].params["range_stages"] == schedule.stages_for("staged")

    pilot_cfg, pilot_agent = pilot.prepare_config("foundation", "control")
    actual_cfg, historical_cfg = asdict(control), asdict(pilot_cfg)
    assert (historical_cfg["seed"], actual_cfg["seed"]) == (967, 977)
    actual_cfg["seed"] = historical_cfg["seed"]
    assert actual_cfg == historical_cfg
    changed = {"seed", "experiment_name"}
    actual = asdict(control_agent)
    historical = asdict(pilot_agent)
    assert {key: value for key, value in actual.items() if key not in changed} == {
        key: value for key, value in historical.items() if key not in changed}
    assert (historical["seed"], actual["seed"]) == (967, 977)
    assert actual["experiment_name"] == replication.PROTOCOL


def test_objects_keep_seed_isolation_and_only_foundation_seed_varies():
    a, b = replication.Experiment(977), replication.Experiment(983)
    assert a.MODES["smoke"] == b.MODES["smoke"] == (64, 5, 947)
    assert a.MODES["benchmark"] == b.MODES["benchmark"] == (256, 10, 953)
    assert a.MODES["foundation"][-1] == 977
    assert b.MODES["foundation"][-1] == 983


def test_seed_983_preflight_is_permitted_but_training_and_compare_are_blocked():
    experiment = replication.Experiment(983)
    experiment.require_admission(training=False)
    with pytest.raises(ValueError, match="separately reviewed seed 977"):
        experiment.require_admission(training=True)
    with pytest.raises(ValueError, match="separately reviewed seed 977"):
        experiment.compare_retained("a" * 40, "run")


def test_cli_seed_983_preflight_delegates_and_training_stops_before_driver(monkeypatch):
    events = []
    monkeypatch.setattr(replication.pilot, "main", lambda *, argv, experiment:
                        events.append((argv, experiment.learning_seed)) or "preflight")
    exp = replication.Experiment(983)
    assert exp.main(["--learning-seed", "983", "--source", "a" * 40,
                     "--run-id", "v8-staged-head-replication-s983-aaaaaaaa-20261014",
                     "--preflight"]) == "preflight"
    assert events == [(["--source", "a" * 40,
                        "--run-id", "v8-staged-head-replication-s983-aaaaaaaa-20261014",
                        "--preflight"], 983)]
    with pytest.raises(ValueError, match="separately reviewed seed 977"):
        exp.main(["--learning-seed", "983", "--source", "a" * 40,
                  "--run-id", "v8-staged-head-replication-s983-aaaaaaaa-20261014",
                  "--arm", "control", "--mode", "smoke"])
    assert len(events) == 1


def _sealed_pilot_payloads():
    comparison = {
        "decision": "staged-head-goals-lesson-rejected",
        "failures": [["control", "original_parent", "walking-idle-retention-failure"]],
        "groups": {
            "control": {"complete_moving_route_worlds": 7, "moving_speed_passes": 67},
            "staged": {"complete_moving_route_worlds": 13, "moving_speed_passes": 72},
        },
    }
    summary = {
        "source": replication.PILOT_SOURCE,
        "protocol": pilot.PROTOCOL,
        "status": "verified-evidence-not-policy-acceptance",
        "comparison": comparison,
        "comparison_sha256": "decision-sha",
        "parent_checkpoint_sha256": replication.PARENT_SHA,
        "policy_acceptance": False,
        "physical_motion_authorized": False,
    }
    manifest = {
        "source": replication.PILOT_SOURCE,
        "total_bytes": 373_251_071,
        "files": {"staged-head-goals-decision.json": {"sha256": "decision-sha"}},
    }
    return summary, manifest


def test_parent_reader_authenticates_pilot_seals_then_reads_original_parent(monkeypatch):
    summary, manifest = _sealed_pilot_payloads()
    payloads = {replication.PILOT_SUMMARY: json.dumps(summary).encode(),
                replication.PILOT_MANIFEST: json.dumps(manifest).encode()}
    calls = []
    monkeypatch.setattr(pilot.headroom, "_owned_payload",
                        lambda path, expected: calls.append((path, expected)) or payloads[path])
    parent = {"parent": "original-headroom"}
    monkeypatch.setattr(pilot, "_read_parent", lambda: parent)
    assert replication.Experiment(977)._read_parent() is parent
    assert calls == [(replication.PILOT_SUMMARY, replication.PILOT_SUMMARY_SHA256),
                     (replication.PILOT_MANIFEST, replication.PILOT_MANIFEST_SHA256)]


@pytest.mark.parametrize("mutation", ["summary_sha", "status", "manifest_sha", "failure", "group",
                                        "parent", "flags", "manifest_bounds"])
def test_parent_reader_rejects_bad_pilot_seal_before_original_parent(monkeypatch, mutation):
    summary, manifest = _sealed_pilot_payloads()
    if mutation == "summary_sha":
        summary["source"] = "b" * 40
    elif mutation == "status":
        summary["status"] = "unsigned"
    elif mutation == "manifest_sha":
        manifest["files"]["staged-head-goals-decision.json"]["sha256"] = "wrong"
    elif mutation == "failure":
        summary["comparison"]["failures"].append(["staged", "other", "unexpected"])
    elif mutation == "group":
        summary["comparison"]["groups"]["staged"]["moving_speed_passes"] = 71
    elif mutation == "parent":
        summary["parent_checkpoint_sha256"] = "c" * 64
    elif mutation == "flags":
        summary["physical_motion_authorized"] = True
    elif mutation == "manifest_bounds":
        manifest["total_bytes"] = 0
    payloads = {replication.PILOT_SUMMARY: json.dumps(summary).encode(),
                replication.PILOT_MANIFEST: json.dumps(manifest).encode()}
    monkeypatch.setattr(pilot.headroom, "_owned_payload", lambda path, _sha: payloads[path])
    parent_calls = []
    monkeypatch.setattr(pilot, "_read_parent", lambda: parent_calls.append(True))
    with pytest.raises(ValueError, match="authenticated rejected staged-head pilot"):
        replication.Experiment(977)._read_parent()
    assert parent_calls == []


def test_cli_seed_mismatch_rejected_before_delegation(monkeypatch):
    monkeypatch.setattr(replication.pilot, "main", lambda **_kwargs: pytest.fail("delegated"))
    with pytest.raises(ValueError, match="CLI learning seed"):
        replication.Experiment(977).main(["--learning-seed", "983", "--preflight"])
