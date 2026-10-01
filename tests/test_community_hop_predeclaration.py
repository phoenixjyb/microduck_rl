"""Check the fixed C1-S declaration, not policies, a plant rollout or admission."""

import hashlib
import json
from pathlib import Path

import pytest

from mjlab_microduck import community_hop_rehearsal as c1


ROOT = Path(__file__).resolve().parents[1]
DECLARATION = ROOT / "docs/experiments/2026-10-01-community-hop-c1s-predeclaration.json"


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        assert key not in result, f"duplicate declaration key: {key}"
        result[key] = value
    return result


@pytest.fixture
def plan():
    return json.loads(DECLARATION.read_text(), object_pairs_hook=unique_pairs,
                      parse_constant=lambda value: pytest.fail(f"nonfinite JSON: {value}"))


def test_distinct_experiment_identity_and_no_run_result(plan):
    assert plan["schema"] == "community-hop-substitution-predeclaration-v1"
    assert plan["experiment_id"] == "community-hop-c1s-velstand-substitution-v1"
    assert plan["experiment_id"] != c1.PROTOCOL
    assert plan["status"] == "predeclared-runner-not-implemented"
    assert plan["source_base"] == "61d84991938d0181cf78c560a8a33d98d92f4526"
    assert plan["measurement"]["protocol"] == c1.PROTOCOL
    assert plan["measurement"]["trace_schema"] == c1.TRACE_SCHEMA


@pytest.mark.parametrize("section,path_key,hash_key", [
    ("measurement", "scorer_path", "scorer_sha256"),
    ("plant", "scene_path", "scene_sha256"),
    ("plant", "robot_path", "robot_sha256"),
])
def test_tracked_sources_are_byte_pinned(plan, section, path_key, hash_key):
    record = plan[section]
    assert hashlib.sha256((ROOT / record[path_key]).read_bytes()).hexdigest() == record[hash_key]


def test_dependencies_are_frozen_not_updated(plan):
    assert hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest() == plan["runtime"]["uv_lock_sha256"]
    assert plan["plant"]["bam_git_revision"] in (ROOT / "uv.lock").read_text()
    assert plan["runtime"]["versions"] == {
        "mujoco": "3.10.0", "better-actuator-models": "1.0.1",
        "onnx": "1.22.0", "onnxruntime": "1.24.4",
    }


def test_fixed_policy_identity_does_not_require_downloaded_weights_in_ci(plan):
    for kind, revision, digest, size in (
        ("walk", "1b56c396825c052a4e26e95cf2b8d8298af9e9b4",
         "1c659be55da94bc5753b707de5c6a3e7c49931e05ca3b6991615cef1a8ba9a45", 793705),
        ("hop", "c0447da668255436eea070e862ad8eb89bad2ba1",
         "abd6db1bca2f2a583508cfd9a6fd144a7cce65b302827b9b9a0c5227fc2acae9", 793778),
    ):
        assert plan["policies"][kind]["revision"] == revision
        assert plan["policies"][kind]["sha256"] == digest
        assert plan["policies"][kind]["bytes"] == size
        assert plan["policies"][kind]["exact_training_source_verified"] is False
    assert plan["policies"]["walk"]["upstream_file"] == "velstand.onnx"


def test_schedule_and_trace_budget_are_the_unchanged_measurement(plan):
    assert [(p["phase"], p["duration_s"]) for p in plan["schedule"]] == list(c1.PHASES)
    assert [p["command_vx_m_s"] for p in plan["schedule"]] == [0.2, 0.0, 0.0, 0.0, 0.2]
    assert plan["other_command_slots"] == "all-twelve-zero"
    assert plan["runtime"]["physics_dt_s"] == c1.PHYSICS_DT_S
    assert plan["runtime"]["control_dt_s"] == c1.CONTROL_DT_S
    assert plan["budget"]["max_simulated_s_per_attempt"] == c1.TOTAL_TIME_S
    assert plan["budget"]["states_per_complete_attempt"] == c1.TOTAL_SAMPLES
    assert plan["budget"]["max_attempts"] == 2
    assert plan["budget"]["max_setup_wall_s"] + 2 * plan["budget"]["max_wall_s_per_attempt"] == plan["budget"]["max_campaign_wall_s"] == 180


def test_walk_control_precedes_and_gates_hop_no_retry(plan):
    control, transfer = plan["cases"]
    assert control == dict(id="walk-only-control", policy_sequence=["walk"] * 5,
                           run_if="preflight-passed")
    assert transfer == dict(id="walk-hop-transition", policy_sequence=["walk", "walk", "hop", "walk", "walk"],
                            run_if="walk-only-control-indicators-pass")
    assert plan["baseline_requirements"]["never_apply_hop_overall_decision_to_baseline"] is True
    assert plan["baseline_requirements"]["qualified_airborne_episode_count_4_9"] == 0
    assert plan["baseline_requirements"]["stable_final_zero_command_window_s"] == [8.0, 9.0]
    assert "bilateral_30mm_flight_10ms" not in plan["baseline_requirements"]["shared_measurement_gates"]
    assert plan["budget"]["automatic_retry"] is False
    assert plan["budget"]["sequential_only"] is True
    assert plan["budget"]["save_partial_failure_on_timeout"] is True


def test_queues_and_normalizer_are_explicit_choices_not_training_parity(plan):
    assert plan["runtime"]["action_delay_steps"] == 1
    assert plan["runtime"]["switch_history"] == "carry-all-three-histories-no-reset"
    assert plan["interface"]["joint_velocity_observation_lag_steps"] == 1
    assert plan["interface"]["normalizer"] == "baked-onnx-only-no-external-normalization"
    assert plan["interface"]["normalizer_parity_verified"] is False
    assert len(plan["interface"]["joint_names"]) == len(set(plan["interface"]["joint_names"])) == 14
    assert len(plan["interface"]["home_rad"]) == 14
    assert plan["interface"]["input"] == dict(name="obs", dtype="float32", shape=[1, 61])
    assert plan["interface"]["output"] == dict(name="actions", dtype="float32", shape=[1, 14])


def test_surrogate_is_not_ideal_pd_or_hidden_support(plan):
    assert plan["plant"]["kind"] == "our-controlled-stock-backlash-surrogate-not-author-training-plant"
    assert plan["plant"]["motor_model"] == "BAM-XL330-M6-voltage-current-limiter-not-position-PD"
    assert plan["plant"]["firmware_position_view"] == "servo-plus-backlash"
    assert plan["plant"]["back_emf_and_friction_velocity_view"] == "motor-side"
    assert plan["plant"]["hidden_support"] is False
    assert plan["plant"]["domain_randomization"] is False
    assert plan["plant"]["keep_source_collision_masks_and_contacts"] is True
    assert plan["plant"]["nonfoot_floor_contact_is_fatal"] is True
    assert plan["plant"]["expected_ground_enabled_robot_geom_count"] == 10


def test_no_execution_or_admission_is_created_by_declaration(plan):
    assert all(value is False for value in plan["not_claimed"].values())
    assert plan["runtime"]["backend"] == "native-cpu-mujoco"
    assert plan["runtime"]["ort_provider"] == "CPUExecutionProvider"
    assert plan["runtime"]["cuda_visible_devices"] == ""
    assert plan["excluded"] == ["gpu-workload", "remote-service-changes", "dependency-upgrades", "new-training", "h1-gate-changes", "mp4", "raw-perception", "physical-motion"]
    assert plan["evidence"]["unique_attempt_directories_no_overwrite"] is True
    assert plan["evidence"]["report_identity"] == ["experiment_id", "case_id", "measurement_protocol", "measurement_scorer_sha256", "predeclaration_sha256", "runner_source_commit"]


def test_predeclaration_approval_is_not_policy_execution_authority(plan):
    assert plan["authority"] == dict(predeclaration_only=True,
                                     policy_execution_approved=False,
                                     training_approved=False,
                                     run_go_ahead_required=True)
