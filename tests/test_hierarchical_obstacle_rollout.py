import hashlib
from dataclasses import asdict

import pytest
import torch

from mjlab_microduck.hierarchical_obstacle_rollout import (
    HC1_ATTEMPT_TIMEOUT_S,
    MAX_CASES,
    advance_first_attempt_window,
    fixed_attempt_metrics,
    load_learned_supervisor,
    prepare_rollout_configs,
    range_noise_provenance,
    range_noise_uniform_samples,
    resolved_correction_samples,
    rollout_stage,
    validate_dataset_collection_mode,
    validate_rollout_bounds,
)
from mjlab_microduck.obstacle_supervisor_bc import (
    HC4U1_STAGE,
    HC4U2_STAGE,
    InteractionSpeedOnlySupervisor,
    EpisodeLatchedRangeSpeedSupervisor,
    LateralGatedSupervisor,
    ObstacleSupervisor,
    RangeSpeedGatedSupervisor,
    SupervisorBcCfg,
)


@pytest.mark.parametrize("stage", (HC4U1_STAGE, HC4U2_STAGE))
def test_unified_offline_checkpoint_is_eligible_for_diagnostic_rollout(
    tmp_path, stage
):
    base_checkpoint = tmp_path / "actor.pt"
    base_checkpoint.write_bytes(b"frozen actor")
    cfg = SupervisorBcCfg()
    model = ObstacleSupervisor(cfg)
    checkpoint = tmp_path / "supervisor.pt"
    torch.save(
        {
            "stage": stage,
            "decision": "offline-imitation-pass",
            "source_locomotion_checkpoint_sha256": hashlib.sha256(
                base_checkpoint.read_bytes()
            ).hexdigest(),
            "model_config": asdict(cfg),
            "model_state_dict": model.state_dict(),
        },
        checkpoint,
    )
    loaded = load_learned_supervisor(checkpoint, base_checkpoint, "cpu")
    assert isinstance(loaded, ObstacleSupervisor)


def test_hc4u1_has_a_retained_rollout_stage():
    assert (
        rollout_stage(HC4U1_STAGE)
        == "HC4U1-unified-range-lateral-correction-BC-rollout"
    )


def test_hc4u2_has_a_retained_rollout_stage():
    assert (
        rollout_stage(HC4U2_STAGE)
        == "HC4U2-far-center-student-state-correction-BC-rollout"
    )


def test_rollout_config_keeps_obstacle_physics_but_restores_base_observation():
    env_cfg, agent_cfg = prepare_rollout_configs(8, 0.3, 1.15, -0.27)
    assert "obstacle" in env_cfg.scene.entities
    assert "reset_obstacle" in env_cfg.events
    assert "obstacle_collision" in env_cfg.terminations
    assert "obstacle" not in env_cfg.observations["actor"].terms
    assert "obstacle_ground_truth" not in env_cfg.observations["critic"].terms
    assert env_cfg.events["reset_obstacle"].params["forward_range_m"] == (1.15, 1.15)
    assert env_cfg.events["reset_obstacle"].params["lateral_range_m"] == (-0.27, -0.27)
    assert "lateral_abs_range_m" not in env_cfg.events["reset_obstacle"].params
    assert env_cfg.commands["twist"].rel_forward_envs == 0.0
    assert (
        env_cfg.terminations["obstacle_attempt_timeout"].params[
            "max_attempt_time_s"
        ]
        == HC1_ATTEMPT_TIMEOUT_S
    )
    assert agent_cfg.experiment_name == "run_motor_aware"


def test_rollout_bounds_accept_small_matrix():
    validate_rollout_bounds(64, 400, (0.3,), (1.15,), (-0.27, 0.27), (41,))


@pytest.mark.parametrize(
    "args",
    [
        (0, 10, (0.3,), (1.15,), (0.0,), (41,)),
        (1, 0, (0.3,), (1.15,), (0.0,), (41,)),
        (1, 10, (), (1.15,), (0.0,), (41,)),
        (1, 10, (0.9,), (1.15,), (0.0,), (41,)),
        (1, 10, (0.3,), (), (0.0,), (41,)),
        (1, 10, (0.3,), (0.0,), (0.0,), (41,)),
        (1, 10, (0.3,), (1.15,), (), (41,)),
        (1, 10, (0.3,), (1.15,), (0.0,), ()),
    ],
)
def test_rollout_bounds_reject_invalid_inputs(args):
    with pytest.raises(ValueError):
        validate_rollout_bounds(*args)


def test_rollout_bounds_reject_too_many_cases():
    speeds = tuple(0.1 for _ in range(MAX_CASES + 1))
    with pytest.raises(ValueError, match="case count"):
        validate_rollout_bounds(1, 1, speeds, (1.0,), (0.0,), (1,))


def test_range_noise_samples_are_replayable_and_range_only():
    first = torch.Generator(device="cpu")
    first.manual_seed(3000282)
    first_update = range_noise_uniform_samples(
        4, first, device="cpu", dtype=torch.float64
    )
    second_update = range_noise_uniform_samples(
        4, first, device="cpu", dtype=torch.float64
    )

    replay = torch.Generator(device="cpu")
    replay.manual_seed(3000282)
    replay_first = range_noise_uniform_samples(
        4, replay, device="cpu", dtype=torch.float64
    )
    replay_second = range_noise_uniform_samples(
        4, replay, device="cpu", dtype=torch.float64
    )

    torch.testing.assert_close(first_update, replay_first)
    torch.testing.assert_close(second_update, replay_second)
    assert not torch.equal(first_update[:, 0], second_update[:, 0])
    assert torch.all((first_update[:, 0] >= 0.0) & (first_update[:, 0] <= 1.0))
    torch.testing.assert_close(
        first_update[:, 1:], torch.full((4, 4), 0.5, dtype=torch.float64)
    )


def test_range_noise_provenance_separates_physics_and_noise_streams():
    exact = range_noise_provenance(0.0, 271)
    noisy = range_noise_provenance(0.02, 271)

    assert exact["identity"] == "exact-v1"
    assert exact["noise_seed"] is None
    assert noisy == {
        "identity": "compact-range-uniform-v1",
        "distribution": "bounded-uniform",
        "range_noise_bound_m": 0.02,
        "noise_seed": 3000282,
        "noise_seed_rule": "physics_seed+3000011",
        "supervisor_update_interval_steps": 5,
        "perturbed_fields": ["range"],
        "exact_fields": [
            "bearing_sin",
            "bearing_cos",
            "width",
            "height",
            "closing_rate",
            "valid",
        ],
        "ground_truth_outcomes": True,
    }


def test_range_noise_provenance_rejects_negative_bound():
    with pytest.raises(ValueError, match="non-negative"):
        range_noise_provenance(-0.01, 271)


def test_dataset_collection_modes_require_the_matching_controller(tmp_path):
    checkpoint = tmp_path / "student.pt"
    validate_dataset_collection_mode(
        collect_success_dataset=True,
        collect_teacher_corrections=False,
        supervisor_checkpoint=None,
    )
    validate_dataset_collection_mode(
        collect_success_dataset=False,
        collect_teacher_corrections=True,
        supervisor_checkpoint=checkpoint,
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_dataset_collection_mode(
            collect_success_dataset=True,
            collect_teacher_corrections=True,
            supervisor_checkpoint=checkpoint,
        )
    with pytest.raises(ValueError, match="require a student"):
        validate_dataset_collection_mode(
            collect_success_dataset=False,
            collect_teacher_corrections=True,
            supervisor_checkpoint=None,
        )


def test_fixed_attempt_metrics_uses_the_predeclared_denominator():
    metrics = fixed_attempt_metrics(
        expected_attempts=64,
        completed_attempts=64,
        clean_pass_events=61,
        collision_events=1,
        attempt_timeout_events=2,
        fall_events=0,
        nan_termination_events=0,
        other_terminal_events=0,
    )
    assert metrics == {
        "expected_attempts": 64,
        "completed_attempts": 64,
        "unresolved_attempts": 0,
        "clean_pass_rate_fixed_denominator": 61 / 64,
        "collision_rate_fixed_denominator": 1 / 64,
        "attempt_timeout_rate_fixed_denominator": 2 / 64,
        "hard_failure_events": 0,
        "other_terminal_events": 0,
    }


def test_first_attempt_window_closes_each_environment_once():
    active = torch.tensor([True, True, True])
    active, finished = advance_first_attempt_window(
        active, torch.tensor([True, False, False])
    )
    assert finished.tolist() == [True, False, False]
    assert active.tolist() == [False, True, True]

    active, finished = advance_first_attempt_window(
        active, torch.tensor([True, True, False])
    )
    assert finished.tolist() == [False, True, False]
    assert active.tolist() == [False, False, True]


def test_first_attempt_window_rejects_invalid_masks():
    with pytest.raises(ValueError, match="must be boolean"):
        advance_first_attempt_window(torch.ones(2), torch.ones(2))
    with pytest.raises(ValueError, match="matching vectors"):
        advance_first_attempt_window(
            torch.ones(2, dtype=torch.bool), torch.ones(1, dtype=torch.bool)
        )


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"expected_attempts": 0}, "expected_attempts must be positive"),
        ({"collision_events": -1}, "must be non-negative"),
        ({"completed_attempts": 65}, "cannot exceed"),
        (
            {"completed_attempts": 1, "clean_pass_events": 2},
            "resolved outcomes cannot exceed",
        ),
        (
            {
                "completed_attempts": 2,
                "clean_pass_events": 2,
                "collision_events": 0,
                "attempt_timeout_events": 0,
                "other_terminal_events": 1,
            },
            "resolved and other terminal outcomes cannot exceed",
        ),
    ],
)
def test_fixed_attempt_metrics_rejects_impossible_counts(overrides, message):
    arguments = {
        "expected_attempts": 64,
        "completed_attempts": 64,
        "clean_pass_events": 61,
        "collision_events": 1,
        "attempt_timeout_events": 2,
        "fall_events": 0,
        "nan_termination_events": 0,
        "other_terminal_events": 0,
    }
    arguments.update(overrides)
    with pytest.raises(ValueError, match=message):
        fixed_attempt_metrics(**arguments)


def test_correction_dataset_keeps_only_resolved_episode_samples():
    observations = torch.arange(5 * 17, dtype=torch.float32).reshape(5, 17)
    teacher_commands = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    student_commands = teacher_commands + 0.1
    episode_keys = torch.tensor([10, 10, 11, 12, 12])

    selected = resolved_correction_samples(
        observations,
        teacher_commands,
        student_commands,
        episode_keys,
        {10: 1, 12: 2},
    )

    assert selected["episode_keys"].tolist() == [10, 10, 12, 12]
    assert selected["outcome_codes"].tolist() == [1, 1, 2, 2]
    torch.testing.assert_close(
        selected["observations"], observations[torch.tensor([0, 1, 3, 4])]
    )
    torch.testing.assert_close(
        selected["student_commands"],
        student_commands[torch.tensor([0, 1, 3, 4])],
    )


def test_load_hc3e_wraps_checkpoint_with_speed_only_authority(tmp_path):
    base_checkpoint = tmp_path / "locomotion.pt"
    base_checkpoint.write_bytes(b"frozen locomotion")
    actor = ObstacleSupervisor()
    supervisor_checkpoint = tmp_path / "supervisor.pt"
    torch.save(
        {
            "stage": "HC3E-interaction-speed-PPO",
            "decision": "training-complete-pending-rollout",
            "action_authority": "interaction-speed-only",
            "min_interaction_speed_mps": 0.30,
            "source_locomotion_checkpoint_sha256": hashlib.sha256(
                base_checkpoint.read_bytes()
            ).hexdigest(),
            "model_config": asdict(SupervisorBcCfg()),
            "model_state_dict": actor.state_dict(),
            "anchor_model_state_dict": actor.state_dict(),
        },
        supervisor_checkpoint,
    )
    loaded = load_learned_supervisor(
        supervisor_checkpoint, base_checkpoint, "cpu"
    )
    assert isinstance(loaded, InteractionSpeedOnlySupervisor)
    observation = torch.zeros(1, 17)
    observation[:, 0] = 0.625
    observation[:, -4] = 1.0
    command = loaded(observation)
    torch.testing.assert_close(command, actor(observation))


def test_load_hc3f_wraps_averaged_checkpoint_with_speed_only_authority(tmp_path):
    base_checkpoint = tmp_path / "locomotion.pt"
    base_checkpoint.write_bytes(b"frozen locomotion")
    actor = ObstacleSupervisor()
    supervisor_checkpoint = tmp_path / "supervisor.pt"
    torch.save(
        {
            "stage": "HC3F-seed-averaged-speed-head",
            "decision": "aggregation-complete-pending-rollout",
            "action_authority": "interaction-speed-only",
            "min_interaction_speed_mps": 0.30,
            "source_locomotion_checkpoint_sha256": hashlib.sha256(
                base_checkpoint.read_bytes()
            ).hexdigest(),
            "model_config": asdict(SupervisorBcCfg()),
            "model_state_dict": actor.state_dict(),
            "anchor_model_state_dict": actor.state_dict(),
        },
        supervisor_checkpoint,
    )

    loaded = load_learned_supervisor(supervisor_checkpoint, base_checkpoint, "cpu")

    assert isinstance(loaded, InteractionSpeedOnlySupervisor)


def test_load_hc3g_wraps_consensus_checkpoint_with_speed_only_authority(tmp_path):
    base_checkpoint = tmp_path / "locomotion.pt"
    base_checkpoint.write_bytes(b"frozen locomotion")
    actor = ObstacleSupervisor()
    supervisor_checkpoint = tmp_path / "supervisor.pt"
    torch.save(
        {
            "stage": "HC3G-seed-consensus-speed-head",
            "decision": "aggregation-complete-pending-rollout",
            "action_authority": "interaction-speed-only",
            "min_interaction_speed_mps": 0.30,
            "source_locomotion_checkpoint_sha256": hashlib.sha256(
                base_checkpoint.read_bytes()
            ).hexdigest(),
            "model_config": asdict(SupervisorBcCfg()),
            "model_state_dict": actor.state_dict(),
            "anchor_model_state_dict": actor.state_dict(),
        },
        supervisor_checkpoint,
    )

    loaded = load_learned_supervisor(
        supervisor_checkpoint, base_checkpoint, "cpu"
    )

    assert isinstance(loaded, InteractionSpeedOnlySupervisor)


def test_load_hc4l_accepts_explicit_lateral_bc_stage(tmp_path):
    base_checkpoint = tmp_path / "locomotion.pt"
    base_checkpoint.write_bytes(b"frozen locomotion")
    actor = ObstacleSupervisor()
    supervisor_checkpoint = tmp_path / "supervisor.pt"
    torch.save(
        {
            "stage": "HC4L-lateral-behavioral-cloning",
            "decision": "offline-imitation-pass",
            "source_locomotion_checkpoint_sha256": hashlib.sha256(
                base_checkpoint.read_bytes()
            ).hexdigest(),
            "model_config": asdict(SupervisorBcCfg()),
            "model_state_dict": actor.state_dict(),
        },
        supervisor_checkpoint,
    )

    loaded = load_learned_supervisor(
        supervisor_checkpoint, base_checkpoint, "cpu"
    )

    assert isinstance(loaded, ObstacleSupervisor)


def test_load_hc4r_accepts_explicit_near_range_bc_stage(tmp_path):
    base_checkpoint = tmp_path / "locomotion.pt"
    base_checkpoint.write_bytes(b"frozen locomotion")
    actor = ObstacleSupervisor()
    supervisor_checkpoint = tmp_path / "supervisor.pt"
    torch.save(
        {
            "stage": "HC4R-near-range-behavioral-cloning",
            "decision": "offline-imitation-pass",
            "source_locomotion_checkpoint_sha256": hashlib.sha256(
                base_checkpoint.read_bytes()
            ).hexdigest(),
            "model_config": asdict(SupervisorBcCfg()),
            "model_state_dict": actor.state_dict(),
        },
        supervisor_checkpoint,
    )

    loaded = load_learned_supervisor(
        supervisor_checkpoint, base_checkpoint, "cpu"
    )

    assert isinstance(loaded, ObstacleSupervisor)


def test_load_hc4r2_accepts_student_state_correction_stage(tmp_path):
    base_checkpoint = tmp_path / "locomotion.pt"
    base_checkpoint.write_bytes(b"frozen locomotion")
    actor = ObstacleSupervisor()
    supervisor_checkpoint = tmp_path / "supervisor.pt"
    torch.save(
        {
            "stage": "HC4R2-student-state-correction-BC",
            "decision": "offline-imitation-pass",
            "source_locomotion_checkpoint_sha256": hashlib.sha256(
                base_checkpoint.read_bytes()
            ).hexdigest(),
            "model_config": asdict(SupervisorBcCfg()),
            "model_state_dict": actor.state_dict(),
        },
        supervisor_checkpoint,
    )

    loaded = load_learned_supervisor(
        supervisor_checkpoint, base_checkpoint, "cpu"
    )

    assert isinstance(loaded, ObstacleSupervisor)


def test_load_hc4lh_wraps_center_and_lateral_supervisors(tmp_path):
    base_checkpoint = tmp_path / "locomotion.pt"
    base_checkpoint.write_bytes(b"frozen locomotion")
    actor = ObstacleSupervisor()
    supervisor_checkpoint = tmp_path / "supervisor.pt"
    torch.save(
        {
            "stage": "HC4LH-lateral-gated-supervisor",
            "decision": "composition-complete-pending-rollout",
            "lateral_gate_m": 0.06,
            "source_locomotion_checkpoint_sha256": hashlib.sha256(
                base_checkpoint.read_bytes()
            ).hexdigest(),
            "model_config": asdict(SupervisorBcCfg()),
            "model_state_dict": actor.state_dict(),
            "center_model_state_dict": actor.state_dict(),
        },
        supervisor_checkpoint,
    )

    loaded = load_learned_supervisor(
        supervisor_checkpoint, base_checkpoint, "cpu"
    )

    assert isinstance(loaded, LateralGatedSupervisor)


def test_load_hc4r2h_wraps_far_and_near_supervisors(tmp_path):
    base_checkpoint = tmp_path / "locomotion.pt"
    base_checkpoint.write_bytes(b"frozen locomotion")
    actor = ObstacleSupervisor()
    supervisor_checkpoint = tmp_path / "supervisor.pt"
    torch.save(
        {
            "stage": "HC4R2H-range-speed-gated-supervisor",
            "decision": "composition-complete-pending-rollout",
            "lateral_gate_m": 0.02,
            "near_range_gate_m": 0.95,
            "max_near_nominal_speed_mps": 0.40,
            "source_locomotion_checkpoint_sha256": hashlib.sha256(
                base_checkpoint.read_bytes()
            ).hexdigest(),
            "model_config": asdict(SupervisorBcCfg()),
            "model_state_dict": actor.state_dict(),
            "center_model_state_dict": actor.state_dict(),
            "near_model_state_dict": actor.state_dict(),
        },
        supervisor_checkpoint,
    )

    loaded = load_learned_supervisor(
        supervisor_checkpoint, base_checkpoint, "cpu"
    )

    assert isinstance(loaded, RangeSpeedGatedSupervisor)
    assert isinstance(loaded.far_supervisor, LateralGatedSupervisor)


def test_load_hc4r2l_wraps_episode_latched_supervisor(tmp_path):
    base_checkpoint = tmp_path / "locomotion.pt"
    base_checkpoint.write_bytes(b"frozen locomotion")
    actor = ObstacleSupervisor()
    supervisor_checkpoint = tmp_path / "supervisor.pt"
    torch.save(
        {
            "stage": "HC4R2L-episode-latched-supervisor",
            "decision": "composition-complete-pending-rollout",
            "selector_state": "latched-until-explicit-episode-reset",
            "lateral_gate_m": 0.02,
            "near_range_gate_m": 0.95,
            "max_near_nominal_speed_mps": 0.40,
            "source_locomotion_checkpoint_sha256": hashlib.sha256(
                base_checkpoint.read_bytes()
            ).hexdigest(),
            "model_config": asdict(SupervisorBcCfg()),
            "model_state_dict": actor.state_dict(),
            "center_model_state_dict": actor.state_dict(),
            "near_model_state_dict": actor.state_dict(),
        },
        supervisor_checkpoint,
    )

    loaded = load_learned_supervisor(
        supervisor_checkpoint, base_checkpoint, "cpu"
    )

    assert isinstance(loaded, EpisodeLatchedRangeSpeedSupervisor)
    assert isinstance(loaded.far_supervisor, LateralGatedSupervisor)


def test_load_hc4r2l_rejects_missing_selector_state(tmp_path):
    base_checkpoint = tmp_path / "locomotion.pt"
    base_checkpoint.write_bytes(b"frozen locomotion")
    actor = ObstacleSupervisor()
    supervisor_checkpoint = tmp_path / "supervisor.pt"
    torch.save(
        {
            "stage": "HC4R2L-episode-latched-supervisor",
            "decision": "composition-complete-pending-rollout",
            "source_locomotion_checkpoint_sha256": hashlib.sha256(
                base_checkpoint.read_bytes()
            ).hexdigest(),
            "model_config": asdict(SupervisorBcCfg()),
            "model_state_dict": actor.state_dict(),
        },
        supervisor_checkpoint,
    )

    with pytest.raises(ValueError, match="invalid selector state"):
        load_learned_supervisor(supervisor_checkpoint, base_checkpoint, "cpu")
