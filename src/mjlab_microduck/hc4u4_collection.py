"""Frozen admission gate for HC4-U3 student-state near-envelope corrections."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch

from mjlab_microduck.hc4u1_gate import ACTOR_SHA256, PROTOCOL, _load_report, _sha256
from mjlab_microduck.hierarchical_obstacle_rollout import FIRST_TERMINAL_OUTCOME_PROTOCOL
from mjlab_microduck.hierarchical_obstacle import ObstacleTeacherCfg
from mjlab_microduck.o3a_gate import _reject_nonfinite

COLLECTION_SEEDS = (317, 331, 337)
STUDENT_SHA256 = "6c6546448340530e4cdb7e4381247f644211029c7750146c0da6dbcf7c60aa2d"
STUDENT_STAGE = "HC4U3-phase-separated-BC-rollout"
MIN_SAMPLES = 20_000


def validate_correction_shard(report_path: Path, dataset_path: Path, *, seed: int) -> dict:
    """Validate all six cells and every first-attempt label before any new fit."""
    if seed not in COLLECTION_SEEDS:
        raise ValueError("seed is not predeclared for HC4-U4 collection")
    report = json.loads(report_path.read_text())
    _reject_nonfinite(report)
    _load_report(
        report_path, expected_stage=STUDENT_STAGE,
        expected_supervisor_sha256=STUDENT_SHA256,
        forward_positions=(0.90,), expected_seed=seed,
    )
    if report.get("terminal_outcome_protocol") != FIRST_TERMINAL_OUTCOME_PROTOCOL:
        raise ValueError("collection must use failure-priority accounting")
    sensor = report.get("obstacle_sensor_model", {})
    if any(sensor.get(k) != 0.0 for k in (
        "range_noise_m", "bearing_noise_rad", "width_noise_m", "height_noise_m",
        "closing_rate_noise_mps", "dropout_probability",
    )):
        raise ValueError("collection geometry must be exact")
    if report.get("attempt_timeout_s") != 12.0:
        raise ValueError("collection timeout must remain 12 seconds")
    if Path(report.get("teacher_correction_dataset", "")).resolve() != dataset_path.resolve():
        raise ValueError("report must name the supplied dataset")
    payload = torch.load(dataset_path, map_location="cpu", weights_only=False)
    for key, expected in (
        ("stage", "HC4R2-student-state-teacher-corrections"),
        ("checkpoint_sha256", ACTOR_SHA256),
        ("student_supervisor_checkpoint_sha256", STUDENT_SHA256),
        ("collection_window", PROTOCOL),
        ("terminal_outcome_protocol", FIRST_TERMINAL_OUTCOME_PROTOCOL),
        ("collection_seeds", [seed]),
        ("observation_dim", 17),
        ("command_fields", ["forward_speed_mps", "yaw_rate_rps"]),
        ("outcome_codes", {"clean_pass": 1, "collision": 2, "timeout": 3}),
        ("teacher_config", asdict(ObstacleTeacherCfg())),
    ):
        if payload.get(key) != expected:
            raise ValueError(f"dataset identity mismatch: {key}")
    x, y, student, keys, codes = [payload[k] for k in (
        "observations", "commands", "student_commands", "episode_keys", "sample_outcome_codes"
    )]
    if not all(isinstance(t, torch.Tensor) for t in (x, y, student, keys, codes)):
        raise ValueError("correction fields must be tensors")
    if x.ndim != 2 or x.shape[1] != 17 or len(x) < MIN_SAMPLES:
        raise ValueError("insufficient finite 17D samples")
    if y.shape != (len(x), 2) or student.shape != y.shape or keys.shape != (len(x),) or codes.shape != keys.shape:
        raise ValueError("correction tensor lengths or shapes differ")
    if keys.dtype != torch.int64 or codes.dtype != torch.int8:
        raise ValueError("episode keys and outcome codes must retain int64/int8")
    if not all(bool(torch.isfinite(t).all()) for t in (x, y, student)):
        raise ValueError("correction contains nonfinite values")
    phase = x[:, 13:16]
    if not bool((((phase == 0) | (phase == 1)).all(-1) & (phase.sum(-1) == 1)).all()):
        raise ValueError("invalid phase one-hot code")
    cfg = ObstacleTeacherCfg()
    for commands in (y, student):
        if not bool(((commands[:, 0] >= 0) & (commands[:, 0] <= cfg.max_forward_speed_mps)
                     & (commands[:, 1].abs() <= cfg.max_yaw_rate_rps)).all()):
            raise ValueError("teacher or student commands exceed existing bounds")
    expected_keys = torch.cat([i * 1_000_000 + torch.arange(64) for i in range(6)])
    if not torch.equal(torch.unique(keys), expected_keys):
        raise ValueError("dataset must contain each of 384 first attempts exactly once")
    episode_outcomes = {}
    for key in expected_keys.tolist():
        values = torch.unique(codes[keys == key])
        if len(values) != 1 or int(values[0]) not in (1, 2, 3):
            raise ValueError("each episode must have one valid outcome")
        episode_outcomes[key] = int(values[0])
    totals = dict.fromkeys(("clean_pass_events", "collision_events", "attempt_timeout_events"), 0)
    for i, case in enumerate(report["cases"]):
        expected_cell = ((0.30, 0.40)[i // 3], (-0.08, 0.0, 0.08)[i % 3])
        if (case["nominal_speed_mps"], case["obstacle_lateral_m"]) != expected_cell:
            raise ValueError("case order must match episode namespacing")
        if case.get("steps") != 700 or case.get("terminal_outcome_protocol") != FIRST_TERMINAL_OUTCOME_PROTOCOL:
            raise ValueError("case collection protocol mismatch")
        if not 0 <= case["motor_torque_utilization_p99"] <= 0.60:
            raise ValueError("collection exceeds the torque envelope")
        for code, field in enumerate(totals, start=1):
            observed = sum(episode_outcomes[i * 1_000_000 + e] == code for e in range(64))
            if type(case.get(field)) is not int or case[field] != observed:
                raise ValueError("dataset outcome counts disagree with case report")
            totals[field] += observed
    if any(report["totals"].get(k) != v for k, v in totals.items()):
        raise ValueError("aggregate outcomes disagree with cases")
    if report.get("teacher_correction_dataset_samples") != len(x) or report.get("teacher_correction_dataset_episodes") != 384:
        raise ValueError("report dataset counts disagree")
    disagreement = float((y - student).abs().mean())
    if disagreement <= 0:
        raise ValueError("corrections have no teacher/student disagreement")
    return {
        "protocol": "HC4-U4-near-first-attempt-correction-v1",
        "decision": "admit-correction-shard-not-policy",
        "seed": seed, "samples": len(x), "episodes": 384,
        "report_sha256": _sha256(report_path), "dataset_sha256": _sha256(dataset_path),
        "student_sha256": STUDENT_SHA256, "locomotion_sha256": ACTOR_SHA256,
        "outcomes": totals, "mean_absolute_teacher_student_disagreement": disagreement,
        "physical_motion_authorized": False,
    }
