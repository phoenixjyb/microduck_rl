import json
from pathlib import Path

import pytest

from mjlab_microduck.hop_checkpoint_sweep import summarize_h1_checkpoint_sweep
from mjlab_microduck.hop_evaluation import H1_PROTOCOL, h1_decision


def _case(*, accepted: bool, episode: float = 0.9, cycle: float = 0.95, torque: float = 0.8):
    case = {
        "cycle_success_fraction": cycle,
        "landing_fraction": 0.99,
        "episode_pass_fraction": episode,
        "cycle_rise_p50_m": 0.03,
        "fall_events": 0,
        "nan_termination_events": 0,
        "nonfinite_episodes": 0,
        "cycle_rise_peak_m": 0.08,
        "max_drift_p95_m": 0.08,
        "spring_bottomed_fraction": 0.005,
        "motor_speed_rated_exceed_fraction": 0.0,
        "motor_torque_utilization_p99": torque,
        "motor_torque_near_stall_fraction": 0.001,
    }
    if not accepted:
        case["fall_events"] = 1
    return case


def _evaluation(path: Path, training_seed: int, iteration: int, **metrics) -> Path:
    cases = [_case(**metrics) for _ in (211, 223, 227)]
    decision = h1_decision(cases)
    value = {
        "protocol": H1_PROTOCOL,
        "checkpoint": str(path.parent / f"seed-{training_seed}" / f"model_{iteration}.pt"),
        "checkpoint_sha256": f"hash-{training_seed}-{iteration}",
        "seeds": [211, 223, 227],
        "cases": cases,
        **decision,
    }
    path.write_text(json.dumps(value))
    return path


def _manifest(tmp_path: Path, evaluations: list[tuple[int, int, Path]]) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "campaign_id": "h1-test",
                "training_seeds": [47, 53, 59],
                "candidates": [
                    {
                        "training_seed": seed,
                        "checkpoint_iteration": iteration,
                        "hop_evaluation": str(evaluation),
                    }
                    for seed, iteration, evaluation in evaluations
                ],
            }
        )
    )
    return path


def test_selects_earliest_all_seed_pass_and_deterministic_representative(tmp_path):
    evaluations = []
    for iteration in (500, 1000):
        for seed in (47, 53, 59):
            accepted = iteration == 1000
            episode = {47: 0.91, 53: 0.93, 59: 0.93}[seed]
            torque = {47: 0.80, 53: 0.82, 59: 0.79}[seed]
            path = _evaluation(
                tmp_path / f"eval-{seed}-{iteration}.json",
                seed,
                iteration,
                accepted=accepted,
                episode=episode,
                cycle=0.96,
                torque=torque,
            )
            evaluations.append((seed, iteration, path))
    summary = summarize_h1_checkpoint_sweep(_manifest(tmp_path, evaluations))
    assert summary["selected_checkpoint_iteration"] == 1000
    # Seeds 53 and 59 tie on episode/cycle; lower torque selects seed 59.
    assert summary["selected_candidate"]["training_seed"] == 59
    assert summary["physical_motion_authorized"] is False


def test_incomplete_iteration_cannot_pass(tmp_path):
    evaluations = []
    for seed in (47, 53):
        path = _evaluation(
            tmp_path / f"eval-{seed}.json", seed, 500, accepted=True
        )
        evaluations.append((seed, 500, path))
    summary = summarize_h1_checkpoint_sweep(_manifest(tmp_path, evaluations))
    assert summary["decision"] == "rejected"
    assert summary["iterations"][0]["complete"] is False


def test_rejects_stored_decision_that_disagrees_with_cases(tmp_path):
    evaluation = _evaluation(tmp_path / "eval.json", 47, 500, accepted=True)
    value = json.loads(evaluation.read_text())
    value["decision"] = "rejected"
    evaluation.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="disagrees"):
        summarize_h1_checkpoint_sweep(_manifest(tmp_path, [(47, 500, evaluation)]))
