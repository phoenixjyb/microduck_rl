import hashlib
import json
import subprocess
import sys

import pytest

from mjlab_microduck.body_response_audit import summarize_case


JOINTS = ["left_hip_yaw", "right_hip_yaw"]


def make_case(speed=0.0, yaw=0.0):
    means_yaw = [yaw + delta for delta in (-.1, -.05, 0., .05, .1, -.02, .02, 0.)]
    means_vx = [speed + delta for delta in (-.1, -.05, 0., .05, .1, -.02, .02, 0.)]
    yaw_mae = [abs(delta) + .02 for delta in (-.1, -.05, 0., .05, .1, -.02, .02, 0.)]
    speed_mae = [abs(delta) + .01 for delta in (-.1, -.05, 0., .05, .1, -.02, .02, 0.)]
    return {
        "seed": 839,
        "speed": speed,
        "yaw": yaw,
        "worlds": 8,
        "steps": 240,
        "complete": True,
        "terminal_worlds": [],
        "checkpoint_sha256": "a" * 64,
        "speed_mae": speed_mae,
        "yaw_mae": yaw_mae,
        "response_diagnostics": {
            "protocol": "signed-body-response-and-joint-load-v1",
            "joint_columns": JOINTS[:],
            "steps": 240,
            "startup_steps": 60,
            "settled_steps": 180,
            "mean_body_vx_per_world": means_vx,
            "mean_body_yaw_per_world": means_yaw,
            "mean_abs_body_yaw_per_world": [abs(value) + .01 for value in means_yaw],
            "yaw_correct_sign_fraction_per_world": None if yaw == 0 else [.75] * 8,
            "abs_force_p99_nm_by_joint": [.1, .2],
            "mean_signed_force_nm_by_joint": [0., 0.],
            "soft_limit_fraction_by_joint": [0., .01],
            "hard_stop_proximity_fraction_by_joint": [0., .02],
            "hard_range_violation_fraction_by_joint": [0., 0.],
            "min_distance_to_hard_stop_rad_by_joint": [.1, .2],
            "positions_not_synchronous_with_force": True,
            "velocity_and_position_timing": "pre-action body-frame state; startup excluded only for response",
            "force_timing": "post-decimation pre-reset control-step stream; all steps included",
            "policy_acceptance": False,
            "physical_motion_authorized": False,
        },
    }


def test_zero_command_reports_body_bias_and_no_yaw_sign_fraction():
    result = summarize_case(make_case())
    assert result["status"] == "descriptive-audit-only"
    assert result["policy_acceptance"] is False
    assert result["physical_motion_authorized"] is False
    assert result["worlds"][0]["yaw_correct_sign_fraction"] is None
    assert result["worlds"][0]["yaw_signed_bias"] == pytest.approx(-.1)
    assert result["worlds"][0]["yaw_abs_bias_lower_bound"] == pytest.approx(.1)
    assert result["worlds"][0]["yaw_jensen_gap"] == pytest.approx(.02)


def test_straight_case_reports_analogous_speed_bias_and_yaw_terms():
    result = summarize_case(make_case(speed=.3, yaw=0.))
    world = result["worlds"][0]
    assert world["speed_signed_bias"] == pytest.approx(-.1)
    assert world["speed_abs_bias_lower_bound"] == pytest.approx(.1)
    assert world["speed_mae"] == pytest.approx(.11)
    assert world["speed_jensen_gap"] == pytest.approx(.01)
    assert world["mean_abs_body_yaw"] >= abs(world["mean_body_yaw"])


@pytest.mark.parametrize("yaw", [-.2, .2])
def test_signed_turn_cases_preserve_directional_response_fraction(yaw):
    world = summarize_case(make_case(speed=.2, yaw=yaw))["worlds"][0]
    assert world["yaw_correct_sign_fraction"] == pytest.approx(.75)
    assert world["yaw_signed_bias"] == pytest.approx(-.1)
    assert world["yaw_mae"] == pytest.approx(.12)
    assert world["yaw_jensen_gap"] == pytest.approx(.02)


@pytest.mark.parametrize("mutation", ["incomplete", "worlds", "steps", "terminal", "seed", "hash"])
def test_rejects_wrong_case_identity_or_episode_coverage(mutation):
    row = make_case(yaw=.2)
    if mutation == "incomplete": row["complete"] = False
    elif mutation == "worlds": row["worlds"] = 8.0
    elif mutation == "steps": row["steps"] = 239
    elif mutation == "terminal": row["terminal_worlds"] = [3]
    elif mutation == "seed": row["seed"] = True
    else: row["checkpoint_sha256"] = "A" * 64
    with pytest.raises(ValueError):
        summarize_case(row)


@pytest.mark.parametrize("field,value", [("speed", True), ("yaw", False),
                                          ("speed", float("nan")), ("yaw", float("inf"))])
def test_rejects_boolean_or_nonfinite_commands(field, value):
    row = make_case(yaw=.2)
    row[field] = value
    with pytest.raises(ValueError):
        summarize_case(row)


@pytest.mark.parametrize("mutation", ["missing", "short", "bool", "nan", "inf"])
@pytest.mark.parametrize("key", ["speed_mae", "yaw_mae"])
def test_rejects_malformed_case_world_arrays(mutation, key):
    row = make_case(yaw=.2)
    if mutation == "missing": row.pop(key)
    elif mutation == "short": row[key] = row[key][:-1]
    elif mutation == "bool": row[key][2] = True
    elif mutation == "nan": row[key][2] = float("nan")
    else: row[key][2] = float("inf")
    with pytest.raises(ValueError):
        summarize_case(row)


@pytest.mark.parametrize("mutation", ["protocol", "window", "duplicate_names", "joint_shape",
                                       "nonfinite_joint", "fraction", "timing"])
def test_rejects_malformed_response_diagnostics(mutation):
    row = make_case(yaw=.2)
    diag = row["response_diagnostics"]
    if mutation == "protocol": diag["protocol"] = "other"
    elif mutation == "window": diag["settled_steps"] = 179
    elif mutation == "duplicate_names": diag["joint_columns"] = ["same", "same"]
    elif mutation == "joint_shape": diag["mean_signed_force_nm_by_joint"] = [0.]
    elif mutation == "nonfinite_joint": diag["abs_force_p99_nm_by_joint"][0] = float("nan")
    elif mutation == "fraction": diag["soft_limit_fraction_by_joint"][0] = 1.1
    else: diag["positions_not_synchronous_with_force"] = False
    with pytest.raises(ValueError):
        summarize_case(row)


@pytest.mark.parametrize("mutation", ["missing", "short", "bool", "nonfinite", "range"])
def test_rejects_invalid_per_world_response_identity(mutation):
    row = make_case(yaw=.2)
    diag = row["response_diagnostics"]
    key = "mean_body_yaw_per_world"
    if mutation == "missing": diag.pop(key)
    elif mutation == "short": diag[key] = diag[key][:-1]
    elif mutation == "bool": diag[key][2] = True
    elif mutation == "nonfinite": diag[key][2] = float("inf")
    else: diag["yaw_correct_sign_fraction_per_world"][2] = 1.1
    with pytest.raises(ValueError):
        summarize_case(row)


@pytest.mark.parametrize("mutation", ["mae_lower_bound", "absolute_mean_lower_bound", "absolute_mean_upper_bound",
                                       "speed_lower_bound"])
def test_rejects_impossible_summary_identities(mutation):
    row = make_case(speed=.2, yaw=.2)
    if mutation == "mae_lower_bound":
        row["yaw_mae"][0] = .01
    elif mutation == "absolute_mean_lower_bound":
        row["response_diagnostics"]["mean_abs_body_yaw_per_world"][0] = .01
    elif mutation == "absolute_mean_upper_bound":
        row["response_diagnostics"]["mean_abs_body_yaw_per_world"][0] = 2.
    else:
        row["speed_mae"][0] = .01
    with pytest.raises(ValueError):
        summarize_case(row)


def test_tolerance_is_tiny_descriptive_roundoff_only():
    row = make_case(yaw=.2)
    row["yaw_mae"][0] = .1 - 5e-7
    result = summarize_case(row)
    assert result["worlds"][0]["yaw_jensen_gap"] == pytest.approx(-5e-7)
    assert result["summary_validation_tolerance"] == 1e-6
    assert result["summary_validation_tolerance_is_acceptance_threshold"] is False
    row["yaw_mae"][0] = .1 - 2e-6
    with pytest.raises(ValueError):
        summarize_case(row)


@pytest.mark.parametrize("fault", ["velocity-frame", "force-time", "policy", "motion", "negative-force", "missing-idle-sign"])
def test_rejects_changed_sampling_or_acceptance_identity(fault):
    row = make_case()
    diag = row["response_diagnostics"]
    if fault == "velocity-frame": diag["velocity_and_position_timing"] = "world-frame"
    elif fault == "force-time": diag["force_timing"] = "post-reset"
    elif fault == "policy": diag["policy_acceptance"] = True
    elif fault == "motion": diag["physical_motion_authorized"] = True
    elif fault == "negative-force": diag["abs_force_p99_nm_by_joint"][0] = -.1
    else: diag.pop("yaw_correct_sign_fraction_per_world")
    with pytest.raises(ValueError):
        summarize_case(row)


def test_cli_prints_deterministic_hash_bound_summary_without_writing(tmp_path):
    row = make_case(speed=.2, yaw=-.2)
    raw = json.dumps(row, sort_keys=True).encode("utf-8")
    input_path = tmp_path / "case.json"
    input_path.write_bytes(raw)
    command = [sys.executable, "-m", "mjlab_microduck.body_response_audit", str(input_path)]
    first = subprocess.run(command, capture_output=True, check=False, text=True)
    second = subprocess.run(command, capture_output=True, check=False, text=True)
    assert first.returncode == second.returncode == 0
    assert first.stdout == second.stdout
    report = json.loads(first.stdout)
    assert report["input_sha256"] == hashlib.sha256(raw).hexdigest()
    assert report["policy_acceptance"] is False
    assert set(path.name for path in tmp_path.iterdir()) == {"case.json"}


def test_module_import_does_not_load_simulation_or_tensor_libraries():
    code = (
        "import sys; import mjlab_microduck.body_response_audit; "
        "assert 'torch' not in sys.modules; assert 'mjlab' not in sys.modules"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
