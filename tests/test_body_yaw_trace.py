import hashlib
import json
import math
import subprocess
import sys

import pytest

from mjlab_microduck.body_yaw_trace import PROTOCOL, TIMING, summarize_trace


STEPS, WORLDS, STARTUP = 240, 8, 60
DT = .02
CHECKPOINT = "a" * 64


def rotate(q, v):
    w, x, y, z = q
    vx, vy, vz = v
    cx, cy, cz = y * vz - z * vy, z * vx - x * vz, x * vy - y * vx
    dx, dy, dz = y * cz - z * cy, z * cx - x * cz, x * cy - y * cx
    return [vx + 2 * (w * cx + dx), vy + 2 * (w * cy + dy), vz + 2 * (w * cz + dz)]


def make_inputs(*, yaw=.2, residuals=None, quaternions=None, angular_b=None):
    expected = [.2, 0., yaw]
    commands = [[expected[:] for _ in range(WORLDS)] for _ in range(STEPS)]
    actors = [[expected[:] for _ in range(WORLDS)] for _ in range(STEPS)]
    identity_quaternions = [[[1., 0., 0., 0.] for _ in range(WORLDS)]
                            for _ in range(STEPS)]
    source_quaternions = quaternions if quaternions is not None else identity_quaternions
    qs = [[source_quaternions[t][w][:] for w in range(WORLDS)] for t in range(STEPS)]
    rates_b = []
    rates_w = []
    mae = []
    for t in range(STEPS):
        b_row, w_row = [], []
        for w in range(WORLDS):
            if angular_b is not None:
                b = angular_b[t][w][:]
            else:
                errors = residuals or [0.05] * (STEPS - STARTUP)
                index = max(0, t - STARTUP)
                error = errors[min(index, len(errors) - 1)]
                b = [0.1, -0.1, yaw + error]
            b_row.append(b)
            w_row.append(rotate(qs[t][w], b))
        rates_b.append(b_row)
        rates_w.append(w_row)
    for w in range(WORLDS):
        residual = [rates_b[t][w][2] - yaw for t in range(STARTUP, STEPS)]
        mae.append(sum(abs(value) for value in residual) / len(residual))
    case = {
        "seed": 839,
        "complete": True,
        "steps": STEPS,
        "worlds": WORLDS,
        "terminal_worlds": [],
        "checkpoint_sha256": CHECKPOINT,
        "speed": .2,
        "yaw": yaw,
        "yaw_mae": mae,
    }
    trace = {
        "protocol": PROTOCOL,
        "step_dt": DT,
        "startup_steps": STARTUP,
        "steps": STEPS,
        "worlds": WORLDS,
        "checkpoint_sha256": CHECKPOINT,
        "timing": TIMING,
        "command": commands,
        "actor_command": actors,
        "quat_wxyz": qs,
        "angular_b": rates_b,
        "angular_w": rates_w,
        "policy_acceptance": False,
        "physical_motion_authorized": False,
    }
    return case, trace


def test_untilted_trace_has_equal_body_world_and_heading_rates():
    case, trace = make_inputs()
    report = summarize_trace(case, trace)
    world = report["worlds"][0]
    assert report["status"] == "descriptive-audit-only"
    assert report["policy_acceptance"] is False
    assert world["body_yaw_mae"] == pytest.approx(.05)
    assert world["body_yaw_signed_bias"] == pytest.approx(.05)
    assert world["body_yaw_residual_std"] == pytest.approx(0.)
    assert world["body_yaw_residual_sign_transitions"] == 0
    assert world["world_vertical_rate_mae"] == pytest.approx(.05)
    assert world["projected_body_x_heading_rate_mean"] == pytest.approx(.25)
    assert world["projected_body_x_heading_rate_mae"] == pytest.approx(.05)
    assert world["projected_body_x_heading_rate_defined_samples"] == 180


def test_tilt_can_separate_body_z_world_vertical_and_heading_rates():
    angle = .5
    q = [math.cos(angle / 2), math.sin(angle / 2), 0., 0.]
    quaternions = [[q[:] for _ in range(WORLDS)] for _ in range(STEPS)]
    case, trace = make_inputs(yaw=.2, quaternions=quaternions)
    world = summarize_trace(case, trace)["worlds"][0]
    assert world["body_yaw_mae"] == pytest.approx(.05)
    assert world["world_vertical_rate_mae"] != pytest.approx(world["body_yaw_mae"])
    assert world["projected_body_x_heading_rate_mae"] != pytest.approx(world["body_yaw_mae"])


def test_time_variation_is_distinguished_from_constant_bias_without_spectral_claim():
    constant_case, constant_trace = make_inputs(residuals=[.05] * 180)
    alternating = [.05 if index % 2 else -.05 for index in range(180)]
    varying_case, varying_trace = make_inputs(residuals=alternating)
    constant = summarize_trace(constant_case, constant_trace)["worlds"][0]
    varying = summarize_trace(varying_case, varying_trace)["worlds"][0]
    assert constant["body_yaw_mae"] == varying["body_yaw_mae"] == pytest.approx(.05)
    assert constant["body_yaw_signed_bias"] == pytest.approx(.05)
    assert varying["body_yaw_signed_bias"] == pytest.approx(0.)
    assert constant["body_yaw_residual_std"] == pytest.approx(0.)
    assert varying["body_yaw_residual_std"] == pytest.approx(.05)
    assert varying["body_yaw_residual_sign_transitions"] == 179
    assert "frequency_hz" not in varying


def test_near_vertical_body_x_axis_marks_heading_undefined_without_dropping_body_metric():
    half = math.sqrt(.5)
    q = [half, 0., half, 0.]
    qs = [[q[:] for _ in range(WORLDS)] for _ in range(STEPS)]
    case, trace = make_inputs(quaternions=qs)
    world = summarize_trace(case, trace)["worlds"][0]
    assert world["projected_body_x_heading_rate_defined_samples"] == 0
    assert world["projected_body_x_heading_rate_undefined_samples"] == 180
    assert world["projected_body_x_heading_rate_mean"] is None
    assert world["body_yaw_mae"] == pytest.approx(.05)


@pytest.mark.parametrize("mutation", ["unit", "rotation", "command", "actor", "timing",
                                       "step", "worlds", "policy", "motion", "hash"])
def test_rejects_changed_trace_identity_or_frame(mutation):
    case, trace = make_inputs()
    if mutation == "unit":
        trace["quat_wxyz"][70][0] = [2., 0., 0., 0.]
    elif mutation == "rotation":
        trace["angular_w"][70][0][1] += .01
    elif mutation == "command":
        trace["command"][70][0][2] += .01
    elif mutation == "actor":
        trace["actor_command"][70][0][1] = .1
    elif mutation == "timing":
        trace["timing"] = "post-action"
    elif mutation == "step":
        trace["step_dt"] = .01
    elif mutation == "worlds":
        trace["worlds"] = 7
    elif mutation == "policy":
        trace["policy_acceptance"] = True
    elif mutation == "motion":
        trace["physical_motion_authorized"] = True
    else:
        trace["checkpoint_sha256"] = "b" * 64
    with pytest.raises(ValueError):
        summarize_trace(case, trace)


@pytest.mark.parametrize("key", ["command", "actor_command", "quat_wxyz", "angular_b", "angular_w"])
@pytest.mark.parametrize("bad", ["missing", "shape", "bool", "nan", "inf"])
def test_rejects_missing_malformed_or_nonfinite_arrays(key, bad):
    case, trace = make_inputs()
    if bad == "missing":
        trace.pop(key)
    elif bad == "shape":
        trace[key] = trace[key][:-1]
    elif bad == "bool":
        trace[key][0][0][0] = True
    elif bad == "nan":
        trace[key][0][0][0] = float("nan")
    else:
        trace[key][0][0][0] = float("inf")
    with pytest.raises(ValueError):
        summarize_trace(case, trace)


def test_rejects_case_identity_and_original_mae_mismatch():
    case, trace = make_inputs()
    case["complete"] = False
    with pytest.raises(ValueError):
        summarize_trace(case, trace)
    case["complete"] = True
    case["yaw_mae"][0] += .01
    with pytest.raises(ValueError, match="MAE mismatch"):
        summarize_trace(case, trace)


def test_rejects_non_heldout_case_command():
    case, trace = make_inputs()
    case["speed"] = .25
    for step in range(STEPS):
        for world in range(WORLDS):
            trace["command"][step][world][0] = .25
            trace["actor_command"][step][world][0] = .25
    with pytest.raises(ValueError, match="held-out commands"):
        summarize_trace(case, trace)


def test_cli_is_deterministic_and_binds_raw_input_hash_without_writes(tmp_path):
    case, trace = make_inputs(yaw=-.2)
    raw = json.dumps({"case": case, "trace": trace}, sort_keys=True).encode()
    path = tmp_path / "trace.json"
    path.write_bytes(raw)
    command = [sys.executable, "-m", "mjlab_microduck.body_yaw_trace", str(path)]
    first = subprocess.run(command, capture_output=True, text=True, check=False)
    second = subprocess.run(command, capture_output=True, text=True, check=False)
    assert first.returncode == second.returncode == 0, first.stderr
    assert first.stdout == second.stdout
    report = json.loads(first.stdout)
    assert report["input_sha256"] == hashlib.sha256(raw).hexdigest()
    assert report["policy_acceptance"] is False
    assert set(item.name for item in tmp_path.iterdir()) == {"trace.json"}


def test_import_does_not_load_simulator_or_tensor_library():
    code = (
        "import sys; import mjlab_microduck.body_yaw_trace; "
        "assert 'torch' not in sys.modules; assert 'mjlab' not in sys.modules"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("seed", [True, 839., 1, None])
def test_seed_identity_fails_closed(seed):
    case, trace = make_inputs()
    case["seed"] = seed
    with pytest.raises(ValueError):
        summarize_trace(case, trace)


def test_trace_retains_original_command_tolerance_and_exact_actor_identity():
    case, trace = make_inputs()
    trace["command"][0][0][0] += 5e-7
    trace["actor_command"][0][0][0] = trace["command"][0][0][0]
    summarize_trace(case, trace)
    trace["command"][0][0][0] += 2e-6
    trace["actor_command"][0][0][0] = trace["command"][0][0][0]
    with pytest.raises(ValueError, match="command changed"):
        summarize_trace(case, trace)
    case, trace = make_inputs()
    trace["actor_command"][0][0][0] += 1e-9
    with pytest.raises(ValueError, match="actor command mismatch"):
        summarize_trace(case, trace)
