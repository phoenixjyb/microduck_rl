"""Synthetic scoring fixtures only; no simulator or trained capability evidence."""
import copy
import hashlib
import json
import math
import subprocess
import sys

import pytest

from mjlab_microduck import integrated_transition_gate as gate


def world(sequence="stop-restart", index=0):
    frames, x = [], 0.
    for phase in gate.phases(sequence):
        vx, _, yaw = phase["command"]
        for step in range(phase["start_step"], phase["end_step"]):
            x += vx*gate.DT
            frames.append(dict(step=step, time_s=(step+1)*gate.DT,
                command=list(phase["command"]), actor_command=list(phase["command"]),
                velocity_b=[vx, 0., 0.], yaw_rate=yaw, world_xy=[x, 0.], tilt_rad=0.,
                qpos=[0.]*14, force_nm=[.3]*14, motor_speed_rad_s=[0.]*14,
                action=[.1]*14, grounded=True, terminal=False, fall=False, reset=False, episode_id=0))
    return dict(world=index, initial_xy=[0., 0.], frames=frames)


def score(value=None, sequence="stop-restart", names=None, bounds=None):
    return gate.score_world(world(sequence) if value is None else value, sequence,
        list(gate.JOINT_NAMES) if names is None else names,
        [[-1., 1.] for _ in range(14)] if bounds is None else bounds)


def metadata():
    return dict(protocol=gate.PROTOCOL, source="a"*40, checkpoint_sha256="b"*64,
        plant_sha256="c"*64, runtime=dict(gate.RUNTIME), step_dt=.02, actor_width=61,
        action_width=14, frame_phase="post-physics-pre-reset-last-substep-v1",
        first_episode_only=True, automatic_retry=False, joint_names=list(gate.JOINT_NAMES),
        joint_limits=[[-1., 1.] for _ in range(14)])


def matrix():
    result = metadata()
    result["cases"] = [dict(seed=seed, sequence=sequence,
        worlds=[world(sequence, index) for index in range(8)])
        for seed in gate.SEEDS for sequence in gate.SEQUENCES]
    return result


def test_fixed_plan_covers_stop_restart_and_mirrored_moving_turns():
    p = gate.plan()
    assert p["seeds"] == [1351, 1361, 1367] and p["worlds"] == 8
    assert p["sequences"]["stop-restart"][-1]["end_step"] == 700
    for name in ("left-right-return", "right-left-return"):
        assert p["sequences"][name][-1]["end_step"] == 750
        assert all(phase["command"][0] > 0 for phase in p["sequences"][name])
    assert p["sequences"]["left-right-return"][1]["command"][2] == -.2
    assert p["sequences"]["right-left-return"][1]["command"][2] == .2
    assert p["collector_qualified"] is p["physical_motion_authorized"] is False


@pytest.mark.parametrize("sequence", tuple(gate.SEQUENCES))
def test_ideal_synthetic_world_has_measured_phases_not_skill_acceptance(sequence):
    r = score(sequence=sequence)
    assert r["complete"] and r["failures"] == []
    assert r["torque_p99"] == .5
    assert r["action_rate_p99"] == 0.
    for phase in r["phases"]:
        assert phase["first_settled_end_s"] == .52 and phase["settled_at_exit"]
        assert phase["speed_mae"] == phase["yaw_mae"] == 0.
    if sequence == "stop-restart":
        assert r["phases"][2]["stop_distance_m"] == 0.


@pytest.mark.parametrize("count,passes", [(25, False), (26, True)])
def test_half_second_dwell_counts_intervals_not_samples(count, passes):
    r = gate.recovery([True]*count)
    assert r["settled_at_exit"] is passes
    assert r["final_settled_span_s"] == (count-1)*.02


def test_exact_deadline_and_broken_exit_dwell():
    assert gate.recovery([False]*74+[True]*26)["settled_by_deadline"]
    assert not gate.recovery([False]*75+[True]*26)["settled_by_deadline"]
    r = gate.recovery([True]*26+[False]+[True]*25)
    assert r["settled_by_deadline"] and not r["settled_at_exit"]


@pytest.mark.parametrize("field", ["command", "actor_command"])
def test_stale_or_wrong_signed_actor_command_rejected(field):
    w = world("left-right-return")
    w["frames"][150][field][2] = .2
    with pytest.raises(ValueError, match="actor command"):
        score(w, "left-right-return")


@pytest.mark.parametrize("damage", ["gap", "time", "duplicate", "bool-step", "bool-time"])
def test_step_timestamp_coverage_is_strict(damage):
    w = world()
    if damage == "gap": w["frames"].pop(4)
    elif damage == "time": w["frames"][4]["time_s"] += .02
    elif damage == "duplicate": w["frames"].insert(4, copy.deepcopy(w["frames"][4]))
    elif damage == "bool-step": w["frames"][0]["step"] = False
    else: w["frames"][0]["time_s"] = True
    with pytest.raises(ValueError, match="step|timestamp|bounded first-attempt"):
        score(w)


@pytest.mark.parametrize("field", ["velocity_b", "qpos", "force_nm", "motor_speed_rad_s", "action", "world_xy"])
def test_nonfinite_or_bool_vectors_fail_closed(field):
    for invalid in (float("nan"), float("inf"), True):
        w = world()
        w["frames"][10][field][0] = invalid
        with pytest.raises(ValueError, match="finite"):
            score(w)


def test_reset_post_terminal_and_false_fall_masks_rejected():
    for key, value, match in (("reset", True, "post-reset"), ("episode_id", 1, "second-episode"),
                              ("terminal", True, "post-terminal"), ("fall", True, "terminal signal")):
        w = world()
        w["frames"][50][key] = value
        with pytest.raises(ValueError, match=match): score(w)


def test_terminal_prefix_is_a_failure_not_missing_data_or_reset_laundering():
    w = world()
    w["frames"] = w["frames"][:120]
    w["frames"][-1].update(terminal=True, fall=True)
    r = score(w)
    assert not r["complete"]
    assert r["failures"][0] == dict(metric="first-episode-terminal", step=119, fall=True)
    assert not any(f["metric"] == "incomplete-trace" for f in r["failures"])
    assert r["phases"][-1]["complete"] is False


def test_truncated_nonterminal_trace_is_incomplete():
    w = world()
    w["frames"].pop()
    assert any(f["metric"] == "incomplete-trace" for f in score(w)["failures"])


def test_stop_lateral_motion_cannot_qualify_as_stopped():
    w = world()
    for f in w["frames"][300:500]: f["velocity_b"][1] = .1
    r = score(w)
    stop = r["phases"][2]
    assert stop["speed_mae"] == 0.  # Existing vx-only MAE is not rewritten.
    assert not stop["settled_by_deadline"] and not stop["settled_at_exit"]


@pytest.mark.parametrize("damage", ["airborne", "vertical-motion"])
def test_stop_dwell_requires_ground_support_and_full_linear_stillness(damage):
    w = world()
    for f in w["frames"][300:500]:
        if damage == "airborne": f["grounded"] = False
        else: f["velocity_b"][2] = .1
    stop = score(w)["phases"][2]
    assert stop["speed_mae"] == 0.
    assert not stop["settled_at_exit"] and not stop["settled_by_deadline"]


def test_stop_distance_is_measured_across_entry_sample_and_is_descriptive():
    w = world()
    for i, f in enumerate(w["frames"][300:500]): f["world_xy"][1] = (i+1)*.001
    r = score(w)
    assert r["phases"][2]["stop_distance_m"] == pytest.approx(.2)
    assert not any(f["metric"] == "stop-distance" for f in r["failures"])


def test_raw_safety_including_tracking_startup_is_never_exempted():
    w = world()
    for f in w["frames"][:30]: f["force_nm"] = [.6]*14
    r = score(w)
    assert r["torque_p99"] == 1.
    assert any(f["metric"] == "torque_p99" for f in r["failures"])
    w = world()
    w["frames"][0]["motor_speed_rad_s"][0] = math.nextafter(gate.RATED_SPEED, math.inf)
    assert any(f["metric"] == "rated-speed-exceedance" for f in score(w)["failures"])


def test_existing_rated_speed_reference_and_boundary():
    assert gate.RATED_SPEED == 123.*2*math.pi/60
    w = world()
    for f in w["frames"]: f["force_nm"] = [.36]*14
    w["frames"][0]["motor_speed_rad_s"][0] = gate.RATED_SPEED
    assert score(w)["failures"] == []


def test_extreme_finite_metrics_produce_finite_rejection_and_overflow_fails_closed():
    w = world()
    for f in w["frames"]:
        f["force_nm"] = [1e153]*14
        f["velocity_b"][0] = f["yaw_rate"] = 1e307
    r = score(w)
    assert any(f["metric"] == "torque_p99" for f in r["failures"])
    assert math.isfinite(r["thermal_load_proxy"])
    assert all(math.isfinite(p["speed_mae"]) and math.isfinite(p["yaw_mae"]) for p in r["phases"])
    json.dumps(r, allow_nan=False)
    w["frames"][0]["force_nm"] = [1e308]*14
    with pytest.raises(ValueError, match="derived motor load"):
        score(w)


def test_named_qpos_range_and_stop_gates_survive_column_permutation():
    w, names, bounds = world(), list(gate.JOINT_NAMES), [[-1., 1.] for _ in range(14)]
    column = names.index("right_hip_yaw")
    for f in w["frames"][:8]: f["qpos"][column] = 1.
    r = score(w)
    assert any(f["metric"] == "right-hip-stop-exposure" for f in r["failures"])
    for key in ("qpos", "force_nm", "motor_speed_rad_s"):
        for f in w["frames"]: f[key].reverse()
    rr = score(w, names=list(reversed(names)), bounds=list(reversed(bounds)))
    assert rr["failures"] == r["failures"]
    assert rr["hard_stop_proximity_fraction"] == r["hard_stop_proximity_fraction"]


def test_all_servo_range_exact_threshold_and_next_float():
    w, bounds = world(), [[-1., 1.] for _ in range(14)]
    bounds[0] = [0., 1.]
    w["frames"][0]["qpos"][0] = -.005
    assert not any(f["metric"] == "joint-range" for f in score(w, bounds=bounds)["failures"])
    w["frames"][0]["qpos"][0] = math.nextafter(-.005, -math.inf)
    assert any(f["metric"] == "joint-range" for f in score(w, bounds=bounds)["failures"])


@pytest.mark.parametrize("key,value", [("source", "bad"), ("plant_sha256", ""), ("actor_width", 68),
    ("action_width", 15), ("step_dt", .04), ("first_episode_only", 1), ("automatic_retry", 0),
    ("frame_phase", "post-reset")])
def test_incompatible_metadata_rejected(key, value):
    t = metadata(); t[key] = value
    with pytest.raises(ValueError): gate.validate_metadata(t)


def test_runtime_and_joint_metadata_are_not_free_form():
    t = metadata(); t["runtime"]["mjlab"] = "1.6.0"
    with pytest.raises(ValueError, match="runtime"): gate.validate_metadata(t)
    t = metadata(); t["joint_names"][0] = t["joint_names"][1]
    with pytest.raises(ValueError, match="servo coverage"): gate.validate_metadata(t)
    t = metadata(); t["joint_limits"][0] = [1., 0.]
    with pytest.raises(ValueError, match="ranges"): gate.validate_metadata(t)


def test_full_matrix_has_per_world_failures_and_diagnostic_only_decision():
    t = matrix()
    r = gate.evaluate(t)
    assert len(r["cases"]) == 9 and sum(len(c["worlds"]) for c in r["cases"]) == 72
    assert r["decision"] == "transition-matrix-pass-diagnostic-only" and r["numerical_pass"]
    assert r["policy_acceptance"] is r["simulator_qualified"] is r["physical_motion_authorized"] is False
    assert r["collector_qualified"] is False and r["foundation_gate_required"]
    bad = t["cases"][4]["worlds"][6]
    for f in bad["frames"]: f["yaw_rate"] += .2
    r = gate.evaluate(t)
    assert r["decision"] == "transition-matrix-not-ready"
    assert all((f["seed"], f["sequence"], f["world"]) == (1361, "left-right-return", 6) for f in r["failures"])
    bad["frames"].pop()
    assert gate.evaluate(t)["decision"] == "transition-evidence-incomplete"


def test_missing_duplicate_matrix_and_world_coverage_fail_closed():
    t = metadata(); t["cases"] = []
    with pytest.raises(ValueError, match="matrix"): gate.evaluate(t)
    dummy_worlds = [dict(world=i) for i in range(8)]
    t["cases"] = [dict(seed=seed, sequence=s, worlds=dummy_worlds) for seed in gate.SEEDS for s in gate.SEQUENCES]
    t["cases"][0]["seed"] = 1361
    with pytest.raises(ValueError, match="case identity"): gate.evaluate(t)
    t["cases"][0]["seed"] = 1351
    dummy_worlds[-1]["world"] = 6
    with pytest.raises(ValueError, match="eight ordered worlds"): gate.evaluate(t)


def test_cli_plan_is_stdlib_only_and_never_overwrites(tmp_path):
    output = tmp_path/"plan.json"
    subprocess.run([sys.executable, "-m", "mjlab_microduck.integrated_transition_gate", "--plan", "--output", str(output)], check=True)
    old = output.read_bytes()
    assert json.loads(old) == gate.plan()
    assert subprocess.run([sys.executable, "-m", "mjlab_microduck.integrated_transition_gate", "--plan", "--output", str(output)], capture_output=True).returncode != 0
    assert output.read_bytes() == old
    script = "import mjlab_microduck.integrated_transition_gate; import sys; assert not any(n in sys.modules for n in ('torch','warp','mujoco','mjlab'))"
    subprocess.run([sys.executable, "-c", script], check=True)


def test_cli_binds_full_retained_input_hash(tmp_path):
    source, output = tmp_path/"trace.json", tmp_path/"report.json"
    raw = json.dumps(matrix(), separators=(",", ":")).encode()
    assert len(raw) < 64*1024*1024
    source.write_bytes(raw)
    assert gate.main(["--input", str(source), "--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["input_sha256"] == hashlib.sha256(raw).hexdigest()
    assert report["decision"] == "transition-matrix-pass-diagnostic-only"
