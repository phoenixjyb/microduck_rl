"""Stdlib-only command plan and offline first-attempt transition trace scorer.

No policy, simulator, GPU, collector or physical motion is executed here.
Passing supplied traces is diagnostic, never a replacement foundation gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

PROTOCOL = "integrated-command-transitions-v1"
SEEDS = (1351, 1361, 1367)
WORLDS = 8
DT = .02
STARTUP_STEPS = 60
DWELL_SAMPLES = 26  # 25 intervals = .50s, not25 samples.
RECOVERY_DEADLINE_STEPS = 100
# Exact existing tasks/run.py reference; do not import its GPU/task stack here.
RATED_SPEED = 123.*2*math.pi/60
JOINT_NAMES = tuple(f"left_{joint}" for joint in
                    ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")) + (
                    "neck_pitch", "head_pitch", "head_yaw", "head_roll") + tuple(
                    f"right_{joint}" for joint in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle"))
RUNTIME = {"torch": "2.9.1", "warp-lang": "1.12.0", "mujoco": "3.10.0",
           "mujoco-warp": "3.8.1", "mjlab": "1.3.0", "better-actuator-models": "1.0.1"}
SEQUENCES = {
    "stop-restart": (("idle", 100, 0., 0.), ("walk", 200, .3, 0.),
                     ("stop", 200, 0., 0.), ("restart", 200, .3, 0.)),
    "left-right-return": (("straight", 150, .2, 0.), ("negative", 150, .2, -.2),
                          ("return", 150, .2, 0.), ("positive", 150, .2, .2),
                          ("final-straight", 150, .2, 0.)),
    "right-left-return": (("straight", 150, .2, 0.), ("positive", 150, .2, .2),
                          ("return", 150, .2, 0.), ("negative", 150, .2, -.2),
                          ("final-straight", 150, .2, 0.)),
}
FRAME_FIELDS = {"step", "time_s", "command", "actor_command", "velocity_b",
                "yaw_rate", "world_xy", "tilt_rad", "qpos", "force_nm",
                "motor_speed_rad_s", "action", "grounded", "terminal", "fall", "reset", "episode_id"}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def vector(value, width):
    return type(value) is list and len(value) == width and all(number(x) for x in value)


def phases(sequence):
    need(type(sequence) is str and sequence in SEQUENCES, "known transition sequence")
    start = 0
    result = []
    for name, steps, vx, yaw in SEQUENCES[sequence]:
        result.append(dict(name=name, start_step=start, end_step=start+steps,
                           command=[vx, 0., yaw]))
        start += steps
    return result


def plan():
    """Machine-readable fixed matrix, not a launch or collector admission."""
    return dict(protocol=PROTOCOL, seeds=list(SEEDS), worlds=WORLDS, step_dt=DT,
        sequences={name: phases(name) for name in SEQUENCES},
        phase_tracking_startup_steps=STARTUP_STEPS, dwell_samples=DWELL_SAMPLES,
        recovery_deadline_steps=RECOVERY_DEADLINE_STEPS,
        runtime=RUNTIME, actor_width=61, action_width=14,
        frame_phase="post-physics-pre-reset-last-substep-v1",
        foundation_gate_required=True, collector_qualified=False,
        policy_acceptance=False, simulator_qualified=False, physical_motion_authorized=False)


def quantile(values, probability=.99):
    ordered = sorted(values)
    need(bool(ordered), "nonempty quantile samples")
    index = (len(ordered)-1)*probability
    lo, hi = math.floor(index), math.ceil(index)
    return ordered[lo]+(ordered[hi]-ordered[lo])*(index-lo)


def recovery(in_band):
    """Integer-index dwell/deadline accounting, independent of float timestamps."""
    run, first = 0, None
    for index, good in enumerate(in_band):
        run = run+1 if good else 0
        if run >= DWELL_SAMPLES and first is None:
            first = index+1
    return dict(first_settled_end_s=None if first is None else first*DT,
        settled_by_deadline=first is not None and first <= RECOVERY_DEADLINE_STEPS,
        final_settled_span_s=max(0, run-1)*DT,
        settled_at_exit=run >= DWELL_SAMPLES)


def validate_metadata(trace):
    need(type(trace) is dict and trace.get("protocol") == PROTOCOL, "exact transition protocol")
    for key, width in (("source", 40), ("checkpoint_sha256", 64), ("plant_sha256", 64)):
        need(type(trace.get(key)) is str and re.fullmatch(rf"[0-9a-f]{{{width}}}", trace[key]),
             f"exact {key} identity")
    need(trace.get("runtime") == RUNTIME, "frozen runtime pins")
    need(type(trace.get("step_dt")) is float and trace["step_dt"] == DT
         and type(trace.get("actor_width")) is int and trace["actor_width"] == 61
         and type(trace.get("action_width")) is int and trace["action_width"] == 14,
         "unchanged 20ms/61-to14 layout")
    need(trace.get("frame_phase") == "post-physics-pre-reset-last-substep-v1"
         and trace.get("first_episode_only") is True and trace.get("automatic_retry") is False,
         "first-attempt pre-reset phase and no retry")
    names, bounds = trace.get("joint_names"), trace.get("joint_limits")
    need(type(names) is list and len(names) == 14 and all(type(n) is str for n in names)
         and len(set(names)) == 14 and set(names) == set(JOINT_NAMES), "exact named servo coverage")
    need(type(bounds) is list and len(bounds) == 14
         and all(vector(pair, 2) and pair[0] < pair[1] for pair in bounds), "finite joint ranges")
    return names, bounds


def validate_world(world, sequence):
    need(type(world) is dict and type(world.get("world")) is int
         and 0 <= world["world"] < WORLDS and vector(world.get("initial_xy"), 2), "world identity/initial pose")
    frames = world.get("frames")
    schedule = phases(sequence)
    need(type(frames) is list and 0 < len(frames) <= schedule[-1]["end_step"], "bounded first-attempt frames")
    for step, frame in enumerate(frames):
        need(type(frame) is dict and set(frame) == FRAME_FIELDS, "closed frame schema")
        need(type(frame["step"]) is int and frame["step"] == step
             and number(frame["time_s"]) and abs(frame["time_s"]-(step+1)*DT) <= 1e-9,
             "contiguous step and exact response timestamp")
        phase = next(p for p in schedule if p["start_step"] <= step < p["end_step"])
        need(vector(frame["command"], 3) and vector(frame["actor_command"], 3)
             and frame["actor_command"] == frame["command"]
             and all(abs(a-b) <= 1e-7 for a, b in zip(frame["command"], phase["command"])),
             "fresh actor command equals declared command (float32 target tolerance)")
        for key, width in (("velocity_b", 3), ("world_xy", 2), ("qpos", 14),
                           ("force_nm", 14), ("motor_speed_rad_s", 14), ("action", 14)):
            need(vector(frame[key], width), f"finite {key} coverage")
        need(number(frame["yaw_rate"]) and number(frame["tilt_rad"])
             and 0 <= frame["tilt_rad"] <= math.pi, "finite yaw/tilt")
        need(all(type(frame[key]) is bool for key in ("grounded", "terminal", "fall", "reset"))
             and type(frame["episode_id"]) is int, "typed episode flags")
        need(not frame["reset"] and frame["episode_id"] == 0, "no post-reset or second-episode frames")
        need(not frame["fall"] or frame["terminal"], "fall must retain terminal signal")
        need(not frame["terminal"] or step == len(frames)-1, "no post-terminal continuation")
    return frames


def score_world(world, sequence, names, bounds):
    frames = validate_world(world, sequence)
    failures, phase_rows = [], []
    expected_steps = phases(sequence)[-1]["end_step"]
    terminal = frames[-1]["terminal"]
    complete = len(frames) == expected_steps and not terminal
    if terminal:
        failures.append(dict(metric="first-episode-terminal", step=frames[-1]["step"], fall=frames[-1]["fall"]))
    if not complete and not terminal:
        failures.append(dict(metric="incomplete-trace", observed=len(frames), expected=expected_steps))
    for phase in phases(sequence):
        subset = frames[phase["start_step"]:phase["end_step"]]
        phase_complete = len(subset) == phase["end_step"]-phase["start_step"] and not any(f["terminal"] for f in subset)
        if not phase_complete:
            phase_rows.append(dict(name=phase["name"], complete=False, sample_count=len(subset)))
            continue
        vx, _, yaw = phase["command"]
        speed_limit = .03 if vx == 0 else .10
        scored = subset[STARTUP_STEPS:]
        speed_mae = math.fsum(abs(f["velocity_b"][0]-vx)/len(scored) for f in scored)
        yaw_mae = math.fsum(abs(f["yaw_rate"]-yaw)/len(scored) for f in scored)
        need(number(speed_mae) and number(yaw_mae), "finite derived tracking errors")
        in_band = [(f["grounded"] and math.hypot(*f["velocity_b"]) <= .03 if vx == 0 else
                    abs(f["velocity_b"][0]-vx) <= .10) and abs(f["yaw_rate"]-yaw) <= .10 for f in subset]
        settled = recovery(in_band)
        previous = (frames[phase["start_step"]-1]["world_xy"] if phase["start_step"] else world["initial_xy"])
        distance = 0.
        for f in subset:
            distance += math.dist(previous, f["world_xy"])
            previous = f["world_xy"]
        need(number(distance), "finite derived planar path length")
        row = dict(name=phase["name"], complete=True, sample_count=len(subset),
                   speed_mae=speed_mae, yaw_mae=yaw_mae, planar_path_length_m=distance,
                   stop_distance_m=distance if phase["name"] == "stop" else None,
                   max_tilt_rad=max(f["tilt_rad"] for f in subset), **settled)
        phase_rows.append(row)
        for metric, failed in (("speed_mae", speed_mae > speed_limit), ("yaw_mae", yaw_mae > .10),
                               ("settled_by_deadline", not settled["settled_by_deadline"]),
                               ("settled_at_exit", not settled["settled_at_exit"])):
            if failed:
                failures.append(dict(phase=phase["name"], metric=metric, value=row[metric]))
    force = [abs(value)/.60 for f in frames for value in f["force_nm"]]
    need(all(number(v) and number(v*v) for v in force), "finite derived motor load")
    torque_p99 = quantile(force)
    exceed = sum(abs(value) > RATED_SPEED for f in frames for value in f["motor_speed_rad_s"])
    minimum, exposure = {}, {}
    for column, name in enumerate(names):
        distances = [min(f["qpos"][column]-bounds[column][0], bounds[column][1]-f["qpos"][column]) for f in frames]
        need(all(number(v) for v in distances), "finite derived joint range distances")
        minimum[name] = min(distances)
        exposure[name] = sum(distance <= .05 for distance in distances)/len(frames)
        if minimum[name] < -.005:
            failures.append(dict(metric="joint-range", joint=name, value=minimum[name]))
    if torque_p99 > .60:
        failures.append(dict(metric="torque_p99", value=torque_p99))
    if exceed:
        failures.append(dict(metric="rated-speed-exceedance", value=exceed))
    if exposure["right_hip_yaw"] > .01:
        failures.append(dict(metric="right-hip-stop-exposure", value=exposure["right_hip_yaw"]))
    rates = [abs(b-a)/DT for previous, current in zip(frames, frames[1:])
             for a, b in zip(previous["action"], current["action"])]
    power = [math.fsum(abs(a*b) for a, b in zip(f["force_nm"], f["motor_speed_rad_s"])) for f in frames]
    need(all(number(v) for v in rates+power), "finite derived action rates and mechanical power")
    return dict(world=world["world"], complete=complete, sample_count=len(frames),
        phases=phase_rows, failures=failures, torque_p99=torque_p99,
        rated_speed_exceed_fraction=exceed/(len(frames)*14),
        thermal_load_proxy=math.fsum(v*v/len(force) for v in force),
        mean_abs_mechanical_power_w=math.fsum(v/len(frames) for v in power),
        min_distance_to_hard_stop_rad=minimum, hard_stop_proximity_fraction=exposure,
        action_magnitude_p99=quantile([abs(v) for f in frames for v in f["action"]]),
        action_rate_p99=None if not rates else quantile(rates))


def evaluate(trace):
    names, bounds = validate_metadata(trace)
    cases = trace.get("cases")
    expected = [(seed, sequence) for seed in SEEDS for sequence in SEQUENCES]
    need(type(cases) is list and len(cases) == len(expected), "complete ordered seed/sequence matrix")
    results, failures = [], []
    for case, (seed, sequence) in zip(cases, expected):
        need(type(case) is dict and type(case.get("seed")) is int
             and case["seed"] == seed and case.get("sequence") == sequence, "exact ordered case identity")
        worlds = case.get("worlds")
        need(type(worlds) is list and len(worlds) == WORLDS
             and all(type(w) is dict and type(w.get("world")) is int and w["world"] == i for i, w in enumerate(worlds)),
             "all eight ordered worlds retained")
        scores = [score_world(world, sequence, names, bounds) for world in worlds]
        results.append(dict(seed=seed, sequence=sequence, worlds=scores))
        failures.extend(dict(seed=seed, sequence=sequence, world=row["world"], **failure)
                        for row in scores for failure in row["failures"])
    incomplete = any(f["metric"] == "incomplete-trace" for f in failures)
    return dict(protocol=PROTOCOL, source=trace["source"], checkpoint_sha256=trace["checkpoint_sha256"],
        plant_sha256=trace["plant_sha256"], runtime=trace["runtime"], cases=results, failures=failures,
        decision=("transition-evidence-incomplete" if incomplete else "transition-matrix-not-ready" if failures
                  else "transition-matrix-pass-diagnostic-only"),
        numerical_pass=not failures, foundation_gate_required=True, collector_qualified=False,
        policy_acceptance=False, simulator_qualified=False, physical_motion_authorized=False)


def write_new(path, value):
    text = json.dumps(value, indent=2, allow_nan=False)+"\n"
    with Path(path).open("x") as stream:
        stream.write(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path)
    source.add_argument("--plan", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.plan:
        result = plan()
    else:
        need(args.input.is_file() and args.input.stat().st_size <= 64*1024*1024, "bounded trace input <=64MiB")
        raw = args.input.read_bytes()
        need(len(raw) <= 64*1024*1024, "bounded retained trace bytes")
        result = evaluate(json.loads(raw))
        result["input_sha256"] = hashlib.sha256(raw).hexdigest()
    write_new(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
