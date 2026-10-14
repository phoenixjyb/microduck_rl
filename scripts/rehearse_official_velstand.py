#!/usr/bin/env python3
"""Finite CPU MuJoCo deployment rehearsal for the pinned official VelStand ONNX."""

import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

import mujoco
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "artifacts/community/intake-20261013/official-velstand-v8/policy.onnx"
SCENES = {
    "allcollisions": ROOT / "src/mjlab_microduck/robot/microduck/scene_allcollisions.xml",
    "groundcontact": ROOT / "src/mjlab_microduck/robot/microduck/scene.xml",
}
POLICY_SHA256 = "2fc7f8ee6acd7d0d3f66a99927b5d5a56589aeed02ec714de329b252d5ccf492"
_IP_SPEC = importlib.util.spec_from_file_location("infer_policy_for_velstand_rehearsal",
                                                   ROOT / "scripts/infer_policy.py")
ip = importlib.util.module_from_spec(_IP_SPEC)
_IP_SPEC.loader.exec_module(ip)
CASES = {
    "idle": (0.0, 0.0, 0.0),
    "straight": (0.3, 0.0, 0.0),
    "left": (0.2, 0.0, 0.2),
    "right": (0.2, 0.0, -0.2),
}
DELAY_STEPS = 4
CONTROL_DT = 0.02


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(values):
    return bool(np.isfinite(np.asarray(values)).all())


def yaw_from_quat(q):
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def angle_delta(a, b):
    return math.atan2(math.sin(b - a), math.cos(b - a))


def run_case(policy, model, data, bam, case, seconds):
    vx, vy, wz = CASES[case]
    with contextlib.redirect_stdout(sys.stderr):
        policy.set_vel_cmd(vx, vy, wz)
    steps = int(round(seconds / CONTROL_DT))
    trunk = policy.trunk_base_id
    initial_pos = data.xpos[trunk].copy()
    initial_yaw = yaw_from_quat(data.xquat[trunk])
    max_tilt = 0.0
    min_height = float("inf")
    fall_steps = 0
    torque = []
    near_stop = 0
    finite_state = True
    failure_reason = None
    completed = 0
    body_velocities = []
    previous_yaw = initial_yaw
    heading_delta = 0.0
    limit = float(model.actuator_forcerange[0, 1])

    for _ in range(steps):
        obs = policy.get_observations()
        if obs.shape != (61,) or not finite(obs) or not np.allclose(obs[48:51], (vx, vy, wz), atol=1e-6):
            finite_state = False
            failure_reason = "observation shape, finiteness, or command identity check failed"
            break
        action = policy.infer()
        if action.shape != (14,) or not finite(action):
            finite_state = False
            failure_reason = "raw action shape or finiteness check failed"
            break
        policy.apply_action(action)
        for _ in range(4):
            bam.update()
            torque.extend(np.abs(data.ctrl).tolist())
            near_stop += int(np.count_nonzero(np.abs(data.ctrl) >= 0.95 * limit))
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        completed += 1
        if not finite(data.qpos) or not finite(data.qvel) or not finite(data.ctrl):
            finite_state = False
            failure_reason = "MuJoCo state or actuator force became non-finite"
            break
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, trunk, velocity, 1)
        body_velocities.append(velocity)
        current_yaw = yaw_from_quat(data.xquat[trunk])
        heading_delta += angle_delta(previous_yaw, current_yaw)
        previous_yaw = current_yaw
        gravity = policy.get_projected_gravity()
        tilt = math.acos(float(np.clip(-gravity[2], -1.0, 1.0)))
        height = float(data.xpos[trunk, 2])
        max_tilt = max(max_tilt, tilt)
        min_height = min(min_height, height)
        fall_steps += int(height < 0.07 or tilt > math.radians(60))
    end_pos = data.xpos[trunk].copy()
    elapsed = completed * CONTROL_DT
    delta_world = end_pos - initial_pos
    cy, sy = math.cos(initial_yaw), math.sin(initial_yaw)
    delta_body = np.array([cy * delta_world[0] + sy * delta_world[1],
                           -sy * delta_world[0] + cy * delta_world[1], delta_world[2]])
    return {
        "command": {"vx": vx, "vy": vy, "wz": wz},
        "requested_seconds": seconds,
        "completed_control_steps": completed,
        "elapsed_seconds": elapsed,
        "mean_initial_heading_displacement_mps": (delta_body / elapsed).tolist() if elapsed else [0.0] * 3,
        "mean_projected_heading_rate_rad_s": heading_delta / elapsed if elapsed else 0.0,
        "mean_body_linear_velocity_mps": np.mean(body_velocities, axis=0)[3:].tolist() if body_velocities else [0.0] * 3,
        "mean_body_yaw_rate_rad_s": float(np.mean(body_velocities, axis=0)[2]) if body_velocities else 0.0,
        "torso_height_min_m": min_height if completed else None,
        "torso_tilt_max_deg": math.degrees(max_tilt),
        "fall_control_steps": fall_steps,
        "finite": finite_state,
        "termination_reason": failure_reason,
        "raw_motor_force_abs_p99_Nm": float(np.percentile(torque, 99)) if torque else None,
        "near_voltage_force_limit_samples": near_stop,
        "force_limit_Nm": limit,
    }


def rehearse(cases, seconds, output, scene="allcollisions"):
    if Path(output).exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    if not cases or any(case not in CASES for case in cases):
        raise ValueError("At least one declared command case is required")
    if scene not in SCENES:
        raise ValueError("Unknown declared deployment scene")
    scene_path = SCENES[scene]
    if not math.isfinite(seconds) or seconds < CONTROL_DT:
        raise ValueError("Duration must be finite and at least one control step")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to the empty string before running")
    providers = ort.get_available_providers()
    if any("CUDA" in p.upper() or "TENSORRT" in p.upper() for p in providers):
        raise RuntimeError(f"GPU providers are available; refusing ONNX Runtime session: {providers}")
    if digest(POLICY) != POLICY_SHA256:
        raise RuntimeError("Pinned official VelStand policy SHA256 does not match")
    if seconds * len(cases) > 30.0:
        raise ValueError("Total requested simulation duration must be at most 30 seconds")

    with contextlib.redirect_stdout(sys.stderr):
        bam_model = ip.load_bam_model(ip.BAM_KP_FW, 7.4, ip.BAM_MAX_CURRENT)
        model, data, bam, _ = ip.load_mujoco_with_bam(str(scene_path), bam_model, 0.005, 0.1,
                                                     ip.BAM_VIN_MIN)
    model.opt.timestep = 0.005
    with contextlib.redirect_stdout(sys.stderr):
        policy = ip.PolicyInference(model, data, walking_onnx_path=str(POLICY), bam_ctrl=bam,
                                    action_scale=1.0, delay_min_lag=DELAY_STEPS,
                                    delay_max_lag=DELAY_STEPS, use_projected_gravity=True,
                                    new_cmd_obs=True)
    session_providers = policy.walking_session.get_providers()
    if session_providers != ["CPUExecutionProvider"]:
        raise RuntimeError(f"ONNX session is not CPU-only: {session_providers}")

    freejoint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    qa = int(model.jnt_qposadr[freejoint])
    mujoco.mj_resetData(model, data)
    data.qpos[qa:qa + 7] = [0.0, 0.0, 0.125, 1.0, 0.0, 0.0, 0.0]
    for i, qidx in enumerate(policy.joint_qpos_indices):
        data.qpos[qidx] = ip.DEFAULT_POSE[i]
    bam.reset(data.qpos)
    bam.q_target[:] = ip.DEFAULT_POSE
    data.qvel[:] = 0.0
    policy.reset_state()
    mujoco.mj_forward(model, data)

    segments = []
    for case in cases:
        segment = run_case(policy, model, data, bam, case, seconds)
        segments.append(segment)
        if not segment["finite"]:
            break
    result = {
        "classification": "simulation-deployment-rehearsal-only",
        "full_upstream_training_reproduction": False,
        "qualification": False,
        "physical_acceptance": False,
        "policy": {"path": str(POLICY.relative_to(ROOT)), "sha256": digest(POLICY), "bytes": POLICY.stat().st_size},
        "scene": {"choice": scene, "path": str(scene_path.relative_to(ROOT)), "sha256": digest(scene_path), "bytes": scene_path.stat().st_size},
        "rehearsal_source": {"path": str(Path(__file__).relative_to(ROOT)), "sha256": digest(__file__)},
        "runtime": {"providers": session_providers, "simulation_timestep_s": 0.005,
                    "control_timestep_s": CONTROL_DT, "decimation": 4,
                    "bam": "XL330 M6", "vin_V": 7.4, "kp_fw": 200.0,
                    "vin_drop_gain": 0.1, "action_delay_control_steps": DELAY_STEPS,
                    "action_delay_reference": "DR 3-6; fixed 4 control steps for this rehearsal",
                    "observation": "raw 61D; new_cmd_obs; projected gravity; command identity at obs[48:51]",
                    "actions": "raw 14D; no clipping, filtering, or remapping"},
        "measurement": "post-physics trunk body COM velocities; separately reported initial-heading displacement and unwrapped projected heading",
        "reset_protocol": "one HOME reset, then continuous declared command segments; falls counted without autoreset",
        "requested_total_seconds": seconds * len(cases),
        "segments": segments,
    }
    with open(output, "x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="new, exclusive JSON output path")
    parser.add_argument("--case", choices=tuple(CASES), action="append", dest="cases",
                        help="case to rehearse; repeat as needed (default: all four)")
    parser.add_argument("--seconds-per-case", type=float, default=4.0)
    parser.add_argument("--scene", choices=tuple(SCENES), default="allcollisions",
                        help="existing deployment scene; neither is the rough/backlash training plant")
    parser.add_argument("--combined", action="store_true",
                        help="divide a 12-second total budget over the selected cases")
    args = parser.parse_args(argv)
    cases = args.cases or list(CASES)
    if args.combined:
        seconds = 12.0 / len(cases)
    else:
        seconds = args.seconds_per_case
    if not math.isfinite(seconds) or seconds <= 0:
        parser.error("duration must be positive and finite")
    if seconds * len(cases) > 30:
        parser.error("total simulation duration must be at most 30 seconds")
    rehearse(cases, seconds, args.output, scene=args.scene)


if __name__ == "__main__":
    main()
