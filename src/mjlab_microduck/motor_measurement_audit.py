"""Opt-in, pre-reset motor measurements; never a policy acceptance protocol."""

from __future__ import annotations

import math

import torch

PROTOCOL = "motor-pre-reset-step-audit-v1"
TERM = "microduck_motor_measurement_audit"
PHASES = ("approach", "interaction", "recovery")


def validate_mode(enabled, *, first_attempt_only, collecting_dataset,
                  recording, num_envs, steps, case_count):
    if enabled and (not first_attempt_only or collecting_dataset or recording
                    or not 1 <= num_envs <= 64 or not 1 <= steps <= 1000
                    or not 1 <= case_count <= 12):
        raise ValueError("motor audit requires first-only, no dataset/recorder, <=64 envs/1000 steps/12 cases")


def motor_layout(robot):
    """Bind force columns to named unit-gear hinge joints, not shape alone."""
    import mujoco

    actuators = tuple(robot.spec.actuators)
    joints = {j.name: j for j in robot.spec.joints}
    names, ids = [], []
    for actuator in actuators:
        joint = joints.get(actuator.target)
        if (actuator.trntype != mujoco.mjtTrn.mjTRN_JOINT or joint is None
                or joint.type != mujoco.mjtJoint.mjJNT_HINGE
                or tuple(actuator.gear) != (1., 0., 0., 0., 0., 0.)):
            raise ValueError("motor audit supports only direct unit-gear hinge actuators")
        name = joint.name.split("/")[-1]
        names.append(name)
        ids.append(robot.joint_names.index(name))
    if (not names or len(names) > 64 or len(set(names)) != len(names)
            or tuple(a.name.split("/")[-1] for a in actuators) != robot.actuator_names):
        raise ValueError("ambiguous motor column mapping")
    return tuple(names), tuple(ids)


def install_metric(cfg):
    """Use mjlab's existing pre-reset metrics hook; never add a forward call."""
    from mjlab.managers.metrics_manager import MetricsTermCfg

    if TERM in cfg.metrics:
        raise ValueError("motor audit metric already registered")
    cfg.metrics[TERM] = MetricsTermCfg(func=capture_metric, per_substep=False)


def capture_metric(env):
    """Called after decimation/rewards, before reset and the final forward()."""
    audit = env._microduck_motor_measurement_audit
    audit.capture(env.scene["robot"].data, env.reset_buf)
    # A zero logging-only metric: no reward, command, RNG or simulation writes.
    return torch.zeros(env.num_envs, device=env.device)


def _stats(values):
    values = values.flatten().double()
    count = values.numel()
    bad = int((~torch.isfinite(values)).sum())
    result = dict(samples=count, nonfinite_samples=bad, abs_p99=None,
                  abs_max=None, rms=None)
    if count and not bad:
        scale = values.abs().max()
        rms = scale * torch.sqrt(torch.square(values / scale).mean()) if scale > 0 else scale
        result.update(abs_p99=float(torch.quantile(values.abs(), .99)),
                      abs_max=float(values.abs().max()),
                      rms=float(rms))
    return result


class MotorMeasurementAudit:
    """Bounded first-attempt samples, cloned before reset and reduced afterward.

    Forces are the derived values from the last physics substep (one integration
    lag); velocity is the integrated state at the same capture boundary. This is
    not substep-peak or power/temperature measurement. Legacy metrics stay intact.
    """

    def __init__(self, num_envs, max_steps, names, joint_ids, *, stall_reference_nm):
        if (not 1 <= num_envs <= 64 or not 1 <= max_steps <= 1000
                or not 1 <= len(names) <= 64 or len(names) != len(joint_ids)
                or len(set(names)) != len(names) or len(set(joint_ids)) != len(joint_ids)
                or any(type(i) is not int or i < 0 for i in joint_ids)
                or not math.isfinite(stall_reference_nm) or stall_reference_nm <= 0):
            raise ValueError("invalid bounded motor audit layout/reference")
        self.num_envs, self.max_steps = num_envs, max_steps
        self.names, self.joint_ids = tuple(names), tuple(joint_ids)
        self.stall_reference_nm = stall_reference_nm
        self.rows = []
        self.pending = None
        self.snapshot = None
        self.finished = None

    def begin(self, step, active, phase):
        if self.pending is not None or type(step) is not int or step != len(self.rows) or step >= self.max_steps:
            raise ValueError("motor audit steps must be bounded and sequential")
        if (active.shape != (self.num_envs,) or active.dtype != torch.bool
                or phase.shape != active.shape or phase.dtype not in (torch.int32, torch.int64)
                or not bool(active.any()) or not bool(((phase[active] >= 0) & (phase[active] <= 2)).all())):
            raise ValueError("invalid active mask/phase")
        expected = torch.ones_like(active) if self.finished is None else ~self.finished
        if not torch.equal(active, expected):
            raise ValueError("active mask must cover exactly the unfinished first attempts")
        self.pending = dict(step=step, active=active.detach().clone(), phase=phase.detach().clone())

    def capture(self, data, terminal):
        if self.pending is None or self.snapshot is not None:
            raise ValueError("exactly one pre-reset capture per armed step required")
        force = data.actuator_force
        speed = data.joint_vel[:, self.joint_ids]
        if force.shape != (self.num_envs, len(self.names)) or speed.shape != force.shape:
            raise ValueError("motor force/velocity layout mismatch")
        if terminal.shape != (self.num_envs,) or terminal.dtype != torch.bool:
            raise ValueError("invalid terminal mask")
        self.snapshot = dict(force=force.detach().clone(), speed=speed.detach().clone(),
                             terminal=terminal.detach().clone())

    def finish(self, terminal, post_return_force):
        if self.pending is None or self.snapshot is None:
            raise ValueError("pre-reset capture missing")
        snap, pending = self.snapshot, self.pending
        if (terminal.dtype != torch.bool or not torch.equal(terminal, snap["terminal"])
                or post_return_force.shape != snap["force"].shape):
            raise ValueError("terminal identity or post-return layout mismatch")
        active = pending["active"]
        self.rows.append(dict(step=pending["step"],
                              environment_ids=active.nonzero().flatten(),
                              phase=pending["phase"][active], terminal=terminal[active].clone(),
                              force=snap["force"][active], speed=snap["speed"][active],
                              post_force=post_return_force[active].detach().clone()))
        self.finished = (terminal & active) if self.finished is None else self.finished | (terminal & active)
        self.pending = self.snapshot = None

    def report(self):
        if self.pending is not None or not self.rows:
            raise ValueError("motor audit has missing or incomplete captures")
        force = torch.cat([r["force"] for r in self.rows])
        speed = torch.cat([r["speed"] for r in self.rows])
        phase = torch.cat([r["phase"] for r in self.rows])
        terminal = torch.cat([r["terminal"] for r in self.rows])
        after = torch.cat([r["post_force"] for r in self.rows])
        utilization = force.double() / self.stall_reference_nm
        groups = {"all": torch.ones_like(terminal), **{
            name: phase == i for i, name in enumerate(PHASES)}}
        summaries = {name: {
            "environment_steps": int(mask.sum()),
            "force_nm": _stats(force[mask]), "speed_rad_s": _stats(speed[mask]),
            "stall_reference_utilization": _stats(utilization[mask]),
            "by_joint": {joint: _stats(utilization[mask, i]) for i, joint in enumerate(self.names)},
        } for name, mask in groups.items()}
        return dict(protocol=PROTOCOL, decision="diagnostic-only-not-admission",
                    physical_motion_authorized=False, policy_acceptance=False,
                    runtime_equivalence_validated=False, training_data_admitted=False,
                    sampling="post-decimation-metrics-hook-before-reset-and-final-forward",
                    force_timing="last-physics-substep-derived-force; one-integration lag",
                    speed_timing="integrated-joint-velocity-at-capture",
                    peak_scope="control-step samples only; not all physics substeps",
                    legacy_metrics_replaced=False, stall_reference_nm=self.stall_reference_nm,
                    joint_columns=list(self.names), steps_captured=len(self.rows),
                    terminal_environment_steps=int(terminal.sum()),
                    incomplete_first_attempts=self.num_envs - int(self.finished.sum()),
                    terminal_force_nm=_stats(force[terminal]),
                    terminal_post_return_force_nm=_stats(after[terminal]),
                    terminal_post_return_minus_pre_reset_nm=_stats((after.double() - force.double())[terminal]),
                    finite=bool(torch.isfinite(force).all() & torch.isfinite(speed).all()
                                & torch.isfinite(after).all() & torch.isfinite(utilization).all()),
                    summary_precision="float64; scaled RMS", groups=summaries)
