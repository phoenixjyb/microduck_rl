"""Constant-retention pre-reset motor samples for a future supervisor trainer.

The installed metrics hook runs after mjlab rewards but before autoreset. A
supervisor can consume this sample after step() to construct its own reward.
This module does NOT add an environment reward or wire itself into any trainer.
"""

from dataclasses import asdict, dataclass
import math

import torch

from mjlab_microduck.motor_measurement_audit import motor_layout


PROTOCOL = "motor-pre-reset-step-stream-v1"
TERM = "microduck_motor_step_stream"


@dataclass(frozen=True)
class MotorStepCostCfg:
    stall_reference_nm: float = .60
    soft_limit_fraction: float = .70
    over_limit_gain: float = 4.

    def __post_init__(self):
        if (any(type(v) not in (int, float) or not math.isfinite(v) for v in asdict(self).values())
                or self.stall_reference_nm <= 0 or not 0 < self.soft_limit_fraction <= 1
                or self.over_limit_gain < 0):
            raise ValueError("invalid motor cost model references")


@dataclass(frozen=True)
class MotorStepSample:
    step: int
    joint_names: tuple[str, ...]
    episode_generation: torch.Tensor
    phase: torch.Tensor
    terminal: torch.Tensor
    force_nm: torch.Tensor
    speed_rad_s: torch.Tensor
    joint_cost: torch.Tensor
    mean_cost: torch.Tensor


class MotorStepStream:
    """Arm/capture/consume exactly once per control step, across autoresets.

    Keeps no trajectory history or growing lists. Episode generations advance
    ONLY after the terminal sample has been consumed; all environments reenter
    on the next step. Manual resets require a new stream at a clean boundary.
    """

    def __init__(self, num_envs, names, joint_ids, *, device, cost_cfg):
        if (type(num_envs) is not int or not 1 <= num_envs <= 256
                or not 1 <= len(names) <= 64 or len(names) != len(joint_ids)
                or any(not isinstance(n, str) or not n for n in names)
                or len(set(names)) != len(names) or len(set(joint_ids)) != len(joint_ids)
                or any(type(i) is not int or i < 0 for i in joint_ids)
                or not isinstance(cost_cfg, MotorStepCostCfg)):
            raise ValueError("invalid bounded motor stream layout/configuration")
        self.num_envs, self.names, self.joint_ids = num_envs, tuple(names), tuple(joint_ids)
        self.cost_cfg = cost_cfg
        self._generation = torch.zeros(num_envs, dtype=torch.int64, device=device)
        self.next_step = 0
        self._phase = self._snapshot = None

    @classmethod
    def from_robot(cls, robot, num_envs, *, device, cost_cfg):
        names, ids = motor_layout(robot)
        return cls(num_envs, names, ids, device=device, cost_cfg=cost_cfg)

    def begin(self, step, phase):
        if (type(step) is not int or step != self.next_step or self._phase is not None
                or self._snapshot is not None):
            raise ValueError("consume the preceding sample before arming the next sequential step")
        if (phase.shape != (self.num_envs,) or phase.dtype not in (torch.int32, torch.int64)
                or phase.device != self._generation.device
                or not bool(((phase >= 0) & (phase <= 2)).all())):
            raise ValueError("invalid pre-action phase identity")
        self._phase = phase.detach().clone()

    def capture(self, data, terminal):
        if self._phase is None or self._snapshot is not None:
            raise ValueError("exactly one armed pre-reset capture required")
        force, velocity = data.actuator_force, data.joint_vel
        if (force.shape != (self.num_envs, len(self.names)) or velocity.ndim != 2
                or velocity.shape[0] != self.num_envs or max(self.joint_ids) >= velocity.shape[1]
                or force.dtype not in (torch.float32, torch.float64) or velocity.dtype != force.dtype
                or force.device != self._generation.device or velocity.device != force.device
                or terminal.shape != (self.num_envs,) or terminal.dtype != torch.bool
                or terminal.device != force.device):
            raise ValueError("motor stream force/velocity/terminal layout mismatch")
        self._snapshot = (force.detach().clone(), velocity[:, self.joint_ids].detach().clone(),
                          terminal.detach().clone())

    def consume(self, terminal):
        if self._snapshot is None:
            raise ValueError("pre-reset motor capture missing")
        force, speed, captured_terminal = self._snapshot
        if (terminal.dtype != torch.bool or terminal.device != captured_terminal.device
                or not torch.equal(terminal, captured_terminal)):
            raise ValueError("returned terminal identity differs from pre-reset capture")
        if not bool(torch.isfinite(force).all() & torch.isfinite(speed).all()):
            raise FloatingPointError("nonfinite motor sample; do not admit a reward")
        u = force.double().abs() / self.cost_cfg.stall_reference_nm
        joint_cost = u.square() + self.cost_cfg.over_limit_gain * (
            u - self.cost_cfg.soft_limit_fraction).clamp_min(0).square()
        mean_cost = joint_cost.mean(dim=1)
        if not bool(torch.isfinite(joint_cost).all() & torch.isfinite(mean_cost).all()):
            raise FloatingPointError("nonfinite derived motor cost; do not admit a reward")
        result = MotorStepSample(self.next_step, self.names, self._generation.clone(), self._phase,
                                 captured_terminal, force, speed, joint_cost, mean_cost)
        self._generation = self._generation + captured_terminal.to(torch.int64)
        self._phase = self._snapshot = None
        self.next_step += 1
        return result

    def provenance(self):
        return dict(protocol=PROTOCOL, joint_columns=list(self.names), cost_config=asdict(self.cost_cfg),
                    cost_formula="per joint: u^2 + gain * max(u - soft_limit, 0)^2; u=abs(force)/reference",
                    sampling="post-decimation-metrics-hook-before-reset-and-final-forward",
                    force_timing="last-physics-substep-derived-force; one-integration lag",
                    speed_timing="integrated-joint-velocity-at-capture",
                    retention="one control step; no history or full-trajectory quantiles",
                    reward_weight_applied=False, policy_acceptance=False, physical_motion_authorized=False,
                    runtime_equivalence_validated=False, trainer_integration_validated=False)


def capture_metric(env):
    env._microduck_motor_step_stream.capture(env.scene["robot"].data, env.reset_buf)
    return torch.zeros(env.num_envs, device=env.device)


def install_metric(cfg):
    from mjlab.managers.metrics_manager import MetricsTermCfg

    if TERM in cfg.metrics:
        raise ValueError("motor step stream metric already registered")
    if cfg.auto_reset is not True:
        raise ValueError("motor step stream requires automatic reset episode semantics")
    cfg.metrics[TERM] = MetricsTermCfg(func=capture_metric, per_substep=False)
