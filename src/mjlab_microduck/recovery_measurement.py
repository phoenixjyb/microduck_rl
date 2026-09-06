"""Bounded first-attempt speed recovery observer; never controls the robot."""

from __future__ import annotations

import math
import torch


class RecoveryMeasurement:
    """Observe pre-step route speed, excluding all post-terminal reset state.

    Success requires a contiguous sampled span of 0.5 s within +/-0.03 m/s
    of nominal, completed within 2 s of the first recovery observation.
    A missing/short recovery is censored, never silently counted as success.
    """

    def __init__(self, num_envs: int, nominal_speed_mps: float, step_dt_s: float):
        if type(num_envs) is not int or not 1 <= num_envs <= 64:
            raise ValueError("first-attempt observer supports 1..64 environments")
        for value in (nominal_speed_mps, step_dt_s):
            if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
                raise ValueError("finite positive speed and timestep required")
        if step_dt_s > .1 or step_dt_s < .001:
            raise ValueError("unsupported observation timestep")
        self.n = num_envs
        self.nominal = float(nominal_speed_mps)
        self.dt = float(step_dt_s)
        self.entry = [None] * num_envs
        self.streak_start = [None] * num_envs
        self.success = [None] * num_envs
        self.last_recovery = [None] * num_envs
        self.terminal = [False] * num_envs
        self.next_step = 0
        self.pending = False

    def _vector(self, value, dtype):
        if not isinstance(value, torch.Tensor) or value.shape != (self.n,) or value.dtype not in dtype:
            raise ValueError("invalid observer vector")
        return value.detach().cpu().tolist()

    def begin(self, step: int, phase: torch.Tensor, route_speed: torch.Tensor):
        if self.pending or type(step) is not int or step != self.next_step or step >= 1000:
            raise ValueError("observer step lifecycle/bound")
        phases = self._vector(phase, (torch.int32, torch.int64))
        speeds = self._vector(route_speed, (torch.float32, torch.float64))
        if any(p not in (0, 1, 2) for p in phases):
            raise ValueError("invalid phase")
        if any(not math.isfinite(v) for i, v in enumerate(speeds) if not self.terminal[i]):
            raise ValueError("nonfinite active route speed")
        for i, (phase_value, speed) in enumerate(zip(phases, speeds, strict=True)):
            if self.terminal[i]:
                continue
            if phase_value != 2:
                self.streak_start[i] = None
                continue
            if self.entry[i] is None:
                self.entry[i] = step
            self.last_recovery[i] = step
            if abs(speed - self.nominal) <= .03:
                if self.streak_start[i] is None:
                    self.streak_start[i] = step
                if ((step - self.streak_start[i]) * self.dt >= .5 - 1e-12
                        and self.success[i] is None):
                    self.success[i] = step
            else:
                self.streak_start[i] = None
        self.pending = True

    def finish(self, dones: torch.Tensor):
        if not self.pending:
            raise ValueError("finish without pre-step observation")
        done = self._vector(dones, (torch.bool,))
        self.terminal = [old or new for old, new in zip(self.terminal, done, strict=True)]
        self.next_step += 1
        self.pending = False

    def report(self):
        if self.pending or self.next_step == 0:
            raise ValueError("incomplete observer lifecycle")
        rows = []
        for i in range(self.n):
            entry, success, last = self.entry[i], self.success[i], self.last_recovery[i]
            latency = None if success is None else (success - entry) * self.dt
            span = None if entry is None else (last - entry) * self.dt
            status = ("not-observed" if entry is None else
                      "recovered-in-window" if latency is not None and latency <= 2. + 1e-12 else
                      "window-missed" if span >= 2. - 1e-12 else "censored-before-window")
            rows.append(dict(environment=i, first_recovery_step=entry,
                             sampled_recovery_span_s=span, stable_recovery_latency_s=latency,
                             terminal=self.terminal[i], status=status))
        return dict(protocol="first-attempt-recovery-speed-v1", sampling="pre-control-step route speed",
                    nominal_speed_mps=self.nominal, step_dt_s=self.dt,
                    speed_tolerance_mps=.03, stable_span_s=.5, deadline_s=2.,
                    policy_acceptance=False, physical_motion_authorized=False,
                    counts={status: sum(r["status"] == status for r in rows) for status in
                            ("not-observed", "recovered-in-window", "window-missed", "censored-before-window")},
                    environments=rows)
