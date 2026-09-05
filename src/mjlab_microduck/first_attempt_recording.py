"""Passive, bounded first-attempt telemetry; never controller input or training data."""

from __future__ import annotations

import copy
import math

import torch

RECORDING_PROTOCOL = "all-first-attempt-pre-step-v1"
OUTCOME_PROTOCOL = "hard-failure-collision-timeout-pass-v1"
MAX_RECORDED_ENVS = 64
MAX_RECORDED_STEPS = 1000
SAMPLE_INTERVAL_STEPS = 5
FRAME_FIELDS = (
    "route_progress_m", "route_lateral_error_m", "route_heading_error_rad",
    "route_speed_mps", "obstacle_ahead_m", "obstacle_clearance_m", "phase",
    "command_speed_mps", "command_yaw_rate_rps",
)
RAW_FLAGS = ("collision", "pass", "timeout", "fall", "nan")
RESOLVED_FLAGS = ("hard_failure", "collision", "timeout", "pass", "other_terminal")


class FirstAttemptRecorder:
    """Own detached CPU copies; sample every five steps plus terminal pre-steps.

    The terminal callback receives only flags and time. It cannot accidentally
    label an auto-reset pose as the state that caused the terminal transition.
    Nonfinite frame values are retained as null plus explicit field names, not
    silently presented as finite measurements. They do not alter metric gates.
    """

    def __init__(self, num_envs: int, max_steps: int) -> None:
        if type(num_envs) is not int or not 1 <= num_envs <= MAX_RECORDED_ENVS:
            raise ValueError("recording requires 1..64 environments")
        if type(max_steps) is not int or not 1 <= max_steps <= MAX_RECORDED_STEPS:
            raise ValueError("recording requires 1..1000 steps")
        self.num_envs = num_envs
        self.max_steps = max_steps
        self._closed = torch.zeros(num_envs, dtype=torch.bool)
        self._frames: list[list[dict]] = [[] for _ in range(num_envs)]
        self._terminals: list[dict | None] = [None] * num_envs
        self._pending: tuple[int, float, torch.Tensor] | None = None
        self._next_step = 0
        self._last_terminal_time = -math.inf

    def _mask(self, value: torch.Tensor) -> torch.Tensor:
        if not isinstance(value, torch.Tensor) or value.dtype != torch.bool or value.shape != (self.num_envs,):
            raise ValueError("recording masks must be matching boolean vectors")
        return value.detach().to(device="cpu", copy=True)

    def capture_pre_step(
        self, step: int, time_s: float, active: torch.Tensor,
        fields: dict[str, torch.Tensor],
    ) -> None:
        if self._pending is not None or type(step) is not int or step != self._next_step or step >= self.max_steps:
            raise ValueError("recording steps must be paired, sequential, and bounded")
        if not math.isfinite(time_s) or time_s < 0 or time_s < self._last_terminal_time:
            raise ValueError("pre-step time must be finite and monotonic")
        if not torch.equal(self._mask(active), ~self._closed):
            raise ValueError("active mask must contain exactly the unfinished first attempts")
        if set(fields) != set(FRAME_FIELDS) or any(
            not isinstance(v, torch.Tensor) or v.shape != (self.num_envs,)
            for v in fields.values()
        ):
            raise ValueError("recording fields must be matching compact-state vectors")
        values = torch.stack([fields[k].detach() for k in FRAME_FIELDS], dim=1)
        values = values.to(device="cpu", copy=True)
        self._pending = (step, float(time_s), values)
        if step % SAMPLE_INTERVAL_STEPS == 0:
            for env_id in torch.where(~self._closed)[0].tolist():
                self._append_frame(env_id)

    def _append_frame(self, env_id: int) -> None:
        assert self._pending is not None
        step, time_s, values = self._pending
        frames = self._frames[env_id]
        if frames and frames[-1]["step"] == step:
            return
        row = dict(zip(FRAME_FIELDS, values[env_id].tolist(), strict=True))
        nonfinite = [k for k, v in row.items() if not math.isfinite(v)]
        for key in nonfinite:
            row[key] = None
        frames.append({
            "step": step, "time_s": time_s, "state_timing": "pre-step",
            **row, "nonfinite_fields": nonfinite,
        })

    def finish_step(
        self, terminal_time_s: float, dones: torch.Tensor,
        raw_flags: dict[str, torch.Tensor],
        resolved_flags: dict[str, torch.Tensor],
    ) -> None:
        if self._pending is None:
            raise ValueError("terminal recording requires a pending pre-step")
        step, pre_time, _ = self._pending
        if not math.isfinite(terminal_time_s) or terminal_time_s <= pre_time:
            raise ValueError("terminal time must follow the finite pre-step time")
        if set(raw_flags) != set(RAW_FLAGS) or set(resolved_flags) != set(RESOLVED_FLAGS):
            raise ValueError("terminal recording requires the exact flag schema")
        done = self._mask(dones)
        raw = {k: self._mask(v) for k, v in raw_flags.items()}
        resolved = {k: self._mask(v) for k, v in resolved_flags.items()}
        active = ~self._closed
        finished = active & done
        counts = torch.stack(list(resolved.values()), dim=1).sum(dim=1)
        if bool((active & (counts != finished.long())).any()):
            raise ValueError("resolved outcomes must partition first terminal attempts")
        if bool((active & ~done & torch.stack(list(raw.values()), dim=1).any(dim=1)).any()):
            raise ValueError("raw terminal flag without a terminal transition")
        # Validate the caller's resolution without feeding anything back to it.
        terminal_rows = []
        for env_id in torch.where(finished)[0].tolist():
            flags = {k: bool(v[env_id]) for k, v in raw.items()}
            expected = (
                "hard_failure" if flags["fall"] or flags["nan"] else
                "collision" if flags["collision"] else
                "timeout" if flags["timeout"] else
                "pass" if flags["pass"] else "other_terminal"
            )
            if not bool(resolved[expected][env_id]):
                raise ValueError("recorded outcome disagrees with failure-priority accounting")
            terminal_rows.append((env_id, {
                "after_step": step, "time_s": float(terminal_time_s),
                "outcome": expected, "raw_flags": flags,
                "overlap": sum(flags.values()) > 1,
                "state_timing": "flags-only-after-step; no-auto-reset-state",
            }))
        for env_id, terminal in terminal_rows:
            self._append_frame(env_id)
            self._terminals[env_id] = terminal
            self._closed[env_id] = True
        self._pending = None
        self._next_step += 1
        self._last_terminal_time = float(terminal_time_s)

    def report(self) -> dict:
        if self._pending is not None:
            raise ValueError("cannot export an unfinished step")
        return copy.deepcopy({
            "schema_version": 1,
            "protocol": RECORDING_PROTOCOL,
            "terminal_outcome_protocol": OUTCOME_PROTOCOL,
            "purpose": "diagnostic-only; not-training-data; not-policy-acceptance",
            "physical_motion_authorized": False,
            "sample_interval_steps": SAMPLE_INTERVAL_STEPS,
            "terminal_pre_step_always_retained": True,
            "phase_codes": {"0": "approach", "1": "interaction", "2": "recovery"},
            "clearance_kind": "center-distance-minus-0.22m-proxy; not-contact-distance",
            "num_envs": self.num_envs,
            "max_steps": self.max_steps,
            "steps_recorded": self._next_step,
            "max_frames_per_environment": (self.max_steps - 1) // SAMPLE_INTERVAL_STEPS + 2,
            "completed_attempts": int(self._closed.sum()),
            "attempts": [{
                "environment_id": env_id, "attempt_index": 0,
                "status": "terminal" if self._closed[env_id] else "incomplete",
                "frames": self._frames[env_id], "terminal": self._terminals[env_id],
            } for env_id in range(self.num_envs)],
        })
