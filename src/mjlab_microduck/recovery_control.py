"""Opt-in positive acceleration cap; no policy, motor-safety or motion admission."""

from dataclasses import asdict, dataclass
import math

import torch


PROTOCOL = "recovery-positive-acceleration-cap-v1"
ROLLOUT_STAGE = "RecoveryAcceleration-diagnostic-rollout"


@dataclass(frozen=True)
class RecoveryAccelerationCfg:
    # Source-test starting point, not a tuned or physically calibrated value.
    max_acceleration_mps2: float = 0.20

    def __post_init__(self):
        if (type(self.max_acceleration_mps2) not in (float, int)
                or not math.isfinite(self.max_acceleration_mps2)
                or self.max_acceleration_mps2 <= 0):
            raise ValueError("recovery acceleration must be finite and positive")

    def provenance(self):
        return dict(protocol=PROTOCOL, config=asdict(self),
                    scope="positive forward-command changes in recovery only",
                    braking="legacy braking and immediate invalid-observation stop unchanged",
                    target="desired speed unchanged; no forced acceleration at a deadline",
                    policy_acceptance=False, physical_motion_authorized=False)


def cap_recovery_acceleration(command, previous, phase, *, cfg, update_dt_s):
    """Further restrict an already bounded command without changing yaw/braking.

    dt is the actual command-update interval, not a physics timestep. There is
    no extra phase history: reset/clone of the existing previous_command state
    remains sufficient. This bounds commands, not measured robot acceleration.
    """
    if not isinstance(cfg, RecoveryAccelerationCfg):
        raise ValueError("explicit recovery configuration required")
    if (type(update_dt_s) not in (float, int) or not math.isfinite(update_dt_s)
            or update_dt_s <= 0 or not math.isfinite(cfg.max_acceleration_mps2 * update_dt_s)):
        raise ValueError("actual command-update interval must be finite and positive")
    if (command.ndim != 2 or command.shape[1] != 2 or command.shape[0] < 1 or previous.shape != command.shape
            or phase.shape != command.shape[:1] or phase.dtype not in (torch.int32, torch.int64)
            or command.dtype not in (torch.float32, torch.float64) or previous.dtype != command.dtype
            or command.device != previous.device or command.device != phase.device
            or not bool(((phase >= 0) & (phase <= 2)).all())
            or not bool(((command[:, 0] >= 0) & (previous[:, 0] >= 0)).all())
            or not bool(torch.isfinite(command).all() & torch.isfinite(previous).all())):
        raise ValueError("invalid recovery command/phase state")
    ceiling = previous[:, 0] + cfg.max_acceleration_mps2 * update_dt_s
    if not bool(torch.isfinite(ceiling).all()):
        raise ValueError("recovery acceleration ceiling overflow")
    result = command.clone()
    result[:, 0] = torch.where(phase == 2, torch.minimum(command[:, 0], ceiling), command[:, 0])
    return result


def validate_rollout_mode(cfg, *, first_attempt_only, motor_measurement_audit,
                          collecting_dataset, recording, range_noise_m):
    if cfg is not None and (
        not isinstance(cfg, RecoveryAccelerationCfg) or not first_attempt_only
        or not motor_measurement_audit or collecting_dataset or recording or range_noise_m != 0.
    ):
        raise ValueError("recovery diagnostic requires first-only motor audit, no dataset/recorder/noise")
