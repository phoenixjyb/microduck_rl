"""CPU trace scoring for a frozen speed map; no simulator, launcher or admission.

Inputs are caller-supplied measurements. Recomputed arithmetic is not proof of
their producer, checkpoint/runtime identity, or execution on a GPU.
"""

from dataclasses import dataclass
import math

import torch

from mjlab_microduck.first_attempt_smoke import canonical, require
from mjlab_microduck.motor_audit_smoke import JOINTS
from mjlab_microduck.recovery_measurement import RecoveryMeasurement
from mjlab_microduck.tasks.run import XL330_M288_RATED_NO_LOAD_SPEED_RAD_S as RATED_SPEED

PROTOCOL = "foundation-command-map-trace-v1"
SPEEDS = (.1, .2, .3)
SEEDS = (503, 509, 521)  # Development data, never untouched confirmation.
CHECKPOINTS = {
    "original": "080f98ae4d5ce731d143c733181bb89d504cb4b51ff39532efccd0b5fdc09c54",
    "narrow": "7ed703d6b5b8407da912f51755be8a8e57698340f62d0f5d3a80cf195ec1f80f",
}
STEPS, SETTLE, NUM_ENVS, STEP_DT = 400, 100, 8, .02


@dataclass(frozen=True)
class Cell:
    seed: int
    speed_mps: float
    policy: str

    def __post_init__(self):
        require(type(self.seed) is int and self.seed in SEEDS, "development seed")
        require(type(self.speed_mps) is float and self.speed_mps in SPEEDS, "declared moving speed")
        require(type(self.policy) is str and self.policy in CHECKPOINTS, "fixed policy")

    def identity(self):
        return dict(seed=self.seed, speed_mps=self.speed_mps, policy=self.policy,
                    declared_checkpoint_sha256=CHECKPOINTS[self.policy])


def schedule():
    return tuple(Cell(seed, speed, policy)
                 for seed in SEEDS for speed in SPEEDS for policy in CHECKPOINTS)


@dataclass(frozen=True)
class Trace:
    # First dimension: pre-control samples and the corresponding step outcome.
    # velocity columns: body x, initial-route x, initial-route y, heading error.
    velocity: torch.Tensor
    position: torch.Tensor  # Initial-route x/y; final post-step position absent.
    issued: torch.Tensor  # x/y/yaw-rate
    consumed: torch.Tensor  # Raw actor command slice, before normalization.
    legacy_force: torch.Tensor  # Nm; legacy post-step/reset-sensitive stream.
    legacy_speed: torch.Tensor  # rad/s
    pre_force: torch.Tensor  # Nm; pre-reset last-substep-derived force.
    pre_speed: torch.Tensor  # rad/s; pre-reset integrated joint velocity.
    dones: torch.Tensor
    joint_names: tuple[str, ...]


def _tensor(value, shape, *, boolean=False):
    require(isinstance(value, torch.Tensor) and value.device.type == "cpu"
            and value.shape == shape, "exact CPU trace coverage")
    if boolean:
        require(value.dtype == torch.bool, "boolean terminal stream")
    else:
        require(value.dtype in (torch.float32, torch.float64)
                and bool(torch.isfinite(value).all()), "finite floating trace")
    return value.detach().clone() if boolean else value.detach().double().clone()


def _no_authority():
    return dict(policy_acceptance=False, training_admitted=False,
                physical_motion_authorized=False, producer_verified=False,
                checkpoint_bytes_verified=False, runtime_verified=False,
                continuous_speed_range_validated=False)


def score(cell: Cell, trace: Trace):
    """Derive checks from bounded raw CPU tensors, never trust precomputed flags."""
    require(type(cell) is Cell and type(trace) is Trace, "typed cell and trace")
    # Revalidate even if a caller bypassed frozen-dataclass construction.
    cell.__post_init__()
    require(isinstance(trace.velocity, torch.Tensor) and trace.velocity.ndim == 3,
            "velocity dimensions")
    n = trace.velocity.shape[0]
    require(1 <= n <= STEPS, "bounded first attempt")
    require(trace.joint_names == tuple(JOINTS), "ordered named motor identity")
    v = _tensor(trace.velocity, (n, NUM_ENVS, 4))
    pos = _tensor(trace.position, (n, NUM_ENVS, 2))
    issued = _tensor(trace.issued, (n, NUM_ENVS, 3))
    consumed = _tensor(trace.consumed, (n, NUM_ENVS, 3))
    lf, ls, pf, ps = [_tensor(getattr(trace, k), (n, NUM_ENVS, 14))
                      for k in ("legacy_force", "legacy_speed", "pre_force", "pre_speed")]
    dones = _tensor(trace.dones, (n, NUM_ENVS), boolean=True)
    require(not bool(dones[:-1].any()), "no samples after first terminal in any environment")
    require(bool((v[:, :, 3].abs() <= math.pi + 1e-6).all()), "wrapped route heading")

    # Same kp/cap/slew as the retained heading-hold protocol, speed parameterized.
    previous = torch.zeros(NUM_ENVS, dtype=torch.float64)
    expected = torch.zeros_like(issued)
    expected[:, :, 0] = cell.speed_mps
    for step in range(n):
        heading = torch.atan2(v[step, :, 3].sin(), v[step, :, 3].cos())
        previous = previous + ((-heading).clamp(-.35, .35) - previous).clamp(-.02, .02)
        expected[step, :, 2] = previous
    safety = []
    if not torch.equal(issued, consumed):
        safety.append("issued-consumed-mismatch")
    if not torch.allclose(issued, expected, atol=1e-6, rtol=0):
        safety.append("heading-or-speed-command-mismatch")
    if bool(dones.any()):
        safety.append("terminal-including-startup")
    if n != STEPS:
        safety.append("incomplete-control")

    observer = RecoveryMeasurement(NUM_ENVS, cell.speed_mps, STEP_DT)
    for step in range(n):
        observer.begin(step, torch.full((NUM_ENVS,), 0 if step < SETTLE else 2), v[step, :, 1])
        observer.finish(dones[step])
    response = observer.report()
    groups = {}
    for name, start in (("all", 0), ("settled", SETTLE)):
        if n <= start:
            continue
        velocity = v[start:]
        # Preserve historical legacy float32 quantile accounting.
        legacy = lf[start:].float().abs() / .6
        pre = pf[start:].abs() / .6
        group = dict(steps=n-start,
                     body_forward_per_env_mean=velocity[:, :, 0].mean(0).tolist(),
                     route_forward_per_env_mean=velocity[:, :, 1].mean(0).tolist(),
                     cross_route_abs_per_env_mean=velocity[:, :, 2].abs().mean(0).tolist(),
                     heading_abs_per_env_max=velocity[:, :, 3].abs().max(0).values.tolist(),
                     legacy_torque_p99=float(torch.quantile(legacy.flatten(), .99)),
                     legacy_rated_speed_exceed_fraction=float(
                         (ls[start:].float().abs()/RATED_SPEED > 1).float().mean()),
                     pre_reset_rated_speed_exceed_fraction=float((ps[start:].abs()/RATED_SPEED > 1).double().mean()),
                     pre_reset_torque_p99=float(torch.quantile(pre.flatten(), .99)),
                     pre_reset_joint_p99={joint: float(torch.quantile(pre[:, :, i].flatten(), .99))
                                         for i, joint in enumerate(JOINTS)},
                     pre_reset_squared_utilization_mean=float(pre.square().mean()),
                     pre_reset_soft_limit_fraction=float((pre > .7).double().mean()),
                     pre_reset_mechanical_abs_power_mean_w=float((pf[start:]*ps[start:]).abs().mean()),
                     max_abs_cross_route_position_m=pos[start:, :, 1].abs().max(0).values.tolist())
        groups[name] = group
        if group["legacy_torque_p99"] > .60:
            safety.append(name+"-legacy-torque")
        if group["legacy_rated_speed_exceed_fraction"] != 0:
            safety.append(name+"-legacy-rated-speed")
        if group["pre_reset_rated_speed_exceed_fraction"] != 0:
            safety.append(name+"-pre-reset-rated-speed")
    performance = []
    if "settled" in groups:
        g = groups["settled"]
        for key in ("body_forward_per_env_mean", "route_forward_per_env_mean"):
            if any(abs(value-cell.speed_mps) > .03 for value in g[key]):
                performance.append(key+"-outside-band")
        if any(value > .05 for value in g["cross_route_abs_per_env_mean"]):
            performance.append("cross-route-motion")
        if any(value > .25 for value in g["heading_abs_per_env_max"]):
            performance.append("heading-drift")
        if response["counts"]["recovered-in-window"] != NUM_ENVS:
            performance.append("stable-speed-window-missed")
    result = dict(protocol=PROTOCOL, cell=cell.identity(), sample_steps=n,
                sampling="pre-control route; paired control-interval pre-reset motor stream, not simultaneous physics state",
                safety_failures=safety, performance_failures=performance,
                classification=("safety-or-coverage-stop" if safety else
                                "descriptive-performance-miss" if performance else
                                "descriptive-cell-within-checks"),
                groups=groups, stable_route_response=response, **_no_authority())
    canonical(result)  # Reject overflow/nonfinite derived statistics as well.
    return result


def summarize_prefix(cells_and_traces):
    """Recompute an ordered map prefix, refusing reordered or post-stop evidence."""
    require(type(cells_and_traces) in (list, tuple) and 1 <= len(cells_and_traces) <= 18,
            "bounded nonempty map prefix")
    results = []
    planned = schedule()
    for index, (cell, trace) in enumerate(cells_and_traces):
        require(cell == planned[index], "exact seed-major speed-major policy order")
        require(not results or not results[-1]["safety_failures"], "no cell after safety stop")
        results.append(score(cell, trace))
    decision = ("safety-or-coverage-stop" if results[-1]["safety_failures"] else
                "complete-descriptive-map" if len(results) == len(planned) else "incomplete-map")
    return dict(protocol=PROTOCOL, decision=decision, cells=results,
                unexecuted=[cell.identity() for cell in planned[len(results):]],
                development_seeds=list(SEEDS), **_no_authority())
