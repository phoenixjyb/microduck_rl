"""CPU-only consistency audit of retained F1 route traces, never policy admission."""

import torch

from mjlab_microduck.first_attempt_smoke import canonical, require
from mjlab_microduck.recovery_measurement import RecoveryMeasurement


def audit_trace(report):
    canonical(report)
    count, n = report["sample_steps"], report["num_envs"]
    require(type(count) is int and 1 <= count <= 400 and n == 8
            and report["step_dt_s"] == .02 and report["speed_mps"] == .3, "fixed trace contract")
    trace = report["route_trace"]
    require(trace["protocol"] == "initial-route-pre-control-v1"
            and trace["position_columns"] == ["route_forward_m", "cross_route_m"]
            and trace["velocity_columns"] == ["body_forward_mps", "route_forward_mps", "cross_route_mps", "heading_rad"],
            "trace identity and column order")
    p, v = (torch.tensor(trace[k], dtype=torch.float64) for k in ("position", "velocity"))
    require(p.shape == (count,n,2) and v.shape == (count,n,4), "trace coverage")
    require(bool(torch.isfinite(p).all() and torch.isfinite(v).all()), "finite trace")
    require(bool((p[0].abs() < 1e-9).all()), "actual initial-origin position")
    def equal(actual, stored):
        expected = torch.as_tensor(stored, dtype=torch.float64)
        require(actual.shape == expected.shape and bool(torch.allclose(actual, expected, atol=1e-9, rtol=1e-9)),
                "trace agrees with summary")
    equal(p[-1,:,1], trace["last_sample_cross_route_m"])
    equal(p[:,:,1].abs().max(0).values, trace["max_abs_cross_route_m"])
    equal(v[:,:,2].mean(0), trace["signed_cross_route_velocity_mean_mps"])
    for name, start in (("all",0), ("settled",100)):
        if count <= start:
            require(name not in report["groups"], "no unsampled window")
            continue
        w, group = v[start:], report["groups"][name]
        require(group["steps"] == count-start, "window coverage")
        for key, actual in dict(body_forward_per_env_mean=w[:,:,0].mean(0),
            route_forward_per_env_mean=w[:,:,1].mean(0), body_forward_mean=w[:,:,0].mean(),
            route_forward_mean=w[:,:,1].mean(), cross_route_abs_mean=w[:,:,2].abs().mean(),
            cross_route_abs_per_env_mean=w[:,:,2].abs().mean(0), heading_abs_max=w[:,:,3].abs().max(),
            heading_abs_per_env_max=w[:,:,3].abs().max(0).values).items():
            equal(actual, group[key])
    reproduced = not report["terminal_steps"]
    if reproduced:
        observer = RecoveryMeasurement(n,.3,.02)
        for step in range(count):
            observer.begin(step, torch.full((n,), 0 if step<100 else 2), v[step,:,1])
            observer.finish(torch.zeros(n,dtype=torch.bool))
        require(canonical(observer.report()) == canonical(report["stable_route_response"]),
                "stable response reproduced from raw signed route velocities")
    # Aggregate terminal steps cannot reconstruct the per-environment done mask.
    return dict(trace_summary_verified=True, stable_response_reproduced=reproduced,
                sampled_steps=count, policy_acceptance=False)
