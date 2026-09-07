"""Raw control-step motor accounting; no contact-phase or physical safety claim."""

import torch

from mjlab_microduck.first_attempt_smoke import canonical, require
from mjlab_microduck.motor_audit_smoke import JOINTS
from mjlab_microduck.tasks.run import XL330_M288_RATED_NO_LOAD_SPEED_RAD_S as RATED_SPEED

PROTOCOL = "pre-reset-motor-history-v1"
TIMING = dict(force="last-physics-substep-derived-force; one-integration lag",
              speed="integrated-joint-velocity-at-capture",
              capture="post-decimation-metrics-hook-before-reset-and-final-forward",
              route="pre-control-step; same control interval, not simultaneous physics state")


def pack_motor_trace(force, speed):
    require(force.ndim == 3 and force.shape == speed.shape
            and 1 <= force.shape[0] <= 400 and force.shape[1:] == (8, 14), "bounded named motor history")
    require(bool(torch.isfinite(force).all() and torch.isfinite(speed).all()), "finite motor history")
    return dict(protocol=PROTOCOL, joint_columns=list(JOINTS), step_dt_s=.02,
                units=dict(force="Nm", speed="rad/s"), timing=dict(TIMING),
                force_nm=force.detach().cpu().double().tolist(),
                speed_rad_s=speed.detach().cpu().double().tolist())


def audit_motor_trace(report):
    canonical(report)
    trace = report["motor_trace"]
    require(trace["protocol"] == PROTOCOL and trace["joint_columns"] == list(JOINTS)
            and trace["step_dt_s"] == report["step_dt_s"] == .02
            and trace["units"] == dict(force="Nm", speed="rad/s")
            and trace["timing"] == TIMING, "motor trace identity/timing/units")
    f, s = (torch.tensor(trace[k], dtype=torch.float64) for k in ("force_nm", "speed_rad_s"))
    count = report["sample_steps"]
    require(type(count) is int and 1 <= count <= 400 and report["num_envs"] == 8
            and f.shape == s.shape == (count, 8, 14), "exact bounded motor coverage")
    require(bool(torch.isfinite(f).all() and torch.isfinite(s).all()), "finite motor samples")
    u = f.abs()/.6
    power = (f*s).abs()
    for name, start in (("all", 0), ("settled", 100)):
        if count <= start:
            require(name not in report["groups"], "no invented settled motor coverage")
            continue
        g = report["groups"][name]
        require(g["steps"] == count-start, "motor group step count")
        values = dict(pre_reset_torque_p99=float(torch.quantile(u[start:].flatten(), .99)),
            pre_reset_rated_speed_exceed_fraction=float((s[start:].abs() > RATED_SPEED).double().mean()),
            pre_reset_squared_utilization_mean=float(u[start:].square().mean()),
            pre_reset_soft_limit_fraction=float((u[start:] > .7).double().mean()),
            pre_reset_mechanical_abs_power_mean_w=float(power[start:].mean()))
        for key, actual in values.items():
            require(abs(g[key]-actual) <= 1e-9, "raw motor/summary disagreement: "+key)
        require(set(g["pre_reset_joint_p99"]) == set(JOINTS), "all named motors in summary")
        for i, joint in enumerate(JOINTS):
            require(abs(g["pre_reset_joint_p99"][joint]-float(torch.quantile(u[start:,:,i].flatten(), .99)))
                    <= 1e-9, "raw named-joint quantile disagreement: "+joint)
    # Analysis below needs full, terminal-free paired episodes and raw route data.
    # A partial trace is still retained/auditable, but cannot supply this diagnosis.
    if count != 400 or report["terminal_steps"] or report["safety_failures"]:
        return dict(raw_motor_summary_verified=True, decision="incomplete-or-unsafe-no-timing-diagnosis",
                    policy_acceptance=False)
    from mjlab_microduck.foundation_trace_audit import audit_trace
    audit_trace(report)
    route = torch.tensor(report["route_trace"]["velocity"], dtype=torch.float64)
    rows = {}
    total_load = u[100:].square().sum()
    for i, joint in enumerate(JOINTS):
        a = u[100:,:,i]
        peak = int(a.flatten().argmax())
        bins = []
        for start in (100, 150, 200, 250, 300, 350):
            window = u[start:start+50,:,i]
            bins.append(dict(first_step=start, steps=50,
                squared_utilization_mean=float(window.square().mean()),
                torque_p99=float(torch.quantile(window.flatten(), .99)),
                absolute_power_mean_w=float(power[start:start+50,:,i].mean())))
        slow = route[100:,:,1] < .27
        # Conditional association only: masks use pre-control route velocity,
        # force/speed are the documented post-decimation capture, not contact phase.
        rows[joint] = dict(
            settled_torque_p99_by_environment=torch.quantile(a, .99, dim=0).tolist(),
            settled_soft_limit_fraction=float((a > .7).double().mean()),
            settled_squared_load_share=float(a.square().sum()/total_load) if total_load > 0 else 0.,
            peak=dict(step=100+peak//8, environment=peak%8, utilization=float(a.flatten()[peak])),
            slow_samples=int(slow.sum()), other_samples=int((~slow).sum()),
            squared_load_when_route_below_027=float(a[slow].square().mean()) if bool(slow.any()) else None,
            squared_load_other=float(a[~slow].square().mean()) if bool((~slow).any()) else None,
            one_second_bins=bins)
    return dict(raw_motor_summary_verified=True, decision="timing-measurement-only", joints=rows,
                timing=dict(TIMING), contact_phase_measured=False, causal_effect_established=False,
                policy_acceptance=False, physical_motion_authorized=False)
