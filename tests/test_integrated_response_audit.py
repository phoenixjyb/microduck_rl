import copy

import pytest
import torch

from mjlab_microduck import integrated_response_audit as audit
from mjlab_microduck import integrated_turn_foundation as base


def samples():
    velocity = torch.tensor([[[.1, -.2]], [[.3, .2]], [[.3, 0.]]], dtype=torch.float64)
    motors = torch.tensor([[[[-.2, 0.], [.6, 0.]]], [[[.2, 0.], [-.1, 0.]]],
                           [[[0., 0.], [.1, 0.]]]], dtype=torch.float64)
    positions = torch.tensor([[[0., .99]], [[1.1, -.99]], [[0., 0.]]], dtype=torch.float64)
    limits = torch.tensor([[[-1., 1.], [-1., 1.]]], dtype=torch.float64)
    return velocity, motors, positions, limits


def test_signed_world_response_does_not_hide_oscillation_or_claim_zero_yaw_direction():
    v, m, q, lim = samples()
    row = base.response_diagnostics(v, m, q, lim, ("left", "right"), .2, startup=1)
    assert row["mean_body_vx_per_world"] == pytest.approx([.3])
    assert row["mean_body_yaw_per_world"] == pytest.approx([.1])
    assert row["mean_abs_body_yaw_per_world"] == pytest.approx([.1])
    assert row["yaw_correct_sign_fraction_per_world"] == pytest.approx([.5])
    assert row["soft_limit_fraction_by_joint"] == pytest.approx([0., 1/3])
    assert row["mean_signed_force_nm_by_joint"] == pytest.approx([0., .2])
    assert row["abs_force_p99_nm_by_joint"][1] == pytest.approx(.59)
    assert row["hard_stop_proximity_fraction_by_joint"] == pytest.approx([1/3, 2/3])
    assert row["hard_range_violation_fraction_by_joint"] == pytest.approx([1/3, 0.])
    assert row["min_distance_to_hard_stop_rad_by_joint"] == pytest.approx([-.1, .01])
    assert row["positions_not_synchronous_with_force"] and not row["policy_acceptance"]
    assert base.response_diagnostics(v, m, q, lim, ("left", "right"), 0.)["yaw_correct_sign_fraction_per_world"] is None


def test_short_terminal_has_no_invented_settled_response():
    v, m, q, lim = samples()
    row = base.response_diagnostics(v, m, q, lim, ("left", "right"), -.2)
    assert row["settled_steps"] == 0 and row["mean_body_vx_per_world"] is None
    assert row["mean_body_yaw_per_world"] is None
    assert row["steps"] == 3 and len(row["abs_force_p99_nm_by_joint"]) == 2


@pytest.mark.parametrize("bad", ["duplicate", "shape", "nonfinite", "bounds"])
def test_diagnostic_mapping_or_state_failure_is_not_sanitized(bad):
    v, m, q, lim = samples()
    names = ("left", "right")
    if bad == "duplicate":
        names = ("left", "left")
    elif bad == "shape":
        q = q[:, :, :1]
    elif bad == "nonfinite":
        m[0, 0, 0, 0] = float("nan")
    else:
        lim[0, 0, 1] = -1.
    with pytest.raises(ValueError):
        base.response_diagnostics(v, m, q, lim, names, .2)


def test_replay_preserves_all_old_fields_and_does_not_widen_tolerance():
    old = dict(complete=True, seed=839, errors=[.1, .2], value=None, binding="sha")
    same = copy.deepcopy(old)
    same["errors"][0] += 5e-7
    assert audit.equal_replay(same, old)
    same["errors"][0] += 2e-6
    assert not audit.equal_replay(same, old)
    for change in (dict(complete=1), dict(binding="other"), dict(errors=[float("nan"), .2]),
                   dict(value=0.), dict(unknown=1)):
        assert not audit.equal_replay({**old, **change}, old)
    assert not audit.equal_replay({k: v for k, v in old.items() if k != "seed"}, old)
    assert not audit.equal_replay(False, 0.)


def test_only_the_two_declared_final_checkpoints_are_audited():
    assert tuple(audit.CHECKPOINTS) == ("control", "tracking")
    assert len(audit.EVALUATION_SHA) == 64 and all(len(v) == 64 for v in audit.CHECKPOINTS.values())
    assert audit.INPUT_SOURCE == "6dbc0613c7fd55f081bf4cebea292b9acec8a61e"


def test_replay_diff_explains_failed_numerics_without_loosening_tolerance():
    retained = dict(seed=839, speed_mae=[.1, .2], complete=True)
    observed = dict(seed=839, speed_mae=[.1, .20002], complete=True)
    assert audit.replay_differences(observed, retained) == [dict(field="speed_mae[1]", retained=.2, observed=.20002)]
    assert audit.replay_differences(retained, retained) == []
    assert audit.replay_differences(dict(complete=1), dict(complete=True)) == [
        dict(field="complete", retained=True, observed=1)]


def test_target_diagnostics_keep_requested_overshoot_and_constraint_sign():
    _, _, q, lim = samples()
    targets = q + .2
    constraints = -q
    row = base.joint_target_diagnostics(q, targets, constraints, lim, ("left", "right"))
    assert row["target_outside_configured_range_fraction"] == pytest.approx([1/3, 1/3])
    assert row["previous_applied_target_rad"]["maximum"] == pytest.approx([1.3, 1.19])
    assert row["position_rad"]["maximum"] == pytest.approx([1.1, .99])
    assert row["generalized_constraint_force_nm"]["minimum"] == pytest.approx([-1.1, -.99])
    assert row["constraint_includes_contacts_and_limits"] and not row["hard_stop_force_isolated"]
    assert not row["policy_acceptance"] and not row["physical_motion_authorized"]


@pytest.mark.parametrize("fault", ["shape", "nan", "names", "bounds"])
def test_target_capture_fails_closed_on_bad_identity_or_values(fault):
    _, _, q, lim = samples()
    t, c, names = q.clone(), q.clone(), ("left", "right")
    if fault == "shape": t = t[..., :1]
    if fault == "nan": c[0, 0, 0] = float("nan")
    if fault == "names": names = ("left", "left")
    if fault == "bounds": lim[0, 0, 1] = -1.
    with pytest.raises(ValueError): base.joint_target_diagnostics(q, t, c, lim, names)
