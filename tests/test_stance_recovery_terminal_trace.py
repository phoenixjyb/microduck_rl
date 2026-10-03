"""Source/schema math fixtures only; never native runtime or terminal evidence."""
from copy import deepcopy

import pytest
import torch

from mjlab_microduck import stance_recovery_contract as baseline
from mjlab_microduck import stance_recovery_schedule as schedule
from mjlab_microduck import stance_recovery_terminal_trace as trace

SOURCE = "a" * 40
CELLS = ["zero-wrench", "+x-2n-20steps-t250"]


def declaration():
    return schedule.declaration(SOURCE, "dose", "training", CELLS)


def synthetic_plant():
    return {"selected_plant": {"synthetic_test_fixture": True}, "nbody": 3,
        "body_names": ["world", "trunk_base", "other"], "body_name": "trunk_base", "body_id": 1}


def test_protocol_is_new_record_only_sibling_with_fixed_horizon_and_flags():
    from mjlab_microduck import stance_attempt_trace

    assert trace.PROTOCOL not in stance_attempt_trace.ITERATIONS
    assert trace.HORIZON == 250 and trace.WALL_LIMIT == 180.0
    assert trace.LIMIT == 128 * 1024 * 1024
    assert all(value is False for value in baseline.FALSE_FLAGS.values())


def test_dose_declaration_binds_only_fixed_rows_and_pulse_timing():
    value = trace._target_schedule(declaration())
    assert value["worlds"] == 2
    assert value["row_cells"][0]["onset_step"] == 500
    assert value["row_cells"][0]["duration_steps"] == 10
    assert value["row_cells"][1]["onset_step"] == 250
    assert value["row_cells"][1]["duration_steps"] == 20
    assert value["row_cells"][1]["force_world_newtons"] == [2., 0., 0.]


@pytest.mark.parametrize("change", ["cell", "stage", "split", "onset", "dose"])
def test_wrong_schedule_or_dose_is_rejected(change):
    cells = list(CELLS)
    stage, split = "dose", "training"
    if change == "cell":
        cells[1] = "-x-2n-20steps-t250"
    value = schedule.declaration(SOURCE, stage, split, cells)
    if change == "stage":
        value["stage"] = "baseline"
    elif change == "split":
        value["split"] = "held-out"
    elif change == "onset":
        value["row_cells"][1]["onset_step"] = 251
    elif change == "dose":
        value["row_cells"][1]["force_world_newtons"] = [4., 0., 0.]
    with pytest.raises(ValueError):
        trace._target_schedule(value)


def test_binding_is_source_parent_plant_profile_and_protocol_pinned(monkeypatch):
    d, plant = declaration(), synthetic_plant()
    monkeypatch.setattr(baseline.force, "_binding", lambda _value: (3, 1))
    profile = trace.profile.expected_receipt()
    b = trace.binding(d, plant, profile)
    assert b["source"] == SOURCE
    assert b["protocol"] == trace.PROTOCOL
    assert b["parent_checkpoint_sha256"] == baseline.CHECKPOINT_SHA256
    assert b["parent_state_sha256"]
    assert b["schedule_sha256"] == schedule.binding_sha256(d)
    assert b["plant_sha256"]
    assert b["max_policy_calls"] == 250 and b["max_physics_steps"] == 2500

    monkeypatch.setattr(trace.profile, "check_recorded", lambda _value: None)
    assert trace._binding_matches(b, d, plant)
    wrong_source = deepcopy(b)
    wrong_source["source"] = "b" * 40
    assert not trace._binding_matches(wrong_source, d, plant)
    changed_plant = synthetic_plant()
    changed_plant["body_id"] = 2
    assert not trace._binding_matches(b, d, changed_plant)


def test_force_coverage_math_does_not_reuse_old_28_tick_windows():
    assert trace._pulse_complete([10, 20], [0, 20], 250, 2500)
    assert not trace._pulse_complete([0, 20], [0, 20], 250, 2500)
    assert not trace._pulse_complete([10, 19], [0, 19], 250, 2499)
    assert not trace._pulse_complete([0, 0], [0, 0], 0, 0)


def _tick(executed):
    return {"executed_steps": torch.tensor(executed, dtype=torch.int64)}


def test_policy_clock_requires_exact_per_world_euler_advance():
    tick = _tick([10, 10])
    expected = torch.zeros(2, dtype=torch.float32)
    for _ in range(10):
        expected = expected + torch.full_like(expected, .002)
    clock = {"before": torch.zeros(2), "after": expected,
        "accepted_substeps": torch.tensor([10, 10])}
    assert trace._check_policy_clock(clock, tick) is None
    broken = deepcopy(clock)
    broken["after"][1] = .021
    with pytest.raises(ValueError):
        trace._check_policy_clock(broken, tick)
    broken = deepcopy(clock)
    broken["accepted_substeps"][0] = 9
    with pytest.raises(ValueError):
        trace._check_policy_clock(broken, tick)


class _FixedCritic:
    def __call__(self, obs):
        return torch.full((2, 1), float(obs["critic"][0, 0]))


def _terminal_policy_fixture():
    actor = torch.zeros(2, 44)
    critic_obs = torch.zeros(2, 50)
    critic_obs[:, :44] = actor
    actor[0, 0] = .25
    critic_obs[:, :44] = actor
    reward = torch.tensor([1., 2.])
    terminated = torch.tensor([False, False])
    timed_out = torch.tensor([True, False])
    done = terminated | timed_out
    value = torch.tensor([[.25], [.25]])
    return {"payload": {"ticks": [{"reward": reward, "terminated": terminated,
            "timed_out": timed_out, "boundaries": [{"observation":{"actor":actor,
                "critic":critic_obs}}]}]},
        "terminal_policy": {"terminal_observation": {"actor":actor.clone(), "critic":critic_obs.clone()},
            "terminal_critic_value": value, "timeout_reward": torch.tensor([1.2475, 2.]),
            "done_mask": done}}, done


def test_terminal_critic_and_timeout_bootstrap_are_exact_once():
    value, done = _terminal_policy_fixture()
    assert trace._check_terminal_policy(value, done, _FixedCritic()) is None
    for field, edit in (
        ("terminal_critic_value", lambda x: x.add_(.1)),
        ("timeout_reward", lambda x: x.add_(.1)),
        ("done_mask", lambda x: x.logical_not_()),
    ):
        damaged = deepcopy(value)
        edit(damaged["terminal_policy"][field])
        with pytest.raises(ValueError):
            trace._check_terminal_policy(damaged, done, _FixedCritic())


def test_terminal_observation_must_be_pre_reset_observation():
    value, done = _terminal_policy_fixture()
    value["terminal_policy"]["terminal_observation"]["critic"][0, 44] = 1.
    with pytest.raises(ValueError, match="terminal observation"):
        trace._check_terminal_policy(value, done, _FixedCritic())


def test_private_sampler_rng_is_a_closed_ordered_chain():
    initial = torch.Generator().manual_seed(653).get_state()
    middle = initial.clone(); middle[0] ^= 1
    final = middle.clone(); final[0] ^= 2
    rows = [{"rng_before": initial, "rng_after": middle},
            {"rng_before": middle, "rng_after": final}]
    assert trace._check_private_chain(initial, rows, final)
    rows[1]["rng_before"] = initial
    with pytest.raises(ValueError):
        trace._check_private_chain(initial, rows, final)
    rows[1]["rng_before"] = middle
    with pytest.raises(ValueError):
        trace._check_private_chain(initial, rows, initial)


def _sibling_fixture():
    before = {"qpos": torch.tensor([[1.], [2.]]), "qvel": torch.tensor([[3.], [4.]]),
        "time": torch.tensor([.5, .7]), "frame": {"physics_steps": torch.tensor([2500, 120]),
            }, "live": torch.tensor([False, True]), "controls": {"ctrl": torch.tensor([[5.], [6.]])}}
    after = {"qpos": before["qpos"].clone(), "qvel": before["qvel"].clone(),
        "time": before["time"].clone(), "frame": {"physics_steps": torch.tensor([0, 120])},
        "live": torch.tensor([True, True]), "controls": {"ctrl": before["controls"]["ctrl"].clone()}}
    after["qpos"][0] = 0.
    return before, after


def _valid_frame(steps):
    qpos = torch.zeros(2, 21)
    qpos[:, 2] = .2
    qvel = torch.zeros(2, 20)
    actor = torch.zeros(2, 44)
    critic = torch.zeros(2, 50)
    critic[:, :44] = actor
    state = {"tilt": torch.zeros(2), "root_velocity": qvel[:, :3].clone(),
        "height": qpos[:, 2].clone(), "support": torch.ones(2, 2), "torque": torch.zeros(2, 14),
        "joint_velocity": torch.zeros(2, 14), "hard_limit": torch.zeros(2, dtype=torch.bool),
        "forbidden_contact": torch.zeros(2, dtype=torch.bool), "warning": torch.zeros(2, dtype=torch.bool)}
    return {"physics_steps": torch.tensor(steps, dtype=torch.int64), "qpos": qpos, "qvel": qvel,
        "soft_limit_mask": torch.zeros(2, 14, dtype=torch.bool), "state": state,
        "observation": {"actor": actor, "critic": critic}}


def test_policy_inputs_bind_pre_action_not_realized_post_action_correction():
    before = _valid_frame([40, 40])
    policy = {"actor_input": before["observation"]["actor"].clone(),
        "critic_input": before["observation"]["critic"].clone()}
    post_action = deepcopy(before["observation"])
    # set_actions() may update the realized-correction feature block before
    # the runtime records the first boundary. The sampled policy input remains
    # the previous frame, not this after-action observation.
    post_action["actor"][:, 34:44] = .25
    post_action["critic"][:, 34:44] = -.5
    assert not torch.equal(policy["critic_input"], post_action["critic"])
    assert trace._check_pre_action_inputs(before, policy) is None

    wrong_post_action_input = dict(policy, critic_input=post_action["critic"])
    with pytest.raises(ValueError, match="pre-action"):
        trace._check_pre_action_inputs(before, wrong_post_action_input)


def test_policy_inputs_reject_actor_from_after_action_frame_too():
    before = _valid_frame([40, 40])
    policy = {"actor_input": before["observation"]["actor"].clone(),
        "critic_input": before["observation"]["critic"].clone()}
    policy["actor_input"][:, 34:44] = .25
    with pytest.raises(ValueError, match="pre-action"):
        trace._check_pre_action_inputs(before, policy)


def test_terminal_frame_must_be_exact_last_boundary_not_detached_label():
    frame = _valid_frame([2500, 2500])
    before = {"frame": deepcopy(frame), "live": torch.tensor([False, False]),
        "qpos": frame["qpos"].clone(), "qvel": frame["qvel"].clone()}
    tick = {"boundaries": [deepcopy(frame)] * 11, "live": torch.tensor([False, False])}
    assert trace._check_terminal_frame(before, tick)
    tick["boundaries"][-1]["qpos"][0, 0] = .01
    with pytest.raises(ValueError, match="exactly equals"):
        trace._check_terminal_frame(before, tick)


def test_reset_rows_must_return_to_initial_qpos_qvel_and_observation():
    initial = _valid_frame([0, 0])
    after_frame = deepcopy(initial)
    after_frame["physics_steps"] = torch.tensor([0, 300], dtype=torch.int64)
    after = {"frame": after_frame, "qpos": after_frame["qpos"].clone(),
        "qvel": after_frame["qvel"].clone(), "time": torch.tensor([0., .6]),
        "live": torch.tensor([True, True])}
    done = torch.tensor([True, False])
    assert trace._check_reset_rows(after, done, initial)
    bad = deepcopy(after)
    bad["qvel"][0, 0] = .1
    bad["frame"]["qvel"][0, 0] = .1
    bad["frame"]["state"]["root_velocity"][0, 0] = .1
    with pytest.raises(ValueError):
        trace._check_reset_rows(bad, done, initial)


def test_original_reset_qpos_uses_exact_per_world_rows_without_broadcast():
    initial = _valid_frame([0, 0])["qpos"]
    initial[1, 0] = .125
    assert trace._check_reset_initial_qpos(initial.clone(), initial) is None
    # A single row, even when it matches world zero, is not the packed capture.
    with pytest.raises(ValueError, match="tensor layout"):
        trace._check_reset_initial_qpos(initial[0], initial)
    with pytest.raises(ValueError, match="exact original per-world"):
        trace._check_reset_initial_qpos(initial[0:1].expand(2, -1), initial)


@pytest.mark.parametrize("change", ["row", "dtype", "nonfinite", "physical-layout"])
def test_original_reset_qpos_rejects_bad_values_and_layout(change):
    initial = _valid_frame([0, 0])["qpos"]
    recorded = initial.clone()
    if change == "row":
        recorded[1, 4] += .01
    elif change == "dtype":
        recorded = recorded.double()
    elif change == "nonfinite":
        recorded[0, 0] = float("nan")
    elif change == "physical-layout":
        initial = initial[0]
    with pytest.raises(ValueError):
        trace._check_reset_initial_qpos(recorded, initial)


def test_selective_reset_preserves_untouched_sibling_state_and_controls():
    before, after = _sibling_fixture()
    done = torch.tensor([True, False])
    assert trace._check_reset_siblings(before, after, done)
    # A still-live row can legitimately differ from episode-initial controls;
    # only preservation across the selected reset is required.
    assert before["controls"]["ctrl"][1].item() == after["controls"]["ctrl"][1].item() == 6.
    for key in ("qpos", "qvel", "time"):
        damaged = deepcopy(after)
        damaged[key][1] += .01
        with pytest.raises(ValueError):
            trace._check_reset_siblings(before, damaged, done)
    damaged = deepcopy(after)
    damaged["controls"]["ctrl"][1] += 1.
    with pytest.raises(ValueError):
        trace._check_reset_siblings(before, damaged, done)


def test_simultaneous_timeout_is_not_selective_reset_and_early_claim_is_not_full():
    records = [{"timed_out": True, "terminated": False}, {"timed_out": True, "terminated": False}]
    full, selective = trace._reset_claims(torch.ones(2, dtype=torch.bool), records,
        torch.full((2,), 2500), True, False)
    assert full and not selective
    with pytest.raises(ValueError):
        trace._reset_claims(torch.ones(2, dtype=torch.bool), records,
            torch.full((2,), 400), True, False)
    with pytest.raises(ValueError):
        trace._reset_claims(torch.tensor([True, False]), [records[0], None],
            torch.tensor([2500, 200]), False, False)


def test_encode_cap_rejects_oversized_or_bad_value(monkeypatch):
    monkeypatch.setattr(trace, "LIMIT", 1)
    with pytest.raises(ValueError, match="bounded terminal"):
        trace.encode({"synthetic_test_fixture": True})
