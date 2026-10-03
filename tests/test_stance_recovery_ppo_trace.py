"""Synthetic trace-contract tests only; no native runtime, physics, or training claim."""
import pytest
import torch

from mjlab_microduck import stance_checkpoint as checkpoint
from mjlab_microduck import stance_recovery_contract as baseline
from mjlab_microduck import stance_recovery_ppo_trace as trace
from mjlab_microduck import stance_recovery_schedule as schedule


SOURCE = "a" * 40


def declaration():
    return schedule.declaration(SOURCE, "dose", "training",
                                ["zero-wrench", "+x-2n-20steps-t250"])


def test_protocol_is_sibling_and_does_not_widen_historical_allowlists():
    from mjlab_microduck import stance_attempt_trace

    assert trace.PROTOCOL not in stance_attempt_trace.ITERATIONS
    assert checkpoint.EVALUABLE["lean-replication"] == (64, 128, 192, 255)
    assert 653 not in checkpoint.FRESH_SEEDS and 659 not in checkpoint.FRESH_SEEDS
    assert baseline.FALSE_FLAGS["training_admitted"] is False


def test_target_schedule_is_exact_training_dose_pair():
    value = trace._target_schedule(declaration())
    assert value["cell_ids"] == list(trace.WORLD_CELLS)
    assert value["split"] == "training"
    assert value["row_cells"][0]["force_world_newtons"] == [0., 0., 0.]
    assert value["row_cells"][1]["onset_step"] == 250
    assert value["row_cells"][1]["duration_steps"] == 20
    assert value["row_cells"][1]["force_world_newtons"] == [2., 0., 0.]


@pytest.mark.parametrize("cells", [
    ["zero-wrench", "+x-2n-20steps-t250", "+x-4n-10steps-t250"],
    ["zero-wrench", "-x-2n-20steps-t250"],
])
def test_target_schedule_rejects_wrong_row_contract(cells):
    with pytest.raises(ValueError):
        trace._target_schedule(schedule.declaration(SOURCE, "dose", "training", cells))


def test_full_matrix_math_has_zero_row_and_accepted_onset_only():
    d = declaration()
    force, generalized = schedule.expected_wrenches(d, [250, 250], [True, True], 3, 1)
    assert force[0] == [[0.] * 6 for _ in range(3)]
    assert force[1][1][:3] == [2., 0., 0.]
    assert force[1][1][3:] == [0., 0., 0.]
    assert generalized == [[0.] * 20 for _ in range(2)]
    force, _ = schedule.expected_wrenches(d, [250, 250], [False, True], 3, 1)
    assert force[0] == [[0.] * 6 for _ in range(3)]
    assert force[1][1][:3] == [2., 0., 0.]
    force, _ = schedule.expected_wrenches(d, [270, 270], [True, True], 3, 1)
    assert force == [[[0.] * 6 for _ in range(3)] for _ in range(2)]


def _storage_fixture():
    storage = dict(observations={"actor": torch.zeros(28, 2, 44),
                                 "critic": torch.zeros(28, 2, 50)},
        actions=torch.zeros(28, 2, 10), rewards=torch.zeros(28, 2, 1),
        dones=torch.zeros(28, 2, 1, dtype=torch.uint8),
        values=torch.zeros(28, 2, 1), actions_log_prob=torch.zeros(28, 2, 1),
        distribution_params=[torch.zeros(28, 2, 10), torch.zeros(28, 2, 10)],
        returns=torch.zeros(28, 2, 1), advantages=torch.zeros(28, 2, 1))
    return storage


def test_plain_storage_contract_preserves_raw_action_and_no_gae():
    storage = _storage_fixture()
    action = torch.full((2, 10), 1.25)
    mu = torch.full((2, 10), .1)
    sigma = torch.full((2, 10), .3)
    policy = dict(actor_input=torch.zeros(2, 44), critic_input=torch.zeros(2, 50),
        raw_action=action, value=torch.ones(2, 1), log_prob=torch.full((2, 1), -.5),
        stored_reward=torch.tensor([[.1], [.2]]), done=torch.zeros(2, 1, dtype=torch.uint8),
        distribution_params=[mu, sigma])
    storage["observations"]["actor"][0] = policy["actor_input"]
    storage["observations"]["critic"][0] = policy["critic_input"]
    storage["actions"][0] = action
    storage["values"][0] = policy["value"]
    storage["actions_log_prob"][0] = policy["log_prob"]
    storage["rewards"][0] = policy["stored_reward"]
    storage["dones"][0] = policy["done"]
    storage["distribution_params"][0][0] = mu
    storage["distribution_params"][1][0] = sigma
    artifact = dict(storage=storage, policy_ticks=[policy])

    trace._check_storage(artifact, 1)
    assert torch.equal(storage["actions"][0], action)
    assert storage["returns"].count_nonzero() == storage["advantages"].count_nonzero() == 0
    storage["actions"][0, 0, 0] = 1.0
    with pytest.raises(ValueError, match="raw transition"):
        trace._check_storage(artifact, 1)


def test_synthetic_trace_never_claims_native_provenance():
    profile = trace.profile.expected_receipt()
    fake_plant = dict(selected_plant={}, nbody=2, body_names=["world", "trunk_base"],
                      body_name="trunk_base", body_id=1)
    binding = trace.binding(declaration(), fake_plant, profile)
    assert binding["worlds"] == 2 and binding["capture_device"] == "cpu"
    assert binding["learner_seed"] == 653
    assert binding["parent_checkpoint_sha256"] == baseline.CHECKPOINT_SHA256
    assert all(value is False for value in baseline.FALSE_FLAGS.values())


@pytest.mark.parametrize("damage", ["returns-shape", "advantages-shape", "done-mask", "unused-gaussian"])
def test_storage_rejects_bad_uncomputed_layout_and_unused_values(damage):
    storage = _storage_fixture()
    if damage == "returns-shape":
        storage["returns"] = torch.zeros(1)
    elif damage == "advantages-shape":
        storage["advantages"] = torch.zeros(1)
    elif damage == "done-mask":
        storage["dones"][0, 0, 0] = 2
    else:
        storage["distribution_params"][1][0, 0, 0] = .3
    with pytest.raises(ValueError):
        trace._check_storage(dict(storage=storage, policy_ticks=[]), 0)


def test_empty_synthetic_pulse_check_is_not_a_complete_delivery():
    value = dict(payload={"ticks": []}, declaration=declaration(),
        compiled_plant=dict(nbody=2, body_id=1), scheduled_pulse_evidence=[],
        control_evidence={"ticks": []})
    result = trace.check_pulses(value)
    assert result["checked_physics_steps"] == 0
    assert result["window_steps_per_row"] == [0, 0]
    assert result["complete_pulse_delivery"] is False


def test_wall_expired_prefix_rescores_as_rejected_without_becoming_success():
    # Expiry classifies this retained prefix; it does not prevent independent
    # consistency scoring of its already captured transitions.
    assert trace._elapsed_within_cap(121.25) is False
    assert trace._accepted_complete(tick_count=7, storage_step=7,
        stop_reason="wall-budget-exhausted", elapsed_seconds=121.25,
        private_rng_closed=True, caller_rng_unchanged=True, terminals=[None, None]) is False
    assert trace._accepted_complete(tick_count=28, storage_step=28,
        stop_reason="transition-limit", elapsed_seconds=121.25,
        private_rng_closed=True, caller_rng_unchanged=True, terminals=[None, None]) is False
    assert trace._accepted_complete(tick_count=28, storage_step=28,
        stop_reason="transition-limit", elapsed_seconds=119.99,
        private_rng_closed=True, caller_rng_unchanged=True, terminals=[None, None]) is True


@pytest.mark.parametrize("elapsed", [-0.1, float("nan"), float("inf")])
def test_elapsed_codec_rejects_invalid_elapsed_values(elapsed):
    with pytest.raises(ValueError, match="elapsed time"):
        trace._elapsed_within_cap(elapsed)
