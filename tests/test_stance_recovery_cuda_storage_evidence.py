"""Synthetic CPU-only checks; no fixture contains a native CUDA capture."""

from copy import deepcopy
from importlib.metadata import version

import pytest
import torch
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from mjlab_microduck import stance_recovery_cuda_storage_evidence as storage_evidence
from mjlab_microduck import stance_recovery_cuda_transition as transition


N, H = storage_evidence.WORLDS, storage_evidence.HORIZON


def _record(index):
    actor = torch.full((N, 44), float(index), dtype=torch.float32)
    critic = torch.zeros((N, 50), dtype=torch.float32)
    critic[:, :44] = actor
    critic[:, 48:] = index + 0.25
    actions = torch.full((N, 10), 1.25 + index, dtype=torch.float32)
    values = torch.full((N, 1), 0.5 + index, dtype=torch.float32)
    log_prob = torch.full((N,), -0.75 - index, dtype=torch.float32)
    reward = torch.full((N,), 2.0 + index, dtype=torch.float32)
    terminated = torch.zeros(N, dtype=torch.bool)
    timed_out = torch.zeros(N, dtype=torch.bool)
    if index == 1:
        terminated[-1] = True
    return {
        "pre_action_observations": {"actor": actor, "critic": critic},
        "raw_actions": actions,
        "pre_action_values": values,
        "raw_actions_log_prob": log_prob,
        "learner_reward": reward,
        "terminated": terminated,
        "timed_out": timed_out,
        "distribution_params": (
            torch.full((N, 10), 0.1 + index, dtype=torch.float32),
            torch.full((N, 10), 0.8 + index, dtype=torch.float32),
        ),
    }


def _fixture(cursor=2):
    records = [_record(index) for index in range(cursor)]
    tensors = {
        "observations": {
            "actor": torch.zeros((H, N, 44), dtype=torch.float32),
            "critic": torch.zeros((H, N, 50), dtype=torch.float32),
        },
        "actions": torch.zeros((H, N, 10), dtype=torch.float32),
        "rewards": torch.zeros((H, N, 1), dtype=torch.float32),
        "dones": torch.zeros((H, N, 1), dtype=torch.uint8),
        "values": torch.zeros((H, N, 1), dtype=torch.float32),
        "actions_log_prob": torch.zeros((H, N, 1), dtype=torch.float32),
        "returns": torch.zeros((H, N, 1), dtype=torch.float32),
        "advantages": torch.zeros((H, N, 1), dtype=torch.float32),
        "distribution_params": None
        if cursor == 0
        else (
            torch.zeros((H, N, 10), dtype=torch.float32),
            torch.zeros((H, N, 10), dtype=torch.float32),
        ),
    }
    for index, record in enumerate(records):
        for name in ("actor", "critic"):
            tensors["observations"][name][index] = record["pre_action_observations"][
                name
            ]
        tensors["actions"][index] = record["raw_actions"]
        tensors["values"][index] = record["pre_action_values"]
        tensors["actions_log_prob"][index, :, 0] = record["raw_actions_log_prob"]
        tensors["rewards"][index, :, 0] = record["learner_reward"]
        tensors["dones"][index, :, 0] = (record["terminated"] | record["timed_out"]).to(
            torch.uint8
        )
        if cursor:
            for stored, item in zip(
                tensors["distribution_params"], record["distribution_params"]
            ):
                stored[index] = item
    value = {
        "protocol": storage_evidence.PROTOCOL,
        "worlds": N,
        "horizon": H,
        "cursor": cursor,
        "tensors": tensors,
        **transition.FALSE_FLAGS,
    }
    return value, records


@pytest.fixture(autouse=True)
def _cpu_checker(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)


@pytest.mark.parametrize("cursor", [0, 2, H])
def test_synthetic_cpu_storage_receipt_binds_empty_partial_and_full_cursors(cursor):
    value, records = _fixture(cursor)
    result = storage_evidence.check(value, records)
    assert result["cursor"] == cursor
    assert result["records_bound"] == cursor
    assert result["raw_unclipped_actions_bound"] is True
    assert result["zero_returns_and_advantages"] is True
    assert result["unused_tail_zero"] is True
    assert all(result[name] is False for name in transition.FALSE_FLAGS)


def test_synthetic_cpu_raw_action_values_above_one_are_retained():
    value, records = _fixture(1)
    assert value["tensors"]["actions"][0].max() > 1.0
    assert torch.equal(value["tensors"]["actions"][0], records[0]["raw_actions"])
    storage_evidence.check(value, records)


@pytest.mark.parametrize("cursor", [1, 2])
def test_installed_stock_cpu_rollout_storage_slots_bind_to_synthetic_records(cursor):
    assert version("rsl-rl-lib") == "5.0.1"
    records = [_record(index) for index in range(cursor)]
    initial = TensorDict(
        {
            "actor": torch.zeros((N, 44), dtype=torch.float32),
            "critic": torch.zeros((N, 50), dtype=torch.float32),
        },
        batch_size=[N],
    )
    stock = RolloutStorage(
        training_type="rl",
        num_envs=N,
        num_transitions_per_env=H,
        obs=initial,
        actions_shape=(10,),
        device="cpu",
    )
    for record in records:
        transition_item = RolloutStorage.Transition()
        transition_item.observations = TensorDict(
            record["pre_action_observations"], batch_size=[N]
        )
        transition_item.actions = record["raw_actions"]
        transition_item.rewards = record["learner_reward"]
        transition_item.dones = (record["terminated"] | record["timed_out"]).to(
            torch.uint8
        )
        transition_item.values = record["pre_action_values"]
        transition_item.actions_log_prob = record["raw_actions_log_prob"]
        transition_item.distribution_params = record["distribution_params"]
        stock.add_transition(transition_item)

    stock_tensors = {
        "observations": {
            name: stock.observations[name] for name in ("actor", "critic")
        },
        "actions": stock.actions,
        "rewards": stock.rewards,
        "dones": stock.dones,
        "values": stock.values,
        "actions_log_prob": stock.actions_log_prob,
        "returns": stock.returns,
        "advantages": stock.advantages,
        "distribution_params": stock.distribution_params,
    }
    owned = storage_evidence._owned_tree(stock_tensors, allow_cuda=False, clone=True)
    value = {
        "protocol": storage_evidence.PROTOCOL,
        "worlds": N,
        "horizon": H,
        "cursor": stock.step,
        "tensors": owned,
        **transition.FALSE_FLAGS,
    }
    result = storage_evidence.check(value, records)
    assert type(stock) is RolloutStorage
    assert stock.step == cursor == result["records_bound"]
    assert result["unused_tail_zero"] is True
    assert torch.equal(
        stock.actions[:cursor],
        torch.stack([record["raw_actions"] for record in records]),
    )
    assert all(result[name] is False for name in transition.FALSE_FLAGS)


def test_cpu_tree_copy_owns_nested_tuple_tensors_and_preserves_tuple_layout():
    source = {"nested": (torch.arange(4, dtype=torch.float32), [torch.ones(2)])}
    retained = storage_evidence._owned_tree(source, allow_cuda=False, clone=True)
    assert type(retained["nested"]) is tuple
    assert retained["nested"][0].data_ptr() != source["nested"][0].data_ptr()
    assert retained["nested"][1][0].data_ptr() != source["nested"][1][0].data_ptr()
    source["nested"][0][0] = 99
    assert retained["nested"][0][0] == 0


def test_capture_checks_lease_before_storage_validation(monkeypatch):
    events = []

    def lease(fd):
        events.append(("lease", fd))

    def initialized():
        events.append(("initialized", None))
        return True

    def device_count():
        events.append(("device_count", None))
        return 1

    def current_device():
        events.append(("current_device", None))
        return 0

    def inspect(_storage):
        events.append(("storage", None))
        raise ValueError("synthetic storage guard")

    monkeypatch.setattr(storage_evidence.training, "inherited_lease", lease)
    monkeypatch.setattr(torch.cuda, "is_initialized", initialized)
    monkeypatch.setattr(torch.cuda, "device_count", device_count)
    monkeypatch.setattr(torch.cuda, "current_device", current_device)
    monkeypatch.setattr(storage_evidence, "_check_source_storage", inspect)
    with pytest.raises(ValueError, match="synthetic storage guard"):
        storage_evidence.capture(object(), lease_fd=17)
    assert events == [
        ("lease", 17),
        ("initialized", None),
        ("device_count", None),
        ("current_device", None),
        ("storage", None),
    ]


def test_capture_stops_at_invalid_lease_before_cuda_or_storage(monkeypatch):
    events = []

    def lease(_fd):
        events.append("lease")
        raise ValueError("invalid inherited lease")

    def cuda_access():
        events.append("cuda")
        return False

    def inspect(_storage):
        events.append("storage")
        raise AssertionError("storage inspected before lease")

    monkeypatch.setattr(storage_evidence.training, "inherited_lease", lease)
    monkeypatch.setattr(torch.cuda, "is_initialized", cuda_access)
    monkeypatch.setattr(storage_evidence, "_check_source_storage", inspect)
    with pytest.raises(ValueError, match="invalid inherited lease"):
        storage_evidence.capture(object(), lease_fd=17)
    assert events == ["lease"]


@pytest.mark.parametrize(
    "mutate, message",
    [
        (
            lambda value, records: value["tensors"]["actions"].__setitem__(
                (0, 0, 0), 9
            ),
            "stored raw actions",
        ),
        (
            lambda value, records: value["tensors"]["actions_log_prob"].__setitem__(
                (0, 0, 0), 9
            ),
            "stored raw actions_log_prob",
        ),
        (
            lambda value, records: value["tensors"]["rewards"].__setitem__(
                (0, 0, 0), 9
            ),
            "stored raw rewards",
        ),
        (
            lambda value, records: value["tensors"]["dones"].__setitem__((0, 0, 0), 2),
            "binary stored done",
        ),
        (
            lambda value, records: value["tensors"]["advantages"].__setitem__(
                (0, 0, 0), 1
            ),
            "returns and advantages",
        ),
        (
            lambda value, records: value["tensors"]["returns"].__setitem__(
                (0, 0, 0), 1
            ),
            "returns and advantages",
        ),
        (
            lambda value, records: value["tensors"]["values"].__setitem__(
                (H - 1, 0, 0), 1
            ),
            "unused storage tail",
        ),
        (
            lambda value, records: value["tensors"]["observations"][
                "actor"
            ].__setitem__((0, 0, 0), float("nan")),
            "finite storage evidence",
        ),
        (
            lambda value, records: value.__setitem__(
                "native_transition_qualified", True
            ),
            "non-admitting CUDA64",
        ),
        (
            lambda value, records: value.__setitem__("cursor", True),
            "non-admitting CUDA64",
        ),
    ],
)
def test_rejects_storage_mismatch_and_invalid_payloads(mutate, message):
    value, records = _fixture(2)
    value = deepcopy(value)
    mutate(value, records)
    with pytest.raises(ValueError, match=message):
        storage_evidence.check(value, records)


def test_rejects_record_cursor_and_gaussian_parameter_mismatches():
    value, records = _fixture(1)
    wrong_cursor = deepcopy(value)
    wrong_cursor["cursor"] = 2
    with pytest.raises(ValueError, match="cursor matches retained records"):
        storage_evidence.check(wrong_cursor, records)
    wrong_param = deepcopy(value)
    wrong_param["tensors"]["distribution_params"][0][0, 0, 0] += 1
    with pytest.raises(ValueError, match="stored Gaussian parameters"):
        storage_evidence.check(wrong_param, records)
    absent_params = deepcopy(value)
    absent_params["tensors"]["distribution_params"] = None
    with pytest.raises(ValueError, match="stored Gaussian parameter pair"):
        storage_evidence.check(absent_params, records)


@pytest.mark.parametrize("bad", ["rank", "broadcast"])
def test_rejects_rank_or_expanded_storage_tensor(bad):
    value, records = _fixture(1)
    if bad == "rank":
        value["tensors"]["actions"] = torch.zeros((H, N, 10, 1))
    else:
        value["tensors"]["actions"] = torch.zeros((1, N, 10)).expand(H, N, 10)
    with pytest.raises(
        ValueError,
        match="dense supported storage evidence tensor|storage evidence tensor layout: actions",
    ):
        storage_evidence.check(value, records)


def test_zero_cursor_requires_absent_params_and_no_records():
    value, records = _fixture(0)
    storage_evidence.check(value, records)
    value["tensors"]["distribution_params"] = (
        torch.zeros((H, N, 10)),
        torch.zeros((H, N, 10)),
    )
    with pytest.raises(ValueError, match="empty cursor has no Gaussian storage"):
        storage_evidence.check(value, records)


def test_rejects_unsupported_tensor_and_excessive_or_unsafe_tree_shapes():
    sparse = torch.sparse_coo_tensor(torch.tensor([[0]]), torch.tensor([1.0]), (1,))
    with pytest.raises(ValueError, match="dense supported storage evidence tensor"):
        storage_evidence._owned_tree({"x": sparse}, allow_cuda=False, clone=False)
    deep = 0
    for _ in range(26):
        deep = [deep]
    with pytest.raises(ValueError, match="bounded storage evidence tree"):
        storage_evidence._owned_tree(deep, allow_cuda=False, clone=False)
