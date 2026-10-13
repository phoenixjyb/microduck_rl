"""CPU-only orchestration tests for the bounded v8 route measurement CLI."""

from __future__ import annotations

import json

import pytest

from mjlab_microduck import integrated_turn_foundation as base
from mjlab_microduck import v8_route_qualification as qualification


SOURCE = "a" * 40
STOCK_ROOT_NAME = "v8-actor-foundation-722e7a6a-20261013"
CAUTIOUS_ROOT_NAME = qualification.CAUTIOUS_RUN
INITIAL_SHA = "80084814096f99a56a11f78d6e0526c99261b1d0eaae3a57da79517733b61163"
STOCK_FINAL_SHA = "4631f81d36f7cc846cbed42efdedb9a16e2a40b2d1cadbcbc2834b43e016d7bb"
CAUTIOUS_FINAL_SHA = "820c86e28b3d05d8bd292b63777b78cfa0b4657d69cc028ad17f8afe1ab91738"


def test_checkpoint_arms_are_three_exact_pinned_locations(tmp_path, monkeypatch):
    monkeypatch.setattr(qualification.base, "ROOT", tmp_path)
    monkeypatch.setattr(qualification.learning, "validate_training_receipt",
                        lambda *_args, **_kwargs: pytest.fail("legacy arms do not validate learning receipts"))

    initial_path, initial_sha = qualification.checkpoint("initial")
    stock_path, stock_sha = qualification.checkpoint("stock")

    assert initial_path == (tmp_path / "artifacts/training" / STOCK_ROOT_NAME / "benchmark/initial.pt")
    assert initial_sha == INITIAL_SHA
    assert stock_path == (tmp_path / "artifacts/training" / STOCK_ROOT_NAME / "foundation/model_999.pt")
    assert stock_sha == STOCK_FINAL_SHA


def test_cautious_arm_requires_completed_verified_receipt_and_exact_final_sha(tmp_path, monkeypatch):
    monkeypatch.setattr(qualification.base, "ROOT", tmp_path)
    calls = []

    def verified_receipt(root, mode, source, *, profile):
        calls.append((root, mode, source, profile))
        return {"checkpoint_sha256": CAUTIOUS_FINAL_SHA}

    monkeypatch.setattr(qualification.learning, "validate_training_receipt", verified_receipt)
    path, digest = qualification.checkpoint("cautious")

    assert path == (tmp_path / "artifacts/training" / CAUTIOUS_ROOT_NAME / "foundation/model_999.pt")
    assert digest == CAUTIOUS_FINAL_SHA
    assert calls == [(tmp_path / "artifacts/training" / CAUTIOUS_ROOT_NAME,
                      "foundation", qualification.CAUTIOUS_SOURCE, "cautious")]

    monkeypatch.setattr(qualification.learning, "validate_training_receipt",
                        lambda *_args, **_kwargs: {"checkpoint_sha256": "0" * 64})
    with pytest.raises(ValueError, match="complete sealed cautious final999"):
        qualification.checkpoint("cautious")


@pytest.mark.parametrize("arm", ["", "best", "final", "stock-999", "cautious-998", None])
def test_checkpoint_enumeration_rejects_every_unlisted_arm(arm):
    with pytest.raises(ValueError, match="predeclared route checkpoint arm"):
        qualification.checkpoint(arm)


class _TypedCheckpoint:
    instances = []

    def __init__(self, path, expected_sha256):
        self.path = path
        self.sha256 = expected_sha256
        self.instances.append(self)

    def require_initial_seed(self):
        self.initial_seed_checked = True

    def provenance(self):
        return {"backend": "frozen-v8-ppo-actor-only", "sha256": self.sha256,
                "ppo_resume": False, "physical_motion_authorized": False}


def _install_route_mocks(monkeypatch, tmp_path, *, arm="stock", fail_summary=False,
                         wrong_digest_at=None):
    digest = {"initial": INITIAL_SHA, "stock": STOCK_FINAL_SHA,
              "cautious": CAUTIOUS_FINAL_SHA}[arm]
    checkpoint_path = tmp_path / f"{arm}-checkpoint.pt"
    _TypedCheckpoint.instances = []
    monkeypatch.setattr(qualification, "FrozenV8Checkpoint", _TypedCheckpoint)
    monkeypatch.setattr(qualification, "checkpoint", lambda selected: (checkpoint_path, digest))
    calls = {"health": [], "evaluate": [], "summaries": [], "decisions": [], "writes": []}
    legacy_gate = {"decision": "foundation-ready-for-next-experiment", "failures": []}
    route_gate = {"decision": "route-qualified-simulation-only", "failures": [], "cases": 12}

    def fake_health():
        calls["health"].append(len(calls["health"]))

    def fake_evaluate(path, seed, speed, yaw, health, **kwargs):
        case_index = len(calls["evaluate"])
        calls["evaluate"].append((path, seed, speed, yaw, health, kwargs))
        return {
            "seed": seed, "speed": speed, "yaw": yaw, "complete": True,
            "steps": 240, "worlds": 8, "terminal_worlds": [],
            "checkpoint_sha256": "0" * 64 if case_index == wrong_digest_at else digest,
            "response_diagnostics": {"torque_p99": .4},
            "gait_path_trace": {"case_index": case_index},
        }

    def fake_summarize(case, trace):
        calls["summaries"].append((case, trace))
        if fail_summary:
            raise ValueError("synthetic route qualification failure")
        return {"protocol": "mock-route-summary-v1", "seed": case["seed"],
                "speed": case["speed"], "yaw": case["yaw"], "qualified": True}

    def fake_route_decision(rows):
        calls["decisions"].append(("route", rows))
        return route_gate

    def fake_legacy_decision(rows):
        calls["decisions"].append(("legacy", rows))
        return legacy_gate

    real_write_new = base.write_new

    def observed_write_new(path, data):
        calls["writes"].append(path.name)
        return real_write_new(path, data)

    monkeypatch.setattr(qualification.base, "evaluate_case", fake_evaluate)
    monkeypatch.setattr(qualification.base, "write_new", observed_write_new)
    monkeypatch.setattr(qualification.route, "summarize_case", fake_summarize)
    monkeypatch.setattr(qualification.route, "decision", fake_route_decision)
    monkeypatch.setattr(qualification.legacy, "decision", fake_legacy_decision)
    return calls, digest, checkpoint_path, legacy_gate, route_gate, fake_health


def test_evaluate_runs_ordered_fixed_matrix_with_actor_only_checkpoint_and_separate_gates(
        tmp_path, monkeypatch):
    output = tmp_path / "stock"
    output.mkdir()
    calls, digest, checkpoint_path, legacy_gate, route_gate, health = _install_route_mocks(
        monkeypatch, tmp_path)

    result = qualification.evaluate("stock", output, health)

    identities = [(seed, speed, yaw) for seed in base.SEEDS for speed, yaw in base.CASES]
    assert [(call[1], call[2], call[3]) for call in calls["evaluate"]] == identities
    assert calls["health"] == list(range(12))
    assert len(_TypedCheckpoint.instances) == 1
    actor = _TypedCheckpoint.instances[0]
    assert actor.path == checkpoint_path and actor.sha256 == digest
    assert all(call[0] == checkpoint_path and call[4] is health for call in calls["evaluate"])
    assert all(call[5] == {
        "diagnostics": True, "yaw_trace": True, "gait_trace": True,
        "gait_path_trace": True, "frozen_v8_checkpoint": actor,
    } for call in calls["evaluate"])
    assert len(calls["summaries"]) == 12
    assert all(trace is case["gait_path_trace"] for case, trace in calls["summaries"])
    assert [decision[0] for decision in calls["decisions"]] == ["route", "legacy"]
    assert calls["decisions"][0][1] == result["cases"]
    assert calls["decisions"][1][1] == [entry[0] for entry in calls["summaries"]]
    assert result["legacy_gate"] is legacy_gate
    assert result["route_gate"] is route_gate
    assert result["checkpoint_sha256"] == digest
    assert result["inference_backend"] == actor.provenance()
    assert result["policy_acceptance"] is False
    assert result["physical_motion_authorized"] is False

    expected_names = []
    for index, (seed, _speed, _yaw) in enumerate(identities):
        case_index = index % len(base.CASES)
        raw_name = f"s{seed}-c{case_index}.json"
        expected_names.extend((raw_name, raw_name.removesuffix(".json") + "-route.json"))
        assert json.loads((output / raw_name).read_text())["checkpoint_sha256"] == digest
        assert json.loads((output / (raw_name.removesuffix(".json") + "-route.json")).read_text()) == result["cases"][index]
    assert calls["writes"] == expected_names


def test_initial_arm_checks_seeded_baseline_before_route_rollouts(tmp_path, monkeypatch):
    output = tmp_path / "initial"
    output.mkdir()
    calls, _digest, _path, _legacy_gate, _route_gate, health = _install_route_mocks(
        monkeypatch, tmp_path, arm="initial")

    qualification.evaluate("initial", output, health)

    assert _TypedCheckpoint.instances[0].initial_seed_checked is True
    assert len(calls["evaluate"]) == len(calls["summaries"]) == 12


def test_case_sha_is_checked_before_any_route_summary_or_raw_case_write(tmp_path, monkeypatch):
    output = tmp_path / "stock"
    output.mkdir()
    calls, _digest, _path, _legacy_gate, _route_gate, health = _install_route_mocks(
        monkeypatch, tmp_path, wrong_digest_at=0)

    with pytest.raises(ValueError, match="route case bound to exact predeclared actor bytes"):
        qualification.evaluate("stock", output, health)

    assert len(calls["evaluate"]) == 1
    assert calls["summaries"] == []
    assert calls["writes"] == []
    assert list(output.iterdir()) == []


def test_raw_case_is_retained_before_route_qualification_failure(tmp_path, monkeypatch):
    output = tmp_path / "stock"
    output.mkdir()
    calls, digest, _path, _legacy_gate, _route_gate, health = _install_route_mocks(
        monkeypatch, tmp_path, fail_summary=True)
    seed = base.SEEDS[0]
    raw_path = output / f"s{seed}-c0.json"
    route_path = output / f"s{seed}-c0-route.json"

    with pytest.raises(ValueError, match="synthetic route qualification failure"):
        qualification.evaluate("stock", output, health)

    assert raw_path.is_file() and not route_path.exists()
    assert json.loads(raw_path.read_text())["checkpoint_sha256"] == digest
    assert calls["writes"] == [raw_path.name]
    assert len(calls["summaries"]) == 1

