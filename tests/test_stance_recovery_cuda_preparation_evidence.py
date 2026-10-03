"""Retained-record consistency tests; current host/services are SYNTHETIC.

Optional authentic old bytes do not authenticate a current native host or CUDA
execution. Production has no caller-supplied hash, source or path override.
"""

from copy import deepcopy
from pathlib import Path

import pytest

from mjlab_microduck import stance_recovery_cuda_preparation_evidence as evidence

SOURCE = "b" * 40


@pytest.fixture
def retained():
    native = evidence.probe.execution.ROOT
    candidates = [native]
    repo = Path(evidence.__file__).resolve().parents[2]
    candidates.extend(
        sorted(
            (repo / "artifacts/retained").glob("cuda64-reader-fixture-48653c58121d.*")
        )
    )
    for root in candidates:
        external = root / "artifacts/tools/cuda64-verified-48653c58121d/receipt.json"
        raw0 = (
            root
            / "artifacts/evaluations/stance-wsl-cuda64-policy-preparation-48653c58121d/seed-653.pt"
        )
        if external.is_file() and raw0.is_file():
            raw = evidence.probe._read_file(external, evidence.probe.JSON_LIMIT)
            assert evidence.probe.digest(raw) == evidence.EXTERNAL_SHA256
            return root, evidence.probe.parse_json(raw)
    if native == Path("/home/yanbo/work/microduck_rl-stance-replication-20260930"):
        pytest.fail("configured native host is missing closed CUDA64 evidence")
    pytest.skip(
        "optional authentic retained CUDA64 bytes absent from external checkout"
    )


def current_binding(record):
    value = deepcopy(record["native_prerequisites"])
    value["source"]["source"] = SOURCE
    value["current_context"]["source_identity"]["source"] = SOURCE
    return value


def test_only_two_source_ids_change_without_mutating_inputs(retained):
    _, record = retained
    original = record["native_prerequisites"]
    before = deepcopy(original)
    current = current_binding(record)
    assert (
        evidence.check_source_delta(original, current, evaluator_source=SOURCE) is True
    )
    assert original == before
    assert record["source"] == evidence.ARTIFACT_SOURCE


@pytest.mark.parametrize(
    "field",
    [
        "gap_inventory",
        "gap_receipts",
        "preflight_failure_binding",
        "source",
        "current_context",
    ],
)
def test_any_non_source_prerequisite_drift_refused(retained, field):
    _, record = retained
    current = current_binding(record)
    current[field]["synthetic_unexpected_drift"] = True
    with pytest.raises(ValueError, match="only evaluator source IDs"):
        evidence.check_source_delta(
            record["native_prerequisites"], current, evaluator_source=SOURCE
        )


@pytest.mark.parametrize("bad_source", [evidence.ARTIFACT_SOURCE, "b" * 39, True])
def test_source_schema_and_old_namespace_refused(retained, bad_source):
    _, record = retained
    with pytest.raises(ValueError):
        evidence.check_source_delta(
            record["native_prerequisites"],
            current_binding(record),
            evaluator_source=bad_source,
        )


def test_actual_external_record_consistency_is_not_new_admission(retained):
    _, record = retained
    assert evidence._validate_external(record) is True
    assert all(record[k] is False for k in evidence.probe.PREPARATION_FALSE_FLAGS)


@pytest.mark.parametrize("flag", sorted(evidence.probe.PREPARATION_FALSE_FLAGS))
def test_each_external_admission_flag_refused(retained, flag):
    _, original = retained
    record = deepcopy(original)
    record[flag] = True
    with pytest.raises(ValueError, match="non-admitting preparation"):
        evidence._validate_external(record)


@pytest.mark.parametrize(
    "field,value",
    [
        ("native_tests_passed", True),
        ("optimizer_steps", 1),
        ("action_samples", 1),
        ("cuda_math_replayed", True),
        ("decision", "training-admitted"),
        ("service_invocations", {}),
    ],
)
def test_external_counts_protocol_and_invocations_refused(retained, field, value):
    _, original = retained
    record = deepcopy(original)
    record[field] = value
    with pytest.raises(ValueError, match="non-admitting preparation"):
        evidence._validate_external(record)


def _synthetic_native(monkeypatch, retained):
    root, record = retained
    calls = []
    monkeypatch.setattr(evidence.probe.execution, "ROOT", root)
    monkeypatch.setattr(evidence.probe.host, "ROOT", root)
    monkeypatch.setattr(
        evidence.probe.execution,
        "PROFILE",
        evidence.probe.execution.select(evidence.probe.execution.WSL),
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(evidence.probe.torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(
        evidence.probe.training_smoke,
        "inherited_lease",
        lambda fd: calls.append(("SYNTHETIC_LEASE", fd)),
    )
    states = {
        "MainPID": "0",
        "ActiveState": "inactive",
        "NRestarts": "0",
        "ExecMainStatus": "0",
        "Result": "success",
        "InvocationID": "",
    }
    monkeypatch.setattr(evidence.probe.host, "read", lambda *args: states[args[-2]])
    current = {
        "launch_binding": current_binding(record),
        "raw_parent": b"SYNTHETIC_CURRENT_PARENT_CALLER_BYTES",
    }
    monkeypatch.setattr(evidence.probe, "_native_prerequisites", lambda source: current)
    return calls, current, states


def test_whole_original_records_rescore_with_synthetic_current_context(
    monkeypatch, retained
):
    calls, current, _ = _synthetic_native(monkeypatch, retained)
    result = evidence.checked(SOURCE, lease_fd=17)
    assert calls == [("SYNTHETIC_LEASE", 17), ("SYNTHETIC_LEASE", 17)]
    assert result["raw_parent"] == current["raw_parent"]
    binding = result["launch_binding"]
    assert binding["artifact_source"] == evidence.ARTIFACT_SOURCE
    assert binding["evaluator_source"] == SOURCE
    assert [s["seed"] for s in binding["scores"]] == [653, 659]
    assert all(binding[k] is False for k in evidence.probe.PREPARATION_FALSE_FLAGS)
    assert binding["cuda_math_replayed"] is False


def test_bad_lease_precedes_any_cuda_probe(monkeypatch):
    def refuse(_fd):
        raise ValueError("SYNTHETIC missing lease")

    monkeypatch.setattr(evidence.probe.training_smoke, "inherited_lease", refuse)
    monkeypatch.setattr(
        evidence.probe.torch.cuda,
        "is_initialized",
        lambda: pytest.fail("CUDA access before lease refusal"),
    )
    with pytest.raises(ValueError, match="missing lease"):
        evidence.checked(SOURCE, lease_fd=17)


def test_visible_cuda_reader_refused_before_record_reads(monkeypatch):
    monkeypatch.setattr(
        evidence.probe.training_smoke, "inherited_lease", lambda fd: None
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(
        evidence.probe,
        "_read_file",
        lambda *args: pytest.fail("record read before CPU-only guard"),
    )
    with pytest.raises(ValueError, match="CUDA-hidden"):
        evidence.checked(SOURCE, lease_fd=17)


@pytest.mark.parametrize(
    "field,value",
    [
        ("Result", "exit-code"),
        ("MainPID", "100"),
        ("NRestarts", "1"),
        ("InvocationID", "f" * 32),
    ],
)
def test_failed_restarted_or_replaced_original_tool_refused(monkeypatch, field, value):
    states = {
        "MainPID": "0",
        "ActiveState": "inactive",
        "NRestarts": "0",
        "ExecMainStatus": "0",
        "Result": "success",
        "InvocationID": "",
    }
    states[field] = value
    monkeypatch.setattr(evidence.probe.host, "read", lambda *args: states[args[-2]])
    with pytest.raises(ValueError, match="original successful terminal"):
        evidence._completed_tool("tests")


def test_original_whole_record_hash_refused_before_schema(monkeypatch, retained):
    _synthetic_native(monkeypatch, retained)
    original = evidence.probe._read_file

    def changed(path, limit):
        raw = original(path, limit)
        return (
            raw + b" "
            if str(path).endswith("cuda64-verified-48653c58121d/receipt.json")
            else raw
        )

    monkeypatch.setattr(evidence.probe, "_read_file", changed)
    with pytest.raises(ValueError, match="whole original external"):
        evidence.checked(SOURCE, lease_fd=17)
