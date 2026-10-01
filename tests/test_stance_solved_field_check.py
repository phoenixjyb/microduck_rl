"""Finite-check equivalence fixtures, not simulation or performance evidence."""

import subprocess
import sys

import pytest
import torch

from mjlab_microduck import stance_solved_field_check as checks


def values():
    return {name: torch.arange(index + 1, dtype=torch.float32).reshape(1, -1)
            for index, name in enumerate(checks.FIELDS)}


def error(check, data):
    with pytest.raises(ValueError) as exc:
        check(data)
    return str(exc.value)


def test_finite_fields_and_input_bytes_are_unchanged():
    data = values()
    data["qpos"][0, 0] = -0.
    before = {name: value.view(torch.uint8).clone() for name, value in data.items()}
    assert checks.legacy_check(data) is checks.packed_check(data) is None
    assert all(torch.equal(before[name], value.view(torch.uint8)) for name, value in data.items())


@pytest.mark.parametrize("name", checks.FIELDS)
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_each_nonfinite_field_has_identical_error(name, bad):
    data = values()
    data[name][0, -1] = bad
    assert error(checks.legacy_check, data) == error(checks.packed_check, data) == (
        "nonfinite solved stance field: " + name)


def test_multiple_faults_retain_original_first_field():
    data = values()
    data["qvel"][0, 0] = float("nan")
    data["xquat"][0, 0] = float("inf")
    assert error(checks.packed_check, data) == error(checks.legacy_check, data)
    assert error(checks.packed_check, data).endswith("qvel")


def test_empty_and_strided_views_are_checked_without_mutation():
    data = values()
    data["qpos"] = torch.empty((0, 21), dtype=torch.float32)
    source = torch.arange(48, dtype=torch.float32).reshape(6, 8)
    data["qvel"] = source[:, ::2].T
    before = source.clone()
    checks.packed_check(data)
    assert torch.equal(source, before)
    source[0, 0] = float("inf")
    assert error(checks.packed_check, data) == error(checks.legacy_check, data)


def test_all_empty_fields_match_vacuous_legacy_check():
    data = {name: torch.empty(0) for name in checks.FIELDS}
    checks.legacy_check(data)
    checks.packed_check(data)


def test_inputs_are_read_again_each_call_never_cached():
    data = values()
    checks.packed_check(data)
    data["time"][0, 0] = float("nan")
    assert error(checks.packed_check, data).endswith("time")
    data["time"][0, 0] = 1.
    checks.packed_check(data)


@pytest.mark.parametrize("damage", ["missing", "extra", "reordered", "float64", "integer", "non_tensor", "sparse"])
def test_out_of_scope_input_rejected_by_both_checkers(damage):
    data = values()
    if damage == "missing":
        del data["qpos"]
    elif damage == "extra":
        data["other"] = torch.zeros(1)
    elif damage == "reordered":
        data = dict(reversed(list(data.items())))
    elif damage == "non_tensor":
        data["qpos"] = [0.]
    elif damage == "sparse":
        data["qpos"] = data["qpos"].to_sparse()
    else:
        data["qpos"] = data["qpos"].to(torch.float64 if damage == "float64" else torch.int64)
    assert error(checks.legacy_check, data) == error(checks.packed_check, data)


def test_mixed_device_and_byte_cap_rejected():
    data = values()
    data["qpos"] = torch.zeros(1, device="meta")
    assert "one solved-field device" in error(checks.packed_check, data)
    data = {name: torch.empty(checks.MAX_BYTES // 4, device="meta") for name in checks.FIELDS}
    assert "bounded" in error(checks.packed_check, data)


def test_backend_error_propagates(monkeypatch):
    def failure(*args, **kwargs):
        raise RuntimeError("synthetic allocation failure")
    monkeypatch.setattr(torch, "cat", failure)
    with pytest.raises(RuntimeError, match="allocation failure"):
        checks.packed_check(values())


def test_aggregate_failure_cannot_silently_pass_if_replay_passes(monkeypatch):
    monkeypatch.setattr(torch, "cat", lambda *args, **kwargs: torch.tensor([float("nan")]))
    with pytest.raises(RuntimeError, match="inconsistent packed"):
        checks.packed_check(values())


def test_import_is_stdlib_only_and_offers_no_runtime_hook():
    code = "import sys; from mjlab_microduck import stance_solved_field_check as c; assert not hasattr(c,'enable'); print(','.join(sorted(set(sys.modules)&{'torch','warp','mujoco','mjlab','bam','onnxruntime'})))"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert not result.stdout.strip()
