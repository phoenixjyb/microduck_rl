import numpy as np
import pytest

from mjlab_microduck import stance_friction_prefix_cpu_fixture as fixture


def _values(nworld=2, rows=1, capacity=16):
    frictionloss = np.zeros((rows, 20), dtype="<f4")
    invweight = np.full((rows, 20), 0.75, dtype="<f4")
    solref = np.empty((rows, 20, 2), dtype="<f4")
    solref[..., 0] = 0.02
    solref[..., 1] = 1.0
    solimp = np.empty((rows, 20, 5), dtype="<f4")
    solimp[..., 0] = 0.9
    solimp[..., 1] = 0.95
    solimp[..., 2] = 0.01
    solimp[..., 3] = 0.5
    solimp[..., 4] = 2.0
    return {
        "frictionloss": frictionloss,
        "qvel": np.arange(nworld * 20, dtype="<f4").reshape(nworld, 20) / 100,
        "invweight": invweight,
        "solref": solref,
        "solimp": solimp,
        "timestep": np.full((rows,), 0.002, dtype="<f4"),
        "njmax": capacity,
        "initial_nefc": np.zeros((nworld,), dtype="<i4"),
    }


def _run(**changes):
    values = _values()
    values.update(changes)
    return fixture.run_prefix_cpu_fixture(**values)


def _row_values(values, world=0, dofs=(1, 4, 9)):
    values["frictionloss"][world % values["frictionloss"].shape[0], list(dofs)] = 0.5


def test_nonzero_mixed_prefixes_preserve_prefix_and_compare_all_rows_by_address():
    values = _values(nworld=4, rows=1, capacity=12)
    values["initial_nefc"] = np.array([0, 2, 5, 8], dtype="<i4")
    values["frictionloss"][0, [1, 4, 9]] = 0.5
    report = fixture.run_prefix_cpu_fixture(**values)
    assert report["device"] == "cpu"
    assert report["initial_nf"]["i32"] == [0, 0, 0, 0]
    assert report["counts_and_addresses_complete"]
    assert report["all_prefix_rows_preserved"]
    assert report["all_inactive_suffix_poison_preserved"]
    assert report["all_dense_sparse_scratch_unchanged"]
    assert report["candidate_rows_ascending"]
    assert report["candidate_replay_full_bank_bit_identical"]
    assert report["candidate_replay_addressed_exact"]
    assert report["original_candidate_addressed_exact"]
    assert report["component_exact_without_overflow"]
    assert [row["expected_counts"] for row in report["worlds"]] == [
        {"nf": 3, "nefc": 3, "stored_rows": 3},
        {"nf": 3, "nefc": 5, "stored_rows": 5},
        {"nf": 3, "nefc": 8, "stored_rows": 8},
        {"nf": 3, "nefc": 11, "stored_rows": 11},
    ]
    assert all(value is False for value in report["flags"].values())


def test_empty_friction_with_full_prefix_and_poisoned_suffix_is_exact():
    values = _values(nworld=2, rows=1, capacity=8)
    values["initial_nefc"] = np.array([8, 4], dtype="<i4")
    report = fixture.run_prefix_cpu_fixture(**values)
    assert report["component_exact_without_overflow"]
    assert report["worlds"][0]["expected_counts"] == {
        "nf": 0,
        "nefc": 8,
        "stored_rows": 8,
    }
    assert report["worlds"][0]["candidate0"]["rows"] == []
    assert report["all_inactive_suffix_poison_preserved"]


def test_exact_fill_and_prefix_boundary_append_are_not_overflow():
    values = _values(nworld=1, rows=1, capacity=5)
    values["initial_nefc"][:] = 2
    values["frictionloss"][0, [0, 7, 19]] = 0.4
    report = fixture.run_prefix_cpu_fixture(**values)
    assert not report["overflow_negative"]
    assert report["worlds"][0]["expected_counts"] == {
        "nf": 3,
        "nefc": 5,
        "stored_rows": 5,
    }
    assert [row["dof"] for row in report["worlds"][0]["candidate0"]["rows"]] == [
        0,
        7,
        19,
    ]
    assert report["component_exact_without_overflow"]


def test_overflow_keeps_full_counts_and_visible_subset_but_never_qualifies():
    values = _values(nworld=2, rows=2, capacity=3)
    values["initial_nefc"] = np.array([1, 2], dtype="<i4")
    values["frictionloss"][:, [2, 6, 10]] = 0.6
    report = fixture.run_prefix_cpu_fixture(**values)
    assert report["overflow_negative"]
    assert report["overflow_decision"] == "overflow-negative-no-qualification"
    assert not report["component_exact_without_overflow"]
    assert report["worlds"][0]["expected_counts"] == {
        "nf": 3,
        "nefc": 4,
        "stored_rows": 3,
    }
    assert report["worlds"][1]["expected_counts"] == {
        "nf": 3,
        "nefc": 5,
        "stored_rows": 3,
    }
    assert all(value is False for value in report["flags"].values())


def test_signed_zero_and_nan_poison_bits_survive_untouched_prefix_and_suffix():
    values = _values(nworld=1, rows=1, capacity=8)
    values["initial_nefc"][:] = 2
    report = fixture.run_prefix_cpu_fixture(**values)
    before = report["runs"]["candidate0"]["before"]
    after = report["runs"]["candidate0"]["after"]
    # Empty friction leaves every dense row bit untouched, including NaN
    # payloads and the negative-zero poison in the seeded fields.
    assert before == after
    for field in ("pos", "vel"):
        assert 0x80000000 in before[field]["bits"]
    for field in ("J", "margin", "D", "aref", "frictionloss"):
        assert any((bit & 0x7F800000) == 0x7F800000 for bit in before[field]["bits"])


@pytest.mark.parametrize(
    "mutation", ["dtype", "shape", "negative", "too-large", "noncontiguous"]
)
def test_invalid_prefix_is_rejected(mutation):
    values = _values()
    if mutation == "dtype":
        values["initial_nefc"] = values["initial_nefc"].astype(np.int64)
    elif mutation == "shape":
        values["initial_nefc"] = np.zeros((2, 1), dtype="<i4")
    elif mutation == "negative":
        values["initial_nefc"][0] = -1
    elif mutation == "too-large":
        values["initial_nefc"][0] = values["njmax"] + 1
    else:
        values["initial_nefc"] = np.zeros((4,), dtype="<i4")[::2]
    with pytest.raises(ValueError, match="initial_nefc"):
        fixture.run_prefix_cpu_fixture(**values)


@pytest.mark.parametrize("mutation", ["wrong-f32-dtype", "bad-shape", "nonfinite"])
def test_base_fixture_input_validation_is_preserved(mutation):
    values = _values()
    if mutation == "wrong-f32-dtype":
        values["qvel"] = values["qvel"].astype(np.float64)
    elif mutation == "bad-shape":
        values["qvel"] = np.zeros((2, 19), dtype="<f4")
    else:
        values["frictionloss"][0, 0] = np.inf
    with pytest.raises(ValueError):
        fixture.run_prefix_cpu_fixture(**values)


def test_nonfinite_constructed_active_row_is_rejected():
    values = _values(nworld=1, rows=1, capacity=8)
    values["frictionloss"][0, 6] = 0.5
    values["solref"][..., 1] = 0.0
    with pytest.raises(ValueError, match="finite complete active friction row"):
        fixture.run_prefix_cpu_fixture(**values)


def test_fixture_inputs_are_not_mutated_and_all_three_full_banks_are_retained():
    values = _values(nworld=2, rows=1, capacity=9)
    values["initial_nefc"] = np.array([1, 3], dtype="<i4")
    _row_values(values)
    before = {
        name: value.tobytes()
        for name, value in values.items()
        if isinstance(value, np.ndarray)
    }
    report = fixture.run_prefix_cpu_fixture(**values)
    assert all(values[name].tobytes() == raw for name, raw in before.items())
    assert set(report["runs"]) == {"original", "candidate0", "candidate1"}
    for run in report["runs"].values():
        assert set(run) == {
            "inputs_before",
            "inputs_after",
            "allocations",
            "before",
            "after",
        }
        assert run["inputs_before"] == run["inputs_after"]
        assert all(row["device"] == "cpu" for row in run["inputs_before"].values())
        assert len(run["allocations"]["inputs"]) == 6
        assert len(run["allocations"]["outputs"]) == len(fixture._BANK_DTYPES)
        assert set(run["before"]) == set(fixture._BANK_DTYPES)
        assert set(run["after"]) == set(fixture._BANK_DTYPES)
        assert run["before"]["J"]["shape"] == [2, 9, 20]
    assert len(report["prefix_module_sha256_before"]) == 64
    assert report["prefix_module_sha256_before"] == report["prefix_module_sha256_after"]


def test_original_fixture_source_pin_is_enforced(monkeypatch):
    values = _values()
    monkeypatch.setattr(fixture.base.constraint, "_friction_dof", object())
    with pytest.raises(ValueError, match="pinned original friction kernel"):
        fixture.run_prefix_cpu_fixture(**values)


def test_borrowed_helper_pin_is_enforced(monkeypatch):
    values = _values()
    monkeypatch.setattr(fixture.base, "_copies", lambda *args: args)
    with pytest.raises(ValueError, match="pinned borrowed dense-row helper _copies"):
        fixture.run_prefix_cpu_fixture(**values)


def test_report_is_bounded_and_all_admission_flags_are_false():
    report = _run()
    import json

    payload = (
        json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()
    assert 0 < len(payload) <= fixture.OUTPUT_CAP
    assert report["all_actual_inputs_unchanged"]
    assert report["all_run_buffers_independent"]
    assert report["qualification"] is False
    assert report["runtime_cause_proven"] is False
    assert report["native_qualified"] is False
    assert report["full_window_qualified"] is False
    assert report["training_authorized"] is False
    assert report["physical_acceptance"] is False


def test_maximum_fixture_remains_within_complete_report_cap():
    values = _values(nworld=4, rows=4, capacity=32)
    values["initial_nefc"] = np.array([0, 1, 16, 32], dtype="<i4")
    values["frictionloss"][:] = 0.5
    report = fixture.run_prefix_cpu_fixture(**values)
    import json

    payload = (
        json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()
    assert len(payload) <= fixture.OUTPUT_CAP
    assert report["overflow_negative"]
    assert report["counts_and_addresses_complete"]


@pytest.mark.parametrize(
    "helper", ["_launch_candidate", "_new_wp_bank", "_input_record", "_bank_record"]
)
def test_prefix_control_and_capture_helpers_are_pinned(monkeypatch, helper):
    monkeypatch.setattr(fixture, helper, lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="pinned prefix helper"):
        _run()


def test_literal_component_bounds_are_not_mutable_admission_knobs(monkeypatch):
    monkeypatch.setattr(fixture, "OUTPUT_CAP", 1024)
    with pytest.raises(ValueError, match="literal CPU component bounds"):
        _run()


def _decoded_after(report, side="candidate0"):
    result = {}
    for name, packet in report["runs"][side]["after"].items():
        value = np.asarray(packet["bits"], dtype=packet["dtype"]).reshape(
            packet["shape"]
        )
        result[name] = (
            value.view("<f4") if fixture._BANK_DTYPES[name] == "u32" else value
        )
    return result


def test_independent_capture_rejects_equal_but_wrong_dense_jacobian():
    values = _values(nworld=1, capacity=8)
    values["frictionloss"][0, 6] = 0.5
    report = fixture.run_prefix_cpu_fixture(**values)
    arrays = _decoded_after(report)
    arrays["J"][0, 0] = 0
    with pytest.raises(ValueError, match="one-hot addressed Jacobian"):
        fixture._capture_active(arrays, 1, 8, np.array([0], dtype="<i4"))


def test_capture_rejects_unbounded_counters_before_row_iteration():
    report = fixture.run_prefix_cpu_fixture(**_values(nworld=1, capacity=8))
    arrays = _decoded_after(report)
    arrays["nf"][0] = 1000
    arrays["nefc"][0] = 1000
    with pytest.raises(ValueError, match="bounded prefix/friction counters"):
        fixture._capture_active(arrays, 1, 8, np.array([0], dtype="<i4"))


def test_full_prefix_overflow_retains_all_proposals_without_visible_friction():
    values = _values(nworld=1, capacity=3)
    values["initial_nefc"][0] = 3
    values["frictionloss"][0, [1, 4, 9]] = 0.5
    report = fixture.run_prefix_cpu_fixture(**values)
    assert (
        report["overflow_negative"] and not report["component_exact_without_overflow"]
    )
    assert report["counts_and_addresses_complete"]
    for side in ("original", "candidate0", "candidate1"):
        assert report["worlds"][0][side]["rows"] == []
        assert report["worlds"][0][side]["counts"] == {
            "nf": 3,
            "nefc": 6,
            "stored_rows": 3,
        }
        assert report["runs"][side]["before"]["J"] == report["runs"][side]["after"]["J"]


def test_all_63_live_cpu_allocation_ranges_are_nonoverlapping():
    report = _run()
    records = [
        r
        for run in report["runs"].values()
        for group in run["allocations"].values()
        for r in group.values()
    ]
    assert len(records) == 63
    ranges = sorted((r["pointer"], r["pointer"] + r["bytes"]) for r in records)
    assert all(a[1] <= b[0] for a, b in zip(ranges, ranges[1:]))
    assert all(r["device"] == "cpu" and r["bytes"] > 0 for r in records)


def test_active_appended_velocity_preserves_signed_zero_in_both_candidates():
    values = _values(nworld=1, capacity=8)
    values["initial_nefc"][0] = 2
    values["frictionloss"][0, 6] = 0.5
    values["qvel"][0, 6] = np.array([0x80000000], dtype="<u4").view("<f4")[0]
    report = fixture.run_prefix_cpu_fixture(**values)
    for side in ("original", "candidate0", "candidate1"):
        row = report["worlds"][0][side]["rows"][0]
        assert row["row"] == 2 and row["dof"] == 6 and row["vel"] == [0x80000000]
    assert report["component_exact_without_overflow"]


@pytest.mark.parametrize(
    "case",
    [
        "empty",
        "broadcast",
        "per-world",
        "mixed-broadcast",
        "all-dofs",
        "signed-zero",
        "direct-solref",
        "overflow",
        "append-exact-fill",
        "maximum",
    ],
)
def test_predeclared_prefix_matrix_has_expected_complete_component_result(case):
    report = fixture.run_prefix_cpu_fixture(
        **fixture.predeclared_prefix_fixture_values()[case]
    )
    overflow = case in ("overflow", "maximum")
    assert report["overflow_negative"] == overflow
    assert report["component_exact_without_overflow"] == (not overflow)
    assert report["all_actual_inputs_unchanged"]
    assert report["all_prefix_rows_preserved"]
    assert report["all_inactive_suffix_poison_preserved"]
    assert report["all_dense_sparse_scratch_unchanged"]
    assert report["counts_and_addresses_complete"]
    assert report["candidate_replay_full_bank_bit_identical"]
    assert all(x is False for x in report["flags"].values())


def test_replaced_exported_entrypoint_is_rejected_by_independently_held_checker(
    monkeypatch,
):
    checker, code = fixture._OWN_HELPERS["_own_source_binding"]
    monkeypatch.setattr(fixture, "run_prefix_cpu_fixture", lambda *args, **kwargs: {})
    assert checker.__code__ is code
    with pytest.raises(ValueError, match="pinned prefix helper run_prefix_cpu_fixture"):
        checker()


@pytest.mark.parametrize("mutation", ["replacement", "code"])
def test_actual_entrypoint_rejects_changed_binding_checker_before_using_it(
    monkeypatch, mutation
):
    def fake():
        pytest.fail("mutated checker must not execute")

    if mutation == "replacement":
        monkeypatch.setattr(fixture, "_own_source_binding", fake)
    else:
        monkeypatch.setattr(fixture._own_source_binding, "__code__", fake.__code__)
    with pytest.raises(
        ValueError, match="pinned prefix orchestration and binding checker"
    ):
        _run()


@pytest.mark.parametrize(
    "helper", ["_need", "_module_sha256", "_check_base_helpers", "_base_source_binding"]
)
@pytest.mark.parametrize("mutation", ["replacement", "code"])
def test_guard_helpers_cannot_bypass_the_frozen_checker(monkeypatch, helper, mutation):
    def fake(*args, **kwargs):
        pytest.fail("mutated guard helper must not execute")

    original = getattr(fixture, helper)
    if mutation == "replacement":
        monkeypatch.setattr(fixture, helper, fake)
    else:
        monkeypatch.setattr(original, "__code__", fake.__code__)
    with pytest.raises(ValueError, match="pinned prefix"):
        _run()
    checker, checker_code = fixture._OWN_HELPERS["_own_source_binding"]
    assert checker.__code__ is checker_code
    with pytest.raises(ValueError, match="pinned prefix helper " + helper):
        checker()
