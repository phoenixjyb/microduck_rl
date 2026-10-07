import numpy as np
import pytest

from mjlab_microduck import stance_friction_row_cpu_fixture as fixture


def _values(nworld=2, rows=1):
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
    timestep = np.full((rows,), 0.002, dtype="<f4")
    qvel = np.arange(nworld * 20, dtype="<f4").reshape(nworld, 20) / 100
    return dict(
        frictionloss=frictionloss,
        qvel=qvel,
        invweight=invweight,
        solref=solref,
        solimp=solimp,
        timestep=timestep,
        njmax=32,
    )


def _run(**changes):
    values = _values()
    values.update(changes)
    return fixture.run_cpu_fixture(**values)


def test_broadcast_friction_rows_match_original_by_dof_with_full_dense_fields():
    values = _values(nworld=3, rows=1)
    values["frictionloss"][0, [1, 6, 19]] = [0.3, 0.8, 1.1]
    result = fixture.run_cpu_fixture(**values)
    assert result["device"] == "cpu"
    assert result["worlds"] == 3 and result["njmax"] == 32
    assert result["launches"] == {
        "original_friction_dof": 1,
        "ascending_dense_candidate": 1,
    }
    assert result["candidate_dof_order_ascending"]
    assert result["active_addressed_rows_exact_without_overflow"]
    assert [row["dof"] for row in result["per_world"][0]["candidate_rows"]] == [
        1,
        6,
        19,
    ]
    assert all(
        row["baseline_candidate_addressed_rows_equal"] for row in result["per_world"]
    )
    assert all(
        all(row["baseline_candidate_addressed_field_exact"].values())
        for row in result["per_world"]
    )
    row = result["per_world"][0]["candidate_rows"][0]
    assert row["J_u32"] == [0x3F800000 if i == 1 else 0 for i in range(20)]
    assert row["type"] == 1 and row["dof"] == 1
    assert all(value is False for value in result["flags"].values())


def test_per_world_parameter_rows_and_zero_or_negative_loss_are_supported():
    values = _values(nworld=4, rows=4)
    values["frictionloss"][0, [0, 4]] = [0.1, 0.2]
    values["frictionloss"][1, [1, 5, 9]] = [0.3, 0.4, 0.5]
    values["frictionloss"][2, 12] = -1.0
    values["frictionloss"][3, 18] = 0.0
    values["qvel"][:, [0, 1, 4, 5, 9, 18]] += np.arange(4, dtype="<f4")[:, None]
    result = fixture.run_cpu_fixture(**values)
    assert [row["candidate_counts"]["nf"] for row in result["per_world"]] == [
        2,
        3,
        0,
        0,
    ]
    assert [
        [entry["dof"] for entry in row["candidate_rows"]] for row in result["per_world"]
    ] == [[0, 4], [1, 5, 9], [], []]
    assert result["active_addressed_rows_exact_without_overflow"]


def test_each_model_field_uses_its_own_per_world_broadcast_modulo():
    values = _values(nworld=4, rows=1)
    values["frictionloss"][0, [2, 8]] = [0.3, 0.9]
    values["invweight"] = np.full((4, 20), 0.75, dtype="<f4")
    values["invweight"][1] *= 1.2
    values["solimp"] = np.repeat(values["solimp"], 4, axis=0)
    values["solimp"][2, :, 0] = 0.85
    values["timestep"] = np.array([0.002], dtype="<f4")
    result = fixture.run_cpu_fixture(**values)
    assert all(
        row["baseline_candidate_addressed_rows_equal"] for row in result["per_world"]
    )
    assert all(row["candidate_counts"]["nefc"] == 2 for row in result["per_world"])


def test_all_twenty_dofs_and_njmax_capacity_are_retained():
    values = _values(nworld=1, rows=1)
    values["frictionloss"][:] = np.linspace(0.01, 1.0, 20, dtype="<f4")
    values["njmax"] = 20
    result = fixture.run_cpu_fixture(**values)
    rows = result["per_world"][0]["candidate_rows"]
    assert len(rows) == 20
    assert [row["dof"] for row in rows] == list(range(20))
    assert result["per_world"][0]["candidate_counts"] == {
        "nf": 20,
        "nefc": 20,
        "stored_rows": 20,
    }
    assert result["active_addressed_rows_exact_without_overflow"]


def test_signed_zero_bits_are_retained_in_dense_row_velocity():
    values = _values(nworld=1, rows=1)
    values["frictionloss"][0, 6] = 0.5
    values["qvel"][0, 6] = np.asarray([0x80000000], dtype="<u4").view("<f4")[0]
    result = fixture.run_cpu_fixture(**values)
    for side in ("original_rows", "candidate_rows"):
        row = result["per_world"][0][side][0]
        assert row["dof"] == 6
        assert row["vel_u32"] == [0x80000000]
    assert result["per_world"][0]["baseline_candidate_addressed_field_exact"]["vel_u32"]


def test_overflow_keeps_both_observed_subsets_and_never_qualifies():
    values = _values(nworld=2, rows=2)
    values["frictionloss"][:, [1, 3, 7, 11, 18]] = 0.5
    values["njmax"] = 2
    result = fixture.run_cpu_fixture(**values)
    assert result["overflow_negative"]
    assert not result["active_addressed_rows_exact_without_overflow"]
    assert result["fixture_decision"] == "dense-rows-negative-or-overflow"
    for row in result["per_world"]:
        assert row["original_counts"] == {
            "nf": 5,
            "nefc": 5,
            "stored_rows": 2,
        }
        assert row["candidate_counts"] == row["original_counts"]
        assert len(row["original_rows"]) == len(row["candidate_rows"]) == 2
        assert [entry["dof"] for entry in row["candidate_rows"]] == [1, 3]
        if not row["baseline_candidate_address_multiset_equal"]:
            assert not row["baseline_candidate_addressed_rows_equal"]
    assert all(value is False for value in result["flags"].values())


@pytest.mark.parametrize(
    "mutation",
    ["wrong-dtype", "wrong-shape", "nonfinite", "wrong-worlds", "bad-njmax"],
)
def test_fixture_rejects_invalid_numpy_values_and_caps(mutation):
    values = _values()
    if mutation == "wrong-dtype":
        values["qvel"] = values["qvel"].astype(np.float64)
    elif mutation == "wrong-shape":
        values["qvel"] = np.zeros((2, 19), dtype="<f4")
    elif mutation == "nonfinite":
        values["frictionloss"][0, 0] = np.nan
    elif mutation == "wrong-worlds":
        values["qvel"] = np.zeros((5, 20), dtype="<f4")
    elif mutation == "bad-njmax":
        values["njmax"] = 0
    with pytest.raises(ValueError):
        fixture.run_cpu_fixture(**values)


def test_fixture_rejects_nonempty_cuda_visibility(monkeypatch):
    values = _values()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="CUDA must be hidden"):
        fixture.run_cpu_fixture(**values)


@pytest.mark.parametrize("pin", ["kernel", "helper", "candidate", "launch"])
def test_fixture_rejects_changed_installed_function_pins(monkeypatch, pin):
    values = _values()
    if pin == "kernel":
        monkeypatch.setattr(fixture.constraint, "_friction_dof", object())
    elif pin == "helper":
        monkeypatch.setattr(fixture.constraint, "_efc_row", object())
    elif pin == "candidate":
        monkeypatch.setattr(fixture, "_ascending_dense_friction", object())
    else:
        monkeypatch.setattr(fixture.wp, "launch", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="pinned original friction kernel"):
        fixture.run_cpu_fixture(**values)


def test_finite_inputs_do_not_accept_nonfinite_constructed_rows():
    values = _values()
    values["frictionloss"][0, 6] = 0.5
    values["solref"][..., 1] = 0.0
    with pytest.raises(ValueError, match="finite complete active dense row fields"):
        fixture.run_cpu_fixture(**values)


def test_caller_arrays_are_preserved_and_actual_counts_match_positive_losses():
    values = _values()
    values["frictionloss"][0, [6, 7, 15]] = [0.1, 0.2, 0.3]
    before = {k: v.tobytes() for k, v in values.items() if isinstance(v, np.ndarray)}
    result = fixture.run_cpu_fixture(**values)
    assert result["counts_and_addresses_complete"]
    assert all(row["counts_and_addresses_complete"] for row in result["per_world"])
    assert all(values[k].tobytes() == raw for k, raw in before.items())
    for name, raw in before.items():
        retained = result["fixture_inputs"][name]
        assert retained["shape"] == list(values[name].shape)
        assert retained["dtype"] == "<f4"
        assert np.asarray(retained["u32"], dtype="<u4").tobytes() == raw
    assert len(result["source_binding_before"]["fixture_module_sha256"]) == 64


def test_undeclared_cpu_platform_is_rejected(monkeypatch):
    monkeypatch.setattr(fixture.platform, "python_version", lambda: "3.12.99")
    with pytest.raises(ValueError, match="frozen CPU Python"):
        fixture.run_cpu_fixture(**_values())


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
    ],
)
def test_predeclared_tiny_matrix_cases_keep_complete_component_decisions(case):
    result = fixture.run_cpu_fixture(**fixture.predeclared_fixture_values()[case])
    assert result["counts_and_addresses_complete"]
    assert result["candidate_dof_order_ascending"]
    assert result["active_addressed_rows_exact_without_overflow"] == (
        case != "overflow"
    )
    assert result["overflow_negative"] == (case == "overflow")
    assert all(value is False for value in result["flags"].values())


def test_predeclared_matrix_keeps_all_cases_and_fresh_independent_inputs():
    first = fixture.predeclared_fixture_values()
    second = fixture.predeclared_fixture_values()
    first["broadcast"]["qvel"][:] = 123
    assert not np.array_equal(first["broadcast"]["qvel"], second["broadcast"]["qvel"])
    report = fixture.run_predeclared_cpu_matrix()
    assert set(report["cases"]) == set(first) == set(report["case_expectations_met"])
    assert report["component_expectations_met"]
    assert all(report["case_expectations_met"].values())
    assert all(value is False for value in report["flags"].values())


def test_matrix_cli_keeps_complete_exclusive_report(tmp_path):
    import json

    output = tmp_path / "report.json"
    fixture.main(["--output", str(output)])
    raw = output.read_bytes()
    report = json.loads(raw)
    assert report["component_expectations_met"]
    assert len(report["cases"]) == 8
    assert 0 < len(raw) <= fixture.OUTPUT_CAP
    with pytest.raises(ValueError, match="new exclusive fixture output"):
        fixture.main(["--output", str(output)])
    assert output.read_bytes() == raw


@pytest.mark.parametrize("target", ["directory-symlink", "file-symlink"])
def test_matrix_cli_rejects_symlink_output_before_launch(tmp_path, monkeypatch, target):
    actual = tmp_path / "actual"
    actual.mkdir()
    link = tmp_path / "link"
    if target == "directory-symlink":
        link.symlink_to(actual, target_is_directory=True)
        output = link / "report.json"
    else:
        output = link
        output.symlink_to(actual / "absent.json")
    monkeypatch.setattr(
        fixture,
        "run_predeclared_cpu_matrix",
        lambda: pytest.fail("must reject before CPU launches"),
    )
    with pytest.raises((ValueError, OSError)):
        fixture.main(["--output", str(output)])


@pytest.mark.parametrize("target", ["ancestor-symlink", "parent-traversal"])
def test_matrix_cli_rejects_ancestor_redirection_before_launch(
    tmp_path, monkeypatch, target
):
    actual = tmp_path / "actual"
    (actual / "nested").mkdir(parents=True)
    link = tmp_path / "link"
    link.symlink_to(actual, target_is_directory=True)
    output = (
        link / "nested" / "report.json"
        if target == "ancestor-symlink"
        else actual / "nested" / ".." / "report.json"
    )
    monkeypatch.setattr(
        fixture,
        "run_predeclared_cpu_matrix",
        lambda: pytest.fail("must reject before CPU launches"),
    )
    with pytest.raises((ValueError, OSError)):
        fixture.main(["--output", str(output)])
    assert not list(actual.rglob("*.json"))


def test_matrix_cli_held_parent_cannot_be_redirected_during_computation(
    tmp_path, monkeypatch
):
    parent = tmp_path / "parent"
    retained = tmp_path / "retained"
    elsewhere = tmp_path / "elsewhere"
    parent.mkdir()
    elsewhere.mkdir()

    def replace_path():
        parent.rename(retained)
        parent.symlink_to(elsewhere, target_is_directory=True)
        return {"component_expectations_met": True, "fixture_only": True}

    monkeypatch.setattr(fixture, "run_predeclared_cpu_matrix", replace_path)
    fixture.main(["--output", str(parent / "report.json")])
    assert (retained / "report.json").is_file()
    assert not list(elsewhere.iterdir())


def test_matrix_cli_rejects_oversize_payload_before_file_creation(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        fixture,
        "run_predeclared_cpu_matrix",
        lambda: {
            "component_expectations_met": True,
            "oversize": "x" * fixture.OUTPUT_CAP,
        },
    )
    with pytest.raises(ValueError, match="bounded complete CPU matrix"):
        fixture.main(["--output", str(tmp_path / "report.json")])
    assert not (tmp_path / "report.json").exists()


def test_matrix_cli_retains_expectation_failure_and_returns_error(
    tmp_path, monkeypatch
):
    import json

    monkeypatch.setattr(
        fixture,
        "run_predeclared_cpu_matrix",
        lambda: {"component_expectations_met": False, "fixture_only": True},
    )
    output = tmp_path / "negative.json"
    with pytest.raises(ValueError, match="expectation failure; retained report"):
        fixture.main(["--output", str(output)])
    assert json.loads(output.read_bytes())["component_expectations_met"] is False
