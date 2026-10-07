"""Self-contained numerical checks using CPU-generated, CUDA-shaped samples."""

from copy import deepcopy
import ast
from pathlib import Path
import struct

import pytest

from mjlab_microduck import stance_friction_prefix_cpu_fixture as fixture
from mjlab_microduck import stance_friction_prefix_cuda_numerical as checker
from mjlab_microduck.stance_friction_prefix_cuda_numerical import (
    PROTOCOL,
    audit_cases,
)


def _sample_cases():
    cases = {}
    for name, value in fixture.predeclared_prefix_fixture_values().items():
        old = fixture.run_prefix_cpu_fixture(
            value["frictionloss"],
            value["qvel"],
            value["invweight"],
            value["solref"],
            value["solimp"],
            value["timestep"],
            value["njmax"],
            value["initial_nefc"],
        )
        runs = {}
        for side, run in old["runs"].items():
            runs[side] = {
                "before": run["before"],
                "after": run["after"],
                "inputs_before": {
                    key: {
                        field: run["inputs_before"][key][field]
                        for field in ("shape", "dtype", "u32")
                    }
                    for key in old["input_snapshot"]
                },
                "inputs_after": {
                    key: {
                        field: run["inputs_after"][key][field]
                        for field in ("shape", "dtype", "u32")
                    }
                    for key in old["input_snapshot"]
                },
            }
        cases[name] = {
            "device": "cuda:0",
            "input_snapshot": old["input_snapshot"],
            "initial_nefc": old["initial_nefc"],
            "initial_nf": old["initial_nf"],
            "runs": runs,
        }
    return cases


@pytest.fixture(scope="module")
def cases():
    return _sample_cases()


def test_audits_ten_raw_numerical_cases_without_admission(cases):
    report = audit_cases(cases)
    assert report["protocol"] == PROTOCOL
    assert report["case_count"] == 10
    assert all(
        report["cases"][name]["component_exact_without_overflow"]
        for name in set(report["cases"]) - {"overflow", "maximum"}
    )
    assert report["cases"]["overflow"]["overflow_negative"] is True
    assert report["cases"]["maximum"]["overflow_negative"] is True
    assert all(value is False for value in report["flags"].values())
    assert report["runtime_cause_proven"] is False
    assert report["native_qualified"] is False
    assert report["full_window_qualified"] is False
    assert report["training_authorized"] is False
    assert report["physical_acceptance"] is False


def test_signed_zero_remains_an_exact_input_bit(cases):
    report = audit_cases(cases)
    assert report["cases"]["signed-zero"]["component_exact_without_overflow"] is True
    assert cases["signed-zero"]["input_snapshot"]["qvel"]["u32"][6] == 0x80000000


def test_numerical_mismatch_is_an_explicit_negative_decision(cases):
    changed = deepcopy(cases)
    changed["broadcast"]["runs"]["candidate0"]["after"]["nf"]["bits"][0] += 1
    result = audit_cases(changed)["cases"]["broadcast"]
    assert result["component_exact_without_overflow"] is False
    assert result["counts_and_addresses_complete"] is False
    assert result["fixture_decision"] == "dense-prefix-negative-or-overflow"


def test_original_rows_compare_by_address_without_assuming_order(cases):
    changed = deepcopy(cases)
    bank = changed["broadcast"]["runs"]["original"]["after"]
    for field in ("type", "id", "pos", "margin", "D", "vel", "aref", "frictionloss"):
        bank[field]["bits"][0], bank[field]["bits"][1] = (
            bank[field]["bits"][1],
            bank[field]["bits"][0],
        )
    jacobian = bank["J"]["bits"]
    jacobian[:20], jacobian[20:40] = jacobian[20:40], jacobian[:20]
    report = audit_cases(changed)["cases"]["broadcast"]
    assert report["original_candidate_addressed_exact"] is True
    assert report["component_exact_without_overflow"] is True


def test_malformed_bank_or_prefix_is_rejected(cases):
    changed = deepcopy(cases)
    changed["broadcast"]["runs"]["candidate0"]["after"]["J"]["bits"].pop()
    with pytest.raises(ValueError):
        audit_cases(changed)

    changed = deepcopy(cases)
    changed["broadcast"]["initial_nefc"]["i32"][0] += 1
    with pytest.raises(ValueError):
        audit_cases(changed)


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_requires_exact_ten_case_names(cases, mutation):
    changed = deepcopy(cases)
    if mutation == "missing":
        changed.pop("empty")
    else:
        changed["extra"] = deepcopy(changed["empty"])
    with pytest.raises(ValueError):
        audit_cases(changed)


@pytest.mark.parametrize(
    ("target", "field", "value"),
    [
        ("case", "device", "cpu"),
        ("snapshot", "shape", [1, 19]),
        ("snapshot", "dtype", "<f8"),
        ("snapshot", "word", True),
        ("input-record", "word", True),
        ("input-record", "complete", None),
        ("prefix", "shape", [2]),
        ("before-bank", "bit", None),
    ],
)
def test_rejects_invalid_schema_device_shape_wordtype_and_pins(
    cases, target, field, value
):
    changed = deepcopy(cases)
    case = changed["broadcast"]
    if target == "case":
        case[field] = value
    elif target == "snapshot":
        row = case["input_snapshot"]["frictionloss"]
        if field == "word":
            row["u32"][0] = value
        else:
            row[field] = value
    elif target == "input-record":
        row = case["runs"]["original"]["inputs_before"]["qvel"]
        if field == "word":
            row["u32"][0] = value
        else:
            row["u32"].pop()
    elif target == "prefix":
        case["initial_nefc"][field] = value
    else:
        case["runs"]["original"]["before"]["id"]["bits"][0] += 1
    with pytest.raises(ValueError):
        audit_cases(changed)


@pytest.mark.parametrize(
    "mutation",
    [
        "poison-suffix",
        "scratch",
        "counters",
        "missing-dof",
        "duplicate-dof",
        "onehot",
        "nonfinite",
        "velocity-bits",
        "frictionloss-bits",
        "position",
        "margin",
        "candidate-ascending",
        "candidate-replay",
        "candidate-full-bank",
        "original-field",
    ],
)
def test_numerical_mutations_are_explicit_negative_decisions(cases, mutation):
    changed = deepcopy(cases)
    case = changed["broadcast"]
    dofs = [
        dof
        for dof, bits in enumerate(case["input_snapshot"]["frictionloss"]["u32"][:20])
        if struct.unpack("<f", struct.pack("<I", bits))[0] > 0.0
    ]
    assert len(dofs) >= 2
    candidate0 = case["runs"]["candidate0"]["after"]
    candidate1 = case["runs"]["candidate1"]["after"]

    def swap_rows(bank):
        for field in (
            "type",
            "id",
            "pos",
            "margin",
            "D",
            "vel",
            "aref",
            "frictionloss",
        ):
            bank[field]["bits"][0], bank[field]["bits"][1] = (
                bank[field]["bits"][1],
                bank[field]["bits"][0],
            )
        bank["J"]["bits"][:20], bank["J"]["bits"][20:40] = (
            bank["J"]["bits"][20:40],
            bank["J"]["bits"][:20],
        )

    if mutation == "poison-suffix":
        index = len(dofs)
        candidate0["id"]["bits"][index] += 1
        candidate1["id"]["bits"][index] += 1
    elif mutation == "scratch":
        candidate0["row_adr"]["bits"][0] += 1
    elif mutation == "counters":
        candidate0["nefc"]["bits"][0] += 1
    elif mutation == "missing-dof":
        candidate0["id"]["bits"][0] = -1
    elif mutation == "duplicate-dof":
        candidate0["id"]["bits"][1] = candidate0["id"]["bits"][0]
    elif mutation == "onehot":
        candidate0["J"]["bits"][dofs[0]] ^= 1
    elif mutation == "nonfinite":
        candidate0["D"]["bits"][0] = 0x7FC00001
    elif mutation == "velocity-bits":
        candidate0["vel"]["bits"][0] ^= 1
    elif mutation == "frictionloss-bits":
        candidate0["frictionloss"]["bits"][0] ^= 1
    elif mutation == "position":
        candidate0["pos"]["bits"][0] = 0x3F800000
    elif mutation == "margin":
        candidate0["margin"]["bits"][0] = 0x3F800000
    elif mutation == "candidate-ascending":
        swap_rows(candidate0)
        swap_rows(candidate1)
    elif mutation == "candidate-replay":
        candidate1["aref"]["bits"][0] ^= 1
    elif mutation == "candidate-full-bank":
        candidate1["efc_nnz"]["bits"][0] += 1
    else:
        original = case["runs"]["original"]["after"]
        original["D"]["bits"][0] ^= 1

    report = audit_cases(changed)["cases"]["broadcast"]
    assert report["component_exact_without_overflow"] is False
    assert report["fixture_decision"] == "dense-prefix-negative-or-overflow"


def test_overflow_may_have_a_different_valid_visible_original_subset(cases):
    changed = deepcopy(cases)
    case = changed["overflow"]
    original = case["runs"]["original"]["after"]
    cap = 2
    world, row, replacement_dof = 0, 1, 3
    assert replacement_dof in [1, 3, 7, 11, 18]
    original["type"]["bits"][world * cap + row] = 1
    original["id"]["bits"][world * cap + row] = replacement_dof
    jac_offset = (world * cap + row) * 20
    original["J"]["bits"][jac_offset : jac_offset + 20] = [
        0x3F800000 if dof == replacement_dof else 0 for dof in range(20)
    ]
    original["vel"]["bits"][world * cap + row] = case["input_snapshot"]["qvel"]["u32"][
        replacement_dof
    ]
    original["frictionloss"]["bits"][world * cap + row] = case["input_snapshot"][
        "frictionloss"
    ]["u32"][replacement_dof]
    original["pos"]["bits"][world * cap + row] = 0
    original["margin"]["bits"][world * cap + row] = 0

    report = audit_cases(changed)["cases"]["overflow"]
    assert report["overflow_negative"] is True
    assert report["original_candidate_addressed_exact"] is True
    assert report["component_exact_without_overflow"] is False
    assert report["fixture_decision"] == "dense-prefix-negative-or-overflow"


def test_numerical_checker_does_not_import_warp_or_producer():
    tree = ast.parse(Path(checker.__file__).read_text())
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )
    assert "warp" not in imported
    assert "mujoco_warp" not in imported
    assert all("cpu_fixture" not in name for name in imported)
