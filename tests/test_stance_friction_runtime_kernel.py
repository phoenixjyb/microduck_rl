"""CPU-only ABI and bank-equivalence checks for the runtime kernel component."""

import inspect
import os

import numpy as np
import pytest
import warp as wp

from mujoco_warp._src import constraint as frozen_constraint
from mjlab_microduck import stance_friction_prefix_cpu_fixture as prefix_fixture
from mjlab_microduck import stance_friction_row_cpu_fixture as row_fixture
from mjlab_microduck.stance_friction_runtime_kernel import ascending_friction_dof


EXPECTED_CASES = frozenset(
    {
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
    }
)
CASE_NAMES = tuple(prefix_fixture.predeclared_prefix_fixture_values())
BANK_NAMES = tuple(prefix_fixture._BANK_DTYPES)
NV = row_fixture.NV


def _inputs(values):
    return prefix_fixture._wp_inputs(
        tuple(
            values[name]
            for name in (
                "frictionloss",
                "qvel",
                "invweight",
                "solref",
                "solimp",
                "timestep",
            )
        )
    )


def _launch_runtime(inputs, bank, worlds, capacity):
    frictionloss, qvel, invweight, solref, solimp, timestep = inputs
    wp.launch(
        ascending_friction_dof,
        dim=(worlds, NV),
        inputs=[
            NV,
            timestep,
            0,
            solref,
            solimp,
            frictionloss,
            invweight,
            False,
            qvel,
            capacity,
            capacity * NV,
        ],
        outputs=[bank[name] for name in BANK_NAMES],
        device="cpu",
        block_dim=256,
        record_tape=False,
    )


def _bank_bytes(bank):
    return {name: bank[name].numpy().tobytes() for name in BANK_NAMES}


def _input_bytes(inputs):
    return tuple(array.numpy().tobytes() for array in inputs)


def test_kernel_signature_matches_frozen_26_argument_abi_exactly():
    assert len(CASE_NAMES) == 10 and set(CASE_NAMES) == EXPECTED_CASES
    expected = inspect.signature(frozen_constraint._friction_dof.func)
    actual = inspect.signature(ascending_friction_dof.func)
    assert tuple(actual.parameters) == tuple(expected.parameters)
    assert tuple(actual.parameters.values()) == tuple(expected.parameters.values())
    assert len(actual.parameters) == 26
    assert ascending_friction_dof.options["enable_backward"] is False


@pytest.mark.parametrize("case_name", CASE_NAMES)
def test_two_dimensional_kernel_matches_old_one_dimensional_candidate_full_bank(
    case_name,
):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    values = prefix_fixture.predeclared_prefix_fixture_values()[case_name]
    worlds = values["qvel"].shape[0]
    capacity = values["njmax"]
    initial = prefix_fixture._prefix_values(worlds, capacity, values["initial_nefc"])
    old_bank = prefix_fixture._new_wp_bank(initial)
    runtime_bank = prefix_fixture._new_wp_bank(initial)
    old_inputs = _inputs(values)
    runtime_inputs = _inputs(values)
    old_input_bits = _input_bytes(old_inputs)
    runtime_input_bits = _input_bytes(runtime_inputs)
    assert old_input_bits == runtime_input_bits

    with wp.ScopedDevice("cpu"):
        prefix_fixture._launch_candidate(old_inputs, old_bank, worlds, capacity)
        _launch_runtime(runtime_inputs, runtime_bank, worlds, capacity)

    assert _bank_bytes(old_bank) == _bank_bytes(runtime_bank)
    expected_input_bits = tuple(
        np.asarray(values[name]).tobytes()
        for name in (
            "frictionloss",
            "qvel",
            "invweight",
            "solref",
            "solimp",
            "timestep",
        )
    )
    assert _input_bytes(old_inputs) == old_input_bits == expected_input_bits
    assert _input_bytes(runtime_inputs) == runtime_input_bits == expected_input_bits


def test_sparse_invocation_is_an_explicit_noop_and_preserves_every_output_bank():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    values = prefix_fixture.predeclared_prefix_fixture_values()["broadcast"]
    worlds = values["qvel"].shape[0]
    capacity = values["njmax"]
    initial = prefix_fixture._prefix_values(worlds, capacity, values["initial_nefc"])
    bank = prefix_fixture._new_wp_bank(initial)
    before = _bank_bytes(bank)
    arrays = _inputs(values)
    frictionloss, qvel, invweight, solref, solimp, timestep = arrays

    with wp.ScopedDevice("cpu"):
        wp.launch(
            ascending_friction_dof,
            dim=(worlds, NV),
            inputs=[
                NV,
                timestep,
                0,
                solref,
                solimp,
                frictionloss,
                invweight,
                True,
                qvel,
                capacity,
                capacity * NV,
            ],
            outputs=[bank[name] for name in BANK_NAMES],
            device="cpu",
            block_dim=256,
            record_tape=False,
        )

    assert _bank_bytes(bank) == before
    assert ascending_friction_dof.options["enable_backward"] is False
