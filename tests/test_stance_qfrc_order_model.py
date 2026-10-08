"""Exact finite binary32 arithmetic and virtual captured-order contracts."""

from fractions import Fraction
from pathlib import Path
import os
import random
import struct
import subprocess
import sys

import pytest

from mjlab_microduck import stance_qfrc_order_model as model
from test_stance_solver_init_row_view import fixture, set_words


def bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def rational(word):
    c, e = model.finite_parts(word)
    return Fraction(c << e) if e >= 0 else Fraction(c, 1 << -e)


@pytest.mark.parametrize(
    "word",
    (
        0,
        0x80000000,
        1,
        0x80000001,
        0x007FFFFF,
        0x00800000,
        0x3F800000,
        0x7F7FFFFF,
        0xFF7FFFFF,
    ),
)
def test_decode_round_identity_at_edges(word):
    assert model.round_dyadic(*model.finite_parts(word), zero_sign=word >> 31) == word


def test_decode_round_identity_random_finite_words():
    rng = random.Random(2711)
    for _ in range(1500):
        word = rng.randrange(1 << 32)
        if (word >> 23) & 255 == 255:
            continue
        assert (
            model.round_dyadic(*model.finite_parts(word), zero_sign=word >> 31) == word
        )


def test_rounding_midpoints_both_sides_and_even_significands():
    rng = random.Random(2713)
    for _ in range(150):
        low = rng.randrange(1, 0x7F7FFFFE)
        a, b = rational(low), rational(low + 1)
        midpoint = (a + b) / 2
        # Independent exact midpoint construction, including rare subnormals.
        exponent = -(midpoint.denominator.bit_length() - 1)
        coefficient = midpoint.numerator
        even = low if low & 1 == 0 else low + 1
        assert model.round_dyadic(coefficient, exponent) == even
        assert model.round_dyadic(-coefficient, exponent) == even | 0x80000000
        assert model.round_dyadic((coefficient << 30) - 1, exponent - 30) == low
        assert model.round_dyadic((coefficient << 30) + 1, exponent - 30) == low + 1


def test_subnormal_ties_gradual_underflow_and_negative_zero():
    assert model.round_dyadic(1, -150) == 0
    assert model.round_dyadic(-1, -150) == 0x80000000
    assert model.round_dyadic(3, -150) == 2
    assert model.round_dyadic(-3, -150) == 0x80000002
    assert model.round_dyadic((1 << 24) - 1, -150) == 0x00800000
    assert model.multiply_bits(1, bits(0.5)) == 0
    assert model.multiply_bits(3, bits(0.5)) == 2


def test_normal_ties_and_carry():
    assert model.round_dyadic((1 << 24) + 1, 0) == bits(1 << 24)
    assert model.round_dyadic((1 << 24) + 3, 0) == bits((1 << 24) + 4)
    assert model.round_dyadic((1 << 25) - 1, -24) == bits(2)


def test_fma_does_not_round_product_or_use_double_rounding():
    a, b, c = 0x3F800001, 0x3F7FFFFE, bits(-1)
    assert model.add_bits(model.multiply_bits(a, b), c) == 0
    assert model.fma_bits(a, b, c) == model.round_dyadic(-1, -46)
    # Exact product can overflow while one-round fused result is finite.
    with pytest.raises(OverflowError):
        model.multiply_bits(0x7F7FFFFF, bits(2))
    assert model.fma_bits(0x7F7FFFFF, bits(2), 0xFF7FFFFF) == 0x7F7FFFFF


@pytest.mark.parametrize(
    "a,b,expected",
    ((0, 0, 0), (0, 0x80000000, 0), (0x80000000, 0x80000000, 0x80000000)),
)
def test_add_zero_signs(a, b, expected):
    assert model.add_bits(a, b) == expected
    assert model.add_bits(bits(-1), bits(1)) == 0


def test_multiply_and_fma_zero_signs():
    assert model.multiply_bits(0x80000000, bits(1)) == 0x80000000
    assert model.multiply_bits(0x80000000, bits(-1)) == 0
    assert model.fma_bits(0x80000000, bits(1), 0x80000000) == 0x80000000
    assert model.fma_bits(0x80000000, bits(1), 0) == 0


@pytest.mark.parametrize(
    "word", (0x7F800000, 0xFF800000, 0x7FC00001, True, -1, 1 << 32)
)
def test_nonfinite_or_nonliteral_inputs_refused(word):
    with pytest.raises(ValueError):
        model.finite_parts(word)


def test_overflow_and_bad_dyadic_limits_refused():
    with pytest.raises(OverflowError):
        model.add_bits(0x7F7FFFFF, 0x7F7FFFFF)
    with pytest.raises(OverflowError):
        model.round_dyadic((1 << 25) - 1, 103)
    for c, e in ((True, 0), (1, True), (1, -1025), (1 << 2049, 0)):
        with pytest.raises(ValueError):
            model.round_dyadic(c, e)


@pytest.mark.parametrize("mode", model.MODES)
def test_left_fold_order_sensitivity(mode):
    a, b, c = bits(1 << 24), bits(1), bits(-(1 << 24))
    one = bits(1)
    assert model.left_fold([(a, one), (b, one), (c, one)], mode) == bits(0)
    assert model.left_fold([(a, one), (c, one), (b, one)], mode) == bits(1)
    assert model.left_fold([], mode) == 0


def order_fixture():
    left, right = fixture()
    for bank in (left, right):
        set_words(bank, "efc.force", {0: 1, 1: 1, 2: 1})
    set_words(left, "efc.J", {0: 1 << 24, 20: 1, 40: -(1 << 24)})
    set_words(right, "efc.J", {0: 1 << 24, 20: -(1 << 24), 40: 1})
    set_words(right, "data.qfrc_constraint", {0: 1})
    return left, right


def test_paired_orders_reproduce_difference_without_mutation():
    left, right = order_fixture()
    before = (
        {k: v["raw"] for k, v in left.items()},
        {k: v["raw"] for k, v in right.items()},
    )
    result = model.compare_qfrc(left, right)
    assert result["active_rows"] == 3 and result["matched_term_pairs"] == 60
    assert result["changed_contact_row_offsets"] == 2
    assert result["observed_candidate_repeat"]["differing_words"] == 1
    for mode in model.MODES:
        r = result["models"][mode]
        assert r["candidate0_stored_vs_observed"]["exact"] is True
        assert r["candidate1_stored_vs_observed"]["exact"] is True
        assert r["virtual_common_order_repeat"]["exact"] is True
        assert r["stored_candidate_repeat"]["differing_words"] == 1
        assert r["stored_delta_mask_matches_observed"] is True
    assert before == (
        {k: v["raw"] for k, v in left.items()},
        {k: v["raw"] for k, v in right.items()},
    )
    assert all(x is False for x in result["flags"].values())


def test_inactive_nonfinite_padding_is_not_a_term():
    left, right = order_fixture()
    for bank in (left, right):
        field = bank["efc.J"]
        index = 511 * 20 * 4
        field["raw"] = (
            field["raw"][:index] + bytes.fromhex("0100c07f") + field["raw"][index + 4 :]
        )
    assert model.compare_qfrc(left, right)["active_rows"] == 3


@pytest.mark.parametrize("name", ("efc.J", "efc.force", "data.qfrc_constraint"))
def test_active_nonfinite_inputs_refused(name):
    left, right = order_fixture()
    for bank in (left, right):
        bank[name]["raw"] = bytes.fromhex("0100c07f") + bank[name]["raw"][4:]
    with pytest.raises(ValueError, match="nonfinite"):
        model.compare_qfrc(left, right)


def test_partial_or_done_input_refused_not_normalized():
    left, right = order_fixture()
    set_words(left, "context.done", {0: 1})
    with pytest.raises(ValueError, match="coverage"):
        model.compare_qfrc(left, right)


def test_standard_library_core_import_and_cli_guards():
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"),
    )
    code = "import sys; from mjlab_microduck import stance_qfrc_order_model; assert not {'torch','warp','numpy','mujoco','mujoco_warp'} & set(sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, timeout=15
    )
    assert result.returncode == 0, result.stderr
    for flags, cuda in (([], "0"), (["-O"], "")):
        env["CUDA_VISIBLE_DEVICES"] = cuda
        result = subprocess.run(
            [
                sys.executable,
                *flags,
                "-m",
                "mjlab_microduck.stance_qfrc_order_model",
                "--source",
                "0" * 40,
            ],
            env=env,
            capture_output=True,
            timeout=15,
        )
        assert result.returncode != 0 and b"CUDA-hidden unoptimized" in result.stderr


def test_wrong_analysis_root_refused():
    with pytest.raises(ValueError, match="arithmetic worktree"):
        model.analysis_binding(Path("/tmp"), "0" * 40)
