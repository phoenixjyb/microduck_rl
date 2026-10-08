"""CPU-only initialized row side-view contracts; no solver execution."""

from copy import deepcopy
from hashlib import sha256
from math import prod
from pathlib import Path
import os
import struct
import subprocess
import sys

import pytest

from mjlab_microduck import stance_solver_init_control as control
from mjlab_microduck import stance_solver_init_row_view as view


def fields():
    return {
        name: {
            "shape": list(shape),
            "dtype": {"float32": "<f4", "int32": "<i4", "bool": "|b1"}[dtype],
            "raw": bytes(prod(shape) * (1 if dtype == "bool" else 4)),
        }
        for name, (_, shape, dtype) in control.SOLVER_INIT_SPECS.items()
    }


def set_words(bank, name, values):
    item = bank[name]
    raw = bytearray(item["raw"])
    fmt = {"<f4": "<f", "<i4": "<i", "|b1": "B"}[item["dtype"]]
    width = struct.calcsize(fmt)
    for index, value in values.items():
        struct.pack_into(fmt, raw, index * width, value)
    item["raw"] = bytes(raw)


def fixture():
    left, right = fields(), fields()
    for bank in (left, right):
        set_words(bank, "contact.nacon", {0: 2})
        set_words(bank, "contact.dim", {0: 1, 1: 1})
        set_words(bank, "contact.type", {0: 1, 1: 1})
        set_words(bank, "data.nefc", {0: 3})
        set_words(bank, "efc.type", {0: 1, 1: 5, 2: 5})
        set_words(bank, "efc.id", {0: 4, 1: 0, 2: 1})
        set_words(bank, "contact.efc_address", {0: 1, 4: 2})
    set_words(left, "contact.geom", {0: 11, 2: 22})
    set_words(right, "contact.geom", {0: 22, 2: 11})
    for name in view.ROW_FIELDS:
        set_words(left, name, {0: 3, 1: 1, 2: 2})
        set_words(right, name, {0: 3, 1: 2, 2: 1})
    return left, right


def test_payload_permutation_compares_original_linked_offsets_without_mutation():
    left, right = fixture()
    before = deepcopy((left, right))
    result = view.compare_initialized(left, right)
    assert result["full_active_row_coverage"] is True
    assert result["active_rows_per_arm"] == 3
    assert result["covered_rows_per_arm"] == [3, 3]
    for name in view.ROW_FIELDS:
        assert left[name]["raw"] != right[name]["raw"]
        for kind, count in (("contact", 2), ("noncontact", 1)):
            total = result["row_comparisons"][kind][name]
            assert total["exact"] is True and total["compared_words"] == count
    assert (left, right) == before
    assert all(value is False for value in result["flags"].values())


@pytest.mark.parametrize("name", view.ROW_FIELDS)
def test_initialized_difference_preserves_original_offsets_and_raw_words(name):
    left, right = fixture()
    set_words(right, name, {2: 7})
    result = view.compare_initialized(left, right)
    total = result["row_comparisons"]["contact"][name]
    assert total["differing_words"] == 1 and total["exact"] is False
    delta = total["first_difference"]
    assert (delta["left_row"], delta["right_row"]) == (1, 2)
    assert (delta["left_byte_offset"], delta["right_byte_offset"]) == (4, 8)


@pytest.mark.parametrize("name", view.ROW_FIELDS)
def test_inactive_padding_is_not_compared_as_initialized_row(name):
    left, right = fixture()
    set_words(right, name, {511: 99})
    result = view.compare_initialized(left, right)
    assert all(
        result["row_comparisons"][kind][name]["exact"] is True
        for kind in ("contact", "noncontact")
    )


def test_done_world_is_excluded_not_vacuously_accepted():
    left, right = fixture()
    set_words(left, "context.done", {0: 1})
    result = view.compare_initialized(left, right)
    assert result["full_active_row_coverage"] is False
    assert result["covered_rows_per_arm"] == [0, 0]
    assert result["excluded_done_worlds"] == [0]
    assert all(
        x["exact"] is None
        for values in result["row_comparisons"].values()
        for x in values.values()
    )


def test_empty_active_coverage_is_null():
    result = view.compare_initialized(fields(), fields())
    assert result["full_active_row_coverage"] is None
    assert result["active_rows_per_arm"] == 0
    assert all(
        x["exact"] is None
        for values in result["row_comparisons"].values()
        for x in values.values()
    )


def test_ambiguous_payloads_stay_unpaired():
    left, right = fixture()
    for bank in (left, right):
        set_words(bank, "contact.geom", {0: 11, 2: 11})
    result = view.compare_initialized(left, right)
    assert result["full_active_row_coverage"] is False
    assert result["payload_link_exclusions"]["ambiguous_payload_groups"] == 1
    assert result["covered_rows_per_arm"] == [1, 1]
    assert result["row_comparisons"]["contact"]["efc.force"]["exact"] is None


def test_noncontact_marker_movement_is_not_matched_by_search():
    left, right = fixture()
    set_words(right, "efc.id", {0: 99})
    result = view.compare_initialized(left, right)
    assert result["full_active_row_coverage"] is False
    assert result["unpaired_noncontact_offset_count"] == 1
    assert result["row_comparisons"]["noncontact"]["efc.force"]["exact"] is None


def test_broken_contact_backlink_refused():
    left, right = fixture()
    set_words(right, "contact.efc_address", {0: 2})
    with pytest.raises(ValueError, match="backlink"):
        view.compare_initialized(left, right)


def test_direct_world_dof_fields_are_not_row_aligned():
    left, right = fixture()
    set_words(right, "data.qfrc_constraint", {7: 1})
    result = view.compare_initialized(left, right)
    total = result["direct_full_carrier_comparisons"]["data.qfrc_constraint"]
    assert total["compared_words"] == 1280 and total["differing_words"] == 1
    assert total["first_difference"]["left_byte_offset"] == 28


@pytest.mark.parametrize("bits", ("00000080", "0100c07f"))
def test_signed_zero_and_nan_payloads_remain_literal(bits):
    left, right = fixture()
    name = "context.Jaref"
    for bank in (left, right):
        set_words(bank, name, {0: 0})
    right[name]["raw"] = bytes.fromhex(bits) + right[name]["raw"][4:]
    result = view.compare_initialized(left, right)
    total = result["row_comparisons"]["noncontact"][name]
    assert total["differing_words"] == 1
    assert total["first_difference"]["right_word_le_hex"] == bits


@pytest.mark.parametrize(
    "damage", ("missing", "schema", "dtype", "shape", "bool-shape", "short", "done")
)
def test_literal_abi_damage_refused(damage):
    left, right = fixture()
    if damage == "missing":
        del right["context.Jaref"]
    elif damage == "schema":
        right["context.Jaref"]["extra"] = 1
    elif damage == "dtype":
        right["context.Jaref"]["dtype"] = "<i4"
    elif damage == "shape":
        right["context.Jaref"]["shape"] = [512, 64]
    elif damage == "bool-shape":
        right["model.opt.timestep"]["shape"] = [True]
    elif damage == "short":
        right["context.Jaref"]["raw"] = bytes(4)
    else:
        right["context.done"]["raw"] = bytes([2]) + bytes(63)
    with pytest.raises(ValueError):
        view.compare_initialized(left, right)


def packet_fixture():
    bank = fields()
    packed, metadata, offset = [], {}, 0
    for name in control.SOLVER_INIT_ORDER:
        item = bank[name]
        size = len(item["raw"])
        metadata[name] = {
            "offset": offset,
            "bytes": size,
            "shape": item["shape"],
            "dtype": item["dtype"],
        }
        packed.append(item["raw"])
        offset += size
    packet = b"".join(packed)
    path = "solver-init/candidate0/forward-04.initialized.bin"
    return (
        {
            "phase": "initialized-before-search",
            "forward": 4,
            "path": path,
            "bytes": len(packet),
            "sha256": sha256(packet).hexdigest(),
            "fields": metadata,
        },
        {path: packet},
        bank,
    )


def test_exact_packet_slice_decoder():
    snapshot, raw, bank = packet_fixture()
    assert view.unpack_initialized(snapshot, raw) == bank


@pytest.mark.parametrize(
    "damage", ("hash", "offset", "phase", "forward-bool", "tail", "fieldset")
)
def test_packet_decoder_refuses_damage(damage):
    snapshot, raw, _ = packet_fixture()
    if damage == "hash":
        snapshot["sha256"] = "0" * 64
    elif damage == "offset":
        snapshot["fields"]["context.Jaref"]["offset"] += 4
    elif damage == "phase":
        snapshot["phase"] = "construction"
    elif damage == "forward-bool":
        snapshot["forward"] = True
    elif damage == "tail":
        raw[snapshot["path"]] += b"tail"
        snapshot["bytes"] += 4
        snapshot["sha256"] = sha256(raw[snapshot["path"]]).hexdigest()
    else:
        del snapshot["fields"]["context.Jaref"]
    with pytest.raises(ValueError):
        view.unpack_initialized(snapshot, raw)


def test_whole_anchor_rejects_wrong_hash_and_symlink(tmp_path):
    path = tmp_path / "leaf"
    path.write_bytes(b"whole")
    anchor = sha256(b"whole").hexdigest()
    assert view.anchored(path, 5, anchor) == b"whole"
    with pytest.raises(ValueError):
        view.anchored(path, 5, "0" * 64)
    with pytest.raises(ValueError):
        view.anchored(path, 4, anchor)
    link = tmp_path / "link"
    link.symlink_to(path)
    with pytest.raises(ValueError):
        view.anchored(link, 5, anchor)


def test_git_binding_uses_historical_whole_blobs_and_current_files(tmp_path):
    def git(*args):
        return (
            subprocess.check_output(["git", "-C", str(tmp_path), *args])
            .decode()
            .strip()
        )

    git("init", "-q")
    directory = tmp_path / "src" / "nested"
    directory.mkdir(parents=True)
    path = directory / "source.py"
    path.write_bytes(b"first")
    git("add", "src/nested/source.py")
    git(
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-qm",
        "fixture",
    )
    source = git("rev-parse", "HEAD")
    binding = view.git_binding(tmp_path, source, worktree=True)
    path.write_bytes(b"later")
    assert view.git_binding(tmp_path, source, worktree=False) == binding
    with pytest.raises(ValueError):
        view.git_binding(tmp_path, source, worktree=True)


def test_import_is_pure_and_cli_refuses_visible_cuda():
    # The inherited pure numerical receiver uses CPU NumPy. No CUDA/physics
    # runtime may be imported; absence of NumPy is not the protocol requirement.
    code = "import sys; from mjlab_microduck import stance_solver_init_row_view; assert not {'torch','warp','mujoco','mujoco_warp'} & set(sys.modules)"
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"),
    )
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, timeout=15
    )
    assert result.returncode == 0, result.stderr
    env["CUDA_VISIBLE_DEVICES"] = "0"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mjlab_microduck.stance_solver_init_row_view",
            "--source",
            "0" * 40,
        ],
        env=env,
        capture_output=True,
        timeout=15,
    )
    assert result.returncode != 0 and b"CUDA-hidden" in result.stderr


def test_cli_refuses_optimized_python():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    result = subprocess.run(
        [
            sys.executable,
            "-O",
            "-m",
            "mjlab_microduck.stance_solver_init_row_view",
            "--source",
            "0" * 40,
        ],
        env=env,
        capture_output=True,
        timeout=15,
    )
    assert result.returncode != 0 and b"unoptimized" in result.stderr


def test_analysis_binding_refuses_wrong_root_or_source():
    with pytest.raises(ValueError, match="analysis worktree"):
        view.analysis_binding(Path("/tmp"), "0" * 40)
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="clean analysis source"):
        view.analysis_binding(root, "0" * 40)
