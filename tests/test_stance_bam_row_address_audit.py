import hashlib

import numpy as np
import pytest

from mjlab_microduck import stance_bam_load_receiver as receiver
from mjlab_microduck import stance_bam_row_address_audit as audit


def _fields():
    return {
        name: np.zeros((receiver.SUBSTEPS, receiver.WORLDS, *shape), dtype=dtype)
        for name, (dtype, shape) in receiver.LOAD_FIELDS.items()
    }


def _pairs():
    return {case: _fields() for case in audit.CASE_ORDER}


def _set_row(fields, case, step, world, row, dof, force_bits, *, active=True):
    fields[case]["efc_type"][step, world, row] = 1 if active else 0
    fields[case]["efc_id"][step, world, row] = dof
    fields[case]["efc_force"][step, world, row] = np.asarray(
        [force_bits], dtype="<u4"
    ).view("<f4")[0]


def test_row_ticket_permutation_retains_addressed_bits_and_negative_gate():
    fields = _pairs()
    fields["serial0"]["nefc"][1, 7] = 2
    fields["serial1"]["nefc"][1, 7] = 2
    _set_row(fields, "serial0", 1, 7, 0, 2, 0x3F800000)
    _set_row(fields, "serial0", 1, 7, 1, 9, 0x40000000)
    _set_row(fields, "serial1", 1, 7, 0, 9, 0x40000000)
    _set_row(fields, "serial1", 1, 7, 1, 2, 0x3F800000)

    result = audit.audit_decoded_fields(fields)
    row = next(v for v in result["per_world"] if v["substep"] == 1 and v["world"] == 7)
    assert row["rows"] == {
        "serial0": [[0, 2, 0x3F800000], [1, 9, 0x40000000]],
        "serial1": [[0, 9, 0x40000000], [1, 2, 0x3F800000]],
    }
    assert row["address_multiset_equal"]
    assert row["addressed_force_bits_equal"]
    assert row["row_ticket_address_permutation_only"]
    assert not row["row_force_sequence_equal"]
    assert result["first_affected"] == {"substep": 1, "world": 7}
    assert result["first_affected_worlds"] == [7]
    assert result["first_ticket_address_difference"] == {"substep": 1, "world": 7}
    assert result["first_addressed_force_bit_change"] is None
    assert result["declared_predecessor_decision"] == "fresh-bam-load-repeat-negative"
    assert result["authenticated_closeout"] is False
    assert result["raw_provenance_enforced_by_this_pure_view"] is False
    assert "original_receiver_decision" not in result
    assert all(value is False for value in result["flags"].values())


def test_address_and_force_bit_changes_are_separate_and_signed_zero_is_literal():
    fields = _pairs()
    fields["serial0"]["nefc"][0, 0] = 1
    fields["serial1"]["nefc"][0, 0] = 1
    _set_row(fields, "serial0", 0, 0, 0, 3, 0x00000000)
    _set_row(fields, "serial1", 0, 0, 0, 4, 0x80000000)
    result = audit.audit_decoded_fields(fields)
    row = result["per_world"][0]
    assert row["address_multiset_changed"]
    assert row["addressed_force_bit_change"]
    assert row["zero_force_rows"] == {
        "serial0": {"positive": 1, "negative": 0},
        "serial1": {"positive": 0, "negative": 1},
    }
    assert result["summary"]["address_multiset_changed"] == 1
    assert result["summary"]["addressed_force_bit_change"] == 1
    assert result["row_address_decision"] == "addressed-force-multiset-negative"
    assert row["force_bit_change_with_equal_address_multisets"] is None


def test_signed_zero_force_change_at_same_dof_is_not_an_address_change():
    fields = _pairs()
    for case in audit.CASE_ORDER:
        fields[case]["nefc"][0, 0] = 1
    _set_row(fields, "serial0", 0, 0, 0, 3, 0x00000000)
    _set_row(fields, "serial1", 0, 0, 0, 3, 0x80000000)
    row = audit.audit_decoded_fields(fields)["per_world"][0]
    assert row["address_multiset_equal"]
    assert not row["addressed_force_bits_equal"]
    assert row["force_bit_change_with_equal_address_multisets"] is True


def test_duplicate_rows_are_preserved_as_multisets():
    fields = _pairs()
    fields["serial0"]["nefc"][0, 0] = 2
    fields["serial1"]["nefc"][0, 0] = 1
    _set_row(fields, "serial0", 0, 0, 0, 5, 0x3F000000)
    _set_row(fields, "serial0", 0, 0, 1, 5, 0x3F000000)
    _set_row(fields, "serial1", 0, 0, 0, 5, 0x3F000000)
    result = audit.audit_decoded_fields(fields)
    row = result["per_world"][0]
    assert row["rows"]["serial0"] == [
        [0, 5, 0x3F000000],
        [1, 5, 0x3F000000],
    ]
    assert row["rows"]["serial1"] == [[0, 5, 0x3F000000]]
    assert row["duplicate_address_rows"] == {"serial0": 1, "serial1": 0}
    assert not row["address_multiset_equal"]
    assert not row["addressed_force_bits_equal"]
    assert not row["active_counts_equal"]


def test_inactive_rows_after_nefc_are_excluded():
    fields = _pairs()
    fields["serial0"]["nefc"][0, 0] = 1
    fields["serial1"]["nefc"][0, 0] = 1
    _set_row(fields, "serial0", 0, 0, 0, 2, 0x3F800000)
    _set_row(fields, "serial1", 0, 0, 0, 2, 0x3F800000)
    _set_row(fields, "serial0", 0, 0, 1, 99, 0x3F800000)
    _set_row(fields, "serial1", 0, 0, 1, 88, 0x3F800000)
    result = audit.audit_decoded_fields(fields)
    row = result["per_world"][0]
    assert row["rows"]["serial0"] == row["rows"]["serial1"] == [[0, 2, 0x3F800000]]
    assert row["row_force_sequence_equal"]


@pytest.mark.parametrize(
    "mutation",
    ["count-negative", "count-too-large", "invalid-dof", "wrong-dtype", "nan"],
)
def test_malformed_active_rows_fail_closed(mutation):
    fields = _pairs()
    fields["serial0"]["nefc"][0, 0] = 1
    _set_row(fields, "serial0", 0, 0, 0, 2, 0x3F800000)
    if mutation == "count-negative":
        fields["serial0"]["nefc"][0, 0] = -1
    elif mutation == "count-too-large":
        fields["serial0"]["nefc"][0, 0] = 513
    elif mutation == "invalid-dof":
        fields["serial0"]["efc_id"][0, 0, 0] = 20
    elif mutation == "wrong-dtype":
        fields["serial0"]["efc_id"] = fields["serial0"]["efc_id"].astype(np.int64)
    elif mutation == "nan":
        fields["serial0"]["efc_force"][0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        audit.audit_decoded_fields(fields)


def test_closeout_hashes_are_checked_before_any_json_decode(tmp_path, monkeypatch):
    contents = {name: ("not-json-" + name).encode() for name in audit.CLOSEOUT_CAPS}
    for name, raw in contents.items():
        (tmp_path / name).write_bytes(raw)
    anchors = {
        name: {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        for name, raw in contents.items()
    }
    anchors["verification.json"]["sha256"] = "0" * 64
    monkeypatch.setattr(audit, "CLOSEOUT_CAPS", anchors)
    decoded = []
    monkeypatch.setattr(audit, "_json", lambda raw, label: decoded.append(label))
    with pytest.raises(ValueError, match="SHA-256"):
        audit.analyze(tmp_path, tmp_path)
    assert decoded == []


def test_anchored_reader_rejects_path_hash_and_symlink_tampering(tmp_path):
    file_path = tmp_path / "payload.bin"
    file_path.write_bytes(b"authenticated")
    anchors = {
        "payload.bin": {
            "bytes": len(b"authenticated"),
            "sha256": hashlib.sha256(b"authenticated").hexdigest(),
        }
    }
    assert (
        audit._read_anchored_files(tmp_path, anchors, total_cap=1024)["payload.bin"]
        == b"authenticated"
    )
    file_path.write_bytes(b"tampered!!")
    with pytest.raises(ValueError, match="regular anchored file|SHA-256"):
        audit._read_anchored_files(tmp_path, anchors, total_cap=1024)
    file_path.unlink()
    target = tmp_path.parent / (tmp_path.name + "-target.bin")
    target.write_bytes(b"authenticated")
    file_path.symlink_to(target)
    with pytest.raises(OSError):
        audit._read_anchored_files(tmp_path, anchors, total_cap=1024)


def test_raw_inventory_rejects_extra_or_missing_paths_before_return(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(receiver, "CAPS", {"one.bin": 64})
    (tmp_path / "one.bin").write_bytes(b"one")
    good = {"one.bin": {"bytes": 3, "sha256": hashlib.sha256(b"one").hexdigest()}}
    assert receiver.read_inventory(tmp_path, good) == {"one.bin": b"one"}
    (tmp_path / "extra.bin").write_bytes(b"x")
    with pytest.raises(ValueError, match="inventory"):
        receiver.read_inventory(tmp_path, good)


def test_report_cap_refuses_instead_of_truncating(monkeypatch):
    monkeypatch.setattr(audit, "OUTPUT_CAP", 1)
    with pytest.raises(ValueError, match="two-MiB report cap"):
        audit.audit_decoded_fields(_pairs())


def test_exclusive_writer_rejects_unbounded_and_preserves_existing(tmp_path):
    output = tmp_path / "result.json"
    with pytest.raises(ValueError, match="payload before file creation"):
        audit._write_exclusive(output, b"x" * (audit.OUTPUT_CAP + 1))
    assert not output.exists()
    audit._write_exclusive(output, b"whole\n")
    with pytest.raises(FileExistsError):
        audit._write_exclusive(output, b"replacement\n")
    assert output.read_bytes() == b"whole\n"
