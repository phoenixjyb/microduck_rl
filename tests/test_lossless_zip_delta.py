import gzip
import hashlib
import json
import os
import stat
import struct
import sys
import warnings
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import microduck_lossless_zip_delta as delta
from microduck_lossless_zip_delta import (
    MAGIC,
    DeltaError,
    decode,
    encode,
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_zip(path, *, comment=b"", first=b"same control bytes", second=b"old trace"):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.comment = comment
        zf.writestr("control.pt", first)
        zf.writestr("trace.pt", second)
        zf.writestr("other.bin", b"literal")


def unpack_patch(path):
    with gzip.open(path, "rb") as f:
        raw = f.read()
    assert raw.startswith(MAGIC)
    n = struct.unpack(">I", raw[len(MAGIC):len(MAGIC) + 4])[0]
    start = len(MAGIC) + 4
    header_raw = raw[start:start + n]
    return json.loads(header_raw), raw[:start + n], raw[start + n:]


def repack_patch(path, prefix, body):
    with path.open("wb") as f:
        with gzip.GzipFile(fileobj=f, mode="wb", mtime=0, filename="") as gz:
            gz.write(prefix + body)


def test_roundtrip_uses_base_offsets_after_target_header_shift(tmp_path):
    base, target = tmp_path / "base.zip", tmp_path / "target.zip"
    make_zip(base, comment=b"short", first=b"small", second=b"old trace")
    make_zip(target, comment=b"changed comment length", first=b"a much longer first entry",
             second=b"new trace")
    patch, output = tmp_path / "delta.gz", tmp_path / "out.zip"
    base_info = delta._zip_payloads(base.open("rb"), base.stat().st_size)
    target_info = delta._zip_payloads(target.open("rb"), target.stat().st_size)
    assert base_info["trace.pt"][1] != target_info["trace.pt"][1]
    encode(base, target, patch)
    header, _, _ = unpack_patch(patch)
    trace_delta = next(d for d in header["segments"] if d["kind"] == "xor"
                       and d["length"] == len(b"new trace"))
    assert trace_delta["base_offset"] == base_info["trace.pt"][1]
    assert trace_delta["base_offset"] != target_info["trace.pt"][1]
    decode(base, patch, output, expected_base_sha256=sha(base),
           expected_target_sha256=sha(target))
    assert output.read_bytes() == target.read_bytes()


def test_patch_is_deterministic(tmp_path):
    base, target = tmp_path / "base.zip", tmp_path / "target.zip"
    make_zip(base); make_zip(target, comment=b"different", second=b"changed")
    a, b = tmp_path / "a.gz", tmp_path / "b.gz"
    encode(base, target, a); encode(base, target, b)
    assert a.read_bytes() == b.read_bytes()


def test_decode_rejects_wrong_expected_hashes_and_corrupt_body(tmp_path):
    base, target = tmp_path / "base.zip", tmp_path / "target.zip"
    make_zip(base); make_zip(target, second=b"target")
    patch = tmp_path / "delta.gz"
    encode(base, target, patch)
    with pytest.raises(DeltaError):
        decode(base, patch, tmp_path / "bad-base", expected_base_sha256="0" * 64,
               expected_target_sha256=sha(target))
    with pytest.raises(DeltaError):
        decode(base, patch, tmp_path / "bad-target", expected_base_sha256=sha(base),
               expected_target_sha256="0" * 64)
    header, prefix, body = unpack_patch(patch)
    repack_patch(patch, prefix, bytes([body[0] ^ 1]) + body[1:])
    with pytest.raises(DeltaError):
        decode(base, patch, tmp_path / "corrupt", expected_base_sha256=sha(base),
               expected_target_sha256=sha(target))


@pytest.mark.parametrize("mutate", ["gap", "oversize", "negative_base", "wrong_kind"])
def test_decode_rejects_invalid_descriptors(tmp_path, mutate):
    base, target = tmp_path / "base.zip", tmp_path / "target.zip"
    make_zip(base); make_zip(target, second=b"changed")
    patch = tmp_path / "delta.gz"
    encode(base, target, patch)
    header, _, _ = unpack_patch(patch)
    if mutate == "gap":
        header["segments"][0]["length"] -= 1
    elif mutate == "oversize":
        header["segments"][0]["length"] = header["target_size"] + 1
    elif mutate == "negative_base":
        item = next(x for x in header["segments"] if x["kind"] == "xor")
        item["base_offset"] = -1
    else:
        header["segments"][0]["kind"] = "mystery"
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    repack_patch(patch, MAGIC + struct.pack(">I", len(encoded)) + encoded,
                 unpack_patch(patch)[2])
    with pytest.raises(DeltaError):
        decode(base, patch, tmp_path / "out", expected_base_sha256=sha(base),
               expected_target_sha256=sha(target))


def test_missing_and_trailing_payload_rejected(tmp_path):
    base, target = tmp_path / "base.zip", tmp_path / "target.zip"
    make_zip(base); make_zip(target, second=b"changed")
    patch = tmp_path / "delta.gz"
    encode(base, target, patch)
    _, prefix, body = unpack_patch(patch)
    repack_patch(patch, prefix, body[:-1])
    with pytest.raises(DeltaError):
        decode(base, patch, tmp_path / "short", expected_base_sha256=sha(base),
               expected_target_sha256=sha(target))
    repack_patch(patch, prefix, body + b"hidden")
    with pytest.raises(DeltaError):
        decode(base, patch, tmp_path / "long", expected_base_sha256=sha(base),
               expected_target_sha256=sha(target))


@pytest.mark.parametrize("suffix", [b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x02\x03\x03\x00\x00\x00\x00\x00\x00\x00\x00",
                                     b"\x00\x00", b"\x01"])
def test_rejects_concatenated_member_padding_and_truncated_crc(tmp_path, suffix):
    base, target = tmp_path / "base.zip", tmp_path / "target.zip"
    make_zip(base); make_zip(target, second=b"changed")
    patch = tmp_path / "delta.gz"
    encode(base, target, patch)
    patch.write_bytes(patch.read_bytes() + suffix)
    with pytest.raises(DeltaError):
        decode(base, patch, tmp_path / "out", expected_base_sha256=sha(base),
               expected_target_sha256=sha(target))


def test_rejects_truncated_gzip_footer(tmp_path):
    base, target = tmp_path / "base.zip", tmp_path / "target.zip"
    make_zip(base); make_zip(target, second=b"changed")
    patch = tmp_path / "delta.gz"
    encode(base, target, patch)
    patch.write_bytes(patch.read_bytes()[:-4])
    with pytest.raises(DeltaError):
        decode(base, patch, tmp_path / "out", expected_base_sha256=sha(base),
               expected_target_sha256=sha(target))


def test_exclusive_outputs_and_zip_constraints(tmp_path):
    base, target = tmp_path / "base.zip", tmp_path / "target.zip"
    make_zip(base)
    make_zip(target)
    patch = tmp_path / "delta.gz"
    patch.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        encode(base, target, patch)
    patch.unlink()
    encode(base, target, patch)
    out = tmp_path / "already"
    out.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        decode(base, patch, out, expected_base_sha256=sha(base),
               expected_target_sha256=sha(target))

    compressed = tmp_path / "compressed.zip"
    with zipfile.ZipFile(compressed, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("x.pt", b"x" * 100)
    with pytest.raises(DeltaError):
        encode(base, compressed, tmp_path / "refused.gz")


def test_encrypted_entries_and_symlink_inputs_outputs_rejected(tmp_path):
    base, target = tmp_path / "base.zip", tmp_path / "target.zip"
    make_zip(base); make_zip(target)
    encrypted = tmp_path / "encrypted.zip"
    make_zip(encrypted)
    raw = bytearray(encrypted.read_bytes())
    pos = 0
    while True:
        pos = raw.find(b"PK\x03\x04", pos)
        if pos < 0:
            break
        raw[pos + 6] |= 1
        pos += 4
    pos = 0
    while True:
        pos = raw.find(b"PK\x01\x02", pos)
        if pos < 0:
            break
        raw[pos + 8] |= 1
        pos += 4
    encrypted.write_bytes(raw)
    with pytest.raises(DeltaError):
        encode(base, encrypted, tmp_path / "encrypted.gz")

    link = tmp_path / "base-link.zip"
    link.symlink_to(base)
    with pytest.raises((DeltaError, OSError)):
        encode(link, target, tmp_path / "symlink-input.gz")
    if hasattr(os, "mkfifo"):
        fifo = tmp_path / "input.fifo"
        os.mkfifo(fifo)
        with pytest.raises(DeltaError):
            encode(fifo, target, tmp_path / "fifo-input.gz")
    patch = tmp_path / "delta.gz"
    encode(base, target, patch)
    output_link = tmp_path / "out-link.zip"
    output_link.symlink_to(tmp_path / "victim")
    with pytest.raises((FileExistsError, OSError)):
        decode(base, patch, output_link, expected_base_sha256=sha(base),
               expected_target_sha256=sha(target))


def test_regular_files_without_posix_type_are_supported(tmp_path):
    p = tmp_path / "permissions.zip"
    with zipfile.ZipFile(p, "w") as zf:
        info = zipfile.ZipInfo("member.pt")
        info.external_attr = 0o600 << 16
        zf.writestr(info, b"content")
    assert delta._zip_payloads(p.open("rb"), p.stat().st_size)["member.pt"][0] == 7


@pytest.mark.parametrize("entry_type", [stat.S_IFDIR, stat.S_IFLNK])
def test_directory_and_symlink_zip_entries_rejected(tmp_path, entry_type):
    base, target = tmp_path / "base.zip", tmp_path / "typed.zip"
    make_zip(base)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED) as zf:
        info = zipfile.ZipInfo("typed")
        info.external_attr = (entry_type | 0o777) << 16
        zf.writestr(info, b"payload")
    with pytest.raises(DeltaError):
        encode(base, target, tmp_path / "refused.gz")


def test_duplicate_names_are_literal_and_roundtrip_exactly(tmp_path):
    base, target = tmp_path / "base.zip", tmp_path / "target.zip"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for path, values in ((base, (b"first", b"second")),
                             (target, (b"first", b"changed"))):
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
                zf.writestr("duplicate.pt", values[0])
                zf.writestr("duplicate.pt", values[1])
    patch, output = tmp_path / "delta.gz", tmp_path / "out.zip"
    encode(base, target, patch)
    header, _, _ = unpack_patch(patch)
    assert all(d["kind"] == "literal" for d in header["segments"])
    decode(base, patch, output, expected_base_sha256=sha(base),
           expected_target_sha256=sha(target))
    assert output.read_bytes() == target.read_bytes()


def test_limits_and_small_stream_chunks(tmp_path, monkeypatch):
    base, target = tmp_path / "base.zip", tmp_path / "target.zip"
    make_zip(base); make_zip(target, second=b"changed")
    monkeypatch.setattr(delta, "CHUNK", 7)
    patch, output = tmp_path / "delta.gz", tmp_path / "out.zip"
    encode(base, target, patch)
    decode(base, patch, output, expected_base_sha256=sha(base),
           expected_target_sha256=sha(target))
    assert output.read_bytes() == target.read_bytes()

    monkeypatch.setattr(delta, "MAX_ARTIFACT", 8)
    with pytest.raises(DeltaError):
        encode(base, target, tmp_path / "oversize.gz")


def test_descriptor_and_header_limits_and_exact_schema(tmp_path, monkeypatch):
    header = {"base_sha256": "a" * 64, "base_size": 2,
              "target_sha256": "b" * 64, "target_size": 1,
              "segments": [{"kind": "literal", "length": 1, "extra": 3}]}
    with pytest.raises(DeltaError):
        delta._validate_header(header, 2, "a" * 64, "b" * 64)
    header["segments"] = [{"kind": "literal", "length": 1}]
    monkeypatch.setattr(delta, "MAX_DESCRIPTORS", 0)
    with pytest.raises(DeltaError):
        delta._validate_header(header, 2, "a" * 64, "b" * 64)
    monkeypatch.setattr(delta, "MAX_DESCRIPTORS", 200_000)


def test_decode_enforces_header_limit(tmp_path, monkeypatch):
    base, target = tmp_path / "base.zip", tmp_path / "target.zip"
    make_zip(base); make_zip(target, second=b"changed")
    patch = tmp_path / "delta.gz"
    encode(base, target, patch)
    monkeypatch.setattr(delta, "MAX_HEADER", 1)
    with pytest.raises(DeltaError):
        decode(base, patch, tmp_path / "out", expected_base_sha256=sha(base),
               expected_target_sha256=sha(target))


def test_decode_enforces_patch_size_and_lowercase_expected_hashes(tmp_path, monkeypatch):
    base, target = tmp_path / "base.zip", tmp_path / "target.zip"
    make_zip(base); make_zip(target, second=b"changed")
    patch = tmp_path / "delta.gz"
    encode(base, target, patch)
    monkeypatch.setattr(delta, "MAX_PATCH", patch.stat().st_size - 1)
    with pytest.raises(DeltaError):
        decode(base, patch, tmp_path / "too-large", expected_base_sha256=sha(base),
               expected_target_sha256=sha(target))
    monkeypatch.setattr(delta, "MAX_PATCH", 600 * 1024 * 1024)
    with pytest.raises(DeltaError):
        decode(base, patch, tmp_path / "uppercase", expected_base_sha256=sha(base).upper(),
               expected_target_sha256=sha(target))


def test_base_is_verified_before_output_creation(tmp_path):
    base, target = tmp_path / "base.zip", tmp_path / "target.zip"
    make_zip(base); make_zip(target, second=b"changed")
    patch = tmp_path / "delta.gz"
    encode(base, target, patch)
    wrong = tmp_path / "wrong.zip"
    make_zip(wrong, first=b"not base")
    out = tmp_path / "must-not-exist"
    with pytest.raises(DeltaError):
        decode(wrong, patch, out, expected_base_sha256=sha(base),
               expected_target_sha256=sha(target))
    assert not out.exists()
