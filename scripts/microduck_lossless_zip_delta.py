#!/usr/bin/env python3
"""Lossless transport for compatible ZIP artifacts; never interprets contents."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import stat
import struct
import sys
import zipfile
import zlib
from pathlib import Path
from typing import BinaryIO

import numpy as np

MAGIC = b"MDZD1\n"
MAX_ARTIFACT = 512 * 1024 * 1024
MAX_HEADER = 16 * 1024 * 1024
MAX_PATCH = MAX_ARTIFACT + MAX_HEADER + 1024 * 1024
MAX_DESCRIPTORS = 200_000
CHUNK = 1024 * 1024
_LOCAL = struct.Struct("<IHHHHHIIIHH")
_LOCAL_SIG = 0x04034B50


class DeltaError(ValueError):
    """Invalid or incompatible patch/artifact."""


def _open_regular(path: Path, limit: int) -> BinaryIO:
    try:
        before = path.lstat()
    except OSError:
        raise
    if not stat.S_ISREG(before.st_mode):
        raise DeltaError("input must be a regular non-symlink file")
    flags = (os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
             | getattr(os, "O_NONBLOCK", 0))
    fd = os.open(path, flags)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise DeltaError("input must be a regular non-symlink file")
        if info.st_size > limit:
            raise DeltaError("input exceeds configured size limit")
        return os.fdopen(fd, "rb")
    except Exception:
        os.close(fd)
        raise


def _hash_open(f: BinaryIO) -> tuple[str, int]:
    f.seek(0)
    h, size = hashlib.sha256(), 0
    while block := f.read(min(CHUNK, MAX_ARTIFACT + 1 - size)):
        size += len(block)
        if size > MAX_ARTIFACT:
            raise DeltaError("artifact exceeds 512 MiB limit")
        h.update(block)
    f.seek(0)
    return h.hexdigest(), size


def _entry_data_offset(f: BinaryIO, info: zipfile.ZipInfo, total: int) -> int:
    if info.flag_bits & 1 or info.compress_type != zipfile.ZIP_STORED:
        raise DeltaError("ZIP entries must be unencrypted and ZIP_STORED")
    if info.file_size != info.compress_size:
        raise DeltaError("invalid stored ZIP entry sizes")
    f.seek(info.header_offset)
    fields = _LOCAL.unpack(f.read(_LOCAL.size))
    if fields[0] != _LOCAL_SIG or fields[2] & 1:
        raise DeltaError("invalid or encrypted ZIP local header")
    name_len, extra_len = fields[-2:]
    offset = info.header_offset + _LOCAL.size + name_len + extra_len
    if offset < 0 or offset > total or info.file_size > total - offset:
        raise DeltaError("ZIP entry data outside artifact")
    return offset


def _regular(info: zipfile.ZipInfo) -> bool:
    file_type = stat.S_IFMT(info.external_attr >> 16)
    # Some ZIP writers store permission bits without a POSIX file type.
    return not info.is_dir() and file_type in (0, stat.S_IFREG)


def _zip_payloads(f: BinaryIO, size: int) -> dict[str, tuple[int, int]]:
    result: dict[str, tuple[int, int]] = {}
    duplicates: set[str] = set()
    f.seek(0)
    try:
        with zipfile.ZipFile(f) as zf:
            infos = zf.infolist()
            if len(infos) > MAX_DESCRIPTORS:
                raise DeltaError("ZIP exceeds entry-count limit")
            for info in infos:
                if not _regular(info):
                    raise DeltaError("ZIP may contain only regular non-symlink entries")
                data = _entry_data_offset(f, info, size)
                if info.filename in result or info.filename in duplicates:
                    result.pop(info.filename, None)
                    duplicates.add(info.filename)
                    continue
                result[info.filename] = (info.file_size, data)
    except (OSError, zipfile.BadZipFile, NotImplementedError, struct.error) as exc:
        raise DeltaError(f"unsupported or malformed ZIP: {exc}") from exc
    f.seek(0)
    return result


class _GzipReader:
    """Single-member gzip reader with bounded output and strict physical EOF."""

    def __init__(self, f: BinaryIO):
        self.f = f
        self.dec = zlib.decompressobj(16 + zlib.MAX_WBITS)
        self.pending = b""
        self.compressed_tail = b""
        self.done = False
        self.compressed_bytes = 0

    def _read_compressed(self) -> bytes:
        # The one-byte probe at the cap detects files that grow after fstat.
        amount = min(CHUNK, MAX_PATCH - self.compressed_bytes + 1)
        data = self.f.read(amount)
        self.compressed_bytes += len(data)
        if self.compressed_bytes > MAX_PATCH:
            raise DeltaError("patch grew beyond compressed-size limit")
        return data

    def read_exact(self, size: int) -> bytes:
        out = bytearray()
        while len(out) < size:
            if self.pending:
                data = self.pending[:min(CHUNK, size - len(out))]
                self.pending = self.pending[len(data):]
                out.extend(data)
                continue
            if self.done:
                raise DeltaError("truncated patch payload")
            compressed = self.compressed_tail or self._read_compressed()
            self.compressed_tail = b""
            if not compressed:
                raise DeltaError("truncated gzip stream")
            try:
                self.pending = self.dec.decompress(compressed, min(CHUNK, size - len(out)))
            except zlib.error as exc:
                raise DeltaError(f"invalid gzip stream: {exc}") from exc
            self.compressed_tail = self.dec.unconsumed_tail
            if self.dec.unused_data:
                raise DeltaError("trailing bytes or concatenated gzip member")
            if self.dec.eof:
                self.done = True
                if self._read_compressed():
                    raise DeltaError("trailing bytes after gzip member")
        return bytes(out)

    def finish(self) -> None:
        while not self.done:
            compressed = self.compressed_tail or self._read_compressed()
            self.compressed_tail = b""
            if not compressed:
                raise DeltaError("truncated gzip stream")
            try:
                extra = self.dec.decompress(compressed, CHUNK)
            except zlib.error as exc:
                raise DeltaError(f"invalid gzip stream: {exc}") from exc
            self.compressed_tail = self.dec.unconsumed_tail
            if extra:
                raise DeltaError("trailing decompressed patch payload")
            if self.dec.unused_data:
                raise DeltaError("trailing bytes or concatenated gzip member")
            if self.dec.eof:
                self.done = True
                if self._read_compressed():
                    raise DeltaError("trailing bytes after gzip member")
        if self.pending:
            raise DeltaError("trailing decompressed patch payload")


def _canonical_sha(value: object, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise DeltaError(f"{label} must be a lowercase SHA256 hex digest")
    return value


def _fsync_parent(path: Path) -> None:
    fd = os.open(path.parent or Path("."), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _hash_rewind(f: BinaryIO) -> tuple[str, int]:
    return _hash_open(f)


def encode(base: str | Path, target: str | Path, patch: str | Path) -> None:
    """Create an exclusive deterministic patch from two ZIP byte streams."""
    base, target, patch = Path(base), Path(target), Path(patch)
    with _open_regular(base, MAX_ARTIFACT) as bf, _open_regular(target, MAX_ARTIFACT) as tf:
        base_sha, base_size = _hash_open(bf)
        target_sha, target_size = _hash_open(tf)
        base_entries = _zip_payloads(bf, base_size)
        target_entries = _zip_payloads(tf, target_size)
        matches = {name: (sz, base_entries[name][1])
                   for name, (sz, _) in target_entries.items()
                   if name in base_entries and base_entries[name][0] == sz}
        descriptors: list[dict[str, object]] = []
        cursor = 0
        for name, (length, off) in sorted(target_entries.items(), key=lambda item: item[1][1]):
            if length == 0:
                continue
            if off < cursor:
                raise DeltaError("overlapping target ZIP entry data")
            if off > cursor:
                descriptors.append({"kind": "literal", "length": off - cursor})
            matched = matches.get(name)
            if matched is not None:
                descriptors.append({"kind": "xor", "length": length, "base_offset": matched[1]})
            else:
                descriptors.append({"kind": "literal", "length": length})
            cursor = off + length
        if cursor < target_size:
            descriptors.append({"kind": "literal", "length": target_size - cursor})
        if len(descriptors) > MAX_DESCRIPTORS:
            raise DeltaError("too many patch descriptors")
        header = {"base_sha256": base_sha, "base_size": base_size,
                  "target_sha256": target_sha, "target_size": target_size,
                  "segments": descriptors}
        encoded_header = json.dumps(header, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True).encode("ascii")
        if len(encoded_header) > MAX_HEADER:
            raise DeltaError("patch header exceeds 16 MiB limit")

        plan: list[tuple[int, int, int | None]] = []
        cursor = 0
        for d in descriptors:
            length = d["length"]
            base_off = d["base_offset"] if d["kind"] == "xor" else None
            plan.append((cursor, length, base_off))
            cursor += length
        if cursor != target_size:
            raise DeltaError("internal target coverage error")

        fd = os.open(patch, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as out:
            with gzip.GzipFile(fileobj=out, mode="wb", mtime=0, filename="") as gz:
                gz.write(MAGIC + struct.pack(">I", len(encoded_header)) + encoded_header)
                for target_off, length, base_off in plan:
                    tf.seek(target_off)
                    if base_off is not None:
                        bf.seek(base_off)
                    remaining = length
                    while remaining:
                        count = min(CHUNK, remaining)
                        t = tf.read(count)
                        if len(t) != count:
                            raise DeltaError("target changed during encoding")
                        if base_off is None:
                            gz.write(t)
                        else:
                            b = bf.read(count)
                            if len(b) != count:
                                raise DeltaError("base changed during encoding")
                            x = np.bitwise_xor(np.frombuffer(t, dtype=np.uint8),
                                               np.frombuffer(b, dtype=np.uint8))
                            gz.write(x.tobytes())
                        remaining -= count
            out.flush()
            os.fsync(out.fileno())
        _fsync_parent(patch)
        if _hash_rewind(tf) != (target_sha, target_size):
            raise DeltaError("target changed during encoding")
        if _hash_rewind(bf) != (base_sha, base_size):
            raise DeltaError("base changed during encoding")


def _validate_header(header: object, base_size: int, expected_base: str,
                     expected_target: str) -> tuple[int, list[dict[str, object]]]:
    if not isinstance(header, dict) or set(header) != {
        "base_sha256", "base_size", "target_sha256", "target_size", "segments"
    }:
        raise DeltaError("invalid patch header keys")
    if _canonical_sha(header["base_sha256"], "base_sha256") != expected_base:
        raise DeltaError("base_sha256 does not match caller expectation")
    if _canonical_sha(header["target_sha256"], "target_sha256") != expected_target:
        raise DeltaError("target_sha256 does not match caller expectation")
    for key in ("base_size", "target_size"):
        if type(header[key]) is not int or header[key] < 0:
            raise DeltaError(f"invalid {key}")
    if header["base_size"] != base_size or header["target_size"] > MAX_ARTIFACT:
        raise DeltaError("patch size mismatch or target exceeds 512 MiB limit")
    segments = header["segments"]
    if not isinstance(segments, list) or len(segments) > MAX_DESCRIPTORS:
        raise DeltaError("invalid or excessive descriptor list")
    cursor = 0
    for d in segments:
        if not isinstance(d, dict) or set(d) not in ({"kind", "length"}, {"kind", "length", "base_offset"}):
            raise DeltaError("invalid descriptor keys")
        if type(d.get("length")) is not int or d["length"] <= 0:
            raise DeltaError("invalid descriptor length")
        if d.get("kind") not in ("literal", "xor") or (d["kind"] == "xor") != ("base_offset" in d):
            raise DeltaError("invalid descriptor kind")
        if d["kind"] == "xor":
            off = d["base_offset"]
            if type(off) is not int or off < 0 or off > base_size or d["length"] > base_size - off:
                raise DeltaError("XOR descriptor outside base")
        cursor += d["length"]
        if cursor > header["target_size"]:
            raise DeltaError("out-of-bounds target descriptors")
    if cursor != header["target_size"]:
        raise DeltaError("descriptor coverage has a gap")
    return header["target_size"], segments


def decode(base: str | Path, patch: str | Path, output: str | Path, *,
           expected_base_sha256: str, expected_target_sha256: str) -> None:
    """Decode after authenticating the complete base; never overwrites output."""
    base, patch, output = Path(base), Path(patch), Path(output)
    expected_base_sha256 = _canonical_sha(expected_base_sha256, "expected base SHA256")
    expected_target_sha256 = _canonical_sha(expected_target_sha256, "expected target SHA256")
    with _open_regular(base, MAX_ARTIFACT) as bf, _open_regular(patch, MAX_PATCH) as pf:
        base_sha, base_size = _hash_open(bf)
        if base_sha != expected_base_sha256:
            raise DeltaError("base SHA256 does not match caller expectation")
        reader = _GzipReader(pf)
        if reader.read_exact(len(MAGIC)) != MAGIC:
            raise DeltaError("invalid patch magic")
        header_len = struct.unpack(">I", reader.read_exact(4))[0]
        if header_len > MAX_HEADER:
            raise DeltaError("patch header exceeds 16 MiB limit")
        raw_header = reader.read_exact(header_len)
        try:
            header = json.loads(raw_header)
            canonical = json.dumps(header, sort_keys=True, separators=(",", ":"),
                                   ensure_ascii=True).encode("ascii")
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError, ValueError) as exc:
            raise DeltaError(f"invalid patch header JSON: {exc}") from exc
        if canonical != raw_header:
            raise DeltaError("patch header is not canonical JSON")
        target_size, segments = _validate_header(header, base_size, expected_base_sha256,
                                                   expected_target_sha256)
        fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        h = hashlib.sha256()
        with os.fdopen(fd, "wb") as out:
            for d in segments:
                remaining = d["length"]
                if d["kind"] == "xor":
                    bf.seek(d["base_offset"])
                while remaining:
                    count = min(CHUNK, remaining)
                    data = reader.read_exact(count)
                    if d["kind"] == "xor":
                        b = bf.read(count)
                        if len(b) != count:
                            raise DeltaError("base truncated during decode")
                        data = np.bitwise_xor(np.frombuffer(data, dtype=np.uint8),
                                              np.frombuffer(b, dtype=np.uint8)).tobytes()
                    out.write(data)
                    h.update(data)
                    remaining -= count
            reader.finish()
            out.flush()
            os.fsync(out.fileno())
        _fsync_parent(output)
        if _hash_rewind(bf) != (base_sha, base_size):
            raise DeltaError("base changed during decode")
        if h.hexdigest() != expected_target_sha256 or output.stat().st_size != target_size:
            raise DeltaError("reconstructed target SHA256 or size mismatch")
        with _open_regular(output, MAX_ARTIFACT) as reconstructed:
            if _hash_open(reconstructed) != (expected_target_sha256, target_size):
                raise DeltaError("reconstructed target whole-file SHA256 mismatch")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    enc = sub.add_parser("encode")
    enc.add_argument("base"); enc.add_argument("target"); enc.add_argument("patch")
    dec = sub.add_parser("decode")
    dec.add_argument("base"); dec.add_argument("patch"); dec.add_argument("output")
    dec.add_argument("--expected-base-sha256", required=True)
    dec.add_argument("--expected-target-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "encode":
            encode(args.base, args.target, args.patch)
        else:
            decode(args.base, args.patch, args.output,
                   expected_base_sha256=args.expected_base_sha256,
                   expected_target_sha256=args.expected_target_sha256)
    except (DeltaError, FileExistsError, OSError) as exc:
        print(f"lossless ZIP delta: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
