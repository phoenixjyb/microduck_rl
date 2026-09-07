"""CPU-only byte-bound skill descriptors; never authorize policy switching.

This checks declared evidence identity, not whether a declaration accurately
describes a checkpoint or compiled robot. Artifact binding and behavioral
retention are separate gates. No torch, simulator, pickle loading or writes.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path, PurePosixPath


PROTOCOL = "skill-static-compatibility-v1"
DESCRIPTOR_SCHEMA = "skill-descriptor-v1"
# A matching action count is insufficient: ordering, units, scaling, clipping,
# preprocessing, reset state and mechanics all affect the policy function.
STATIC_FIELDS = (
    "mechanics", "actor_interface", "action_interface", "control_timing",
    "command_sensor_envelope", "runtime",
)
ARTIFACT_FIELDS = ("checkpoint", "evaluation_protocol", "evaluation_report") + STATIC_FIELDS
DESCRIPTOR_FIELDS = {"schema", "skill_id", "training_source", *ARTIFACT_FIELDS}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _hex(value: object, length: int) -> bool:
    return type(value) is str and re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is not None


def _verify_reference(root: Path, field: str, reference: object) -> dict:
    _require(type(reference) is dict and set(reference) == {"path", "sha256", "bytes"},
             f"{field}: exact artifact reference required")
    raw = reference["path"]
    _require(type(raw) is str and bool(raw) and "\\" not in raw and "\x00" not in raw,
             f"{field}: relative POSIX path required")
    relative = PurePosixPath(raw)
    _require(not relative.is_absolute() and ".." not in relative.parts
             and relative.as_posix() == raw and raw != ".", f"{field}: unsafe path")
    _require(_hex(reference["sha256"], 64), f"{field}: SHA256 required")
    _require(type(reference["bytes"]) is int and reference["bytes"] > 0,
             f"{field}: positive byte count required")
    path = root
    for component in relative.parts:
        path = path / component
        _require(not path.is_symlink(), f"{field}: symlink not allowed")
    _require(path.is_file(), f"{field}: regular retained file required")
    digest, count = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            count += len(chunk)
            digest.update(chunk)
    _require(count == reference["bytes"], f"{field}: byte count mismatch")
    _require(digest.hexdigest() == reference["sha256"], f"{field}: SHA256 mismatch")
    return dict(path=raw, sha256=digest.hexdigest(), bytes=count)


def verify_descriptor(root: Path, descriptor: object) -> dict:
    """Verify every referenced byte without interpreting model/report contents.

    Call only on quiescent retained artifact roots. This is a point-in-time
    audit, not a concurrent-filesystem lock or authenticated provenance proof.
    Paths may differ between mirrors; hash and byte identity must not.
    """
    _require(type(descriptor) is dict and set(descriptor) == DESCRIPTOR_FIELDS,
             "exact descriptor fields required")
    _require(descriptor["schema"] == DESCRIPTOR_SCHEMA, "unsupported descriptor schema")
    _require(type(descriptor["skill_id"]) is str
             and re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,127}", descriptor["skill_id"]) is not None,
             "invalid skill id")
    _require(_hex(descriptor["training_source"], 40), "exact training source SHA1 required")
    root = Path(root).resolve(strict=True)
    _require(root.is_dir(), "artifact root must be a directory")
    verified = {field: _verify_reference(root, field, descriptor[field]) for field in ARTIFACT_FIELDS}
    return dict(schema=DESCRIPTOR_SCHEMA, skill_id=descriptor["skill_id"],
                training_source=descriptor["training_source"], **verified)


def compare_descriptors(left_root: Path, left: dict, right_root: Path, right: dict) -> dict:
    """Conservative exact declared-spec match only; never a skill admission.

    No inferred mechanics from filenames, dimension-only matching, widened
    envelope union, automatic migration or acceptance extracted from a report.
    Even identical specs with distinct weights need numerical retention tests.
    """
    left = verify_descriptor(left_root, left)
    right = verify_descriptor(right_root, right)
    mismatches = [field for field in STATIC_FIELDS
                  if (left[field]["sha256"], left[field]["bytes"])
                  != (right[field]["sha256"], right[field]["bytes"])]
    return dict(
        protocol=PROTOCOL,
        decision="declared-spec-mismatch" if mismatches else "declared-spec-match-only",
        mismatches=mismatches,
        left=left,
        right=right,
        referenced_bytes_verified=True,
        descriptor_to_runtime_binding_verified=False,
        behavioral_retention_verified=False,
        policy_acceptance=False,
        transition_authorized=False,
        physical_motion_authorized=False,
    )
