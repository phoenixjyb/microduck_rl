"""Source-only mocked checks for the bounded CUDA probe controller."""

from hashlib import sha1, sha256
import os
from types import SimpleNamespace

import pytest

from mjlab_microduck import stance_friction_prefix_cuda_probe as probe


SOURCE = "0123456789abcdef0123456789abcdef01234567"


def test_process_local_compiler_settings_disable_pch_without_file_edits():
    config = SimpleNamespace(
        **(probe.COMPILER_CONFIG | {"use_precompiled_headers": True})
    )
    wp = SimpleNamespace(config=config)
    assert probe.configure_compiler(wp) == probe.COMPILER_CONFIG
    assert config.use_precompiled_headers is False


def test_process_local_compiler_settings_reject_nonfrozen_defaults():
    config = SimpleNamespace(
        **(probe.COMPILER_CONFIG | {"use_precompiled_headers": True, "llvm_cuda": True})
    )
    with pytest.raises(ValueError, match="fresh frozen compiler defaults"):
        probe.configure_compiler(SimpleNamespace(config=config))
    assert config.use_precompiled_headers is True


@pytest.mark.parametrize("reserve", [0, 1, 240, 540, 600])
def test_check_window_accepts_only_full_reserve_before_strict_cutoff(
    monkeypatch, reserve
):
    now = probe.CUTOFF - reserve - 1
    monkeypatch.setattr(probe.time, "time", lambda: now)
    probe.check_window(reserve)


@pytest.mark.parametrize("reserve", [-1, 1.0, True, "1", None])
def test_check_window_requires_plain_nonnegative_integer_reserve(reserve):
    with pytest.raises(ValueError):
        probe.check_window(reserve, now=probe.CUTOFF - 1000)


@pytest.mark.parametrize("offset", [0, 0.01, 60, 299, 300])
def test_check_window_rejects_deadline_equality_or_insufficient_time(offset):
    with pytest.raises(ValueError):
        probe.check_window(300, now=probe.CUTOFF - offset)


@pytest.mark.parametrize("mode", ["tests", "run"])
def test_unit_and_output_use_exact_source_prefix_and_separate_roots(mode):
    expected = f"microduck-friction-prefix-cuda-{mode}-{SOURCE[:12]}.service"
    assert probe.unit(SOURCE, mode) == expected
    root = "tools" if mode == "tests" else "evaluations"
    assert (
        probe.output(SOURCE, mode)
        == probe.ROOT
        / "artifacts"
        / root
        / f"friction-prefix-cuda-{mode}-{SOURCE[:12]}"
    )


@pytest.mark.parametrize(
    "source,mode",
    [
        ("A" * 40, "tests"),
        (SOURCE[:-1], "run"),
        (SOURCE, "child"),
        (None, "run"),
        (True, "tests"),
    ],
)
def test_unit_rejects_unpinned_sha_or_role(source, mode):
    with pytest.raises(ValueError):
        probe.unit(source, mode)


def test_launch_arguments_preserve_original_signature_and_bank_order():
    inputs = {name: object() for name in probe.INPUT_NAMES}
    bank = {name: object() for name in probe.OUTPUT_NAMES}
    dim, args, outputs = probe.launch_arguments("original", inputs, bank, 3, 17)
    fl, vel, inv, ref, imp, dt = (inputs[name] for name in probe.INPUT_NAMES)
    assert dim == (3, 20)
    assert args == [20, dt, 0, ref, imp, fl, inv, False, vel, 17, 340]
    assert outputs == [bank[name] for name in probe.OUTPUT_NAMES]


@pytest.mark.parametrize("role", ["candidate0", "candidate1"])
def test_launch_arguments_preserve_candidate_signature_and_role(role):
    inputs = {name: object() for name in probe.INPUT_NAMES}
    bank = {name: object() for name in probe.OUTPUT_NAMES}
    dim, args, outputs = probe.launch_arguments(role, inputs, bank, 4, 32)
    fl, vel, inv, ref, imp, dt = (inputs[name] for name in probe.INPUT_NAMES)
    names = (
        "nf",
        "nefc",
        "type",
        "id",
        "J",
        "pos",
        "margin",
        "D",
        "vel",
        "aref",
        "frictionloss",
    )
    assert dim == 4
    assert args == [20, dt, 0, ref, imp, fl, inv, vel, 32]
    assert outputs == [bank[name] for name in names]


@pytest.mark.parametrize("role", ["Candidate0", "candidate", "", None, True])
def test_launch_arguments_reject_nonliteral_role(role):
    inputs = {name: object() for name in probe.INPUT_NAMES}
    with pytest.raises(ValueError):
        probe.launch_arguments(role, inputs, {}, 1, 1)


def test_digest_file_hashes_the_complete_bounded_regular_file(tmp_path):
    path = tmp_path / "receipt.json"
    raw = b"first line\nsecond line\x00"
    path.write_bytes(raw)
    assert probe.digest_file(path, cap=len(raw)) == {
        "bytes": len(raw),
        "sha256": sha256(raw).hexdigest(),
    }


def test_digest_file_rejects_file_above_cap_and_nonregular_path(tmp_path):
    large = tmp_path / "large"
    large.write_bytes(b"12345")
    with pytest.raises(ValueError):
        probe.digest_file(large, cap=4)
    with pytest.raises(ValueError):
        probe.digest_file(tmp_path, cap=16)


def test_digest_file_rejects_symlink_even_when_target_is_regular(tmp_path):
    target = tmp_path / "target"
    target.write_bytes(b"regular")
    alias = tmp_path / "alias"
    alias.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        probe.digest_file(alias)


def test_digest_file_rejects_metadata_change_during_read(tmp_path, monkeypatch):
    path = tmp_path / "stable-check"
    path.write_bytes(b"payload")
    actual_fstat = os.fstat

    def changed_fstat(fd):
        value = actual_fstat(fd)
        return SimpleNamespace(
            st_dev=value.st_dev,
            st_ino=value.st_ino,
            st_size=value.st_size + 1,
            st_mtime_ns=value.st_mtime_ns,
            st_ctime_ns=value.st_ctime_ns,
        )

    monkeypatch.setattr(probe.os, "fstat", changed_fstat)
    with pytest.raises(ValueError, match="stable"):
        probe.digest_file(path)


def test_write_emits_canonical_json_exclusively_and_fsyncs_file_and_directory(
    tmp_path, monkeypatch
):
    target = tmp_path / "packet.json"
    real_fsync = os.fsync
    calls = []

    def recording_fsync(fd):
        calls.append(fd)
        return real_fsync(fd)

    monkeypatch.setattr(probe.os, "fsync", recording_fsync)
    probe.write(target, {"z": 3, "a": [1, 2]})
    assert target.read_bytes() == b'{"a":[1,2],"z":3}\n'
    assert len(calls) == 2
    with pytest.raises(FileExistsError):
        probe.write(target, {"replacement": True})
    assert target.read_bytes() == b'{"a":[1,2],"z":3}\n'


@pytest.mark.parametrize("value,cap", [(b"12345", 4), ({"long": "payload"}, 2)])
def test_write_rejects_payload_over_cap_without_creating_file(tmp_path, value, cap):
    target = tmp_path / "too-large"
    with pytest.raises(ValueError, match="bounded"):
        probe.write(target, value, cap=cap)
    assert not target.exists()


def test_write_preserves_exact_bytes_payload(tmp_path):
    target = tmp_path / "bytes.bin"
    payload = b"\x00\xffarbitrary"
    probe.write(target, payload)
    assert target.read_bytes() == payload


def test_library_binding_hashes_exact_runtime_library_bytes(tmp_path, monkeypatch):
    library_root = tmp_path / "warp/bin"
    library_root.mkdir(parents=True)
    payloads = {"libprobe-a.so": b"runtime-a", "libprobe-b.so": b"runtime-b"}
    pins = {}
    for name, payload in payloads.items():
        (library_root / name).write_bytes(payload)
        pins[name] = (len(payload), sha256(payload).hexdigest())
    monkeypatch.setattr(probe, "LIBRARIES", pins)
    monkeypatch.setattr(
        probe.importlib.metadata,
        "distribution",
        lambda name: SimpleNamespace(locate_file=lambda relative: tmp_path / relative),
    )
    binding = probe.library_binding()
    assert set(binding) == set(pins)
    assert {
        name: (row["bytes"], row["sha256"]) for name, row in binding.items()
    } == pins


def test_library_binding_rejects_a_runtime_library_byte_change(tmp_path, monkeypatch):
    library_root = tmp_path / "warp/bin"
    library_root.mkdir(parents=True)
    path = library_root / "libprobe.so"
    path.write_bytes(b"verified")
    monkeypatch.setattr(
        probe, "LIBRARIES", {"libprobe.so": (8, sha256(b"verified").hexdigest())}
    )
    monkeypatch.setattr(
        probe.importlib.metadata,
        "distribution",
        lambda name: SimpleNamespace(locate_file=lambda relative: tmp_path / relative),
    )
    path.write_bytes(b"changed!")
    with pytest.raises(ValueError, match="library bytes"):
        probe.library_binding()


def test_library_binding_rejects_symlinked_runtime_library(tmp_path, monkeypatch):
    library_root = tmp_path / "warp/bin"
    library_root.mkdir(parents=True)
    payload = b"runtime"
    target = tmp_path / "runtime-copy"
    target.write_bytes(payload)
    (library_root / "libprobe.so").symlink_to(target)
    monkeypatch.setattr(
        probe, "LIBRARIES", {"libprobe.so": (len(payload), sha256(payload).hexdigest())}
    )
    monkeypatch.setattr(
        probe.importlib.metadata,
        "distribution",
        lambda name: SimpleNamespace(locate_file=lambda relative: tmp_path / relative),
    )
    with pytest.raises(ValueError, match="path"):
        probe.library_binding()


def test_packages_pins_versions_python_architecture_and_loader_source(
    tmp_path, monkeypatch
):
    context = tmp_path / "context.py"
    context.write_bytes(b"mock pinned loader")
    monkeypatch.setattr(
        probe.importlib.metadata, "version", lambda name: probe.VERSIONS[name]
    )
    monkeypatch.setattr(
        probe.importlib.metadata,
        "distribution",
        lambda name: SimpleNamespace(locate_file=lambda relative: context),
    )
    monkeypatch.setattr(probe.platform, "python_version", lambda: "3.12.13")
    monkeypatch.setattr(probe.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(
        probe,
        "digest_file",
        lambda path: {"bytes": 408818, "sha256": probe.CONTEXT_SHA},
    )
    assert probe.packages() == probe.VERSIONS


def test_packages_rejects_wrong_runtime_version(tmp_path, monkeypatch):
    context = tmp_path / "context.py"
    context.write_bytes(b"mock pinned loader")
    monkeypatch.setattr(
        probe.importlib.metadata,
        "version",
        lambda name: "0.0.0" if name == "warp-lang" else probe.VERSIONS[name],
    )
    monkeypatch.setattr(
        probe.importlib.metadata,
        "distribution",
        lambda name: SimpleNamespace(locate_file=lambda relative: context),
    )
    monkeypatch.setattr(probe.platform, "python_version", lambda: "3.12.13")
    monkeypatch.setattr(probe.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(
        probe,
        "digest_file",
        lambda path: {"bytes": 408818, "sha256": probe.CONTEXT_SHA},
    )
    with pytest.raises(ValueError, match="packages"):
        probe.packages()


def test_prior_test_inventory_requires_authenticated_exact_plain_test_list(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(probe, "ROOT", tmp_path)
    path = (
        tmp_path
        / "artifacts/tools/cuda-artifact-guard-release-mac-oct7/source-bindings.json"
    )
    path.parent.mkdir(parents=True)
    files = [f"tests/test_{index:02}.py" for index in range(82)]
    path.write_text(probe.canonical({"files": files}).decode())
    monkeypatch.setattr(
        probe,
        "digest_file",
        lambda candidate, cap=16 * 1024**2: {
            "bytes": 4190,
            "sha256": "015b5290c3b998300e5dee30c1dc8237a34fbdfaced6e584e6337b056c1434da",
        },
    )
    result = probe.prior_test_files()
    owned = sorted(name for name in probe.OWN if name.startswith("tests/"))
    assert result == files + owned


@pytest.mark.parametrize(
    "files",
    [
        [f"tests/test_{index:02}.py" for index in range(81)],
        ["tests/test_duplicate.py"] * 82,
        [f"src/test_{index:02}.py" for index in range(82)],
        [f"tests/test_{index:02}.txt" for index in range(82)],
    ],
)
def test_prior_test_inventory_rejects_wrong_count_duplicates_or_paths(
    tmp_path, monkeypatch, files
):
    monkeypatch.setattr(probe, "ROOT", tmp_path)
    path = (
        tmp_path
        / "artifacts/tools/cuda-artifact-guard-release-mac-oct7/source-bindings.json"
    )
    path.parent.mkdir(parents=True)
    path.write_text(probe.canonical({"files": files}).decode())
    monkeypatch.setattr(
        probe,
        "digest_file",
        lambda candidate, cap=16 * 1024**2: {
            "bytes": 4190,
            "sha256": "015b5290c3b998300e5dee30c1dc8237a34fbdfaced6e584e6337b056c1434da",
        },
    )
    with pytest.raises(ValueError, match="scope"):
        probe.prior_test_files()


def _mock_source_checkout(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    root.mkdir()
    monkeypatch.chdir(root)
    monkeypatch.setattr(probe, "ROOT", root)
    monkeypatch.setattr(
        probe,
        "__file__",
        str(root / "src/mjlab_microduck/stance_friction_prefix_cuda_probe.py"),
    )
    for name in probe.OWN:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("source leaf: " + name + "\n")
    leaves = []
    for name in sorted(probe.OWN):
        raw = (root / name).read_bytes()
        oid = sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        leaves.append(f"100644 blob {oid}\t{name}".encode())
    tree = b"\0".join(leaves) + b"\0"
    state = {"branch": probe.BRANCH, "status": "", "paths": sorted(probe.OWN)}

    def read(*args):
        if args == ("git", "rev-parse", "HEAD"):
            return SOURCE
        if args == ("git", "branch", "--show-current"):
            return state["branch"]
        if args == ("git", "status", "--porcelain"):
            return state["status"]
        if args == ("git", "diff", "--name-only", probe.BASE, SOURCE):
            return "\n".join(state["paths"])
        if args == ("git", "rev-parse", SOURCE + "^{tree}"):
            return "a" * 40
        raise AssertionError("unexpected mocked git read: " + repr(args))

    monkeypatch.setattr(probe, "read", read)
    monkeypatch.setattr(probe.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(probe.subprocess, "check_output", lambda *args, **kwargs: tree)
    return root, state


def test_source_binding_accepts_exact_clean_branch_and_seven_path_fence(
    tmp_path, monkeypatch
):
    _root, _state = _mock_source_checkout(tmp_path, monkeypatch)
    assert len(probe.OWN) == 7
    report = probe.source_binding(SOURCE)
    assert report["source"] == SOURCE
    assert report["branch"] == probe.BRANCH
    assert set(report["leaves"]) == probe.OWN
    assert report["tree"] == "a" * 40


@pytest.mark.parametrize("change", ["dirty", "branch", "missing-path", "extra-path"])
def test_source_binding_rejects_dirty_wrong_branch_or_nonexact_fence(
    tmp_path, monkeypatch, change
):
    _root, state = _mock_source_checkout(tmp_path, monkeypatch)
    if change == "dirty":
        state["status"] = " M src/file.py"
    elif change == "branch":
        state["branch"] = "main"
    elif change == "missing-path":
        state["paths"] = state["paths"][:-1]
    else:
        state["paths"] = state["paths"] + ["src/unowned.py"]
    with pytest.raises(ValueError):
        probe.source_binding(SOURCE)


def test_source_binding_rejects_symlink_in_committed_tree(tmp_path, monkeypatch):
    root, _state = _mock_source_checkout(tmp_path, monkeypatch)
    name = sorted(probe.OWN)[0]
    target = root / (name + ".target")
    target.write_bytes((root / name).read_bytes())
    (root / name).unlink()
    (root / name).symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        probe.source_binding(SOURCE)


def test_source_binding_rejects_blob_content_that_does_not_match_tree(
    tmp_path, monkeypatch
):
    root, _state = _mock_source_checkout(tmp_path, monkeypatch)
    (root / sorted(probe.OWN)[0]).write_text("changed after tree capture")
    with pytest.raises(ValueError, match="blob"):
        probe.source_binding(SOURCE)
