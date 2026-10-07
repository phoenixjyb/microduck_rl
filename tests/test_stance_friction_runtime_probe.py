"""CPU-only guards for the runtime-tick owner and its commit fence."""

import ast
from hashlib import sha1, sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

from mjlab_microduck import stance_friction_runtime_probe as probe


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "kernel-object",
        "kernel-func",
        "kernel-code",
        "row-object",
        "row-func",
        "row-code",
    ],
)
def test_dispatch_entries_pin_shared_helper_and_kernel_code(monkeypatch, mutation):
    def kernel_func():
        return 1

    def row_func():
        return 2

    def foreign():
        return 3

    kernel, row = SimpleNamespace(func=kernel_func), SimpleNamespace(func=row_func)
    constraint = SimpleNamespace(_friction_dof=kernel, _efc_row=row)
    fixture = SimpleNamespace(
        _ORIGINAL_KERNEL=kernel,
        _ORIGINAL_KERNEL_FUNC=kernel_func,
        _ORIGINAL_KERNEL_CODE=kernel_func.__code__,
        _ORIGINAL_EFC_ROW=row,
        _ORIGINAL_EFC_FUNC=row_func,
        _ORIGINAL_EFC_CODE=row_func.__code__,
    )
    if mutation is None:
        probe.frozen_dispatch_entries(constraint, fixture)
        return
    kind, slot = mutation.split("-")
    name, function, wrapped = (
        ("_friction_dof", kernel_func, kernel)
        if kind == "kernel"
        else ("_efc_row", row_func, row)
    )
    if slot == "object":
        setattr(constraint, name, SimpleNamespace(func=function))
    elif slot == "func":
        wrapped.func = foreign
    else:
        monkeypatch.setattr(function, "__code__", foreign.__code__)
    with pytest.raises(ValueError, match="shared row helper identity/code"):
        probe.frozen_dispatch_entries(constraint, fixture)


def _git_tree_entry(name, raw):
    oid = sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    return f"100644 blob {oid}\t{name}\0".encode()


def _source_binding_fixture(monkeypatch, tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    path = root / "sample.txt"
    raw = b"committed source\n"
    path.write_bytes(raw)
    fake_file = root / "src/mjlab_microduck/probe.py"
    monkeypatch.setattr(probe, "ROOT", root)
    monkeypatch.setattr(probe, "__file__", str(fake_file))
    monkeypatch.setattr(probe, "OWN", {"sample.txt"})
    monkeypatch.chdir(root)
    source = "a" * 40

    def fake_read(*args):
        if args == ("git", "rev-parse", "HEAD"):
            return source
        if args == ("git", "branch", "--show-current"):
            return probe.BRANCH
        if args == ("git", "status", "--porcelain"):
            return ""
        if args == ("git", "diff", "--name-only", probe.BASE, source):
            return "sample.txt"
        if args == ("git", "rev-parse", source + "^{tree}"):
            return "b" * 40
        raise AssertionError(args)

    monkeypatch.setattr(probe, "read", fake_read)
    monkeypatch.setattr(probe.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        probe.subprocess,
        "check_output",
        lambda *args, **kwargs: _git_tree_entry("sample.txt", raw),
    )
    return root, path, raw, source


def test_protocol_and_admission_flags_are_literal_non_admission():
    assert probe.PROTOCOL == "microduck-friction-runtime-tick-oct8-v1"
    assert probe.FLAGS == {
        "runtime_cause_proven": False,
        "native_qualified": False,
        "full_window_qualified": False,
        "training_authorized": False,
        "physical_acceptance": False,
    }


def test_cpu_prerequisite_cache_is_preserved_auxiliary_not_cuda_evidence(tmp_path):
    raw = b"complete prerequisite bytes"
    (tmp_path / "receipt.json").write_bytes(raw)
    (tmp_path / "inventory.json").write_bytes(b"{}")
    cache = tmp_path / "private-cpu-warp-cache"
    cache.mkdir()
    (cache / "compiled.o").write_bytes(b"not admitted or decoded")
    anchors = {"receipt.json": {"bytes": len(raw), "sha256": sha256(raw).hexdigest()}}
    assert probe._raw_inventory(
        tmp_path, anchors, external_inventory=True, cpu_cache=True
    ) == {"receipt.json": raw}
    assert (cache / "compiled.o").read_bytes() == b"not admitted or decoded"
    with pytest.raises(ValueError, match="exact closed"):
        probe._raw_inventory(tmp_path, anchors, external_inventory=True)


def test_cpu_auxiliary_exception_does_not_hide_extra_raw_packets(tmp_path):
    (tmp_path / "receipt.json").write_bytes(b"x")
    (tmp_path / "inventory.json").write_bytes(b"{}")
    (tmp_path / "extra.json").write_bytes(b"unclaimed")
    anchors = {"receipt.json": {"bytes": 1, "sha256": sha256(b"x").hexdigest()}}
    with pytest.raises(ValueError, match="exact closed"):
        probe._raw_inventory(tmp_path, anchors, external_inventory=True, cpu_cache=True)


def test_reviewed_source_fence_is_exactly_eleven_new_paths():
    assert len(probe.OWN) == 11
    assert probe.OWN == {
        "src/mjlab_microduck/stance_friction_runtime_kernel.py",
        "src/mjlab_microduck/stance_friction_runtime_numerical.py",
        "src/mjlab_microduck/stance_friction_runtime_control.py",
        "src/mjlab_microduck/stance_friction_runtime_probe.py",
        "src/mjlab_microduck/stance_friction_runtime_receiver.py",
        "tests/test_stance_friction_runtime_kernel.py",
        "tests/test_stance_friction_runtime_numerical.py",
        "tests/test_stance_friction_runtime_control.py",
        "tests/test_stance_friction_runtime_probe.py",
        "tests/test_stance_friction_runtime_receiver.py",
        "docs/experiments/2026-10-08-dense-friction-runtime-tick.md",
    }


def test_prefix_probe_usage_is_limited_to_allowed_pure_helpers():
    tree = ast.parse(Path(probe.__file__).read_text())
    forbidden = {
        "source_binding",
        "run",
        "tests",
        "unit",
        "output",
        "service_properties",
        "child",
    }
    assert not any(
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "prefix"
        and node.attr in forbidden
        for node in ast.walk(tree)
    )


@pytest.mark.parametrize("mode", ["tests", "run"])
def test_unit_and_output_are_source_derived_and_bounded(mode):
    source = "1234567890" + "a" * 30
    expected_mode = "tests" if mode == "tests" else "run"
    assert (
        probe.unit(source, mode)
        == f"microduck-friction-runtime-tick-{expected_mode}-1234567890aa.service"
    )
    leaf = probe.output(source, mode)
    assert leaf.name == f"friction-runtime-tick-{mode}-1234567890aa"
    assert leaf.parent.name == ("tools" if mode == "tests" else "evaluations")


@pytest.mark.parametrize(
    "source,mode", [("short", "run"), ("g" * 40, "child"), ("a" * 39 + "G", "tests")]
)
def test_unit_rejects_nonliteral_source_or_mode(source, mode):
    with pytest.raises(ValueError):
        probe.unit(source, mode)


def test_cutoff_reserve_accepts_only_strictly_before_fixed_cutoff():
    assert probe.check_window(60, now=probe.CUTOFF - 61) is None
    with pytest.raises(ValueError, match="cutoff"):
        probe.check_window(60, now=probe.CUTOFF - 60)
    with pytest.raises(ValueError):
        probe.check_window(True, now=probe.CUTOFF - 10)


def test_source_binding_covers_committed_blob_and_full_tree(monkeypatch, tmp_path):
    root, path, raw, source = _source_binding_fixture(monkeypatch, tmp_path)
    result = probe.source_binding(source)
    entry = result["leaves"]["sample.txt"]
    assert result == {
        "source": source,
        "tree": "b" * 40,
        "branch": probe.BRANCH,
        "leaves": {
            "sample.txt": {
                "bytes": len(raw),
                "git_blob": sha1(
                    b"blob " + str(len(raw)).encode() + b"\0" + raw
                ).hexdigest(),
                "sha256": sha256(raw).hexdigest(),
            }
        },
    }
    assert path.read_bytes() == raw
    assert entry["bytes"] == len(raw)


def test_source_binding_rejects_out_of_fence_changed_path(monkeypatch, tmp_path):
    _, _, _, source = _source_binding_fixture(monkeypatch, tmp_path)
    original_read = probe.read

    def changed(*args):
        if args == ("git", "diff", "--name-only", probe.BASE, source):
            return "sample.txt\nunrelated.py"
        return original_read(*args)

    monkeypatch.setattr(probe, "read", changed)
    with pytest.raises(ValueError, match="eleven-path"):
        probe.source_binding(source)


def test_source_binding_rejects_changed_bytes_under_committed_blob(
    monkeypatch, tmp_path
):
    _, path, _, source = _source_binding_fixture(monkeypatch, tmp_path)
    path.write_bytes(b"modified after commit")
    with pytest.raises(ValueError, match="whole committed Git blob"):
        probe.source_binding(source)


def test_test_scope_is_exactly_prior_85_plus_five_new_tests(monkeypatch):
    prior = [f"tests/test_retained_{index:02d}.py" for index in range(85)]
    monkeypatch.setattr(probe.prefix, "prior_test_files", lambda: prior)
    expected = prior + sorted(name for name in probe.OWN if name.startswith("tests/"))
    monkeypatch.setattr(
        probe, "TEST_FILES_SHA256", sha256(probe.canonical(expected)).hexdigest()
    )
    files = probe.test_files()
    assert len(files) == len(set(files)) == 90
    monkeypatch.setattr(probe, "TEST_FILES_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="frozen 85"):
        probe.test_files()
    assert files[:85] == prior
    assert files[85:] == sorted(name for name in probe.OWN if name.startswith("tests/"))


def test_parent_module_keeps_gpu_imports_inside_child_only():
    tree = ast.parse(Path(probe.__file__).read_text())
    top_imports = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            top_imports.append(node.module or "")
    assert not any(name == "torch" or name == "warp" for name in top_imports)
    child = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "child"
    )
    child_imports = [
        node.module for node in ast.walk(child) if isinstance(node, ast.ImportFrom)
    ]
    child_imports.extend(
        alias.name
        for node in ast.walk(child)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert "torch" in child_imports and "warp" in child_imports


def test_child_and_owner_protocol_records_keep_fixed_literal_scope():
    source = Path(probe.__file__).read_text()
    assert '"case_order": list(ARMS)' in source
    assert '"child_timeout_seconds": CHILD_SECONDS' in source
    assert '"cutoff_utc": "2026-10-07T23:30:00Z"' in source
    calls = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "receiver"
        and node.func.attr == "verify_run"
    ]
    assert len(calls) == 1
    assert [arg.id for arg in calls[0].args] == ["directory", "inventory"]
    assert {item.arg: item.value.id for item in calls[0].keywords} == {
        "expected_source": "binding",
        "expected_tests_sha": "tests_inventory_sha",
    }


def test_raw_inventory_rejects_symlink_and_unlisted_leaf(tmp_path):
    root = tmp_path / "inventory"
    root.mkdir()
    target = root / "packet.bin"
    target.write_bytes(b"raw")
    alias = root / "alias.bin"
    alias.symlink_to(target)
    anchors = {
        "packet.bin": {
            "bytes": 3,
            "sha256": "d7439bee24773bcbfa2f8ad07e6e6c05e9a90f28f8c8a8f0c0bdb0f80c7ce11b",
        }
    }
    with pytest.raises((ValueError, FileNotFoundError)):
        probe._raw_inventory(root, anchors)
