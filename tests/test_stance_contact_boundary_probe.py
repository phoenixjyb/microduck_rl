"""CPU-only guards for the passive contact-boundary owner probe."""

from hashlib import sha1, sha256
import json
from pathlib import Path
import os
from types import SimpleNamespace

import pytest

from mjlab_microduck import stance_contact_boundary_probe as probe


@pytest.mark.parametrize("parent_value", (None, "64"))
def test_cpu_child_thread_caps_preserve_parent_environment(
    monkeypatch, tmp_path, parent_value
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    for name in probe.CPU_TEST_THREADS:
        if parent_value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, parent_value)
    before = dict(os.environ)
    cache = tmp_path / "private-cpu-warp-cache"
    child = probe.cpu_test_environment(cache)
    assert os.environ == before
    assert {name: child[name] for name in probe.CPU_TEST_THREADS} == {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    assert child["WARP_CACHE_PATH"] == str(cache)
    assert child["CUDA_VISIBLE_DEVICES"] == ""


def test_cpu_child_thread_caps_refuse_visible_cuda(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="CPU tests keep CUDA hidden"):
        probe.cpu_test_environment(tmp_path)


def _git_tree_entry(name, raw):
    oid = sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    return f"100644 blob {oid}\t{name}\0".encode()


def _source_binding_fixture(monkeypatch, tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    path = root / "sample.txt"
    raw = b"committed source\n"
    path.write_bytes(raw)
    monkeypatch.setattr(probe, "ROOT", root)
    monkeypatch.setattr(probe, "__file__", str(root / "src/mjlab_microduck/probe.py"))
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


def test_protocol_source_fence_and_admission_flags_are_literal():
    assert probe.PROTOCOL == "microduck-contact-boundary-tick-oct8-v1"
    assert probe.BASE == "0ad275806cb64383535fa2338f307fc91dccea99"
    assert probe.BRANCH == "feat/athletics-obstacle-curriculum"
    assert probe.OWN == {
        "src/mjlab_microduck/stance_contact_boundary_control.py",
        "src/mjlab_microduck/stance_contact_boundary_probe.py",
        "src/mjlab_microduck/stance_contact_boundary_receiver.py",
        "tests/test_stance_contact_boundary_control.py",
        "tests/test_stance_contact_boundary_probe.py",
        "tests/test_stance_contact_boundary_receiver.py",
        "docs/experiments/2026-10-08-passive-contact-boundary.md",
    }
    assert probe.FLAGS == {
        "runtime_cause_proven": False,
        "native_qualified": False,
        "full_window_qualified": False,
        "training_authorized": False,
        "physical_acceptance": False,
    }


def test_unit_and_output_are_exact_source_derived_paths():
    source = "1234567890" + "a" * 30
    assert (
        probe.unit(source, "tests")
        == "microduck-contact-boundary-tick-tests-1234567890aa.service"
    )
    assert (
        probe.unit(source, "run")
        == "microduck-contact-boundary-tick-run-1234567890aa.service"
    )
    assert (
        probe.output(source, "tests").name == "contact-boundary-tick-tests-1234567890aa"
    )
    assert probe.output(source, "tests").parent.name == "tools"
    assert probe.output(source, "run").name == "contact-boundary-tick-run-1234567890aa"
    assert probe.output(source, "run").parent.name == "evaluations"


@pytest.mark.parametrize(
    "source,mode",
    [
        ("short", "run"),
        ("g" * 40, "run"),
        ("a" * 39 + "G", "tests"),
        ("a" * 40, "child"),
    ],
)
def test_unit_rejects_nonliteral_source_or_mode(source, mode):
    with pytest.raises(ValueError):
        probe.unit(source, mode)


def test_cutoff_reserve_is_strict_and_plain_integer():
    assert probe.CUTOFF == 1791415800
    assert probe.check_window(60, now=probe.CUTOFF - 61) is None
    with pytest.raises(ValueError, match="cutoff"):
        probe.check_window(60, now=probe.CUTOFF - 60)
    with pytest.raises(ValueError):
        probe.check_window(True, now=probe.CUTOFF - 10)


def test_source_binding_covers_every_committed_blob(monkeypatch, tmp_path):
    root, path, raw, source = _source_binding_fixture(monkeypatch, tmp_path)
    result = probe.source_binding(source)
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
    assert root == Path.cwd()


@pytest.mark.parametrize("damage", ["path-fence", "blob", "symlink", "dirty", "branch"])
def test_source_binding_rejects_source_or_worktree_drift(monkeypatch, tmp_path, damage):
    _, path, _, source = _source_binding_fixture(monkeypatch, tmp_path)
    original_read = probe.read
    if damage == "path-fence":

        def changed(*args):
            if args == ("git", "diff", "--name-only", probe.BASE, source):
                return "sample.txt\nunrelated.py"
            return original_read(*args)

        monkeypatch.setattr(probe, "read", changed)
    elif damage == "blob":
        path.write_bytes(b"changed after commit")
    elif damage == "symlink":
        target = path.with_suffix(".target")
        target.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(target)
    elif damage == "dirty":
        monkeypatch.setattr(
            probe,
            "read",
            lambda *args: (
                " M sample.txt"
                if args == ("git", "status", "--porcelain")
                else original_read(*args)
            ),
        )
    else:
        monkeypatch.setattr(
            probe,
            "read",
            lambda *args: (
                "other-branch"
                if args == ("git", "branch", "--show-current")
                else original_read(*args)
            ),
        )
    with pytest.raises((ValueError, OSError)):
        probe.source_binding(source)


def test_test_scope_is_exact_prior_91_plus_three_new_test_files(monkeypatch):
    from mjlab_microduck import stance_friction_runtime_probe as prior_probe

    prior = [f"tests/test_retained_{index:02d}.py" for index in range(90)]
    monkeypatch.setattr(prior_probe, "test_files", lambda: prior)
    expected = (
        prior
        + ["tests/test_stance_raw_carrier_delta.py"]
        + sorted(name for name in probe.OWN if name.startswith("tests/"))
    )
    monkeypatch.setattr(
        probe, "TEST_FILES_SHA256", sha256(probe.canonical(expected)).hexdigest()
    )
    files = probe.test_files()
    assert len(files) == len(set(files)) == 94
    assert files == expected
    monkeypatch.setattr(probe, "TEST_FILES_SHA256", "f" * 64)
    with pytest.raises(ValueError, match="94-file"):
        probe.test_files()


@pytest.mark.parametrize(
    "xml,expected,match",
    [
        (
            b'<testsuites><testsuite tests="94" errors="0" failures="0" skipped="0"/></testsuites>',
            94,
            None,
        ),
        (
            b'<testsuites><testsuite tests="93" errors="0" failures="0" skipped="0"/></testsuites>',
            94,
            "exact source-test count",
        ),
        (
            b'<testsuites><testsuite tests="94" errors="1" failures="0" skipped="0"/></testsuites>',
            94,
            "exact source-test count",
        ),
        (
            b'<testsuites><testsuite tests="94" errors="0" failures="1" skipped="0"/></testsuites>',
            94,
            "exact source-test count",
        ),
        (
            b'<testsuites><testsuite tests="94" errors="0" failures="0" skipped="1"/></testsuites>',
            94,
            "exact source-test count",
        ),
        (
            b'<testsuites><testsuite tests="94" errors="0" failures="0" skipped="0"/><testsuite tests="0" errors="0" failures="0" skipped="0"/></testsuites>',
            94,
            "exact source-test count",
        ),
    ],
)
def test_junit_requires_exact_collected_count_and_zero_omissions(
    monkeypatch, xml, expected, match
):
    monkeypatch.setattr(probe, "EXPECTED_TESTS", expected)
    if match:
        with pytest.raises(ValueError, match=match):
            probe._checked_junit(xml)
    else:
        assert probe._checked_junit(xml) == expected


def test_junit_refuses_unfrozen_count_and_malformed_or_oversize_input(monkeypatch):
    monkeypatch.setattr(probe, "EXPECTED_TESTS", 0)
    with pytest.raises(ValueError, match="exact source-test count"):
        probe._checked_junit(
            b'<testsuites><testsuite tests="0" errors="0" failures="0" skipped="0"/></testsuites>'
        )
    monkeypatch.setattr(probe, "EXPECTED_TESTS", 94)
    with pytest.raises(ValueError, match="bounded exact JUnit"):
        probe._checked_junit(b"")
    with pytest.raises(ValueError, match="bounded exact JUnit"):
        probe._checked_junit(b"x" * (1024**2 + 1))
    with pytest.raises(Exception):
        probe._checked_junit(b"<not-xml")


def test_raw_inventory_authenticates_whole_leaves_and_caps_before_return(tmp_path):
    root = tmp_path / "raw"
    root.mkdir()
    packet = b"opaque payload"
    (root / "packet.bin").write_bytes(packet)
    anchors = {
        "packet.bin": {"bytes": len(packet), "sha256": sha256(packet).hexdigest()}
    }
    assert probe._raw_inventory(root, anchors) == {"packet.bin": packet}
    with pytest.raises(ValueError, match="per-leaf inventory caps"):
        probe._raw_inventory(
            root, {"packet.bin": {"bytes": 16 * 1024**2 + 1, "sha256": "a" * 64}}
        )
    with pytest.raises(ValueError, match="bounded inventory total"):
        probe._raw_inventory(root, anchors, total_cap=len(packet) - 1)


def test_unclaimed_special_node_refuses_before_reading_any_anchor(
    tmp_path, monkeypatch
):
    (tmp_path / "packet.bin").write_bytes(b"x")
    os.mkfifo(tmp_path / "unclaimed.pipe")
    anchors = {"packet.bin": {"bytes": 1, "sha256": sha256(b"x").hexdigest()}}
    monkeypatch.setattr(
        probe, "digest_file", lambda *_: pytest.fail("must preflight nodes first")
    )
    with pytest.raises(ValueError, match="special inventory nodes"):
        probe._raw_inventory(tmp_path, anchors)


@pytest.mark.parametrize(
    "damage", ["symlink", "unclaimed", "corrupt", "bad-anchor", "unsafe-name"]
)
def test_raw_inventory_rejects_symlinks_unclaimed_bytes_and_bad_anchors(
    tmp_path, damage
):
    root = tmp_path / "raw"
    root.mkdir()
    packet = b"opaque payload"
    target = root / "packet.bin"
    target.write_bytes(packet)
    anchors = {
        "packet.bin": {"bytes": len(packet), "sha256": sha256(packet).hexdigest()}
    }
    if damage == "symlink":
        alias = root / "alias.bin"
        alias.symlink_to(target)
        anchors = {"alias.bin": anchors["packet.bin"]}
    elif damage == "unclaimed":
        (root / "extra.bin").write_bytes(b"unclaimed")
    elif damage == "corrupt":
        target.write_bytes(b"corrupt payload")
    elif damage == "bad-anchor":
        anchors["packet.bin"]["sha256"] = "a" * 64
    else:
        anchors = {"../packet.bin": anchors["packet.bin"]}
    with pytest.raises((ValueError, OSError)):
        probe._raw_inventory(root, anchors)


def _fake_module(monkeypatch):
    import sys
    from types import ModuleType

    class Kernel:
        def __init__(self, func, module):
            self.func = func
            self.module = module

    module = ModuleType("contact_test_module")
    exec(
        "def make_kernel():\n"
        "    IS_ELLIPTIC = IS_SPARSE = False\n"
        "    def kernel_func():\n"
        "        return IS_ELLIPTIC or IS_SPARSE\n"
        "    return kernel_func\n",
        module.__dict__,
    )
    kernel_func = module.make_kernel()
    monkeypatch.setitem(sys.modules, "contact_test_module", module)
    warp_module = object()
    kernel = Kernel(kernel_func, warp_module)

    def factory(cone, sparse):
        assert (cone, sparse) == (0, False)
        return kernel

    return (
        SimpleNamespace(_efc_contact_init=factory),
        factory,
        kernel,
        kernel_func,
        module,
        warp_module,
    )


def test_cached_contact_factory_binding_pins_factory_code_kernel_and_module(
    monkeypatch,
):
    constraint, factory, kernel, function, module, warp_module = _fake_module(
        monkeypatch
    )
    held = probe.contact_factory_binding(constraint)
    assert held == {
        "factory": factory,
        "function": factory,
        "code": factory.__code__,
        "kernel": kernel,
        "kernel_function": function,
        "kernel_code": function.__code__,
        "module": warp_module,
        "python_module_name": "contact_test_module",
        "python_module": module,
    }
    probe.check_contact_factory(constraint, held)
    for mutation in (
        "factory",
        "factory-code",
        "kernel",
        "kernel-func",
        "kernel-code",
        "warp-module",
        "python-module",
        "python-module-name",
        "closure-elliptic",
        "closure-sparse",
    ):
        constraint, factory, kernel, function, module, warp_module = _fake_module(
            monkeypatch
        )
        held = probe.contact_factory_binding(constraint)
        if mutation == "factory":
            constraint._efc_contact_init = lambda *_: kernel
        elif mutation == "factory-code":

            def replacement(cone, sparse):
                return kernel

            factory.__code__ = replacement.__code__
        elif mutation == "kernel":
            kernel.func = lambda: None
        elif mutation == "kernel-func":
            kernel.func = lambda: None
        elif mutation == "kernel-code":
            left = right = False

            def replacement_kernel():
                return left and right

            function.__code__ = replacement_kernel.__code__
        elif mutation.startswith("closure-"):
            function.__closure__[
                0 if mutation == "closure-elliptic" else 1
            ].cell_contents = True
        elif mutation == "warp-module":
            kernel.module = object()
        elif mutation == "python-module-name":
            function.__module__ = "foreign_contact_test_module"
        else:
            monkeypatch.setitem(
                __import__("sys").modules, "contact_test_module", object()
            )
        try:
            probe.check_contact_factory(constraint, held)
        except ValueError:
            continue
        pytest.fail(f"contact binding accepted mutation: {mutation}")


def test_child_source_scope_and_flags_never_claim_native_qualification():
    source = Path(probe.__file__).read_text()
    assert '"case_order": list(ARMS)' in source
    assert '"flags": FLAGS' in source
    assert set(probe.FLAGS.values()) == {False}
    assert probe.CALLER_RNG_SEEDS == {"cpu": 673, "cuda": 677}
    assert probe.ARMS == ("original", "candidate0", "candidate1")


def test_unfrozen_child_refuses_before_cuda_or_source_access(monkeypatch):
    monkeypatch.setattr(probe, "check_window", lambda *_: None)
    monkeypatch.setattr(probe, "EXPECTED_TESTS", 0)
    monkeypatch.setattr(
        probe, "source_binding", lambda *_: pytest.fail("must refuse first")
    )
    with pytest.raises(ValueError, match="fully frozen CPU collection"):
        probe.child("a" * 40, 123, 456, "b" * 64)


class _SizedRaw:
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size


def test_historical_negative_authenticates_all_external_anchors_before_decode(
    monkeypatch, tmp_path
):
    closeout = tmp_path / "artifacts/tools/friction-runtime-tick-closeout-321a72a6b7d3"
    run_dir = tmp_path / "artifacts/evaluations/friction-runtime-tick-run-321a72a6b7d3"
    closeout.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    receiver_report = {
        "decision": "real-model-one-tick-candidate-repeat-negative",
        "flags": dict(probe.FLAGS),
    }
    receiver_raw = probe.canonical(receiver_report)
    inventory_raw = b'{"packet.bin":{"bytes":1,"sha256":"' + "e".encode() * 64 + b'"}}'
    inventory_raw += b" " * (60551 - len(inventory_raw))
    inventory_sha = sha256(inventory_raw).hexdigest()
    replay = {
        "decision": receiver_report["decision"],
        "flags": dict(probe.FLAGS),
        "source": "c" * 40,
        "all_whole_leaves_authenticated": True,
        "native_receiver_bytes_equal": True,
        "raw_inventory": {"bytes": 60551, "sha256": inventory_sha},
    }
    delta = {
        "flags": dict(probe.FLAGS),
        "execution_source": "c" * 40,
        "raw_inventory_sha256": inventory_sha,
        "independent_receiver_sha256": sha256(receiver_raw).hexdigest(),
    }
    contents = {
        "inventory.json": inventory_raw,
        "receiver.json": receiver_raw,
        "mac-receiver.json": receiver_raw,
        "mac-replay.json": probe.canonical(replay),
        "mac-first-delta.json": probe.canonical(delta),
        "terminal.txt": b"retained negative\n",
    }
    anchors = {
        name: (len(raw), sha256(raw).hexdigest()) for name, raw in contents.items()
    }
    monkeypatch.setattr(probe, "NEGATIVE_ANCHORS", anchors)
    monkeypatch.setattr(probe, "NEGATIVE_SOURCE", "c" * 40)
    monkeypatch.setattr(probe, "ROOT", tmp_path)
    monkeypatch.setattr(
        probe,
        "digest_file",
        lambda path, *_: {
            "bytes": len(Path(path).read_bytes()),
            "sha256": sha256(Path(path).read_bytes()).hexdigest(),
        },
    )
    monkeypatch.setattr(
        probe,
        "canonical",
        lambda value: (
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode(),
    )
    for name, raw in contents.items():
        (closeout / name).write_bytes(raw)

    from mjlab_microduck import stance_friction_runtime_receiver as old_receiver

    calls = []
    virtual_raw = {"declaration.json": _SizedRaw(1)}
    per_leaf = (526737887 - 1) // 413
    virtual_raw.update({f"leaf-{index}": _SizedRaw(per_leaf) for index in range(413)})
    # Adjust one virtual leaf so the authenticated inventory has the exact retained total.
    virtual_raw["leaf-0"] = _SizedRaw(
        526737887
        - 1
        - sum(len(virtual_raw[f"leaf-{index}"]) for index in range(1, 413))
    )

    def authenticate(directory, inventory):
        calls.append(("authenticate", directory, inventory))
        return virtual_raw

    def json_packet(raw):
        assert raw is virtual_raw["declaration.json"]
        return {
            "source_binding": {"source": "c" * 40},
            "tests_inventory_sha256": "f" * 64,
        }

    def verify_run(directory, inventory, *, expected_source, expected_tests_sha):
        calls.append(
            ("verify", directory, inventory, expected_source, expected_tests_sha)
        )
        return receiver_report

    monkeypatch.setattr(old_receiver, "authenticate", authenticate)
    monkeypatch.setattr(old_receiver, "json_packet", json_packet)
    monkeypatch.setattr(old_receiver, "verify_run", verify_run)
    result = probe.historical_negative()
    assert result["source"] == "c" * 40
    assert result["decision"] == "real-model-one-tick-candidate-repeat-negative"
    assert result["raw_files"] == 414
    assert result["raw_bytes"] == 526737887
    assert set(result["flags"].values()) == {False}
    assert [call[0] for call in calls] == ["authenticate", "verify"]


def test_historical_negative_anchor_corruption_refuses_before_receiver_decode(
    monkeypatch, tmp_path
):
    closeout = tmp_path / "artifacts/tools/friction-runtime-tick-closeout-321a72a6b7d3"
    closeout.mkdir(parents=True)
    monkeypatch.setattr(probe, "ROOT", tmp_path)
    monkeypatch.setattr(probe, "NEGATIVE_ANCHORS", {"inventory.json": (1, "a" * 64)})
    monkeypatch.setattr(
        probe, "digest_file", lambda *_args, **_kwargs: {"bytes": 1, "sha256": "b" * 64}
    )
    from mjlab_microduck import stance_friction_runtime_receiver as old_receiver

    monkeypatch.setattr(
        old_receiver,
        "authenticate",
        lambda *_: pytest.fail("decode before anchor refusal"),
    )
    with pytest.raises(ValueError, match="whole retained negative anchor"):
        probe.historical_negative()
