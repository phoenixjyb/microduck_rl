"""Synthetic contract tests for the capped CPU CRB archive supervisor."""

from copy import deepcopy
from contextlib import contextmanager
from hashlib import sha256
import os
from pathlib import Path
import subprocess

import pytest
import torch

import test_stance_crb_trace_archive as trace_archive_tests
from mjlab_microduck import stance_crb_archive_cpu_probe as probe
from mjlab_microduck import stance_crb_launch_trace as crb
from mjlab_microduck import stance_crb_trace_archive as archive
from mjlab_microduck import stance_recovery_cuda_inertia_probe as prior
from mjlab_microduck import stance_recovery_cuda_rollout_evidence as evidence
from mjlab_microduck.first_attempt_smoke import canonical

SOURCE = "e" * 40
INVOCATION = "a" * 32


@pytest.fixture(scope="module")
def _module_cuda_hidden():
    prior_value = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        yield
    finally:
        if prior_value is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = prior_value


@pytest.fixture(scope="module")
def _cpu64_case(_module_cuda_hidden):
    return trace_archive_tests.actual_64_case.__wrapped__(_module_cuda_hidden)


def properties(main_pid=None):
    return {
        "MainPID": str(os.getpid()) if main_pid is None else str(main_pid),
        "ActiveState": "active",
        "RuntimeMaxUSec": "5min",
        "MemoryMax": str(probe.MEMORY),
        "CPUQuotaPerSecUSec": "2s",
        "Nice": "10",
        "KillMode": "control-group",
        "InvocationID": INVOCATION,
        "RemainAfterExit": "yes",
    }


def test_exact_caps_distinct_namespaces_and_owner_frozen_test_gate():
    assert probe.CAP == 300
    assert probe.MEMORY == 6 * 1024**3
    assert probe.CPU_QUOTA == "2s" and probe.NICE == "10"
    assert probe.KILL_MODE == "control-group"
    assert probe.RUN_RESERVE == probe.CAP + probe.MARGIN
    assert probe.BASE_SOURCE == "90f8edc7230ed8f7aac3b7e0c2b1ec230d9f1e81"
    assert probe.SYNC_FROM_SOURCE == "766271542f4c168ebe00c56e8e03755c3f44be2b"
    assert probe.SYNC_ROOT == probe.prior.SYNC_ROOT
    assert probe.SYNC_ORIGIN == probe.prior.SYNC_ORIGIN
    assert probe.OWN == (
        "src/mjlab_microduck/stance_crb_archive_cpu_probe.py",
        "tests/test_stance_crb_archive_cpu_probe.py",
        probe.DOC,
    )
    assert type(probe.EXPECTED_TESTS) is int and probe.EXPECTED_TESTS >= 0
    assert len(probe.TEST_FILES) == 44  # 43 frozen inputs plus this new test.
    assert probe.TEST_FILES[-1] == "test_stance_crb_archive_cpu_probe.py"
    assert probe.unit(SOURCE, "run") == (
        f"microduck-crb-archive-cpu-run-{SOURCE[:12]}.service"
    )
    assert probe.unit(SOURCE, "tests") == (
        f"microduck-crb-archive-cpu-tests-{SOURCE[:12]}.service"
    )
    assert probe.output_path(SOURCE, "run").name == (
        f"stance-crb-archive-cpu-witness-{SOURCE[:12]}"
    )
    assert probe.output_path(SOURCE, "tests").name == (
        f"stance-crb-archive-cpu-tests-{SOURCE[:12]}"
    )
    assert probe.source_sync_unit(SOURCE) == (
        f"microduck-crb-archive-cpu-sync-{SOURCE[:12]}.service"
    )
    assert probe.output_path(SOURCE, "run") != probe.prior.output_path(SOURCE)
    for bad_source in ("bad", "F" * 40, SOURCE.upper(), "e" * 39, "e" * 41):
        with pytest.raises(ValueError):
            probe.unit(bad_source, "run")
        with pytest.raises(ValueError):
            probe.source_sync_unit(bad_source)
    with pytest.raises(ValueError):
        probe.unit(SOURCE, "closeout")


@pytest.mark.parametrize(
    "key,value",
    [
        ("MainPID", "0"),
        ("MainPID", True),
        ("ActiveState", "inactive"),
        ("RuntimeMaxUSec", "4min"),
        ("MemoryMax", "infinity"),
        ("CPUQuotaPerSecUSec", "infinity"),
        ("Nice", "0"),
        ("KillMode", "process"),
        ("InvocationID", "bad"),
        ("RemainAfterExit", "no"),
    ],
)
def test_typed_service_record_rejects_wrong_caps_and_values(key, value):
    changed = properties() | {key: value}
    with pytest.raises(ValueError):
        probe._recorded_properties(changed)


def test_source_sync_template_retargets_exactly_two_guards(monkeypatch):
    source, bundle = "f" * 40, "c" * 64
    template = probe.prior.source_sync_script(source, bundle_sha256=bundle)
    result = probe.source_sync_script(source, bundle_sha256=bundle)
    old_head = f'test "$(git rev-parse HEAD)" = {probe.prior.SYNC_FROM_SOURCE}\n'
    new_head = f'test "$(git rev-parse HEAD)" = {probe.SYNC_FROM_SOURCE}\n'
    old_unit = f'if test "$duck_sync_running" != {probe.prior.source_sync_unit(source)}; then\n'
    new_unit = (
        f'if test "$duck_sync_running" != {probe.source_sync_unit(source)}; then\n'
    )
    assert template.count(old_head) == result.count(new_head) == 1
    assert template.count(old_unit) == result.count(new_unit) == 1
    assert result.count(probe.SYNC_FROM_SOURCE) == 1
    assert probe.prior.SYNC_FROM_SOURCE not in result
    assert probe.source_sync_unit(source) in result
    assert "set -euo pipefail" in result
    assert str(probe.SYNC_ROOT) in result and probe.SYNC_ORIGIN in result
    assert str(bundle) in result
    assert "git bundle verify" in result and "git bundle list-heads" in result
    assert "sha256sum" in result and "git merge --ff-only" in result
    assert f"{probe.prior.SYNC_BUNDLE_LIMIT}" in result
    assert "test ! -L" in result and "sha256sum" in result
    for checksum in ("bad", "D" * 64, "d" * 63):
        with pytest.raises(ValueError):
            probe.source_sync_script(source, bundle_sha256=checksum)

    for guard in (old_head, old_unit):
        monkeypatch.setattr(
            probe.prior,
            "source_sync_script",
            lambda *_a, _guard=guard, **_k: template.replace(_guard, ""),
        )
        with pytest.raises(ValueError):
            probe.source_sync_script(source, bundle_sha256=bundle)
        monkeypatch.setattr(
            probe.prior,
            "source_sync_script",
            lambda *_a, _guard=guard, **_k: template + _guard,
        )
        with pytest.raises(ValueError):
            probe.source_sync_script(source, bundle_sha256=bundle)


def _source_fixture(
    tmp_path,
    source,
    changed=None,
    *,
    head=None,
    branch=None,
    origin=None,
    status=b"",
    ancestor=True,
):
    root = tmp_path / "repo"
    contents = {}
    for path in set(probe.FROZEN_FILES) | set(probe.OWN):
        raw = ("new:" if path in probe.OWN else "base:").encode() + path.encode()
        contents[path] = raw
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    source_value = "e" * 40
    original_head = head or source_value
    allowed_changes = set(probe.OWN) if changed is None else set(changed)

    def fake_run(args, **kwargs):
        cmd = args[1:]
        if cmd == ["rev-parse", "HEAD"]:
            stdout = (original_head + "\n").encode()
        elif cmd == ["branch", "--show-current"]:
            stdout = ((branch or "feat/athletics-obstacle-curriculum") + "\n").encode()
        elif cmd == ["remote", "get-url", "origin"]:
            stdout = ((origin or probe.SYNC_ORIGIN) + "\n").encode()
        elif cmd == ["status", "--porcelain"]:
            stdout = status
        elif cmd == ["diff", "--name-only", probe.BASE_SOURCE, source_value]:
            stdout = ("\n".join(sorted(allowed_changes)) + "\n").encode()
        elif (
            len(cmd) == 2 and cmd[0] == "show" and cmd[1].startswith(source_value + ":")
        ):
            stdout = contents[cmd[1].split(":", 1)[1]]
        elif (
            len(cmd) == 2
            and cmd[0] == "show"
            and cmd[1].startswith(probe.BASE_SOURCE + ":")
        ):
            path = cmd[1].split(":", 1)[1]
            stdout = ("base:" + path).encode()
        elif cmd == ["merge-base", "--is-ancestor", probe.BASE_SOURCE, source_value]:
            result = subprocess.CompletedProcess(args, int(not ancestor), b"", b"")
            if kwargs.get("check") and result.returncode:
                raise subprocess.CalledProcessError(result.returncode, args)
            return result
        else:
            raise AssertionError((args, kwargs))
        return subprocess.CompletedProcess(args, 0, stdout, b"")

    return root, source_value, fake_run


def test_source_binding_checks_native_root_ancestor_clean_head_and_only_new_leaves(
    tmp_path, monkeypatch
):
    root, source, fake_run = _source_fixture(tmp_path, "")
    monkeypatch.setattr(probe, "__file__", str(root / "src/mjlab_microduck/probe.py"))
    monkeypatch.setattr(probe, "SYNC_ROOT", str(root))
    monkeypatch.setattr(probe.subprocess, "run", fake_run)
    binding = probe.source_binding(source)
    assert binding["source"] == source
    assert binding["branch"] == "feat/athletics-obstacle-curriculum"
    assert set(binding["leaves"]) == set(probe.FROZEN_FILES) | set(probe.OWN)


def test_source_binding_rejects_worktree_leaf_different_from_committed_bytes(
    tmp_path, monkeypatch
):
    root, source, fake_run = _source_fixture(tmp_path, "")
    monkeypatch.setattr(probe, "__file__", str(root / "src/mjlab_microduck/probe.py"))
    monkeypatch.setattr(probe, "SYNC_ROOT", str(root))

    def mismatch(args, **kwargs):
        result = fake_run(args, **kwargs)
        if args[1:] == ["show", f"{source}:{probe.OWN[0]}"]:
            return subprocess.CompletedProcess(
                args, 0, b"different committed bytes", b""
            )
        return result

    monkeypatch.setattr(probe.subprocess, "run", mismatch)
    with pytest.raises(ValueError, match="committed source leaf"):
        probe.source_binding(source)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"changed": {"src/mjlab_microduck/stance_attempt_trace.py"}},
        {"head": "d" * 40},
        {"branch": "main"},
        {"origin": "https://example.invalid/repo.git"},
        {"status": b" M prior.py\n"},
        {"ancestor": False},
    ],
)
def test_source_binding_rejects_frozen_source_mutation_or_wrong_native_context(
    tmp_path, monkeypatch, kwargs
):
    root, source, fake_run = _source_fixture(tmp_path, "", **kwargs)
    monkeypatch.setattr(probe, "__file__", str(root / "src/mjlab_microduck/probe.py"))
    monkeypatch.setattr(probe, "SYNC_ROOT", str(root))
    monkeypatch.setattr(probe.subprocess, "run", fake_run)
    with pytest.raises((OSError, ValueError, subprocess.CalledProcessError)):
        probe.source_binding(source)


def test_source_binding_refuses_non_native_root(tmp_path, monkeypatch):
    root, source, fake_run = _source_fixture(tmp_path, "")
    monkeypatch.setattr(probe, "__file__", str(root / "src/mjlab_microduck/probe.py"))
    monkeypatch.setattr(probe, "SYNC_ROOT", str(root / "other-checkout"))
    monkeypatch.setattr(probe.subprocess, "run", fake_run)
    with pytest.raises(ValueError, match="exact native WSL source root"):
        probe.source_binding(source)


def test_inventory_enforces_exact_names_and_per_file_limits(tmp_path):
    (tmp_path / "stages.bin").write_bytes(b"stage bytes")
    (tmp_path / "witness.json").write_bytes(b"{}")
    inventory = probe._inventory(tmp_path, {"stages.bin", "witness.json"})
    assert inventory == {
        name: {"bytes": len(raw), "sha256": sha256(raw).hexdigest()}
        for name, raw in {"stages.bin": b"stage bytes", "witness.json": b"{}"}.items()
    }
    (tmp_path / "unexpected").write_bytes(b"bad")
    with pytest.raises(ValueError):
        probe.base._exact_inventory(tmp_path, {"stages.bin", "witness.json"})


def test_execute_refuses_before_service_or_output_side_effect_when_cpu_profile_fails(
    monkeypatch,
):
    calls = []
    monkeypatch.setattr(
        probe, "_hidden", lambda: (_ for _ in ()).throw(ValueError("CPU profile"))
    )
    monkeypatch.setattr(
        probe, "_service_values", lambda *_args: calls.append("service")
    )
    monkeypatch.setattr(probe, "source_binding", lambda *_args: calls.append("source"))
    monkeypatch.setattr(Path, "mkdir", lambda *_args, **_kwargs: calls.append("mkdir"))
    with pytest.raises(ValueError, match="CPU profile"):
        probe.execute(SOURCE, "run")
    assert calls == []


def test_child_lease_and_parent_guards_precede_runtime_creation(monkeypatch):
    calls = []

    def no_lease(_fd):
        calls.append("lease")
        raise ValueError("lease missing")

    monkeypatch.setattr(probe.base.training_smoke, "inherited_lease", no_lease)
    monkeypatch.setattr(probe, "_hidden", lambda: calls.append("hidden"))
    monkeypatch.setattr(
        probe, "ScheduledRecoveryRuntime", lambda *_a, **_k: calls.append("runtime")
    )
    with pytest.raises(ValueError, match="lease missing"):
        probe._child(SOURCE, 7, 10)
    assert calls == ["lease"]


def test_child_parent_identity_refusal_precedes_service_and_runtime(monkeypatch):
    calls = []
    monkeypatch.setattr(probe.base.training_smoke, "inherited_lease", lambda _fd: None)
    monkeypatch.setattr(probe, "_hidden", lambda: calls.append("hidden"))
    monkeypatch.setattr(probe.os, "getppid", lambda: 101)
    monkeypatch.setattr(probe, "_service_values", lambda *_a: calls.append("service"))
    monkeypatch.setattr(
        probe, "ScheduledRecoveryRuntime", lambda *_a, **_k: calls.append("runtime")
    )
    with pytest.raises(ValueError, match="owned CPU witness child parent PID"):
        probe._child(SOURCE, 7, 102)
    assert calls == ["hidden"]


def test_child_authenticates_original_failure_before_constructing_runtime(monkeypatch):
    calls = []
    monkeypatch.setattr(probe.base.training_smoke, "inherited_lease", lambda _fd: None)
    monkeypatch.setattr(probe, "_hidden", lambda: calls.append("hidden"))
    monkeypatch.setattr(probe.os, "getppid", lambda: 101)
    monkeypatch.setattr(
        probe, "_owned_properties", lambda *_a: calls.append("service") or properties()
    )

    def fail_original(_source):
        calls.append("original")
        raise ValueError("original failure changed")

    monkeypatch.setattr(probe, "_authenticate_original_without_loading", fail_original)
    monkeypatch.setattr(
        probe, "ScheduledRecoveryRuntime", lambda *_a, **_k: calls.append("runtime")
    )
    with pytest.raises(ValueError, match="original failure changed"):
        probe._child(SOURCE, 7, 101)
    assert calls == ["hidden", "service", "original"]


def test_child_retains_synthetic_cpu64_witness_without_duplicate_declaration(
    tmp_path, monkeypatch, _cpu64_case, _module_cuda_hidden
):
    case = _cpu64_case
    assert not torch.cuda.is_initialized()
    tmp_path.mkdir(exist_ok=True)
    binding = {"source": SOURCE, "leaves": {}}
    original = {
        "source_binding": binding,
        "original_files": {"old": {"sha256": "a" * 64}},
        "original_anchors": {"smooth": "b" * 64},
        "original_launch_sha256": "c" * 64,
        "original_pair_accepted": False,
        "original_pair_error": "paired rollout semantic state exactness",
    }

    class Runtime:
        def step_with_schedule(self, actions, *, capture_control):
            assert actions.shape == (64, 10) and capture_control is True
            return case["record"]["runtime_result_before_reset"]

    class Observer:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def capture(self):
            return case["trace"]

    monkeypatch.setattr(probe.base.training_smoke, "inherited_lease", lambda _fd: None)
    monkeypatch.setattr(probe, "_hidden", lambda: None)
    monkeypatch.setattr(probe.os, "getppid", lambda: 1234)
    monkeypatch.setattr(probe, "_owned_properties", lambda *_args: properties("1234"))
    monkeypatch.setattr(
        probe, "_authenticate_original_without_loading", lambda _source: original
    )
    monkeypatch.setattr(probe, "source_binding", lambda _source: binding)
    monkeypatch.setattr(probe, "declaration", lambda _source: case["declaration"])
    monkeypatch.setattr(probe, "ScheduledRecoveryRuntime", lambda *_a, **_k: Runtime())
    monkeypatch.setattr(probe.crb, "CrbLaunchTrace", lambda *_a, **_k: Observer())
    monkeypatch.setattr(probe, "output_path", lambda *_args: tmp_path)
    witness = probe._child(SOURCE, 17, 1234)
    assert witness["cpu_only"] is True and witness["shared_lease_verified"] is True
    assert witness["pre_pulse_only"] is True
    assert witness["original_pair_accepted"] is False
    assert set(path.name for path in tmp_path.iterdir()) == (
        probe.ARCHIVE_FILES | {"witness.json"}
    )
    assert not torch.cuda.is_initialized()


def test_test_receipt_requires_frozen_positive_count_hash_and_zero_skips(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(probe, "EXPECTED_TESTS", 2)
    monkeypatch.setattr(probe, "output_path", lambda _source, _mode: tmp_path)
    terminal_state = {
        "MainPID": "0",
        "ActiveState": "active",
        "SubState": "exited",
        "RemainAfterExit": "yes",
        "NRestarts": "0",
        "ExecMainStatus": "0",
        "Result": "success",
        "InvocationID": INVOCATION,
    }

    completed_queries = []

    def terminal_read(*args):
        completed_queries.append(args)
        assert args[3] == probe.unit(SOURCE, "tests")
        return terminal_state[args[-2]]

    monkeypatch.setattr(
        probe.base.host,
        "read",
        terminal_read,
    )
    log = b"2 passed in 0.03s\n"
    (tmp_path / "pytest.log").write_bytes(log)
    receipt = {
        "protocol": probe.PROTOCOL + ":tests",
        "source": SOURCE,
        "mode": "tests",
        "status": "passed",
        "source_binding": {"source": SOURCE},
        "test_files": list(probe.TEST_FILES),
        "passed": 2,
        "skips": 0,
        "pytest_sha256": sha256(log).hexdigest(),
        "service_properties": properties(),
        **probe.FLAGS,
    }
    report = {
        "protocol": probe.PROTOCOL + ":tests",
        "source": SOURCE,
        "mode": "tests",
        "status": "passed",
        "source_binding": {"source": SOURCE},
        "service_properties": properties(),
        "passed": 2,
        "pytest_sha256": sha256(log).hexdigest(),
        "elapsed_seconds": 1.0,
        **probe.FLAGS,
    }
    (tmp_path / "receipt.json").write_bytes((canonical(receipt) + "\n").encode())
    (tmp_path / "report.json").write_bytes((canonical(report) + "\n").encode())
    assert (
        probe._read_tests(SOURCE, {"source": SOURCE})
        == sha256((tmp_path / "receipt.json").read_bytes()).hexdigest()
    )
    assert len(completed_queries) == 8
    assert {call[3] for call in completed_queries} == {probe.unit(SOURCE, "tests")}
    assert {call[-2] for call in completed_queries} == {
        "MainPID",
        "ActiveState",
        "SubState",
        "RemainAfterExit",
        "NRestarts",
        "ExecMainStatus",
        "Result",
        "InvocationID",
    }

    for key, value in (
        ("SubState", "running"),
        ("RemainAfterExit", "no"),
        ("InvocationID", "d" * 32),
    ):
        terminal_state[key] = value
        with pytest.raises(ValueError, match="successful terminal CPU probe service"):
            probe._read_tests(SOURCE, {"source": SOURCE})
        terminal_state[key] = {
            "SubState": "exited",
            "RemainAfterExit": "yes",
            "InvocationID": INVOCATION,
        }[key]

    bad = deepcopy(receipt)
    bad["passed"] = 1
    (tmp_path / "receipt.json").write_bytes((canonical(bad) + "\n").encode())
    with pytest.raises(ValueError):
        probe._read_tests(SOURCE, {"source": SOURCE})

    wrong_log = b"1 passed in 0.03s\n"
    (tmp_path / "pytest.log").write_bytes(wrong_log)
    bad = deepcopy(receipt)
    bad["pytest_sha256"] = sha256(wrong_log).hexdigest()
    (tmp_path / "receipt.json").write_bytes((canonical(bad) + "\n").encode())
    with pytest.raises(ValueError, match="exact owner-frozen CPU test log"):
        probe._read_tests(SOURCE, {"source": SOURCE})

    skipped_log = log + b"1 skipped in 0.01s\n"
    (tmp_path / "pytest.log").write_bytes(skipped_log)
    bad = deepcopy(receipt)
    bad["pytest_sha256"] = sha256(skipped_log).hexdigest()
    (tmp_path / "receipt.json").write_bytes((canonical(bad) + "\n").encode())
    with pytest.raises(ValueError, match="exact owner-frozen CPU test log"):
        probe._read_tests(SOURCE, {"source": SOURCE})

    (tmp_path / "pytest.log").write_bytes(log)
    bad = deepcopy(receipt)
    bad["service_properties"]["MemoryMax"] = str(4 * 1024**3)
    (tmp_path / "receipt.json").write_bytes((canonical(bad) + "\n").encode())
    with pytest.raises(ValueError, match="exact CPU archive service caps"):
        probe._read_tests(SOURCE, {"source": SOURCE})

    (tmp_path / "receipt.json").write_bytes((canonical(receipt) + "\n").encode())
    (tmp_path / "report.json").unlink()
    with pytest.raises(ValueError, match="exact retained evidence inventory"):
        probe._read_tests(SOURCE, {"source": SOURCE})


def _mock_test_execution(tmp_path, monkeypatch, log_bytes):
    binding = {"source": SOURCE, "leaves": {}}
    originals = {
        "original_files": {"source": "pinned"},
        "original_anchors": {"smooth": "pinned"},
    }
    identity = {"source": SOURCE, "host": "synthetic"}
    out = tmp_path / "test-output"

    @contextmanager
    def lease():
        yield 17

    monkeypatch.setattr(probe, "_hidden", lambda: None)
    monkeypatch.setattr(probe.prior, "check_window", lambda **_kwargs: None)
    monkeypatch.setattr(probe, "_service_values", lambda *_args: properties())
    monkeypatch.setattr(probe, "source_binding", lambda _source: binding)
    monkeypatch.setattr(probe.base.host, "identity", lambda _source: identity)
    monkeypatch.setattr(
        probe, "_authenticate_original_without_loading", lambda _source: originals
    )
    monkeypatch.setattr(probe, "output_path", lambda *_args: out)
    monkeypatch.setattr(probe.base.gap.base.files, "gpu_lease", lease)
    monkeypatch.setattr(probe.base.gpu_idle_gate, "wait_idle", lambda: {"idle": True})
    monkeypatch.setattr(
        probe.base.gap.base.retained.d0,
        "filmbrain_state",
        lambda: {"generation": 1},
    )
    monkeypatch.setattr(
        probe.base.gap.base,
        "protected_state",
        lambda: {"filmbrain": "inactive", "vllm": "inactive"},
    )
    monkeypatch.setattr(probe, "_cpu_service_env", lambda: {"CUDA_VISIBLE_DEVICES": ""})

    def run_process(_command, log_path, _timeout, *, env, fd):
        assert env["CUDA_VISIBLE_DEVICES"] == "" and fd == 17
        log_path.write_bytes(log_bytes)

    monkeypatch.setattr(probe.prior, "_run_process", run_process)
    return binding, out


@pytest.mark.parametrize(
    "log_bytes,accepted",
    [
        (b"2 passed in 0.03s\n", True),
        (b"1 passed in 0.03s\n", False),
        (b"2 passed in 0.03s\n1 skipped in 0.01s\n", False),
        (b"2 passed in 0.03s\n1 xfailed in 0.01s\n", False),
    ],
)
def test_supervisor_test_log_requires_exact_positive_count_and_zero_skips(
    tmp_path, monkeypatch, log_bytes, accepted
):
    monkeypatch.setattr(probe, "EXPECTED_TESTS", 2)
    binding, out = _mock_test_execution(tmp_path, monkeypatch, log_bytes)
    if accepted:
        report = probe.execute(SOURCE, "tests")
        assert report["status"] == "passed"
        receipt = probe._parse_canonical_json(
            (out / "receipt.json").read_bytes(), "test receipt"
        )
        assert receipt["passed"] == 2 and receipt["skips"] == 0
        assert receipt["source_binding"] == binding
        assert receipt["pytest_sha256"] == sha256(log_bytes).hexdigest()
    else:
        with pytest.raises(ValueError, match="exact frozen CPU suite, no skips"):
            probe.execute(SOURCE, "tests")
        retained = probe._parse_canonical_json(
            (out / "report.json").read_bytes(), "failed test report"
        )
        assert retained["status"] == "failed-retained"


def test_zero_unfrozen_test_count_fails_closed_before_pytest(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "EXPECTED_TESTS", 0)
    calls = []
    _binding, out = _mock_test_execution(tmp_path, monkeypatch, b"0 passed in 0.01s\n")
    monkeypatch.setattr(
        probe.prior,
        "_run_process",
        lambda *_args, **_kwargs: calls.append("pytest"),
    )
    with pytest.raises(
        ValueError, match="complete test count owner-frozen before launch"
    ):
        probe.execute(SOURCE, "tests")
    assert calls == []
    assert not (out / "receipt.json").exists()
    assert (out / "report.json").is_file()


def _write_cpu_witness(tmp_path, case):
    trace = case["trace"]
    declaration = case["declaration"]
    first_record = prior._producer_tree(case["record"])
    projected = crb._project(trace)
    manifest, inertia_raw, stage_raw = archive.encode(trace, declaration, first_record)
    decoded, score = archive.decode(
        manifest, inertia_raw, stage_raw, projected, declaration, first_record
    )
    assert probe._tree_equal(trace, decoded)
    rng_state = torch.random.get_rng_state().clone()
    files = {
        "declaration.json": (canonical(declaration) + "\n").encode(),
        "inertia.pt": inertia_raw,
        "stages.bin": stage_raw,
        "manifest.json": (canonical(manifest) + "\n").encode(),
        "first-record.pt": evidence.encode(first_record),
        "rng.pt": evidence.encode({"cpu_before": rng_state, "cpu_after": rng_state}),
    }
    for name, raw in files.items():
        (tmp_path / name).write_bytes(raw)
    (tmp_path / "child.log").write_bytes(b"synthetic CPU child summary\n")
    file_inventory = probe._inventory(tmp_path, probe.ARCHIVE_FILES)
    source_binding = {"source": SOURCE, "leaves": {}}
    originals = {
        "original_files": {"old": {"sha256": "a" * 64}},
        "original_anchors": {"smooth": "b" * 64},
        "original_launch_sha256": "c" * 64,
    }
    witness = {
        "protocol": probe.PROTOCOL + ":witness-v1",
        "source": SOURCE,
        "source_binding": source_binding,
        "service_properties": properties(),
        "original_files": originals["original_files"],
        "original_anchors": originals["original_anchors"],
        "original_pair_accepted": False,
        "original_pair_error": "paired rollout semantic state exactness",
        "original_launch_sha256": originals["original_launch_sha256"],
        "files": file_inventory,
        "trace_score": score,
        "trace_tree_sha256_before": probe._tree_digest(trace),
        "trace_tree_sha256_after": probe._tree_digest(decoded),
        "caller_rng_preserved": True,
        "cpu_only": True,
        "pre_pulse_only": True,
        "shared_lease_verified": True,
        "full_native_window_qualified": False,
        "optimizer_steps": 0,
        **probe.FLAGS,
    }
    (tmp_path / "witness.json").write_bytes((canonical(witness) + "\n").encode())
    return source_binding, properties(), originals


def test_witness_authenticates_all_artifacts_before_weights_only_loads(
    tmp_path, monkeypatch, _cpu64_case, _module_cuda_hidden
):
    source_binding, service, originals = _write_cpu_witness(tmp_path, _cpu64_case)
    monkeypatch.setattr(probe, "_hidden", lambda: None)
    monkeypatch.setattr(
        probe, "_authenticate_original_without_loading", lambda _source: originals
    )
    load_calls = []
    original_load = torch.load

    def counted_load(*args, **kwargs):
        load_calls.append(kwargs.copy())
        return original_load(*args, **kwargs)

    monkeypatch.setattr(probe.torch, "load", counted_load)
    result = probe._verify_witness(tmp_path, SOURCE, source_binding, service)
    assert len(load_calls) == 3
    assert all(call.get("weights_only") is True for call in load_calls)
    assert len(result["files"]) == len(probe.ARCHIVE_FILES)
    assert result["trace_score"]["worlds"] == 64


def test_witness_tampering_is_rejected_before_any_tensor_load(
    tmp_path, monkeypatch, _cpu64_case, _module_cuda_hidden
):
    source_binding, service, originals = _write_cpu_witness(tmp_path, _cpu64_case)
    monkeypatch.setattr(probe, "_hidden", lambda: None)
    monkeypatch.setattr(
        probe, "_authenticate_original_without_loading", lambda _source: originals
    )
    with (tmp_path / "stages.bin").open("r+b") as stream:
        first = stream.read(1)
        stream.seek(0)
        stream.write(bytes([first[0] ^ 1]))
    calls = []
    monkeypatch.setattr(probe.torch, "load", lambda *_a, **_k: calls.append("load"))
    with pytest.raises(ValueError):
        probe._verify_witness(tmp_path, SOURCE, source_binding, service)
    assert calls == []


def test_witness_flags_and_cpu_only_metadata_are_non_admitting(_cpu64_case):
    assert _cpu64_case["trace"]["worlds"] == 64
    assert all(value is False for value in probe.FLAGS.values())
    assert probe.FLAGS["full_native_window_qualified"] is False
    assert probe.FLAGS["optimizer_update_qualified"] is False
    assert probe.FLAGS["execution_admitted"] is False
    assert type(probe.EXPECTED_TESTS) is int and probe.EXPECTED_TESTS >= 0
