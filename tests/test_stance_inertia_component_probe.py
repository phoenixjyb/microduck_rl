"""Synthetic CPU supervision tests for component-level inertia diagnosis."""

from contextlib import contextmanager
from pathlib import Path
import subprocess

import pytest

from mjlab_microduck import stance_inertia_component_probe as probe
from mjlab_microduck.first_attempt_smoke import canonical

SOURCE = "b" * 40
INVOCATION = "d" * 32
CAPS = {
    "MainPID": "123",
    "ActiveState": "active",
    "RuntimeMaxUSec": "5min",
    "MemoryMax": str(4 * 1024**3),
    "CPUQuotaPerSecUSec": "2s",
    "Nice": "10",
    "KillMode": "control-group",
    "InvocationID": INVOCATION,
}


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(probe.components.torch.cuda, "is_initialized", lambda: False)


@pytest.fixture
def retained_inputs(tmp_path, monkeypatch):
    for name in probe.INPUT_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"{}\n")
    inventory = probe.prior._inventory(tmp_path, probe.prior.COMPLETE_FILES)
    diagnosis = dict(
        protocol=probe.prior.PROTOCOL + ":failure-diagnosis",
        source=probe.PARENT,
        status="failed-pair-independently-diagnosed",
        pair_error="paired rollout semantic state exactness",
        pair_accepted=False,
        files_rehashed=inventory,
        launch_sha256=inventory["launch.json"]["sha256"],
        **probe.prior.FLAGS,
    )
    raw = (canonical(diagnosis) + "\n").encode()
    (tmp_path / "independent-failure-diagnosis.json").write_bytes(raw)
    monkeypatch.setattr(probe, "DIAGNOSIS_SHA", probe.base.digest(raw))
    monkeypatch.setattr(probe.prior, "output_path", lambda _source: tmp_path)
    launch = {
        "service_properties": {"InvocationID": probe.RUN_INVOCATION},
        "schedule": {"fixture": "schedule"},
        "compiled_plant": {"fixture": "compiled-plant"},
    }
    monkeypatch.setattr(probe.prior, "_read_launch", lambda *_: launch)
    terminals = []
    monkeypatch.setattr(
        probe.prior, "_failed_terminal", lambda *args: terminals.append(args)
    )
    return tmp_path, diagnosis, terminals, launch


def test_caps_separate_namespaces_and_declared_23_file_input_inventory():
    assert probe.CAP == 300 and probe.MEMORY == 4 * 1024**3
    assert len(probe.INPUT_FILES) == 23
    assert len(probe.TEST_FILES) == 39
    assert probe.EXPECTED_TESTS == 1182  # Full frozen suite, never a subset receipt.
    assert probe.unit(SOURCE, "run") != probe.prior.unit(probe.PARENT, "run")
    assert probe.output_path(SOURCE, "run") != probe.prior.output_path(probe.PARENT)
    assert probe.source_sync_unit(SOURCE) != probe.prior.source_sync_unit(SOURCE)
    assert all(value is False for value in probe.FLAGS.values())
    for mode in ("invalid", "preflight", "closeout"):
        with pytest.raises(ValueError):
            probe.unit(SOURCE, mode)


@pytest.mark.parametrize(
    "damage", [None, "diagnosis-hash", "diagnosis-flags", "original-file", "invocation"]
)
def test_authentication_binds_all_23_original_files_before_any_tensor_load(
    retained_inputs, monkeypatch, damage
):
    root, _diagnosis, terminals, _launch = retained_inputs
    monkeypatch.setattr(
        probe.prior.torch,
        "load",
        lambda *_args, **_kwargs: pytest.fail(
            "tensor load before complete byte authentication"
        ),
    )
    if damage == "diagnosis-hash":
        with (root / "independent-failure-diagnosis.json").open("ab") as stream:
            stream.write(b" ")
    elif damage == "diagnosis-flags":
        record = probe.base.parse_json(
            (root / "independent-failure-diagnosis.json").read_bytes()
        )
        key = next(iter(probe.prior.FLAGS))
        record[key] = True
        (root / "independent-failure-diagnosis.json").write_text(
            canonical(record) + "\n"
        )
        monkeypatch.setattr(
            probe,
            "DIAGNOSIS_SHA",
            probe.base.digest(
                (root / "independent-failure-diagnosis.json").read_bytes()
            ),
        )
    elif damage == "original-file":
        (root / "capture.pt").write_bytes(b"changed")
    elif damage == "invocation":
        monkeypatch.setattr(
            probe.prior,
            "_read_launch",
            lambda *_: {"service_properties": {"InvocationID": "a" * 32}},
        )

    if damage is None:
        result = probe.authenticate_inputs()
        assert result[0] == root
        assert result[3] == probe.prior._inventory(root, probe.prior.COMPLETE_FILES)
        assert terminals == [(probe.PARENT, probe.RUN_INVOCATION)]
    else:
        with pytest.raises(ValueError):
            probe.authenticate_inputs()
        assert terminals == []


@pytest.mark.parametrize("outcome", ["same-failure", "different-failure", "success"])
def test_compare_refuses_to_substitute_different_original_pair_failure(
    retained_inputs, monkeypatch, outcome
):
    root, _diagnosis, _terminals, launch = retained_inputs
    inputs = [{"archive": {"records": [{}]}}, {"archive": {"records": [{}]}}]
    traces = [{"trace": 0}, {"trace": 1}]
    monkeypatch.setattr(probe.prior, "_score_inputs", lambda *_: (inputs, [], traces))
    monkeypatch.setattr(probe.prior, "_read_launch", lambda *_: launch)

    def pair(*_args):
        if outcome == "success":
            return {}
        message = (
            "paired rollout semantic state exactness"
            if outcome == "same-failure"
            else "paired early trace exactness"
        )
        raise ValueError(message)

    monkeypatch.setattr(probe.prior, "_strict_pair", pair)
    compared = []

    def compare(*args):
        compared.append(args)
        return {"all_false": True}

    monkeypatch.setattr(probe.components, "compare", compare)
    if outcome == "same-failure":
        result, inventory = probe.read_compare()
        assert result == {"all_false": True}
        assert inventory == probe.prior._inventory(root, probe.prior.COMPLETE_FILES)
        assert len(compared[0]) == 5
        assert compared[0][-1] == launch["compiled_plant"]
    else:
        with pytest.raises(
            ValueError, match="original strict pair failure reproduced again"
        ):
            probe.read_compare()
        assert compared == []
    assert {path.name for path in root.iterdir()} == probe.INPUT_FILES


@pytest.mark.parametrize("damage", [None, "pid", "competitor", "memory"])
def test_exact_caps_and_sole_owned_cpu_service(monkeypatch, damage):
    values = dict(CAPS)
    if damage == "pid":
        values["MainPID"] = "999"
    elif damage == "memory":
        values["MemoryMax"] = str(8 * 1024**3)
    monkeypatch.setattr(probe.os, "getpid", lambda: 123)

    def read(*args):
        if args[2] == "list-units":
            return probe.unit(SOURCE, "run") + (
                "\n microduck-other.service" if damage == "competitor" else ""
            )
        return values[args[-2]]

    monkeypatch.setattr(probe.base.host, "read", read)
    if damage is None:
        assert probe.properties(SOURCE, "run") == values
    else:
        with pytest.raises(ValueError):
            probe.properties(SOURCE, "run")


@pytest.mark.parametrize(
    "damage",
    [None, "active", "status", "log", "service-caps", "invocation", "failed-terminal"],
)
def test_test_receipt_and_complete_log_are_exact_and_fail_closed(
    tmp_path, monkeypatch, damage
):
    monkeypatch.setattr(probe, "EXPECTED_TESTS", 37)
    root = tmp_path
    source = SOURCE
    binding = {"stable": "source-binding"}
    log = b"37 passed in 3.21s\n"
    report = {
        "protocol": probe.PROTOCOL,
        "source": source,
        "mode": "tests",
        "status": "passed",
        "source_binding": binding,
        "passed": 37,
        "pytest_sha256": probe.base.digest(log),
        "service_properties": dict(CAPS),
        **probe.FLAGS,
    }
    if damage == "active":
        report["status"] = "running"
    elif damage == "status":
        report["passed"] = 36
    elif damage == "service-caps":
        report["service_properties"]["MemoryMax"] = str(8 * 1024**3)
    elif damage == "invocation":
        report["service_properties"]["InvocationID"] = "e" * 32
    elif damage == "failed-terminal":
        report["service_properties"]["InvocationID"] = "e" * 32
    (root / "report.json").write_text(canonical(report) + "\n")
    wrong_log = b"37 passed but a different log\n"
    (root / "pytest.log").write_bytes(
        b"36 passed in 3s\n"
        if damage == "status"
        else wrong_log
        if damage == "log"
        else log
    )
    monkeypatch.setattr(probe, "output_path", lambda *_: root)
    terminal = {
        "MainPID": "0",
        "ActiveState": "inactive",
        "NRestarts": "0",
        "ExecMainStatus": "1" if damage == "failed-terminal" else "0",
        "Result": "exit-code" if damage == "failed-terminal" else "success",
        "InvocationID": INVOCATION,
    }
    monkeypatch.setattr(probe.base.host, "read", lambda *args: terminal[args[-2]])
    if damage is None:
        assert probe._read_tests(source, binding) == probe.base.digest(
            (root / "report.json").read_bytes()
        )
    else:
        with pytest.raises(ValueError):
            probe._read_tests(source, binding)


def test_completed_service_requires_inactive_zero_exit_and_exact_invocation(
    monkeypatch,
):
    states = {
        "MainPID": "0",
        "ActiveState": "inactive",
        "NRestarts": "0",
        "ExecMainStatus": "0",
        "Result": "success",
        "InvocationID": INVOCATION,
    }
    monkeypatch.setattr(probe.base.host, "read", lambda *args: states[args[-2]])
    assert probe.completed(SOURCE, "tests", INVOCATION)["Result"] == "success"
    states["ExecMainStatus"] = "1"
    with pytest.raises(
        ValueError, match="successful terminal original CPU test service"
    ):
        probe.completed(SOURCE, "tests", INVOCATION)


def _git(root, *args):
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.mark.parametrize("damage", [None, "unauthorized", "frozen-source"])
def test_real_git_binding_requires_base_ancestry_and_frozen_prior_inputs(
    tmp_path, monkeypatch, damage
):
    _git(tmp_path, "init", "-b", "feat/athletics-obstacle-curriculum")
    _git(tmp_path, "config", "user.email", "fixture@example.invalid")
    _git(tmp_path, "config", "user.name", "CPU fixture")
    stable = set(probe.prior.OWN_FILES) - {probe.OWN[-1]}
    for name in stable:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("frozen " + name)
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "synthetic immutable parent")
    parent = _git(tmp_path, "rev-parse", "HEAD")
    monkeypatch.setattr(probe, "PARENT", parent)
    monkeypatch.setattr(probe, "BASE_SOURCE", parent)
    if damage == "frozen-source":
        path = tmp_path / next(iter(sorted(stable)))
        path.write_text("mutated immutable input")
    for name in probe.OWN:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("new declared " + name)
    if damage == "unauthorized":
        (tmp_path / "unrelated.txt").write_text("not allowed")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "synthetic component probe source")
    source = _git(tmp_path, "rev-parse", "HEAD")
    monkeypatch.setattr(probe.base.execution, "ROOT", tmp_path)
    if damage is None:
        assert set(probe.source_binding(source)) == stable | set(probe.OWN)
    else:
        with pytest.raises(ValueError, match="only newly declared"):
            probe.source_binding(source)


def _unit(source):
    return probe.source_sync_unit(source) + " loaded active running\n"


def _sync_fixture(
    tmp_path,
    *,
    units=None,
    head=None,
    branch="feat/athletics-obstacle-curriculum",
    dirty="",
    root=None,
    origin=None,
    incoming=None,
    bundle_state="regular",
    bundle_size=None,
    checksum=None,
    reported_checksum=None,
    verify=True,
    heads=None,
):
    source = "f" * 40
    checksum = checksum or "e" * 64
    bin_dir, state = tmp_path / "bin", tmp_path / "state"
    bin_dir.mkdir()
    state.mkdir()
    (state / "head").write_text((probe.PARENT if head is None else head) + "\n")
    (state / "incoming").write_text((source if incoming is None else incoming) + "\n")
    (state / "branch").write_text(branch + "\n")
    (state / "dirty").write_text(dirty)
    (state / "units").write_text(_unit(source) if units is None else units)
    (state / "root").write_text(
        (probe.prior.SYNC_ROOT if root is None else root) + "\n"
    )
    (state / "origin").write_text(
        (probe.prior.SYNC_ORIGIN if origin is None else origin) + "\n"
    )
    bundle_path = (
        Path(probe.prior.SYNC_ROOT)
        / "artifacts/tools"
        / f"cuda64-inertia-source-{source[:12]}.bundle"
    )
    if bundle_state != "missing":
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        if bundle_state == "symlink":
            target = bundle_path.with_suffix(".target")
            target.write_bytes(b"synthetic source bundle")
            bundle_path.symlink_to(target)
        elif bundle_state == "oversize":
            bundle_path.write_bytes(b"x")
        else:
            bundle_path.write_bytes(b"synthetic source bundle")
    measured_size = bundle_path.stat().st_size if bundle_path.exists() else 0
    (state / "bundle_size").write_text(
        str(bundle_size if bundle_size is not None else measured_size) + "\n"
    )
    (state / "bundle_hash").write_text((reported_checksum or checksum) + "\n")
    (state / "bundle_verify").write_text("1\n" if verify else "0\n")
    (state / "bundle_heads").write_text(
        (heads or f"{source} refs/heads/feat/athletics-obstacle-curriculum") + "\n"
    )
    git_stub = r"""#!/usr/bin/env bash
set -euo pipefail
printf 'git %s\n' "$*" >> "$SYNC_LOG"
case "$1" in
  status) cat "$SYNC_STATE/dirty" ;;
  rev-parse)
    case "$2" in
      --show-toplevel) cat "$SYNC_STATE/root" ;;
      HEAD) cat "$SYNC_STATE/head" ;;
      refs/remotes/origin/feat/athletics-obstacle-curriculum)
        test -f "$SYNC_STATE/ref" || exit 93
        cat "$SYNC_STATE/ref" ;;
      *) exit 91 ;;
    esac ;;
  remote) test "$2" = get-url && test "$3" = origin; cat "$SYNC_STATE/origin" ;;
  branch) cat "$SYNC_STATE/branch" ;;
  bundle)
    case "$2" in
      verify) test "$(cat "$SYNC_STATE/bundle_verify")" = 1 ;;
      list-heads) cat "$SYNC_STATE/bundle_heads" ;;
      *) exit 95 ;;
    esac ;;
  fetch)
    test "$2" = "$SYNC_BUNDLE_PATH"
    test "$3" = refs/heads/feat/athletics-obstacle-curriculum:refs/remotes/origin/feat/athletics-obstacle-curriculum
    cat "$SYNC_STATE/incoming" > "$SYNC_STATE/ref"
    touch "$SYNC_STATE/fetched" ;;
  merge)
    test "$2" = --ff-only
    printf '%s\n' "$3" > "$SYNC_STATE/head"
    touch "$SYNC_STATE/merged" ;;
  *) exit 92 ;;
esac
"""
    stat_stub = r"""#!/usr/bin/env bash
set -euo pipefail
printf 'stat %s\n' "$*" >> "$SYNC_LOG"
test "$1" = -c && test "$2" = %s
cat "$SYNC_STATE/bundle_size"
"""
    sha_stub = r"""#!/usr/bin/env bash
set -euo pipefail
printf 'sha256sum %s\n' "$*" >> "$SYNC_LOG"
printf '%s  %s\n' "$(cat "$SYNC_STATE/bundle_hash")" "$1"
"""
    systemctl_stub = r"""#!/usr/bin/env bash
set -euo pipefail
printf 'systemctl %s\n' "$*" >> "$SYNC_LOG"
cat "$SYNC_STATE/units"
"""
    for name, source_text in (
        ("git", git_stub),
        ("stat", stat_stub),
        ("sha256sum", sha_stub),
        ("systemctl", systemctl_stub),
    ):
        path = bin_dir / name
        path.write_text(source_text)
        path.chmod(0o755)
    env = {
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "SYNC_LOG": str(tmp_path / "calls.log"),
        "SYNC_STATE": str(state),
        "SYNC_BUNDLE_PATH": str(bundle_path),
    }
    script = probe.source_sync_script(source, bundle_sha256=checksum)
    process = subprocess.run(
        ["bash", "-c", script],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    log_path = tmp_path / "calls.log"
    return (
        process,
        log_path.read_text() if log_path.exists() else "",
        state,
        source,
        script,
    )


def test_source_sync_retargets_only_exact_two_template_guards(monkeypatch):
    source = "f" * 40
    bundle_hash = "e" * 64
    template = probe.prior.source_sync_script(source, bundle_sha256=bundle_hash)
    script = probe.source_sync_script(source, bundle_sha256=bundle_hash)
    expected = template.replace(
        f'test "$(git rev-parse HEAD)" = {probe.prior.SYNC_FROM_SOURCE}\n',
        f'test "$(git rev-parse HEAD)" = {probe.PARENT}\n',
        1,
    ).replace(
        f'if test "$duck_sync_running" != {probe.prior.source_sync_unit(source)}; then\n',
        f'if test "$duck_sync_running" != {probe.source_sync_unit(source)}; then\n',
        1,
    )
    assert script == expected
    assert f'test "$(git rev-parse HEAD)" = {probe.PARENT}' in script
    assert f'if test "$duck_sync_running" != {probe.source_sync_unit(source)}' in script
    assert "set -euo pipefail" in script


@pytest.mark.parametrize(
    "guard,variant",
    [(0, "missing"), (0, "duplicate"), (1, "missing"), (1, "duplicate")],
)
def test_source_sync_fails_closed_when_frozen_template_guard_not_unique(
    monkeypatch, guard, variant
):
    source = "f" * 40
    template = probe.prior.source_sync_script(source, bundle_sha256="e" * 64)
    guard_lines = (
        f'test "$(git rev-parse HEAD)" = {probe.prior.SYNC_FROM_SOURCE}\n',
        f'if test "$duck_sync_running" != {probe.prior.source_sync_unit(source)}; then\n',
    )
    damaged = (
        template.replace(guard_lines[guard], "", 1)
        if variant == "missing"
        else template + guard_lines[guard]
    )
    monkeypatch.setattr(probe.prior, "source_sync_script", lambda *_a, **_k: damaged)
    with pytest.raises(ValueError, match="exact frozen bootstrap guard occurrence"):
        probe.source_sync_script(source, bundle_sha256="e" * 64)


def test_source_sync_accepts_only_unique_self_unit_and_verified_bundle(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(probe.prior, "SYNC_ROOT", str(tmp_path))
    checksum = "e" * 64
    process, log, state, source, script = _sync_fixture(tmp_path, checksum=checksum)
    assert process.returncode == 0, process.stderr
    assert process.stdout.strip() == source
    assert (state / "fetched").exists() and (state / "merged").exists()
    bundle_path = f"{probe.prior.SYNC_ROOT}/artifacts/tools/cuda64-inertia-source-{source[:12]}.bundle"
    ordered = [
        f"stat -c %s {bundle_path}",
        f"sha256sum {bundle_path}",
        f"git bundle verify {bundle_path}",
        f"git bundle list-heads {bundle_path}",
        f"git fetch {bundle_path} refs/heads/feat/athletics-obstacle-curriculum:refs/remotes/origin/feat/athletics-obstacle-curriculum",
        f"git merge --ff-only {source}",
    ]
    assert [log.index(part) for part in ordered] == sorted(
        log.index(part) for part in ordered
    )
    assert "test -f " + bundle_path in script
    assert "test ! -L " + bundle_path in script


@pytest.mark.parametrize(
    "damage",
    [
        "no-unit",
        "foreign-unit",
        "competing-unit",
        "head",
        "branch",
        "dirty",
        "root",
        "origin",
        "incoming",
        "missing-bundle",
        "symlink-bundle",
        "oversize-bundle",
        "bad-hash",
        "verify",
        "heads",
    ],
)
def test_source_sync_refuses_each_bad_context_or_bundle_before_merge(
    tmp_path, monkeypatch, damage
):
    monkeypatch.setattr(probe.prior, "SYNC_ROOT", str(tmp_path))
    kwargs = {}
    if damage == "no-unit":
        kwargs["units"] = ""
    elif damage == "foreign-unit":
        kwargs["units"] = (
            "microduck-inertia-components-run-other.service loaded active running\n"
        )
    elif damage == "competing-unit":
        kwargs["units"] = (
            _unit("f" * 40) + "microduck-other.service loaded active running\n"
        )
    elif damage == "head":
        kwargs["head"] = "a" * 40
    elif damage == "branch":
        kwargs["branch"] = "wrong-branch"
    elif damage == "dirty":
        kwargs["dirty"] = " M source.py\n"
    elif damage == "root":
        kwargs["root"] = "/wrong/checkout"
    elif damage == "origin":
        kwargs["origin"] = "https://example.invalid/microduck_rl.git"
    elif damage == "incoming":
        kwargs["incoming"] = "a" * 40
    elif damage == "missing-bundle":
        kwargs["bundle_state"] = "missing"
    elif damage == "symlink-bundle":
        kwargs["bundle_state"] = "symlink"
    elif damage == "oversize-bundle":
        kwargs.update(
            bundle_state="oversize", bundle_size=probe.prior.SYNC_BUNDLE_LIMIT + 1
        )
    elif damage == "bad-hash":
        kwargs["reported_checksum"] = "a" * 64
    elif damage == "verify":
        kwargs["verify"] = False
    elif damage == "heads":
        kwargs["heads"] = "a" * 40 + " refs/heads/feat/athletics-obstacle-curriculum"
    process, log, state, _source, _script = _sync_fixture(tmp_path, **kwargs)
    assert process.returncode != 0
    assert "git merge" not in log
    assert not (state / "merged").exists()
    if damage in {
        "no-unit",
        "foreign-unit",
        "competing-unit",
        "head",
        "branch",
        "dirty",
        "root",
        "origin",
    }:
        assert "git fetch" not in log
    if damage == "missing-bundle":
        assert "sha256sum" not in log and "git bundle verify" not in log
    if damage == "symlink-bundle":
        assert "sha256sum" not in log and "git bundle verify" not in log
    if damage == "oversize-bundle":
        assert "stat -c %s" in log and "sha256sum" not in log
    if damage == "bad-hash":
        assert "sha256sum" in log and "git bundle verify" not in log
    if damage == "verify":
        assert "git bundle verify" in log and "git bundle list-heads" not in log
    if damage == "heads":
        assert "git bundle list-heads" in log and "git fetch" not in log
    if damage == "incoming":
        assert "git fetch" in log and "git merge" not in log


@pytest.mark.parametrize("checksum", ["bad", "e" * 63, "E" * 64])
def test_malformed_bundle_digest_fails_before_creating_script(checksum):
    with pytest.raises(ValueError, match="whole source bundle SHA256"):
        probe.source_sync_script("f" * 40, bundle_sha256=checksum)


def test_source_sync_unit_source_hex_is_strict():
    for invalid in ("bad", "F" * 40, "f" * 39, "f" * 41):
        with pytest.raises(ValueError):
            probe.source_sync_unit(invalid)
        with pytest.raises(ValueError):
            probe.source_sync_script(invalid, bundle_sha256="e" * 64)


def test_hidden_guard_precedes_source_and_service_access(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(
        probe,
        "properties",
        lambda *_: pytest.fail("service access before hidden CUDA guard"),
    )
    with pytest.raises(ValueError, match="CUDA-hidden exact WSL"):
        probe.execute(SOURCE, "run")


@pytest.mark.parametrize(
    "failure", [None, "reader", "context", "protected-active", "protected-change"]
)
def test_report_retention_is_non_admitting_and_never_overwrites(
    tmp_path, monkeypatch, retained_inputs, failure
):
    retained_root, _diagnosis, _terminals, _launch = retained_inputs
    retained_before = {
        path.name: probe.base.digest(path.read_bytes())
        for path in retained_root.iterdir()
    }
    root = tmp_path / "component-report"
    monkeypatch.setattr(probe.prior, "_hidden", lambda: None)
    monkeypatch.setattr(probe.prior, "check_window", lambda **_kwargs: None)
    monkeypatch.setattr(probe, "properties", lambda *_: dict(CAPS))
    calls = []

    def binding(_source):
        calls.append(True)
        return {"changed": failure == "context" and len(calls) > 1}

    monkeypatch.setattr(probe, "source_binding", binding)
    monkeypatch.setattr(probe.base.host, "identity", lambda _source: {})
    monkeypatch.setattr(probe, "output_path", lambda *_: root)
    monkeypatch.setattr(probe, "_read_tests", lambda *_: "a" * 64)

    @contextmanager
    def lease():
        yield 31

    monkeypatch.setattr(probe.base.gap.base.files, "gpu_lease", lease)
    monkeypatch.setattr(probe.base.gpu_idle_gate, "wait_idle", lambda: {"idle": True})
    monkeypatch.setattr(
        probe.base.gap.base.retained.d0,
        "filmbrain_state",
        lambda: {"unchanged": True},
    )
    protected_calls = []

    def protected_state():
        protected_calls.append(True)
        if failure == "protected-active":
            return {"mission": "active"}
        if failure == "protected-change" and len(protected_calls) > 1:
            return {"mission": "active"}
        return {"mission": "inactive"}

    monkeypatch.setattr(probe.base.gap.base, "protected_state", protected_state)

    def compare():
        if failure == "reader":
            raise ValueError("reader failed")
        return {"protocol": probe.components.PROTOCOL, **probe.FLAGS}, {}

    monkeypatch.setattr(probe, "read_compare", compare)
    monkeypatch.setattr(probe, "authenticate_inputs", lambda: (None, None, None, {}))
    if failure is None:
        assert probe.execute(SOURCE, "run")["status"] == "passed"
    else:
        with pytest.raises(ValueError):
            probe.execute(SOURCE, "run")
    report = probe.base.parse_json((root / "report.json").read_bytes())
    assert report["status"] == ("passed" if failure is None else "failed-retained")
    assert all(report[key] is False for key in probe.FLAGS)
    assert (root / "comparison.json").exists() is (
        failure not in {"reader", "protected-active"}
    )
    original = (root / "report.json").read_bytes()
    with pytest.raises(FileExistsError):
        probe.execute(SOURCE, "run")
    assert (root / "report.json").read_bytes() == original
    assert {
        path.name: probe.base.digest(path.read_bytes())
        for path in retained_root.iterdir()
        if path.is_file()
    } == retained_before
    if failure is None:
        assert (
            report["protected_services"]
            == report["protected_services_after"]
            == {"mission": "inactive"}
        )
    elif failure.startswith("protected-"):
        assert report["status"] == "failed-retained"
