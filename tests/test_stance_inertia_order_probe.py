"""Synthetic CPU tests for whole-byte order-diagnostic supervision."""

from contextlib import contextmanager
from pathlib import Path
import subprocess

import pytest

from mjlab_microduck import stance_inertia_order_probe as probe
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
TERMINAL = {
    "MainPID": "0",
    "ActiveState": "inactive",
    "NRestarts": "0",
    "ExecMainStatus": "0",
    "Result": "success",
    "InvocationID": INVOCATION,
}


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(probe.oracle.torch.cuda, "is_initialized", lambda: False)


@pytest.fixture
def authenticated_anchors(tmp_path, monkeypatch):
    old_root = tmp_path / "frozen-component-inputs"
    old_root.mkdir()
    for name in probe.prior.COMPLETE_FILES:
        path = old_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic frozen original input\n")
    old_inventory = probe.prior._inventory(old_root, probe.prior.COMPLETE_FILES)
    launch = {
        "schedule": {"worlds": 64, "fixture": "full-fixed-schedule"},
        "compiled_plant": {"fixture": "compiled-plant"},
    }
    old_calls = []

    def old_authentication():
        old_calls.append(True)
        return old_root, launch, "c" * 64, old_inventory

    monkeypatch.setattr(
        probe.frozen_component, "authenticate_inputs", old_authentication
    )
    map_root = tmp_path / "map-output"
    map_root.mkdir()
    comparison = (
        canonical(
            {
                "source": probe.PARENT,
                "worlds": 64,
                "earliest_persistent_differing_event": 0,
                "compiled_map": {"topology_sha256": probe.MAP_TOPOLOGY_SHA},
            }
        )
        + "\n"
    ).encode()
    (map_root / "comparison.json").write_bytes(comparison)
    comparison_sha = probe.base.digest(comparison)
    monkeypatch.setattr(probe, "MAP_COMPARISON_SHA", comparison_sha)
    report = {
        "protocol": probe.frozen_component.PROTOCOL,
        "source": probe.FREEZE_SOURCE,
        "mode": "run",
        "status": "passed",
        "comparison_sha256": comparison_sha,
        "diagnosis_sha256": probe.DIAGNOSIS_SHA,
        "service_properties": dict(CAPS, InvocationID=probe.MAP_INVOCATION),
        **probe.frozen_component.FLAGS,
    }
    report_raw = (canonical(report) + "\n").encode()
    (map_root / "report.json").write_bytes(report_raw)
    monkeypatch.setattr(probe, "MAP_REPORT_SHA", probe.base.digest(report_raw))
    monkeypatch.setattr(
        probe.frozen_component,
        "output_path",
        lambda source, mode: (
            map_root
            if source == probe.FREEZE_SOURCE and mode == "run"
            else pytest.fail("unexpected frozen output lookup")
        ),
    )
    completed_calls = []
    monkeypatch.setattr(
        probe.frozen_component,
        "completed",
        lambda *args: completed_calls.append(args),
    )
    smooth = b"# installed synthetic smooth implementation\n"
    smooth_path = tmp_path / "smooth.py"
    smooth_path.write_bytes(smooth)
    monkeypatch.setattr(probe, "SMOOTH_LEAF", "smooth.py")
    monkeypatch.setattr(probe, "SMOOTH_SHA", probe.base.digest(smooth))
    monkeypatch.setattr(probe.base.execution, "ROOT", tmp_path)
    return {
        "tmp_path": tmp_path,
        "old_root": old_root,
        "old_inventory": old_inventory,
        "old_calls": old_calls,
        "launch": launch,
        "map_root": map_root,
        "report": report,
        "comparison": comparison,
        "smooth_path": smooth_path,
        "smooth": smooth,
        "completed_calls": completed_calls,
    }


def test_declared_whole_input_inventory_and_distinct_cpu_service_namespaces():
    assert len(probe.INPUT_FILES) == 23
    assert (
        probe.MAP_COMPARISON_SHA
        == "e1e85c0eb76ee579fd15357f3be098cef951434c1ba83fa6846f487c1435d8e9"
    )
    assert (
        probe.MAP_REPORT_SHA
        == "e1321c1cc3787240e01039f877032869f735ed5fdbb0e93d34726d4c00077e93"
    )
    assert probe.MAP_INVOCATION == "e161d7339d904a4baaac1be2be74c36a"
    assert (
        probe.MAP_TOPOLOGY_SHA
        == "5b445215f52d61d10bc892a14b0d0b15e3041ff3a5dfb6c1964f06211d0cb43a"
    )
    assert (
        probe.SMOOTH_SHA
        == "63b2d4093745762309bb335826a1f741a1baab26d93277ba92859fea1495880f"
    )
    assert probe.SMOOTH_LIMIT == 1024**2
    assert probe.CAP == 300 and probe.MEMORY == 4 * 1024**3
    assert len(probe.TEST_FILES) == 41
    assert type(probe.EXPECTED_TESTS) is int
    assert probe.EXPECTED_TESTS == 1272
    assert probe.unit(SOURCE, "run") != probe.frozen_component.unit(SOURCE, "run")
    assert probe.output_path(SOURCE, "run") != probe.frozen_component.output_path(
        probe.FREEZE_SOURCE, "run"
    )
    assert probe.source_sync_unit(SOURCE) != probe.frozen_component.source_sync_unit(
        SOURCE
    )
    assert all(value is False for value in probe.FLAGS.values())


@pytest.mark.parametrize(
    "damage",
    [
        None,
        "map-comparison",
        "comparison-source",
        "comparison-worlds",
        "comparison-earliest",
        "comparison-topology",
        "map-report",
        "status",
        "diagnosis",
        "map-flags",
        "invocation",
        "service-caps",
        "extra-map",
        "smooth",
        "smooth-oversize",
    ],
)
def test_map_diagnosis_and_installed_smooth_bytes_authenticate_before_tensor_load(
    authenticated_anchors, monkeypatch, damage
):
    fixture = authenticated_anchors
    events = []
    monkeypatch.setattr(
        probe.frozen_component,
        "authenticate_inputs",
        lambda: (
            events.append("original23-auth")
            or (
                fixture["old_root"],
                fixture["launch"],
                "c" * 64,
                fixture["old_inventory"],
            )
        ),
    )
    monkeypatch.setattr(
        probe.prior.torch,
        "load",
        lambda *_args, **_kwargs: pytest.fail("load before all map and smooth bytes"),
    )
    map_root = fixture["map_root"]
    if damage == "map-comparison":
        (map_root / "comparison.json").write_bytes(b"changed\n")
    elif type(damage) is str and damage.startswith("comparison-"):
        comparison = {
            "source": probe.PARENT,
            "worlds": 64,
            "earliest_persistent_differing_event": 0,
            "compiled_map": {"topology_sha256": probe.MAP_TOPOLOGY_SHA},
        }
        if damage == "comparison-source":
            comparison["source"] = "f" * 40
        elif damage == "comparison-worlds":
            comparison["worlds"] = 63
        elif damage == "comparison-earliest":
            comparison["earliest_persistent_differing_event"] = 1
        else:
            comparison["compiled_map"]["topology_sha256"] = "f" * 64
        raw_comparison = (canonical(comparison) + "\n").encode()
        (map_root / "comparison.json").write_bytes(raw_comparison)
        new_comparison_sha = probe.base.digest(raw_comparison)
        monkeypatch.setattr(probe, "MAP_COMPARISON_SHA", new_comparison_sha)
        report = dict(fixture["report"], comparison_sha256=new_comparison_sha)
        raw_report = (canonical(report) + "\n").encode()
        (map_root / "report.json").write_bytes(raw_report)
        monkeypatch.setattr(probe, "MAP_REPORT_SHA", probe.base.digest(raw_report))
    elif damage in {
        "map-report",
        "status",
        "diagnosis",
        "map-flags",
        "invocation",
        "service-caps",
    }:
        report = dict(fixture["report"])
        if damage == "status":
            report["status"] = "failed-retained"
        elif damage == "diagnosis":
            report["diagnosis_sha256"] = "f" * 64
        elif damage == "map-flags":
            report[next(iter(probe.frozen_component.FLAGS))] = True
        elif damage == "invocation":
            report["service_properties"]["InvocationID"] = "f" * 32
        elif damage == "service-caps":
            report["service_properties"]["MemoryMax"] = str(8 * 1024**3)
        elif damage == "map-report":
            report["unexpected"] = "bytes differ from pinned report"
        raw = (canonical(report) + "\n").encode()
        (map_root / "report.json").write_bytes(raw)
        if damage != "map-report":
            monkeypatch.setattr(probe, "MAP_REPORT_SHA", probe.base.digest(raw))
    elif damage == "extra-map":
        (map_root / "unexpected.json").write_text("{}\n")
    elif damage == "smooth":
        fixture["smooth_path"].write_bytes(b"changed installed source\n")
    elif damage == "smooth-oversize":
        fixture["smooth_path"].write_bytes(b"x" * (probe.SMOOTH_LIMIT + 1))

    if damage is None:
        root, launch, sha, inventory, anchors = probe.authenticate_inputs()
        assert root == fixture["old_root"] and launch == fixture["launch"]
        assert sha == "c" * 64 and inventory == fixture["old_inventory"]
        assert anchors == {
            "comparison_sha256": probe.MAP_COMPARISON_SHA,
            "report_sha256": probe.MAP_REPORT_SHA,
            "smooth_sha256": probe.SMOOTH_SHA,
            "compiled_topology_sha256": probe.MAP_TOPOLOGY_SHA,
        }
        assert events == ["original23-auth"]
        assert fixture["completed_calls"] == [
            (probe.FREEZE_SOURCE, "run", probe.MAP_INVOCATION)
        ]
    else:
        with pytest.raises(ValueError):
            probe.authenticate_inputs()
        assert events == ["original23-auth"]


def test_map_terminal_receipt_and_caps_fail_closed(monkeypatch, authenticated_anchors):
    fixture = authenticated_anchors
    monkeypatch.setattr(
        probe.frozen_component,
        "completed",
        lambda *_: (_ for _ in ()).throw(
            ValueError("successful terminal original CPU test service")
        ),
    )
    with pytest.raises(
        ValueError, match="successful terminal original CPU test service"
    ):
        probe.authenticate_map()
    report = dict(fixture["report"])
    report["service_properties"] = dict(
        report["service_properties"], MemoryMax=str(8 * 1024**3)
    )
    raw = (canonical(report) + "\n").encode()
    (fixture["map_root"] / "report.json").write_bytes(raw)
    monkeypatch.setattr(probe, "MAP_REPORT_SHA", probe.base.digest(raw))
    with pytest.raises(ValueError):
        probe.authenticate_map()
    assert fixture["map_root"].exists()


@pytest.mark.parametrize(
    "pair_result", ["original-failure", "different-failure", "success"]
)
def test_order_analysis_requires_original_strict_failure_and_reauthenticates_afterward(
    monkeypatch, authenticated_anchors, pair_result
):
    fixture = authenticated_anchors
    inventory = fixture["old_inventory"]
    anchors = {
        "comparison_sha256": probe.MAP_COMPARISON_SHA,
        "report_sha256": probe.MAP_REPORT_SHA,
        "smooth_sha256": probe.SMOOTH_SHA,
        "compiled_topology_sha256": probe.MAP_TOPOLOGY_SHA,
    }
    source_context = (
        fixture["old_root"],
        fixture["launch"],
        "c" * 64,
        inventory,
        anchors,
    )
    auth_calls = []

    def authenticate():
        auth_calls.append(True)
        return source_context

    monkeypatch.setattr(probe, "authenticate_inputs", authenticate)
    inputs = [
        {"archive": {"records": [{"id": 0}]}},
        {"archive": {"records": [{"id": 1}]}},
    ]
    traces = [{"trace": "capture"}, {"trace": "replay"}]
    monkeypatch.setattr(probe.prior, "_score_inputs", lambda *_: (inputs, [], traces))

    def strict_pair(*_):
        if pair_result == "success":
            return {"paired": True}
        message = (
            "paired rollout semantic state exactness"
            if pair_result == "original-failure"
            else "paired early trace exactness"
        )
        raise ValueError(message)

    monkeypatch.setattr(probe.prior, "_strict_pair", strict_pair)
    compared = []
    monkeypatch.setattr(
        probe.oracle,
        "compare",
        lambda *args: (
            compared.append(args)
            or {
                "source": probe.PARENT,
                "worlds": 64,
                "compiled_topology_sha256": probe.MAP_TOPOLOGY_SHA,
                **probe.oracle.FLAGS,
            }
        ),
    )
    if pair_result == "original-failure":
        result, got_inventory, got_anchors = probe.read_compare()
        assert result["source"] == probe.PARENT
        assert result["worlds"] == 64
        assert all(result[key] is False for key in probe.oracle.FLAGS)
        assert got_inventory == inventory and got_anchors == anchors
        assert auth_calls == [True, True]
        assert compared[0][3] == [record["archive"]["records"][0] for record in inputs]
        assert compared[0][-1] == fixture["launch"]["compiled_plant"]
    else:
        with pytest.raises(
            ValueError, match="original strict pair failure reproduced again"
        ):
            probe.read_compare()
        assert compared == [] and auth_calls == [True]


@pytest.mark.parametrize(
    "changed", ["inventory", "anchors", "smooth", "map-report", "topology"]
)
def test_post_analysis_input_or_map_change_retained_as_failure(
    monkeypatch, authenticated_anchors, changed
):
    fixture = authenticated_anchors
    initial = (
        fixture["old_root"],
        fixture["launch"],
        "c" * 64,
        fixture["old_inventory"],
        {
            "comparison_sha256": probe.MAP_COMPARISON_SHA,
            "report_sha256": probe.MAP_REPORT_SHA,
            "smooth_sha256": probe.SMOOTH_SHA,
            "compiled_topology_sha256": probe.MAP_TOPOLOGY_SHA,
        },
    )
    calls = []

    def authenticate():
        calls.append(True)
        if len(calls) == 1:
            return initial
        if changed == "inventory":
            return (*initial[:3], {"changed": True}, initial[4])
        if changed in {"anchors", "smooth", "map-report", "topology"}:
            other = dict(initial[4])
            key = (
                "smooth_sha256"
                if changed == "smooth"
                else "compiled_topology_sha256"
                if changed == "topology"
                else "report_sha256"
            )
            other[key] = "f" * 64
            return (*initial[:4], other)
        return initial

    monkeypatch.setattr(probe, "authenticate_inputs", authenticate)
    inputs = [
        {"archive": {"records": [{"id": 1}]}},
        {"archive": {"records": [{"id": 2}]}},
    ]
    traces = [{"trace": 1}, {"trace": 2}]
    monkeypatch.setattr(probe.prior, "_score_inputs", lambda *_: (inputs, [], traces))
    monkeypatch.setattr(
        probe.prior,
        "_strict_pair",
        lambda *_: (_ for _ in ()).throw(
            ValueError("paired rollout semantic state exactness")
        ),
    )
    monkeypatch.setattr(
        probe.oracle,
        "compare",
        lambda *_: {
            "source": probe.PARENT,
            "worlds": 64,
            "compiled_topology_sha256": probe.MAP_TOPOLOGY_SHA,
            **probe.oracle.FLAGS,
        },
    )
    with pytest.raises(
        ValueError, match="all 25 original files and installed source unchanged"
    ):
        probe.read_compare()
    assert calls == [True, True]


def test_full_64_world_schedule_is_required_before_oracle(
    monkeypatch, authenticated_anchors
):
    fixture = authenticated_anchors
    launch = dict(fixture["launch"], schedule={"worlds": 63})
    context = (
        fixture["old_root"],
        launch,
        "c" * 64,
        fixture["old_inventory"],
        {},
    )
    monkeypatch.setattr(probe, "authenticate_inputs", lambda: context)
    inputs = [{"archive": {"records": [{}]}}, {"archive": {"records": [{}]}}]
    monkeypatch.setattr(probe.prior, "_score_inputs", lambda *_: (inputs, [], [{}, {}]))
    monkeypatch.setattr(
        probe.prior,
        "_strict_pair",
        lambda *_: (_ for _ in ()).throw(
            ValueError("paired rollout semantic state exactness")
        ),
    )
    monkeypatch.setattr(
        probe.oracle,
        "compare",
        lambda *_: pytest.fail("oracle called for 63-world input"),
    )
    with pytest.raises(ValueError, match="unchanged full CUDA64 input protocol"):
        probe.read_compare()


@pytest.mark.parametrize("damage", ["source", "worlds", "topology", "flag"])
def test_oracle_result_must_match_frozen_map_and_remain_non_admitting(
    monkeypatch, authenticated_anchors, damage
):
    fixture = authenticated_anchors
    anchors = {
        "comparison_sha256": probe.MAP_COMPARISON_SHA,
        "report_sha256": probe.MAP_REPORT_SHA,
        "smooth_sha256": probe.SMOOTH_SHA,
        "compiled_topology_sha256": probe.MAP_TOPOLOGY_SHA,
    }
    context = (
        fixture["old_root"],
        fixture["launch"],
        "c" * 64,
        fixture["old_inventory"],
        anchors,
    )
    monkeypatch.setattr(probe, "authenticate_inputs", lambda: context)
    inputs = [{"archive": {"records": [{}]}}, {"archive": {"records": [{}]}}]
    monkeypatch.setattr(probe.prior, "_score_inputs", lambda *_: (inputs, [], [{}, {}]))
    monkeypatch.setattr(
        probe.prior,
        "_strict_pair",
        lambda *_: (_ for _ in ()).throw(
            ValueError("paired rollout semantic state exactness")
        ),
    )
    result = {
        "source": probe.PARENT,
        "worlds": 64,
        "compiled_topology_sha256": probe.MAP_TOPOLOGY_SHA,
        **probe.oracle.FLAGS,
    }
    if damage == "source":
        result["source"] = "f" * 40
    elif damage == "worlds":
        result["worlds"] = 63
    elif damage == "topology":
        result["compiled_topology_sha256"] = "f" * 64
    else:
        result[next(iter(probe.oracle.FLAGS))] = True
    monkeypatch.setattr(probe.oracle, "compare", lambda *_: result)
    with pytest.raises(ValueError, match="full non-admitting oracle binds"):
        probe.read_compare()


@pytest.mark.parametrize("damage", [None, "pid", "competitor", "memory"])
def test_new_component_service_caps_and_only_one_duck(monkeypatch, damage):
    values = dict(CAPS)
    if damage == "pid":
        values["MainPID"] = "999"
    elif damage == "memory":
        values["MemoryMax"] = str(8 * 1024**3)
    monkeypatch.setattr(probe.os, "getpid", lambda: 123)

    def read(*args):
        if args[2] == "list-units":
            return probe.unit(SOURCE, "run") + (
                "\n microduck-foreign.service" if damage == "competitor" else ""
            )
        return values[args[-2]]

    monkeypatch.setattr(probe.base.host, "read", read)
    if damage is None:
        assert probe.properties(SOURCE, "run") == values
    else:
        with pytest.raises(ValueError):
            probe.properties(SOURCE, "run")


def test_new_test_receipt_requires_exact_count_flags_log_and_terminal_service(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(probe, "EXPECTED_TESTS", 41)
    log = b"41 passed in 4.2s\n"
    binding = {"leaf": "pinned"}
    report = {
        "protocol": probe.PROTOCOL,
        "source": SOURCE,
        "mode": "tests",
        "status": "passed",
        "source_binding": binding,
        "passed": 41,
        "pytest_sha256": probe.base.digest(log),
        "service_properties": dict(CAPS),
        **probe.FLAGS,
    }
    (tmp_path / "report.json").write_text(canonical(report) + "\n")
    (tmp_path / "pytest.log").write_bytes(log)
    monkeypatch.setattr(probe, "output_path", lambda *_: tmp_path)
    monkeypatch.setattr(probe.base.host, "read", lambda *args: TERMINAL[args[-2]])
    assert probe._read_tests(SOURCE, binding) == probe.base.digest(
        (tmp_path / "report.json").read_bytes()
    )

    for field, value in (
        ("passed", 40),
        ("status", "failed-retained"),
        (next(iter(probe.FLAGS)), True),
        ("source_binding", {"leaf": "changed"}),
    ):
        damaged = dict(report, **{field: value})
        (tmp_path / "report.json").write_text(canonical(damaged) + "\n")
        with pytest.raises(ValueError):
            probe._read_tests(SOURCE, binding)
    (tmp_path / "report.json").write_text(canonical(report) + "\n")
    (tmp_path / "pytest.log").write_bytes(b"wrong log\n")
    with pytest.raises(ValueError, match="whole new CPU pytest log"):
        probe._read_tests(SOURCE, binding)
    (tmp_path / "pytest.log").write_bytes(log)
    for field, value in (
        ("service_properties", dict(CAPS, MemoryMax=str(8 * 1024**3))),
        ("service_properties", dict(CAPS, InvocationID="f" * 32)),
    ):
        damaged = dict(report, **{field: value})
        (tmp_path / "report.json").write_text(canonical(damaged) + "\n")
        with pytest.raises(ValueError):
            probe._read_tests(SOURCE, binding)
    (tmp_path / "report.json").write_text(canonical(report) + "\n")
    monkeypatch.setattr(
        probe, "completed", lambda *_: (_ for _ in ()).throw(ValueError("terminal"))
    )
    with pytest.raises(ValueError, match="terminal"):
        probe._read_tests(SOURCE, binding)


def _git(root, *args):
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.mark.parametrize("damage", [None, "unauthorized", "frozen-input"])
def test_real_git_source_binding_freezes_prior_component_and_old_readers(
    tmp_path, monkeypatch, damage
):
    _git(tmp_path, "init", "-b", "feat/athletics-obstacle-curriculum")
    _git(tmp_path, "config", "user.email", "fixture@example.invalid")
    _git(tmp_path, "config", "user.name", "CPU source fixture")
    stable = (set(probe.prior.OWN_FILES) | set(probe.frozen_component.OWN)) - {
        probe.OWN[-1]
    }
    for name in stable:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("frozen " + name)
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "synthetic prior component freeze")
    parent = _git(tmp_path, "rev-parse", "HEAD")
    monkeypatch.setattr(probe, "PARENT", parent)
    monkeypatch.setattr(probe, "FREEZE_SOURCE", parent)
    monkeypatch.setattr(probe, "BASE_SOURCE", parent)
    if damage == "frozen-input":
        path = tmp_path / sorted(stable)[0]
        path.write_text("tampered frozen path")
    for name in probe.OWN:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("declared new order path " + name)
    if damage == "unauthorized":
        (tmp_path / "unrelated.txt").write_text("not owned")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "synthetic order reader source")
    source = _git(tmp_path, "rev-parse", "HEAD")
    monkeypatch.setattr(probe.base.execution, "ROOT", tmp_path)
    if damage is None:
        assert set(probe.source_binding(source)) == stable | set(probe.OWN)
    else:
        with pytest.raises(ValueError, match="only newly declared"):
            probe.source_binding(source)


def _sync_unit(source):
    return probe.source_sync_unit(source) + " loaded active running\n"


def _sync_run(
    tmp_path,
    *,
    units=None,
    head=None,
    branch=None,
    dirty="",
    incoming=None,
    root=None,
    origin=None,
):
    source = "f" * 40
    checksum = "e" * 64
    bin_dir, state = tmp_path / "bin", tmp_path / "state"
    bin_dir.mkdir()
    state.mkdir()
    (state / "head").write_text((probe.FREEZE_SOURCE if head is None else head) + "\n")
    (state / "incoming").write_text((source if incoming is None else incoming) + "\n")
    (state / "branch").write_text(
        ("feat/athletics-obstacle-curriculum" if branch is None else branch) + "\n"
    )
    (state / "dirty").write_text(dirty)
    (state / "units").write_text(_sync_unit(source) if units is None else units)
    (state / "root").write_text(
        (probe.prior.SYNC_ROOT if root is None else root) + "\n"
    )
    (state / "origin").write_text(
        (probe.prior.SYNC_ORIGIN if origin is None else origin) + "\n"
    )
    bundle = (
        Path(probe.prior.SYNC_ROOT)
        / "artifacts/tools"
        / f"cuda64-inertia-source-{source[:12]}.bundle"
    )
    bundle.parent.mkdir(parents=True, exist_ok=True)
    bundle.write_bytes(b"synthetic bundle")
    (state / "bundle-size").write_text(str(bundle.stat().st_size) + "\n")
    (state / "bundle-hash").write_text(checksum + "\n")
    (state / "bundle-heads").write_text(
        f"{source} refs/heads/feat/athletics-obstacle-curriculum\n"
    )

    scripts = {
        "git": r"""#!/usr/bin/env bash
set -euo pipefail
printf 'git %s\n' "$*" >> "$SYNC_LOG"
case "$1" in
 status) cat "$SYNC_STATE/dirty" ;;
 rev-parse) case "$2" in
   --show-toplevel) cat "$SYNC_STATE/root" ;;
   HEAD) cat "$SYNC_STATE/head" ;;
   refs/remotes/origin/feat/athletics-obstacle-curriculum) test -f "$SYNC_STATE/ref"; cat "$SYNC_STATE/ref" ;;
   *) exit 91;; esac ;;
 remote) cat "$SYNC_STATE/origin" ;;
 branch) cat "$SYNC_STATE/branch" ;;
 bundle) case "$2" in verify) true;; list-heads) cat "$SYNC_STATE/bundle-heads";; *) exit 92;; esac ;;
 fetch) test "$2" = "$SYNC_BUNDLE"; printf '%s\n' "$(cat "$SYNC_STATE/incoming")" > "$SYNC_STATE/ref"; touch "$SYNC_STATE/fetched" ;;
 merge) test "$2" = --ff-only; printf '%s\n' "$3" > "$SYNC_STATE/head"; touch "$SYNC_STATE/merged" ;;
 *) exit 93;; esac
""",
        "systemctl": r"""#!/usr/bin/env bash
set -euo pipefail
printf 'systemctl %s\n' "$*" >> "$SYNC_LOG"
cat "$SYNC_STATE/units"
""",
        "stat": r"""#!/usr/bin/env bash
set -euo pipefail
printf 'stat %s\n' "$*" >> "$SYNC_LOG"
test "$1" = -c && test "$2" = %s
cat "$SYNC_STATE/bundle-size"
""",
        "sha256sum": r"""#!/usr/bin/env bash
set -euo pipefail
printf 'sha256sum %s\n' "$*" >> "$SYNC_LOG"
printf '%s  %s\n' "$(cat "$SYNC_STATE/bundle-hash")" "$1"
""",
    }
    for name, script in scripts.items():
        path = bin_dir / name
        path.write_text(script)
        path.chmod(0o755)
    env = {
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "SYNC_LOG": str(tmp_path / "calls.log"),
        "SYNC_STATE": str(state),
        "SYNC_BUNDLE": str(bundle),
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
    logpath = tmp_path / "calls.log"
    return (
        process,
        logpath.read_text() if logpath.exists() else "",
        state,
        source,
        script,
    )


def test_bootstrap_retargets_only_two_unique_parent_self_unit_guards():
    source, checksum = "f" * 40, "e" * 64
    template = probe.prior.source_sync_script(source, bundle_sha256=checksum)
    result = probe.source_sync_script(source, bundle_sha256=checksum)
    expected = template.replace(
        f'test "$(git rev-parse HEAD)" = {probe.prior.SYNC_FROM_SOURCE}\n',
        f'test "$(git rev-parse HEAD)" = {probe.FREEZE_SOURCE}\n',
        1,
    ).replace(
        f'if test "$duck_sync_running" != {probe.prior.source_sync_unit(source)}; then\n',
        f'if test "$duck_sync_running" != {probe.source_sync_unit(source)}; then\n',
        1,
    )
    assert result == expected
    assert f'test "$(git rev-parse HEAD)" = {probe.FREEZE_SOURCE}' in result
    assert (
        f'if test "$duck_sync_running" != {probe.source_sync_unit(source)}; then'
        in result
    )


@pytest.mark.parametrize(
    "guard,kind", [(0, "missing"), (0, "duplicate"), (1, "missing"), (1, "duplicate")]
)
def test_bootstrap_refuses_missing_or_repeated_frozen_template_guard(
    monkeypatch, guard, kind
):
    source, checksum = "f" * 40, "e" * 64
    template = probe.prior.source_sync_script(source, bundle_sha256=checksum)
    lines = (
        f'test "$(git rev-parse HEAD)" = {probe.prior.SYNC_FROM_SOURCE}\n',
        f'if test "$duck_sync_running" != {probe.prior.source_sync_unit(source)}; then\n',
    )
    damaged = (
        template.replace(lines[guard], "", 1)
        if kind == "missing"
        else template + lines[guard]
    )
    monkeypatch.setattr(probe.prior, "source_sync_script", lambda *_a, **_k: damaged)
    with pytest.raises(ValueError, match="exact frozen bootstrap guard occurrence"):
        probe.source_sync_script(source, bundle_sha256=checksum)


def test_bundle_bootstrap_validates_exact_self_head_and_merges_fast_forward_only(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(probe.prior, "SYNC_ROOT", str(tmp_path))
    process, log, state, source, script = _sync_run(tmp_path)
    assert process.returncode == 0, process.stderr
    assert process.stdout.strip() == source
    assert (state / "fetched").exists() and (state / "merged").exists()
    assert f"git merge --ff-only {source}" in log
    assert "test -f " in script and "test ! -L " in script


@pytest.mark.parametrize(
    "damage",
    [
        "wrong-head",
        "old-self-unit",
        "foreign-self-unit",
        "competitor",
        "dirty",
        "wrong-root",
        "wrong-origin",
        "remote-head",
    ],
)
def test_bootstrap_rejects_bad_old_head_or_service_context_before_merge(
    tmp_path, monkeypatch, damage
):
    monkeypatch.setattr(probe.prior, "SYNC_ROOT", str(tmp_path))
    kwargs = {}
    if damage == "wrong-head":
        kwargs["head"] = "a" * 40
    elif damage == "old-self-unit":
        kwargs["units"] = probe.prior.source_sync_unit("f" * 40) + " active running\n"
    elif damage == "foreign-self-unit":
        kwargs["units"] = "microduck-other-sync.service loaded active running\n"
    elif damage == "competitor":
        kwargs["units"] = (
            _sync_unit("f" * 40) + "microduck-foreign.service loaded active running\n"
        )
    elif damage == "dirty":
        kwargs["dirty"] = " M file.py\n"
    elif damage == "wrong-root":
        kwargs["root"] = "/wrong/worktree"
    elif damage == "wrong-origin":
        kwargs["origin"] = "https://invalid.example/microduck_rl.git"
    elif damage == "remote-head":
        kwargs["incoming"] = "a" * 40
    process, log, state, _source, _script = _sync_run(tmp_path, **kwargs)
    assert process.returncode != 0
    assert "git merge" not in log and not (state / "merged").exists()
    if damage != "remote-head":
        assert "git fetch" not in log and not (state / "fetched").exists()
    else:
        assert "git fetch" in log and (state / "fetched").exists()


@pytest.mark.parametrize("damage", [None, "protected-active", "protected-change"])
def test_execute_retains_diagnostic_and_protects_inputs_services_and_admission(
    tmp_path, monkeypatch, authenticated_anchors, damage
):
    fixture = authenticated_anchors
    root = tmp_path / "new-order-output"
    monkeypatch.setattr(probe.prior, "_hidden", lambda: None)
    monkeypatch.setattr(probe.prior, "check_window", lambda **_kwargs: None)
    service = dict(CAPS)
    monkeypatch.setattr(probe, "properties", lambda *_: service)
    monkeypatch.setattr(probe, "source_binding", lambda *_: {"frozen": True})
    monkeypatch.setattr(probe.base.host, "identity", lambda *_: {"source": SOURCE})
    monkeypatch.setattr(probe, "output_path", lambda *_: root)
    monkeypatch.setattr(probe, "_read_tests", lambda *_: "e" * 64)

    @contextmanager
    def lease():
        yield 19

    monkeypatch.setattr(probe.base.gap.base.files, "gpu_lease", lease)
    monkeypatch.setattr(probe.base.gpu_idle_gate, "wait_idle", lambda: {"idle": True})
    monkeypatch.setattr(
        probe.base.gap.base.retained.d0, "filmbrain_state", lambda: {"same": True}
    )
    protected_calls = []

    def protected():
        protected_calls.append(True)
        if damage == "protected-active":
            return {"protected": "active"}
        if damage == "protected-change" and len(protected_calls) > 1:
            return {"protected": "active"}
        return {"protected": "inactive"}

    monkeypatch.setattr(probe.base.gap.base, "protected_state", protected)
    result = {
        "protocol": probe.oracle.PROTOCOL + ":comparison",
        "source": probe.PARENT,
        "worlds": 64,
        "compiled_topology_sha256": probe.MAP_TOPOLOGY_SHA,
        **probe.oracle.FLAGS,
    }
    monkeypatch.setattr(
        probe,
        "read_compare",
        lambda: (
            result,
            fixture["old_inventory"],
            {
                "comparison_sha256": probe.MAP_COMPARISON_SHA,
                "report_sha256": probe.MAP_REPORT_SHA,
                "smooth_sha256": probe.SMOOTH_SHA,
                "compiled_topology_sha256": probe.MAP_TOPOLOGY_SHA,
            },
        ),
    )
    monkeypatch.setattr(
        probe,
        "authenticate_inputs",
        lambda: (
            fixture["old_root"],
            fixture["launch"],
            "c" * 64,
            fixture["old_inventory"],
            {
                "comparison_sha256": probe.MAP_COMPARISON_SHA,
                "report_sha256": probe.MAP_REPORT_SHA,
                "smooth_sha256": probe.SMOOTH_SHA,
                "compiled_topology_sha256": probe.MAP_TOPOLOGY_SHA,
            },
        ),
    )
    retained_before = {
        path.name: probe.base.digest(path.read_bytes())
        for path in fixture["map_root"].iterdir()
        if path.is_file()
    }
    old_files_before = probe.prior._inventory(
        fixture["old_root"], probe.prior.COMPLETE_FILES
    )

    if damage is None:
        assert probe.execute(SOURCE, "run")["status"] == "passed"
    else:
        with pytest.raises(ValueError):
            probe.execute(SOURCE, "run")
    report = probe.base.parse_json((root / "report.json").read_bytes())
    assert all(report[key] is False for key in probe.FLAGS)
    if damage != "protected-active":
        assert report["original_pair_accepted"] is False
        assert (
            report["original_pair_error"] == "paired rollout semantic state exactness"
        )
    if damage is None:
        assert (
            report["protected_services"]
            == report["protected_services_after"]
            == {"protected": "inactive"}
        )
    assert (root / "comparison.json").exists() is (damage != "protected-active")
    assert {
        path.name: probe.base.digest(path.read_bytes())
        for path in fixture["map_root"].iterdir()
        if path.is_file()
    } == retained_before
    assert (
        probe.prior._inventory(fixture["old_root"], probe.prior.COMPLETE_FILES)
        == old_files_before
    )
    original_report = (root / "report.json").read_bytes()
    with pytest.raises(FileExistsError):
        probe.execute(SOURCE, "run")
    assert (root / "report.json").read_bytes() == original_report
