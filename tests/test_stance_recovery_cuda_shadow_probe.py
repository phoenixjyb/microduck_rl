"""Source/record tests only; synthetic envelopes never qualify a native run."""

from copy import deepcopy
import io
import json
import os

import pytest
import torch

from mjlab_microduck import stance_recovery_cuda_shadow_probe as probe
from test_stance_recovery_cuda_shadow_evidence import (  # noqa: F401 - imported pytest fixture
    pair_input as synthetic_pair,
)

SOURCE = "b" * 40


def properties(mode):
    return {
        "MainPID": str(os.getpid()),
        "ActiveState": "active",
        "RuntimeMaxUSec": f"{probe.SECONDS[mode] // 60}min",
        "MemoryMax": str(probe.MEMORY[mode]),
        "CPUQuotaPerSecUSec": "2s",
        "Nice": "10",
        "KillMode": "control-group",
        "InvocationID": "a" * 32,
    }


def test_fixed_four_child_declaration_is_not_training():
    d = probe.declaration()
    assert d["pairs"] == [
        [653, "capture"],
        [653, "replay"],
        [659, "capture"],
        [659, "replay"],
    ]
    assert d["launch_reserve_seconds"] == 840
    assert d["service_seconds"] == {
        "preflight": 180,
        "tests": 240,
        "run": 600,
        "closeout": 180,
    }
    assert d["simulation_steps"] == d["optimizer_steps"] == 0
    assert d["rollout_collection"] is d["physical_motion_authorized"] is False
    assert all(d[k] is False for k in probe.FALSE_FLAGS)
    assert probe.MODULE == "mjlab_microduck.stance_recovery_cuda_shadow_probe"


@pytest.mark.parametrize("mode", list(probe.SECONDS))
def test_exact_service_caps(mode):
    assert probe._recorded_properties(properties(mode), mode) is True


@pytest.mark.parametrize(
    "field,value",
    [
        ("MainPID", "0"),
        ("ActiveState", "inactive"),
        ("RuntimeMaxUSec", "11min"),
        ("MemoryMax", str(4 * 1024**3)),
        ("CPUQuotaPerSecUSec", "3s"),
        ("Nice", "0"),
        ("KillMode", "process"),
        ("InvocationID", "a" * 31),
    ],
)
def test_changed_service_caps_refused(field, value):
    state = properties("run")
    state[field] = value
    with pytest.raises(ValueError):
        probe._recorded_properties(state, "run")


def test_owned_service_pid_and_no_other_duck(monkeypatch):
    state = properties("run")

    def read(*args):
        if "list-units" in args:
            return probe.unit(SOURCE, "run") + " loaded active running"
        return state[args[-2]]

    monkeypatch.setattr(probe.base.host, "read", read)
    assert probe._properties(SOURCE, "run") == state
    state["MainPID"] = "999999"
    with pytest.raises(ValueError, match="owns this process"):
        probe._properties(SOURCE, "run")
    state["MainPID"] = str(os.getpid())
    original = read
    monkeypatch.setattr(
        probe.base.host,
        "read",
        lambda *args: (
            original(*args) + "\nmicroduck-other.service running"
            if "list-units" in args
            else original(*args)
        ),
    )
    with pytest.raises(ValueError, match="only this Duck"):
        probe._properties(SOURCE, "run")


@pytest.mark.parametrize(
    "field,value",
    [
        ("MainPID", "1"),
        ("ActiveState", "active"),
        ("NRestarts", "1"),
        ("ExecMainStatus", "1"),
        ("Result", "exit-code"),
        ("InvocationID", "f" * 32),
    ],
)
def test_failed_active_restarted_replaced_unit_cannot_close(monkeypatch, field, value):
    state = {
        "MainPID": "0",
        "ActiveState": "inactive",
        "NRestarts": "0",
        "ExecMainStatus": "0",
        "Result": "success",
        "InvocationID": "",
    }
    state[field] = value
    monkeypatch.setattr(probe.base.host, "read", lambda *args: state[args[-2]])
    with pytest.raises(ValueError, match="successfully terminal"):
        probe.completed(SOURCE, "run", "a" * 32)


@pytest.mark.parametrize("invocation", ["", "a" * 32])
def test_completed_gc_or_original_invocation(monkeypatch, invocation):
    state = {
        "MainPID": "0",
        "ActiveState": "inactive",
        "NRestarts": "0",
        "ExecMainStatus": "0",
        "Result": "success",
        "InvocationID": invocation,
    }
    monkeypatch.setattr(probe.base.host, "read", lambda *args: state[args[-2]])
    assert probe.completed(SOURCE, "run", "a" * 32) == state


def test_old_namespace_and_unknown_mode_refused():
    with pytest.raises(ValueError, match="separate shadow namespace"):
        probe.output_path(probe.closed.ARTIFACT_SOURCE)
    with pytest.raises(ValueError, match="declared shadow service"):
        probe.unit(SOURCE, "train")


def test_gpu_sample_pairs_keep_fixed_order(monkeypatch):
    children = [{"seed": s, "attempt": a} for s, a in probe.PAIRS]
    calls = []
    monkeypatch.setattr(
        probe.base, "_check_live_gpu_samples", lambda rows: calls.append(rows)
    )
    probe._gpu_samples(children)
    assert calls == [[children[0], children[2]], [children[1], children[3]]]
    children.reverse()
    with pytest.raises(ValueError, match="sequential capture/replay"):
        probe._gpu_samples(children)


def test_child_lease_refusal_precedes_visibility_and_cuda(monkeypatch):
    def reject(_fd):
        raise ValueError("SYNTHETIC missing inherited lease")

    monkeypatch.setattr(probe.base.training_smoke, "inherited_lease", reject)
    monkeypatch.setattr(
        probe, "_read_launch", lambda *args: pytest.fail("read before lease")
    )
    monkeypatch.setattr(
        probe.base.torch.cuda,
        "is_initialized",
        lambda: pytest.fail("CUDA before lease"),
    )
    with pytest.raises(ValueError, match="missing inherited lease"):
        probe.child(SOURCE, "a" * 64, 653, "capture", 19)


def test_child_hidden_visibility_refused_before_launch(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(probe.base.training_smoke, "inherited_lease", lambda fd: None)
    monkeypatch.setattr(
        probe,
        "_read_launch",
        lambda *args: pytest.fail("read before visibility refusal"),
    )
    with pytest.raises(ValueError, match="visible CUDA0 only"):
        probe.child(SOURCE, "a" * 64, 653, "capture", 19)


def test_exact_child_startup_settings_override_inherited_math(monkeypatch):
    monkeypatch.setattr(
        probe.base.campaign,
        "child_environment",
        lambda: {
            "OMP_NUM_THREADS": "99",
            "ATEN_CPU_CAPABILITY": "avx2",
            "MKL_CBWR": "AUTO",
            "CUDA_VISIBLE_DEVICES": "1",
            "MICRODUCK_STANCE_PROFILE": probe.base.execution.WSL,
        },
    )
    value = probe.child_environment()
    assert {k: value[k] for k in ("OMP_NUM_THREADS", "ATEN_CPU_CAPABILITY", "MKL_CBWR")}
    assert value["OMP_NUM_THREADS"] == "1"
    assert value["ATEN_CPU_CAPABILITY"] == "default"
    assert value["MKL_CBWR"] == "COMPATIBLE"
    assert value["CUDA_VISIBLE_DEVICES"] == "0"
    assert value["PYTHONUNBUFFERED"] == "1"
    assert value["PATH"].startswith(str(probe.base.execution.ROOT / ".venv/bin") + ":")


def test_closeout_terminal_refusal_retains_failure_record(monkeypatch, tmp_path):
    monkeypatch.setattr(probe, "_hidden", lambda: None)
    monkeypatch.setattr(probe.base.window, "check", lambda **kwargs: None)
    monkeypatch.setattr(probe, "_properties", lambda *args: properties("closeout"))
    monkeypatch.setattr(
        probe,
        "_read_launch",
        lambda *args: (
            tmp_path,
            {"service_properties": {"InvocationID": "a" * 32}},
            {},
        ),
    )

    def refusal(*_args):
        raise ValueError("SYNTHETIC original run not terminal")

    monkeypatch.setattr(probe, "completed", refusal)
    with pytest.raises(ValueError, match="original run not terminal"):
        probe.closeout(SOURCE, "a" * 64)
    value = json.loads((tmp_path / "independent-closeout.json").read_bytes())
    assert value["status"] == "failed-retained"
    assert value["error_type"] == "ValueError"
    assert value["error"] == "SYNTHETIC original run not terminal"
    assert all(value[k] is False for k in probe.FALSE_FLAGS)


def test_complete_work_deadline_is_not_a_start_only_gate():
    declaration = probe.base.window.declaration()
    cutoff = declaration["cutoff_unix"]
    probe.base.window.check(
        now=cutoff - probe.RUN_RESERVE - 1, reserve_seconds=probe.RUN_RESERVE
    )
    with pytest.raises(ValueError, match="complete closeout"):
        probe.base.window.check(
            now=cutoff - probe.RUN_RESERVE, reserve_seconds=probe.RUN_RESERVE
        )


def test_whole_launch_hash_refused_before_schema(monkeypatch):
    monkeypatch.setattr(probe.base, "_read_file", lambda *args: b"{}\n")
    with pytest.raises(ValueError, match="whole shadow launch before parse"):
        probe._read_launch(SOURCE, "a" * 64)


@pytest.fixture
def synthetic_wire(tmp_path, synthetic_pair):  # noqa: F811 - injected imported fixture
    # The unchanged old preparation is authentic; its new sampler envelope is
    # deliberately synthetic, with no current source/host/service attestation.
    prepared = synthetic_pair["prepared_snapshot"]
    meta = prepared["metadata"]
    source = meta["source"]
    sha = meta["launch_sha256"]
    sampled = deepcopy(synthetic_pair["sampling"])
    sampled["receipt"]["source"] = sampled["scope_receipt"]["source"] = source
    cpu = meta["raw_preparation_receipt"]["cpu_parent_receipt"]
    cpu_raw = (canonical(cpu) + "\n").encode()
    (tmp_path / "cpu-parent-receipt.json").write_bytes(cpu_raw)
    launch = {
        "source": source,
        "cpu_parent_receipt_file_sha256": probe.base.digest(cpu_raw),
        "cpu_parent_receipt_canonical_sha256": probe.sampling.preparation._cpu_parent_receipt_digest(
            cpu
        ),
        "cpu_parent_binding": meta["raw_preparation_receipt"]["caller_binding"],
        "native_prerequisites": {
            "current_context": {
                "source_identity": meta["raw_preparation_receipt"]["source_identity"],
                "cpu_math_profile": cpu["cpu_math_profile"],
            }
        },
    }
    stream = io.BytesIO()
    torch.save(prepared, stream)
    prep_raw = stream.getvalue()
    payload = {
        "protocol": probe.PROTOCOL + ":child-v1",
        "source": source,
        "launch_sha256": sha,
        "seed": 653,
        "attempt": "capture",
        "preparation_raw": prep_raw,
        "sampling": sampled,
    }
    stream = io.BytesIO()
    torch.save(payload, stream)
    raw = stream.getvalue()
    summary = {
        "protocol": probe.PROTOCOL + ":child-summary-v1",
        "source": source,
        "launch_sha256": sha,
        "seed": 653,
        "attempt": "capture",
        "payload_sha256": probe.base.digest(raw),
        "payload_bytes": len(raw),
        "preparation_sha256": probe.base.digest(prep_raw),
        "preparation_metadata": meta,
        "sampling_receipt": sampled["receipt"],
        "optimizer_steps": 0,
        **probe.FALSE_FLAGS,
    }
    (tmp_path / "seed-653-capture.pt").write_bytes(raw)
    (tmp_path / "seed-653-capture.json").write_text(canonical(summary) + "\n")
    return tmp_path, launch, sha, raw, summary


def canonical(value):
    return probe.canonical(value)


def test_complete_embedded_preparation_and_sampling_cpu_score(synthetic_wire):
    root, launch, sha, _, _ = synthetic_wire
    pair, score, summary = probe._score_child(root, launch, sha, 653, "capture")
    assert score["cuda_math_replayed"] is False
    assert summary["attempt"] == "capture"
    assert pair["sampling"]["receipt"]["native_cuda_sampling_qualified"] is False


def test_whole_child_bytes_refused_before_torch_load(synthetic_wire):
    root, launch, sha, raw, _ = synthetic_wire
    (root / "seed-653-capture.pt").write_bytes(raw + b" ")
    with pytest.raises(ValueError, match="whole shadow payload before"):
        probe._score_child(root, launch, sha, 653, "capture")


@pytest.mark.parametrize(
    "field", ["seed", "payload_bytes", "optimizer_steps", "unexpected"]
)
def test_boolean_counter_or_extra_summary_refused(synthetic_wire, field):
    root, launch, sha, _, summary = synthetic_wire
    summary[field] = False
    (root / "seed-653-capture.json").write_text(json.dumps(summary) + "\n")
    with pytest.raises(ValueError, match="typed non-admitting child summary schema"):
        probe._score_child(root, launch, sha, 653, "capture")


@pytest.mark.parametrize(
    "field,value",
    [
        ("attempt", "replay"),
        ("source", "f" * 40),
        ("launch_sha256", "f" * 64),
        ("execution_admitted", True),
        ("sampling_receipt", {}),
    ],
)
def test_outer_summary_cannot_borrow_other_raw(synthetic_wire, field, value):
    root, launch, sha, _, summary = synthetic_wire
    summary[field] = value
    (root / "seed-653-capture.json").write_text(json.dumps(summary) + "\n")
    with pytest.raises(ValueError, match="whole child binding"):
        probe._score_child(root, launch, sha, 653, "capture")
