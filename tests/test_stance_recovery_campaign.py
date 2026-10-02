"""CPU-hidden orchestration tests for measured five-case frozen baseline."""

from copy import deepcopy
from hashlib import sha256 as real_sha256
import json
import os
from pathlib import Path

import pytest

from mjlab_microduck import stance_recovery_campaign as campaign
from mjlab_microduck import stance_recovery_contract as contract
from mjlab_microduck import stance_recovery_matrix as matrix


SOURCE = "a" * 40
LAUNCH_SHA = "b" * 64
REPORT_RAW = b"synthetic retained R1 report bytes"
CAPTURE_RAW = b"synthetic CPU-rescored R1 capture"
CHECKPOINT_RAW = b"fixed selected checkpoint bytes"
DECLARATION = {"source": SOURCE, "plant": {"bound": "synthetic"}}


def r1_score():
    return dict(case=contract.PROBE_CASE,
        numerical_diagnostic={"gates": {"full_duration": True}},
        pulse={"checked_physics_steps": contract.TOTAL_STEPS,
               "complete_pulse_delivery": True}, **contract.FALSE_FLAGS)


def timing_report():
    repeating = 166.40493231918664
    entry = 24.82244214182718
    wrapper = 2.71480189403518
    capture = dict(source=campaign.TIMING_SOURCE, optimizer_steps=0,
        backend={"torch_device": "cuda:0", "warp_is_cuda": True},
        construction_seconds=1., collection={"policy_ticks": contract.POLICY_TICKS,
            "elapsed_seconds": repeating-2.}, serialization_seconds=1.,
        elapsed_seconds=repeating+entry, capture_sha256=real_sha256(CAPTURE_RAW).hexdigest(),
        capture_bytes=len(CAPTURE_RAW), **contract.FALSE_FLAGS)
    score = r1_score()
    child_elapsed = capture["elapsed_seconds"]+wrapper
    return dict(source=campaign.TIMING_SOURCE, decision="frozen-recovery-timing-probe-replayed",
        full_length_probe_completed=True, child={"returncode": 0, "elapsed_s": child_elapsed},
        capture=capture, score=score, elapsed_seconds=244.01035961904563,
        supervisor_replay_seconds=6.111677116947249,
        files={"capture.pt": real_sha256(CAPTURE_RAW).hexdigest(),
               "checkpoint.pt": real_sha256(CHECKPOINT_RAW).hexdigest(),
               "launch.json": real_sha256(b"launch").hexdigest()},
        **contract.FALSE_FLAGS)


def valid_receipt(case="zero-wrench"):
    diagnostic = dict(candidate_pass=True, gates={
        "complete_first_attempt": True, "full_duration": True, "no_hard_failure": True,
        "displacement": True, "soft_limit": True, "final_tilt": True,
        "final_speed": True, "final_height": True, "foot_support": True})
    score = dict(protocol=contract.PROTOCOL, case=case, numerical_diagnostic=diagnostic,
        pulse=dict(checked_physics_steps=contract.TOTAL_STEPS,
            pulse_window_steps=contract.PULSE_STEPS, complete_pulse_delivery=True,
            complete_pulse_window_phase_checks=True, exact_scheduled_forces_checked=True,
            unforced_post_arrays_checked=True), actor_replay_max_abs_error=0.,
        whole_trajectory_physics_resimulated=False, **contract.FALSE_FLAGS)
    return dict(protocol=matrix.PROTOCOL, source=SOURCE, case=case,
        declaration_sha256="d"*64, checkpoint_sha256=contract.CHECKPOINT_SHA256,
        evaluation_seed=contract.EVALUATION_SEED, capture_sha256="e"*64, capture_bytes=10,
        prefix_sha256="f"*64, matched_through_physics_step=contract.ONSET_STEP,
        first_pre_force_action_included=True, post_push_actions_compared=False,
        score=score, whole_trajectory_physics_resimulated=False, **contract.FALSE_FLAGS)


def test_measured_budget_reproduces_predeclared_math_and_caps(monkeypatch):
    report = timing_report()
    result = campaign.measured_budget(report)
    assert result["repeating_case_seconds"] == pytest.approx(166.40493231918664)
    assert result["entry_seconds"] == pytest.approx(24.82244214182718)
    assert result["wrapper_seconds"] == pytest.approx(2.71480189403518)
    assert result["predicted_child_seconds"] == pytest.approx(859.5619056317956)
    assert result["child_rounded_seconds"] == 1075
    assert result["parent_rounded_seconds"] == 202
    assert result["child_seconds"] == 1352 and result["service_seconds"] == 1560
    assert result["case_seconds"] == 250 and result["launch_reserve_seconds"] == 1800
    assert result["memory_bytes"] == 2*1024**3 and result["cpu_quota_percent"] == 200
    assert result["nice"] == 10 and result["kill_mode"] == "control-group"


@pytest.mark.parametrize("damage", [
    "source", "decision", "partial-report", "child-exit", "report-admission",
    "capture-source", "optimizer", "backend", "ticks", "score-case", "partial-score",
    "pulse-steps", "pulse-delivery", "capture-admission", "score-admission",
    "nan-timing", "infinite-timing", "zero-timing", "rounding-drift", "wrapper-drift",
])
def test_measured_budget_rejects_partial_admitting_or_invalid_timing(monkeypatch, damage):
    report = timing_report()
    if damage == "source": report["source"] = "0"*40
    elif damage == "decision": report["decision"] = "failed"
    elif damage == "partial-report": report["full_length_probe_completed"] = False
    elif damage == "child-exit": report["child"]["returncode"] = 1
    elif damage == "report-admission": report["training_admitted"] = True
    elif damage == "capture-source": report["capture"]["source"] = "0"*40
    elif damage == "optimizer": report["capture"]["optimizer_steps"] = 1
    elif damage == "backend": report["capture"]["backend"]["torch_device"] = "cpu"
    elif damage == "ticks": report["capture"]["collection"]["policy_ticks"] -= 1
    elif damage == "score-case": report["score"]["case"] = "-x"
    elif damage == "partial-score": report["score"]["numerical_diagnostic"]["gates"]["full_duration"] = False
    elif damage == "pulse-steps": report["score"]["pulse"]["checked_physics_steps"] -= 1
    elif damage == "pulse-delivery": report["score"]["pulse"]["complete_pulse_delivery"] = False
    elif damage == "capture-admission": report["capture"]["training_admitted"] = True
    elif damage == "score-admission": report["score"]["training_admitted"] = True
    elif damage == "nan-timing": report["capture"]["construction_seconds"] = float("nan")
    elif damage == "infinite-timing": report["capture"]["collection"]["elapsed_seconds"] = float("inf")
    elif damage == "zero-timing": report["child"]["elapsed_s"] = 0.
    elif damage == "rounding-drift": report["elapsed_seconds"] += 20.
    else: monkeypatch.setattr(campaign.files, "LEAN_EVALUATION_CHILD_SECONDS", 1351)
    with pytest.raises(ValueError):
        campaign.measured_budget(report)


def test_launch_window_requires_strictly_more_than_1800_seconds():
    campaign.check_window(launching=True, now=contract.CUTOFF-1801)
    with pytest.raises(ValueError, match="before fixed 08:00 Shanghai cutoff"):
        campaign.check_window(launching=True, now=contract.CUTOFF-1800)
    with pytest.raises(ValueError, match="before fixed 08:00 Shanghai cutoff"):
        campaign.check_window(launching=True, now=contract.CUTOFF-1799)
    campaign.check_window(now=contract.CUTOFF-1)


def test_case_paths_are_five_ordered_distinct_owned_names(tmp_path):
    paths = [campaign.case_paths(tmp_path, index) for index in range(len(contract.CASE_NAMES))]
    assert [p[0].name for p in paths] == [f"case-{i}.pt" for i in range(5)]
    assert [p[1].name for p in paths] == [f"case-{i}.json" for i in range(5)]
    assert len({p for pair in paths for p in pair}) == 10
    for bad in (-1, 5, True, 1.0):
        with pytest.raises(ValueError, match="ordered bounded matrix index"):
            campaign.case_paths(tmp_path, bad)


def test_launch_plan_checks_prefix_before_pinned_timing_and_binds_five_cases(monkeypatch):
    events = []
    prefix = {"selected_checkpoint": {"identity": {"iteration": 255}}, "declaration": DECLARATION}
    timing = {"source": campaign.TIMING_SOURCE, "report_sha256": campaign.TIMING_REPORT_SHA,
              "budget": {"child_seconds": campaign.CHILD_SECONDS}}
    monkeypatch.setattr(campaign.probe, "checked", lambda source, digest: (
        events.append(("checked", source, digest)), prefix)[1])
    monkeypatch.setattr(campaign, "timing_prerequisite", lambda: (events.append(("timing", None)), timing)[1])
    result = campaign.launch_plan(SOURCE, LAUNCH_SHA)
    assert events == [("checked", SOURCE, LAUNCH_SHA), ("timing", None)]
    assert result["protocol"] == matrix.PROTOCOL and result["source"] == SOURCE
    assert result["prefix_launch_sha256"] == LAUNCH_SHA and result["prefix_launch"] == prefix
    assert result["cases"] == list(contract.CASE_NAMES) and result["timing"] == timing
    assert all(result[key] is False for key in contract.FALSE_FLAGS)


def test_prepare_creates_fresh_source_bound_campaign_root_after_prefix_and_timing(monkeypatch, tmp_path):
    root = tmp_path / ("matrix-" + SOURCE[:12]); events = []
    prefix = {"launch_sha256": LAUNCH_SHA}
    launch = {"source": SOURCE, "prefix_launch_sha256": LAUNCH_SHA,
              "cases": list(contract.CASE_NAMES), **contract.FALSE_FLAGS}
    monkeypatch.setattr(campaign, "check_window", lambda **kwargs: events.append(("window", kwargs)))
    monkeypatch.setattr(campaign.probe, "prepare", lambda source: (
        events.append(("prefix-prepare", source)), prefix)[1])
    monkeypatch.setattr(campaign, "launch_plan", lambda source, digest: (
        events.append(("launch-plan", source, digest)), launch)[1])
    monkeypatch.setattr(campaign, "output_path", lambda source: (
        events.append(("output", source)), root)[1])
    monkeypatch.setattr(campaign.files.native, "_plain_path", lambda path: path)
    monkeypatch.setattr(campaign.files, "write_json", lambda path, value: (
        events.append(("write", Path(path).name, value)), path.write_text(json.dumps(value, sort_keys=True)))[1])
    monkeypatch.setattr(campaign, "service_name", lambda source: "five-case.service")
    monkeypatch.setattr(campaign.host, "digest", lambda path: real_sha256(Path(path).read_bytes()).hexdigest())

    result = campaign.prepare(SOURCE)
    assert events.index(("prefix-prepare", SOURCE)) < events.index(("launch-plan", SOURCE, LAUNCH_SHA))
    assert events.index(("launch-plan", SOURCE, LAUNCH_SHA)) < events.index(("output", SOURCE))
    assert root.is_dir() and json.loads((root/"launch.json").read_text()) == launch
    assert result["output"] == str(root) and result["service"] == "five-case.service"
    assert result["launch_sha256"] == real_sha256((root/"launch.json").read_bytes()).hexdigest()


def test_timing_prerequisite_pins_r1_report_and_case_bytes_before_fresh_cpu_rescore(monkeypatch, tmp_path):
    root = tmp_path / "r1"; root.mkdir()
    report = timing_report()
    report_raw = json.dumps(report, sort_keys=True).encode()
    launch = {"declaration": DECLARATION}
    launch_raw = json.dumps(launch, sort_keys=True).encode()
    digest_by_name = {"capture.pt": real_sha256(CAPTURE_RAW).hexdigest(),
                      "checkpoint.pt": real_sha256(CHECKPOINT_RAW).hexdigest(),
                      "launch.json": real_sha256(b"launch").hexdigest()}
    events = []
    monkeypatch.setattr(campaign.probe, "output_path", lambda source: (events.append(("root", source)), root)[1])
    monkeypatch.setattr(campaign.torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(campaign, "TIMING_REPORT_SHA", real_sha256(report_raw).hexdigest())
    monkeypatch.setattr(campaign.host, "digest", lambda path: (
        events.append(("artifact-pin", Path(path).name)), digest_by_name[Path(path).name])[1])
    def file_bytes(path, **_kwargs):
        name = Path(path).name; events.append(("read", name))
        return {"report.json": report_raw, "capture.pt": CAPTURE_RAW,
                "checkpoint.pt": CHECKPOINT_RAW, "launch.json": launch_raw}[name]
    monkeypatch.setattr(campaign.files, "file_bytes", file_bytes)
    def parse(raw):
        events.append(("parse", raw))
        return json.loads(raw)
    monkeypatch.setattr(campaign.files, "parse", parse)
    monkeypatch.setattr(campaign.evidence, "verify", lambda raw, digest, cp, declaration: (
        events.append(("verify", raw, digest, cp, declaration)), report["score"])[1])

    result = campaign.timing_prerequisite()
    assert result["source"] == campaign.TIMING_SOURCE
    assert result["report_sha256"] == real_sha256(report_raw).hexdigest()
    assert result["files"] == report["files"]
    assert events[0] == ("root", campaign.TIMING_SOURCE)
    assert events[1] == ("read", "report.json")
    assert events.index(("read", "report.json")) < events.index(("parse", report_raw))
    verify_index = next(i for i, event in enumerate(events) if event[0] == "verify")
    raw_read = events.index(("read", "capture.pt"))
    assert raw_read < verify_index
    assert all(events.index(("artifact-pin", name)) < raw_read for name in digest_by_name)
    assert events[verify_index] == ("verify", CAPTURE_RAW, real_sha256(CAPTURE_RAW).hexdigest(),
                                    CHECKPOINT_RAW, DECLARATION)


def test_timing_prerequisite_refuses_initialized_cuda_before_r1_reads(monkeypatch):
    events = []
    monkeypatch.setattr(campaign.torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(campaign.probe, "output_path", lambda *_args: events.append("root"))
    monkeypatch.setattr(campaign.files, "file_bytes", lambda *_args, **_kwargs: events.append("read"))
    with pytest.raises(ValueError, match="fresh CPU timing qualification before CUDA initialization"):
        campaign.timing_prerequisite()
    assert events == []


@pytest.mark.parametrize("damage", ["report-hash", "capture-bytes"])
def test_timing_prerequisite_rejects_bad_pinned_bytes_before_replay(monkeypatch, tmp_path, damage):
    root = tmp_path / "r1"; root.mkdir(); events = []
    report = timing_report(); report_raw = json.dumps(report, sort_keys=True).encode()
    monkeypatch.setattr(campaign.probe, "output_path", lambda _source: root)
    monkeypatch.setattr(campaign.torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(campaign, "TIMING_REPORT_SHA", "0"*64 if damage == "report-hash"
                        else real_sha256(report_raw).hexdigest())
    files_seen = []
    def file_bytes(path, **_kwargs):
        name = Path(path).name; files_seen.append(name)
        if name == "report.json": return report_raw
        if name == "capture.pt": return b"tampered case" if damage == "capture-bytes" else CAPTURE_RAW
        if name == "checkpoint.pt": return CHECKPOINT_RAW
        return json.dumps({"declaration": DECLARATION}).encode()
    monkeypatch.setattr(campaign.files, "file_bytes", file_bytes)
    monkeypatch.setattr(campaign.files, "parse", lambda raw: json.loads(raw))
    monkeypatch.setattr(campaign.host, "digest", lambda path: report["files"][Path(path).name])
    monkeypatch.setattr(campaign.evidence, "verify", lambda *_args: pytest.fail("unverified R1 bytes replayed"))
    if damage == "capture-bytes":
        monkeypatch.setattr(campaign.host, "digest", lambda path: report["files"][Path(path).name])
    with pytest.raises(ValueError):
        campaign.timing_prerequisite()
    if damage == "report-hash":
        assert files_seen == ["report.json"]
    else:
        assert "capture.pt" in files_seen


def _child_seams(monkeypatch, tmp_path, *, partial=False):
    root = tmp_path / "matrix"; root.mkdir()
    started = 1000.
    launch = {"prefix_launch": {"selected_checkpoint": {"identity": {"iteration": 255}},
                               "declaration": DECLARATION}}
    events, actors, envs, cases = [], [], [], []
    times = [started+1, started+20]
    for index in range(5):
        base = started+20+index*30
        times.extend((base, base+.1, base+1., base+15., base+16., base+17., base+20.))
    ticks = iter(times)
    monkeypatch.setattr(campaign.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(campaign, "check_window", lambda **kwargs: events.append(("window", kwargs)))
    monkeypatch.setattr(campaign.probe.smoke, "inherited_lease", lambda fd: events.append(("lease", fd)))
    monkeypatch.setattr(campaign, "checked", lambda *_args: (events.append(("checked", None)), launch)[1])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    # This monkeypatch exercises only the branch order; no CUDA runtime is initialized.
    monkeypatch.setattr(campaign.torch.cuda, "is_available", lambda: events.append(("cuda-query", None)) or True)
    monkeypatch.setattr(campaign.host, "wait_idle", lambda: events.append(("idle", None)))
    monkeypatch.setattr(campaign, "output_path", lambda _source: root)
    monkeypatch.setattr(campaign.files, "file_bytes", lambda *_args, **_kwargs: b"checkpoint")
    monkeypatch.setattr(campaign.probe, "seed_reset", lambda: events.append(("seed", None)))
    def restored(raw, identity):
        actor = object(); actors.append(actor); events.append(("actor", actor)); return actor
    monkeypatch.setattr(campaign.probe, "restored_actor", restored)
    runtime_module = __import__("mjlab_microduck.stance_recovery_runtime", fromlist=["RecoveryRuntime"])
    class FakeRuntime:
        def __init__(self, selected, *, device):
            self.cases = tuple(selected); self.device = device
            cases.append(self.cases[0]); envs.append(self); events.append(("env", self.cases[0]))
    monkeypatch.setattr(runtime_module, "RecoveryRuntime", FakeRuntime)
    def collect(env, actor, declaration, identity, *, deadline_monotonic):
        assert actor is actors[-1] and env is envs[-1]
        assert env.cases == (contract.CASE_NAMES[len(cases)-1],) and env.device == "cuda:0"
        events.append(("collect", env.cases[0], deadline_monotonic))
        count = contract.POLICY_TICKS-1 if partial and len(cases) == 1 else contract.POLICY_TICKS
        return {"backend": {"torch_device": "cuda:0", "warp_is_cuda": True},
            "collection": {"policy_ticks": count, "elapsed_seconds": 10.,
                           "stop_reason": "policy-tick-limit"}}
    monkeypatch.setattr(campaign.evidence, "collect", collect)
    monkeypatch.setattr(campaign.evidence, "encode", lambda value: b"capture-"+value["collection"]["policy_ticks"].to_bytes(2, "little")+bytes([len(cases)]))
    original_writer = campaign.probe.write_capture
    def write_capture(path, raw):
        events.append(("capture", Path(path).name, raw))
        return original_writer(path, raw)
    monkeypatch.setattr(campaign.probe, "write_capture", write_capture)
    def write_json(path, value):
        events.append(("record", Path(path).name, value["case"]))
        with Path(path).open("x") as stream: json.dump(value, stream, sort_keys=True)
    monkeypatch.setattr(campaign.files, "write_json", write_json)
    gc_events = []
    monkeypatch.setattr(campaign.gc, "collect", lambda: (gc_events.append(len(cases)), events.append(("gc", len(cases))))[0])
    monkeypatch.setattr(campaign, "CASE_SECONDS", 250)
    return root, launch, events, actors, envs, cases, gc_events, started


def test_child_runs_ordered_fresh_case_lifetimes_and_exclusive_retained_outputs(monkeypatch, tmp_path):
    root, launch, events, actors, envs, cases, gc_events, started = _child_seams(monkeypatch, tmp_path)
    campaign.child(SOURCE, LAUNCH_SHA, 73, started)
    assert events.index(("lease", 73)) < events.index(("checked", None)) < events.index(("cuda-query", None))
    assert cases == list(contract.CASE_NAMES)
    assert len({id(actor) for actor in actors}) == len({id(env) for env in envs}) == 5
    assert gc_events == [1, 2, 3, 4, 5]
    assert len(list(root.glob("case-*.pt"))) == 5 and len(list(root.glob("case-*.json"))) == 5
    for index, case in enumerate(contract.CASE_NAMES):
        record = json.loads((root/f"case-{index}.json").read_text())
        raw = (root/f"case-{index}.pt").read_bytes()
        assert record["case"] == case and record["index"] == index
        assert record["capture_sha256"] == real_sha256(raw).hexdigest()
        assert record["capture_bytes"] == len(raw) and record["optimizer_steps"] == 0
        assert all(record[key] is False for key in contract.FALSE_FLAGS)
        with pytest.raises(FileExistsError):
            campaign.probe.write_capture(root/f"case-{index}.pt", b"overwrite")
    for completed_count in range(1, 5):
        previous_gc = events.index(("gc", completed_count))
        next_seed = events.index(("seed", None), previous_gc+1)
        assert previous_gc < next_seed
    for index in range(5):
        capture_at = next(i for i, event in enumerate(events)
                          if event[0] == "capture" and event[1] == f"case-{index}.pt")
        record_at = events.index(("record", f"case-{index}.json", contract.CASE_NAMES[index]))
        gc_at = events.index(("gc", index+1))
        collect_at = next(i for i, event in enumerate(events)
                          if i < capture_at and event[0] == "collect" and event[1] == contract.CASE_NAMES[index])
        assert collect_at < capture_at < record_at < gc_at


def test_child_retains_partial_case_once_then_stops_without_retry(monkeypatch, tmp_path):
    root, _launch, events, _actors, _envs, cases, gc_events, started = _child_seams(
        monkeypatch, tmp_path, partial=True)
    with pytest.raises(ValueError, match="partial or early-failed matrix case retained"):
        campaign.child(SOURCE, LAUNCH_SHA, 73, started)
    assert cases == [contract.CASE_NAMES[0]] and gc_events == [1]
    assert (root/"case-0.pt").exists() and (root/"case-0.json").exists()
    assert not (root/"case-1.pt").exists() and not (root/"case-1.json").exists()
    assert len([event for event in events if event[0] == "seed"]) == 1


@pytest.mark.parametrize("failed_index", [0, 1])
def test_child_exception_retains_honest_failure_receipt_and_prior_cases(monkeypatch, tmp_path, failed_index):
    root, _launch, _events, _actors, _envs, cases, _gc, started = _child_seams(monkeypatch, tmp_path)
    original = campaign.evidence.collect
    def raised(env, *args, **kwargs):
        if env.cases[0] == contract.CASE_NAMES[failed_index]:
            raise ValueError("synthetic mid-case failure")
        return original(env, *args, **kwargs)
    monkeypatch.setattr(campaign.evidence, "collect", raised)
    with pytest.raises(ValueError, match="synthetic mid-case failure"):
        campaign.child(SOURCE, LAUNCH_SHA, 73, started)
    assert cases == list(contract.CASE_NAMES[:failed_index+1])
    receipt = json.loads((root/f"case-{failed_index}-failure.json").read_text())
    assert receipt["stage"] == "collection" and receipt["error_type"] == "ValueError"
    assert receipt["current_case_trace_retained"] is False
    assert all(receipt[key] is False for key in contract.FALSE_FLAGS)
    assert not (root/f"case-{failed_index}.pt").exists()
    for index in range(failed_index):
        assert (root/f"case-{index}.pt").exists() and (root/f"case-{index}.json").exists()


def test_replay_cases_authenticates_each_case_before_matrix_checker_and_compares(monkeypatch, tmp_path):
    root = tmp_path / "matrix"; root.mkdir()
    prefix_root = tmp_path / "prefix"; prefix_root.mkdir()
    cp = b"checkpoint"; (prefix_root/"checkpoint.pt").write_bytes(cp)
    launch = {"prefix_launch": {"declaration": DECLARATION}}
    events, expected = [], []
    for index, case in enumerate(contract.CASE_NAMES):
        raw = f"authenticated-case-{index}".encode()
        collection = {"stop_reason": "policy-tick-limit", "policy_ticks": contract.POLICY_TICKS,
                      "elapsed_seconds": 10.}
        record = dict(protocol=matrix.PROTOCOL, source=SOURCE, case=case, index=index,
            launch_sha256=LAUNCH_SHA, capture_sha256=real_sha256(raw).hexdigest(), capture_bytes=len(raw),
            backend={"torch_device": "cuda:0", "warp_is_cuda": True}, optimizer_steps=0,
            collection=collection, construction_seconds=1., serialization_seconds=1., case_elapsed_seconds=20.,
            **contract.FALSE_FLAGS)
        (root/f"case-{index}.pt").write_bytes(raw)
        (root/f"case-{index}.json").write_text(json.dumps(record))
        expected.append((raw, record))
    monkeypatch.setattr(campaign, "output_path", lambda _source: root)
    monkeypatch.setattr(campaign.probe, "output_path", lambda _source: prefix_root)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(campaign.torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(campaign.files, "file_bytes", lambda path, **_kwargs: (
        events.append(("read", Path(path).name)), Path(path).read_bytes())[1])
    monkeypatch.setattr(campaign.files, "parse", lambda raw: (events.append(("parse", None)), json.loads(raw))[1])
    monkeypatch.setattr(campaign.files, "write_json", lambda path, value: (
        events.append(("write", Path(path).name)), Path(path).write_text(json.dumps(value, sort_keys=True)))[1])
    def checked(raw, digest, checkpoint, declaration):
        events.append(("checked", raw, digest, checkpoint, declaration))
        receipt_value = valid_receipt(contract.CASE_NAMES[len([x for x in events if x[0] == "checked"])-1])
        receipt_value["score"]["collection"] = {
            "stop_reason": "policy-tick-limit", "policy_ticks": contract.POLICY_TICKS,
            "elapsed_seconds": 10.}
        return receipt_value
    monkeypatch.setattr(campaign.matrix, "checked_case", checked)
    monkeypatch.setattr(campaign.matrix, "compare", lambda receipts: (
        events.append(("compare", [r["case"] for r in receipts])), {"decision": "synthetic"})[1])

    result = campaign.replay_cases(SOURCE, launch, LAUNCH_SHA)
    checked_events = [event for event in events if event[0] == "checked"]
    assert [event[1] for event in checked_events] == [item[0] for item in expected]
    assert [event[2] for event in checked_events] == [item[1]["capture_sha256"] for item in expected]
    assert all(event[3:] == (cp, DECLARATION) for event in checked_events)
    assert [event[1] for event in events if event[0] == "compare"][0] == list(contract.CASE_NAMES)
    assert result == {"decision": "synthetic"}
    for index in range(5): assert (root/f"case-{index}-replay.json").exists()
    assert (root/"comparison.json").exists()


def test_replay_cases_requires_cpu_hidden_before_read_or_matrix_tensor_check(monkeypatch, tmp_path):
    events = []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(campaign.torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(campaign, "output_path", lambda _source: tmp_path)
    monkeypatch.setattr(campaign.files, "file_bytes", lambda *_args, **_kwargs: events.append("read"))
    monkeypatch.setattr(campaign.matrix, "checked_case", lambda *_args: pytest.fail("visible CUDA reached matrix score"))
    with pytest.raises(ValueError, match="independent CPU-only matrix scorer"):
        campaign.replay_cases(SOURCE, {"prefix_launch": {}}, LAUNCH_SHA)
    assert events == []


def _supervisor_setup(monkeypatch, tmp_path, *, bad_property=False):
    root = tmp_path / "matrix"; root.mkdir(); (root/"launch.json").write_text("launch")
    props = dict(MainPID=str(os.getpid()), RuntimeMaxUSec="26min", KillMode="control-group",
        ActiveState="active", MemoryMax=str(2*1024**3), CPUQuotaPerSecUSec="2s", Nice="10")
    if bad_property: props["MemoryMax"] = "1G"
    preserved = {"filmbrain": "unchanged"}; launch = {"prefix_launch": {"preserved_filmbrain": preserved}}
    events, reports = [], []
    times = iter((100., 101., 102.))
    monkeypatch.setattr(campaign.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(campaign, "output_path", lambda _source: root)
    monkeypatch.setattr(campaign, "service_name", lambda _source: "matrix.service")
    monkeypatch.setattr(campaign.host, "read", lambda *args: props[args[-2]] if "systemctl" in args
        else SOURCE if args[:2] == ("git", "rev-parse") else "")
    monkeypatch.setattr(campaign.host, "check_log", lambda *_args: events.append(("log", None)))
    monkeypatch.setattr(campaign, "check_window", lambda **kwargs: events.append(("window", kwargs)))
    monkeypatch.setattr(campaign, "checked", lambda *_args: (events.append(("checked", None)), launch)[1])
    monkeypatch.setattr(campaign.probe.d0, "filmbrain_state", lambda: (events.append(("filmbrain", None)), preserved)[1])
    monkeypatch.setattr(campaign.torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(campaign.host, "wait_idle", lambda: (events.append(("idle", None)), {"idle": True})[1])
    class Lease:
        def __enter__(self): events.append(("lease", None)); return 73
        def __exit__(self, *_args): return False
    monkeypatch.setattr(campaign.files, "gpu_lease", lambda: Lease())
    monkeypatch.setattr(campaign.files, "child_environment", lambda: {"CUDA_VISIBLE_DEVICES": "0"})
    monkeypatch.setattr(campaign.files, "write_json", lambda _path, value: reports.append(deepcopy(value)))
    return root, events, reports, launch, preserved


def test_supervisor_checks_exact_caps_then_lease_and_live_guard_and_retains_failure(monkeypatch, tmp_path):
    root, events, reports, _launch, _preserved = _supervisor_setup(monkeypatch, tmp_path)
    def child(_command, log, **kwargs):
        assert kwargs["lock_fd"] == 73
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "0"
        assert kwargs["env"]["OMP_NUM_THREADS"] == "1"
        assert "timeout" not in kwargs
        assert campaign.CHILD_SECONDS == campaign.files.LEAN_EVALUATION_CHILD_SECONDS == 1352
        assert events.index(("checked", None)) < events.index(("lease", None))
        kwargs["guard"]()
        log.write_text("synthetic child failure\n")
        raise RuntimeError("synthetic matrix failure")
    monkeypatch.setattr(campaign.files, "supervised_lean_evaluation", child)
    with pytest.raises(RuntimeError, match="synthetic matrix failure"):
        campaign.supervise(SOURCE, LAUNCH_SHA)
    assert events.index(("lease", None)) < events.index(("log", None)) < events.index(("filmbrain", None))
    report = reports[-1]
    assert report["decision"] == "frozen-five-case-diagnostic-failed"
    assert report["error"] == "synthetic matrix failure" and "comparison" not in report
    assert report["files"] == {p.name: real_sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}
    assert all(report[key] is False for key in contract.FALSE_FLAGS)


def test_supervisor_refuses_service_cap_drift_before_gpu_lease(monkeypatch, tmp_path):
    _root, events, reports, _launch, _preserved = _supervisor_setup(
        monkeypatch, tmp_path, bad_property=True)
    monkeypatch.setattr(campaign.files, "gpu_lease", lambda: pytest.fail("service drift acquired lease"))
    with pytest.raises(ValueError, match="independently capped matrix service"):
        campaign.supervise(SOURCE, LAUNCH_SHA)
    assert not any(event[0] == "lease" for event in events)
    assert reports[-1]["decision"] == "frozen-five-case-diagnostic-failed"
    assert reports[-1]["error_type"] == "ValueError"
