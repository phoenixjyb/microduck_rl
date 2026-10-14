"""Evidence authentication refuses changed or ambiguous diagnostic inputs."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("cycle_cli",Path(__file__).parents[1]/"scripts/diagnose_cycle_heading.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def test_bound_reader_checks_exact_bytes_owner_and_symlink(tmp_path,monkeypatch):
    p=tmp_path/"input.json";p.write_text("{}")
    sha=hashlib.sha256(p.read_bytes()).hexdigest()
    assert cli.read_bound(p,sha)=={}
    p.write_text('{"changed":true}')
    with pytest.raises(ValueError,match="hash mismatch"):cli.read_bound(p,sha)
    link=tmp_path/"link";link.symlink_to(p)
    with pytest.raises(ValueError,match="non-symlink"):cli.read_bound(link,sha)
    monkeypatch.setattr(cli.os,"getuid",lambda:-1)
    with pytest.raises(ValueError,match="owned regular"):cli.read_bound(p,sha)


def test_isolated_core_executes_authenticated_bytes_only(tmp_path):
    p=tmp_path/"core.py";p.write_text('PROTOCOL="fixture"\ndef summarize_case(a,b):return {"readonly":True}\n')
    sha=hashlib.sha256(p.read_bytes()).hexdigest()
    core=cli.load_core(p,sha)
    assert core.authenticated_source_sha256 == sha
    assert core.PROTOCOL=="fixture"
    assert core.summarize_case({}, {})=={"readonly":True}
    p.write_text('raise AssertionError("not authenticated")\n')
    assert core.authenticated_source_sha256 == sha  # binds executed bytes, not a later reread
    with pytest.raises(ValueError,match="hash mismatch"):cli.load_core(p,sha)
    with pytest.raises(ValueError,match="paired"):cli.load_core(p,None)


def test_reference_arms_are_fixed_unqualified_checkpoints():
    ref=cli.REFERENCE
    assert set(ref["checkpoints"])=={"control","headroom"}
    assert len(ref["source"])==40
    assert all(len(v)==64 for v in (ref["summary"],ref["manifest"],*ref["checkpoints"].values()))


def test_group_statistics_are_cycle_weighted_and_empty_idle_is_not_a_gait():
    from mjlab_microduck import cycle_heading_diagnosis
    from test_gait_path_qualification import make_case_trace
    case, trace = make_case_trace(speed=.2, yaw=.2)
    diagnostic = cycle_heading_diagnosis.summarize_case(case, trace)
    groups = cli.group_cases([diagnostic])
    assert groups[0]["cycle_count"] == 0
    assert groups[0]["cycle_weighted_statistics"]["duration_s"]["mean"] is None
    moving = groups[-1]
    assert sum(moving["phase_exposure_s"].values()) == pytest.approx(
        sum(c["duration_s"] for w in diagnostic["worlds"] for c in w["cycles"]))
    assert sum(moving["peak_only_phase_counts"].values()) == moving["peak_only"] == 0
    assert moving["cycle_weighted_statistics"]["duration_s"]["mean"] == pytest.approx(.8)
    cycle = diagnostic["worlds"][0]["cycles"][0]
    cycle.update(peak_only=True, max_fail=True)
    moving = cli.group_cases([diagnostic])[-1]
    phase = cycle["peak_interval_phase"]
    assert moving["peak_only_phase_counts"][phase] == 1
    assert moving["peak_only_events_per_phase_second"][phase] == pytest.approx(
        1./moving["phase_exposure_s"][phase])


def test_packet_authenticates_all_fixed_raw_cases_before_diagnosis(tmp_path, monkeypatch):
    from types import SimpleNamespace
    ref = {**cli.REFERENCE}
    packet = tmp_path/"artifacts/training"/ref["run"]
    state = tmp_path/"state"/(ref["run"]+"-handoff")
    state.mkdir(parents=True)
    summary = dict(source=ref["source"], status="verified-evidence-not-policy-acceptance",
                   policy_acceptance=False, physical_motion_authorized=False, evaluations={})
    manifest = dict(source=ref["source"], files={})

    def seal(path, value):
        payload = json.dumps(value).encode()
        path.write_bytes(payload)
        return hashlib.sha256(payload).hexdigest(), len(payload)

    for arm, checkpoint in ref["checkpoints"].items():
        stage = packet/arm/"final-eval"
        stage.mkdir(parents=True)
        files = []
        for index, (seed, speed, yaw) in enumerate(cli.route.IDENTITIES):
            name = f"s{seed}-c{index%4}.json"
            digest, size = seal(stage/name, dict(seed=seed, speed=speed, yaw=yaw,
                checkpoint_sha256=checkpoint, complete=True, terminal_worlds=[], gait_path_trace={}))
            manifest["files"][f"{arm}/final-eval/{name}"] = dict(sha256=digest, bytes=size)
            files.append(dict(file=name, sha256=digest))
        result = dict(source=ref["source"], arm=arm, policy_acceptance=False,
                      physical_motion_authorized=False, evaluation=dict(checkpoint_sha256=checkpoint,
                      raw_case_files=files, route_gate={"unchanged":True}, legacy_gate={"unchanged":True}))
        digest, _ = seal(stage/"result.json", result)
        summary["evaluations"][arm] = dict(checkpoint_sha256=checkpoint, result_sha256=digest)
    ref["summary"], _ = seal(state/"headroom-summary.json", summary)
    ref["manifest"], _ = seal(state/"headroom-manifest.json", manifest)
    monkeypatch.setattr(cli,"REFERENCE",ref)
    calls = []
    def diagnose(row, trace):
        calls.append(row)
        return dict(seed=row["seed"], speed=row["speed"], yaw=row["yaw"], worlds=[])
    core = SimpleNamespace(summarize_case=diagnose)
    arms = cli.diagnose_packet(tmp_path,tmp_path/"state",core)
    assert len(calls)==24 and len(arms["control"]["cases"])==12
    assert arms["control"]["original_route_gate"]=={"unchanged":True}
    calls.clear()
    (packet/"control/final-eval/s839-c0.json").write_text("{}")
    with pytest.raises(ValueError,match="hash mismatch"):
        cli.diagnose_packet(tmp_path,tmp_path/"state",core)
    assert calls==[]
