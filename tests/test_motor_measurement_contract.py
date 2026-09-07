"""CPU measurement scope checks, not simulation acceptance."""

import copy

import pytest

from mjlab_microduck import motor_measurement_contract as m
from mjlab_microduck import foundation_replay_control as replay
from mjlab_microduck.motor_trace_audit import pack_motor_trace
from mjlab_microduck.speed_response_control import summarize
from test_speed_response_control import inputs
from test_motor_trace_audit import fixture
from test_foundation_motor_experiment import good

SOURCE = "a"*40


def report(case, force=.1):
    data=inputs(); data["pre_force"].fill_(force)
    r=summarize(**data)
    template=good()
    r.update(source=SOURCE, protocol=m.exp.PROTOCOL, seed=503,
        checkpoint_sha256=m.CHECKPOINTS[case.split("-")[0]], measurement_identity=m.identity(case),
        route_trace=fixture()["route_trace"], command_trace=template["command_trace"],
        command_adapter=template["command_adapter"],
        motor_trace=pack_motor_trace(data["pre_force"],data["pre_speed"]))
    return r


def records():
    state=replay.fingerprint(); state["environment"]=dict(m.EXPECTED_ENV)
    state["cuda_initialized"]=False
    after=copy.deepcopy(state); after["cuda_initialized"]=True
    return [dict(case=case,report=report(case),fingerprint=dict(before=copy.deepcopy(state),after=copy.deepcopy(after)))
            for case in m.ORDER]


def test_reconstructs_own_motor_and_route_data_without_mutation_or_promotion():
    r=report("parent-1"); before=copy.deepcopy(r)
    result=m.measure_case(r,source=SOURCE,case="parent-1")
    assert result["raw_motor_summary_verified"] and r==before
    assert result["analysis"]["joints"] and all(result[k] is False for k in m.CLAIMS)


@pytest.mark.parametrize("mutation", ["source","model","case","seed","causal","historical","admission",
    "command","route","summary","force","timing","missing_joint","unsafe","pre_torque"])
def test_corrupt_scope_identity_commands_raw_metrics_or_unsafe_case_refused(mutation):
    r=report("parent-1",.37 if mutation=="pre_torque" else .1)
    if mutation=="source":r["source"]="b"*40
    if mutation=="model":r["checkpoint_sha256"]=m.CHECKPOINTS["motor"]
    if mutation=="case":r["measurement_identity"]=m.identity("parent-2")
    if mutation=="seed":r["seed"]=509
    if mutation=="causal":r["measurement_identity"]["causal_effect_established"]=True
    if mutation=="historical":r["measurement_identity"]["historical_replay_identity"]=True
    if mutation=="admission":r["training_admitted"]=True
    if mutation=="command":r["command_trace"]["actor_input"][0][0][0]=.5
    if mutation=="route":r["route_trace"]["velocity"][0][0][1]=.2
    if mutation=="summary":r["groups"]["settled"]["pre_reset_squared_utilization_mean"]+=.00001
    if mutation=="force":r["motor_trace"]["force_nm"][0][0][0]=float("nan")
    if mutation=="timing":r["motor_trace"]["timing"]["force"]="simultaneous"
    if mutation=="missing_joint":del r["groups"]["settled"]["pre_reset_joint_p99"][m.JOINTS[0]]
    if mutation=="unsafe":r.update(safety_failures=["all-legacy-torque"],classification="safety-or-coverage-stop")
    with pytest.raises(ValueError):m.measure_case(r,source=SOURCE,case="parent-1")


@pytest.mark.parametrize("reference,candidate,direction", [([1.,2.],[3.,4.],"higher"),
    ([3.,4.],[1.,2.],"lower"),([1.,3.],[2.,4.],"overlapping-or-equal"),([1.,2.],[2.,3.],"overlapping-or-equal")])
def test_observed_ranges_retain_every_replicate_and_not_confidence_intervals(reference,candidate,direction):
    r=m.range_contrast(reference,candidate)
    assert r["reference"]["replicates"]==reference and r["candidate"]["replicates"]==candidate
    assert r["all_pair_deltas"]==[y-x for x in reference for y in candidate]
    assert r["observed_direction"]==direction and not r["confidence_interval"]


def test_six_case_description_keeps_failed_original_gates_and_single_seed_limitation():
    rows=records(); rows[2]["report"]=report("motor-1",.12); rows[3]["report"]=report("motor-2",.13)
    before=copy.deepcopy(rows)
    d=m.describe_dataset(rows,source=SOURCE)
    assert d["decision"]=="descriptive-replicates-only" and rows==before
    assert d["distinct_physics_seeds"]==1 and d["process_replicates_per_arm"]==2
    assert all(d[k] is False for k in m.CLAIMS)
    assert d["unchanged_gate_readouts"]["1"]["failures"] and d["unchanged_gate_readouts"]["2"]["failures"]
    assert d["groups"]["settled"]["pre_reset_squared_utilization_mean"]["control"]["observed_direction"]=="higher"


@pytest.mark.parametrize("mutation", ["omit","reorder","duplicate","settings","hash","entry_cuda","identity","missing_settings"])
def test_dataset_refuses_selection_missing_replicate_or_changed_conditions(mutation):
    rows=records()
    if mutation=="omit":rows.pop()
    if mutation=="reorder":rows.reverse()
    if mutation=="duplicate":rows[2]=copy.deepcopy(rows[0])
    if mutation=="settings":rows[1]["fingerprint"]["after"]["threads"]=99
    if mutation=="hash":rows[1]["fingerprint"]["before"]["environment"]["PYTHONHASHSEED"]=None
    if mutation=="entry_cuda":rows[0]["fingerprint"]["before"]["cuda_initialized"]=True
    if mutation=="identity":rows[1]["report"]["source"]="b"*40
    if mutation=="missing_settings":
        for row in rows:
            for phase in ("before","after"): del row["fingerprint"][phase]["cudnn_enabled"]
    with pytest.raises(ValueError):m.describe_dataset(rows,source=SOURCE)
