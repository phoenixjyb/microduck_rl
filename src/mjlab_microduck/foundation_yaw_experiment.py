"""F1-Y: one predeclared yaw-command contrast, fixed selection and old gates."""

import argparse
import datetime as dt
import functools
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import torch

from mjlab_microduck import foundation_motor_experiment as base
from mjlab_microduck import foundation_motor_replicates as history
from mjlab_microduck.first_attempt_smoke import canonical, require, sha256
from mjlab_microduck.gpu_idle_gate import wait_idle
from mjlab_microduck.motor_trace_audit import audit_motor_trace
from mjlab_microduck.neck_reward_observer import NeckRewardObserver, TERM
from mjlab_microduck.yaw_command_observer import YawCommandObserver, validate_command_evidence
from mjlab_microduck.recovery_ab import verify_source, write_new

PROTOCOL = "f1y-yaw-support-paired-s499-v1"
OUTPUT = base.OUTPUT.parent / PROTOCOL
ARMS, WEIGHTS = ("control","yaw"), dict(control=-.1,yaw=-.1)
YAW_RANGES = dict(control=(0.,0.),yaw=(-.35,.35))
SEEDS, MODES = base.SEEDS, base.MODES
CAMPAIGN_SECONDS = 3600
REWARD_MANAGER_SHA = "8d1679cc4b0581dc0000493e0a661f3976c5f9c54f800bd825aee340a685f2a7"
DESCRIPTIVE_MANIFEST = "44d43f73564e73e67d4e5a456487c4129eff44cd20b128ca051ab3b211f1795e"
DESCRIPTIVE_DECISION = "be1b7a45a94e20d07a0ccc00984bff5745aad82b3f8b5a8aa2c6ab59b7851960"


def prepare_config(mode, *, arm):
    require(arm in ARMS and mode in MODES,"predeclared yaw arm/mode")
    cfg,agent=base.prepare_config(mode,arm="motor")
    cfg.rewards[TERM].weight=WEIGHTS[arm]
    cfg.commands["twist"].ranges.ang_vel_z=YAW_RANGES[arm]
    agent.experiment_name,agent.run_name=PROTOCOL,f"{arm}-{mode}"
    return cfg,agent


def verify_history():
    import mjlab
    require(sha256(Path(mjlab.__file__).parent/"managers/reward_manager.py")==REWARD_MANAGER_SHA,
            "inspected consumed-reward buffer semantics")
    history.verify_history()
    from mjlab_microduck.foundation_command_coverage import audit_retained
    from mjlab_microduck import foundation_neck_experiment as closed
    import inspect
    from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand
    from rsl_rl.algorithms.ppo import PPO
    require(sha256(Path(inspect.getfile(UniformVelocityCommand)))=="b4c60a2c061946fbfafb324e4c6d3d035411a34d2b3b22a0202ea8ab2d20acaf", "inspected original uniform sampler")
    require(sha256(Path(inspect.getfile(PPO)))=="a2d35e7ad7b884c80b7434e7d2ce785a6da1e18d93c96f1179fbd3a208669f8c", "inspected PPO action boundary")
    coverage=audit_retained(closed.OUTPUT)
    p=history.OUTPUT
    require(sha256(p/"manifest.json")==DESCRIPTIVE_MANIFEST and sha256(p/"decision.json")==DESCRIPTIVE_DECISION,
            "closed descriptive evidence identity")
    files=json.loads((p/"manifest.json").read_text())["files"]
    require(len(files)==32,"six-case descriptive evidence")
    for name,row in files.items():
        f=p/name
        require(f.resolve().is_relative_to(p.resolve()) and f.is_file()
                and f.stat().st_size==row["bytes"] and sha256(f)==row["sha256"],"unchanged descriptive payload")
    require(set(files)=={str(f.relative_to(p)) for f in p.rglob("*") if f.is_file() and f.name!="manifest.json"},
            "exact descriptive coverage")
    return dict(descriptive_manifest=DESCRIPTIVE_MANIFEST,descriptive_decision=DESCRIPTIVE_DECISION,
                reward_manager_sha256=REWARD_MANAGER_SHA,checkpoints=base.preserved(),
                closed_neck_manifest=coverage["manifest_sha256"],closed_neck_decision=coverage["decision_sha256"],
                yaw_sampler_sha256="b4c60a2c061946fbfafb324e4c6d3d035411a34d2b3b22a0202ea8ab2d20acaf",
                ppo_sha256="a2d35e7ad7b884c80b7434e7d2ce785a6da1e18d93c96f1179fbd3a208669f8c")


def validate_training(mode, arm, source):
    p=OUTPUT/f"{arm}-{mode}"; r=json.loads((p/"result.json").read_text())
    updates=MODES[mode][0]
    require(r["source"]==source and r["status"]=="training-complete-not-accepted" and r["updates"]==updates
            and r["parent_step"]==204000 and r["common_step"]==204000+24*updates,"complete fixed training counters")
    final=p/f"model_{8498+updates}.pt"
    require(r["final_checkpoint"]==str(final) and sha256(final)==r["final_sha256"],"fixed final checkpoint, no selection")
    labels={i for i in range(8499,8499+updates) if i%50==0}|{8498+updates}
    require({f.name for f in p.glob("model_*.pt")}=={f"model_{i}.pt" for i in labels}
            and not list(p.glob("*.pending")),"complete fixed checkpoint cadence")
    for label in sorted(labels):
        payload=torch.load(p/f"model_{label}.pt",map_location="cpu",weights_only=False)
        require(payload["iter"]==label and payload["infos"]["env_state"]["common_step_counter"]
                ==204000+24*(label-8498),"checkpoint curriculum counter")
        def finite(value):
            if isinstance(value,torch.Tensor): return bool(torch.isfinite(value).all())
            if isinstance(value,dict): return all(finite(v) for v in value.values())
            if isinstance(value,(list,tuple)): return all(finite(v) for v in value)
            if isinstance(value,float): return math.isfinite(value)
            return True
        require(all(k in payload for k in ("actor_state_dict","critic_state_dict","optimizer_state_dict"))
                and finite(payload),"finite complete learned checkpoint state")
    activity=r["reward_observer"]
    require(activity["protocol"]=="f1n-live-neck-reward-v1" and activity["control_steps"]==24*updates
            and activity["neck_weight"]==WEIGHTS[arm] and activity["positive_samples"]>0
            and activity["raw_neck_cost_mean"]>0,"complete active neck reward evidence")
    rows=[json.loads(line) for line in (p/"neck-reward-activity.jsonl").read_text().splitlines()]
    canonical([r,rows])
    require(len(rows)==updates,"every update has live reward evidence")
    for i,row in enumerate(rows,1):
        require(row["control_steps"]==24*i and row["common_step"]==204000+24*i
                and row["neck_weight"]==WEIGHTS[arm] and row["motor_weight"]==-4.
                and row["lateral_weight"]==-.5 and row["samples"]==24*256,"all live weights/counters unchanged")
    validate_command_evidence(p, updates, YAW_RANGES[arm])
    return r


def checkpoint(arm,source):
    if arm=="parent":return base.checkpoint("parent",source)
    require(arm in ARMS,"fixed checkpoint arm")
    r=validate_training("pilot",arm,source)
    return Path(r["final_checkpoint"]),r["final_sha256"]


def paired_decision(parent,control,yaw):
    return base.lateral.paired_decision(parent,control,yaw,protocol=PROTOCOL,seeds=SEEDS)


def child(kind,arm,seed,source):
    verify_source(source);verify_history();base.training.runtime_identity();base.check_host()
    before=history.replay.fingerprint()
    expected_seed=(491 if kind=="smoke" else 499) if kind in MODES else seed
    require(before["environment"]["PYTHONHASHSEED"]==str(expected_seed),"fixed child-start hash seed")
    if kind in MODES:
        require(arm in ARMS,"training arm")
        p=OUTPUT/f"{arm}-{kind}";p.mkdir(exist_ok=False)
        write_new(p/"launch.json",dict(source=source,protocol=PROTOCOL,arm=arm,mode=kind,
            parent_sha256=base.MODEL_SHA,parent_iteration=8498,parent_step=204000,
            neck_weight=WEIGHTS[arm],yaw_range=YAW_RANGES[arm],motor_weight=-4.,lateral_weight=-.5,seed=expected_seed))
        observer=NeckRewardObserver(WEIGHTS[arm],p/"neck-reward-activity.jsonl")
        commands=YawCommandObserver(YAW_RANGES[arm],p/"command-activity.jsonl")
        try:
            result=base.training.train(kind,p,config_factory=functools.partial(prepare_config,arm=arm),
                protocol=PROTOCOL,parent=base.MODEL,parent_iteration=8498,parent_step=204000,
                max_seconds=MODES[kind][1],stop_at=min(base.DEADLINE,dt.datetime.now(dt.timezone.utc)+dt.timedelta(seconds=MODES[kind][1])),
                reward_observer=observer,command_validator=commands)
        except Exception as error:
            write_new(p/"failure.json",dict(source=source,error=f"{type(error).__name__}: {error}"));raise
        write_new(p/"result.json",{**result,"source":source,"policy_acceptance":False})
        write_new(p/"fingerprint.json",dict(before=before,after=history.replay.fingerprint()))
    else:
        require(kind=="evaluate" and seed in SEEDS,"predeclared evaluation seed")
        path,expected=checkpoint(arm,source)
        report=base.run_control(checkpoint=path,seed=seed,protocol=PROTOCOL,retain_route_trace=True,
                               command_adapter=base.HeadingHold(True),retain_motor_trace=True)
        report.update(source=source,retained_checkpoint_hashes=base.preserved())
        write_new(OUTPUT/f"{arm}-s{seed}.json",report)
        write_new(OUTPUT/f"{arm}-s{seed}-fingerprint.json",dict(before=before,after=history.replay.fingerprint()))
        base.validate_controller_trace(report,True,protocol=PROTOCOL,seeds=SEEDS,checkpoint_sha=expected)
        write_new(OUTPUT/f"{arm}-s{seed}-motor-analysis.json",audit_motor_trace(report))


def run_child(kind,arm,source,end,*,seed=SEEDS[0]):
    require(kind in (*MODES,"evaluate") and arm in (*ARMS,"parent"),"declared child")
    seconds=MODES[kind][1] if kind in MODES else 120
    require(time.monotonic()+seconds+60<end and dt.datetime.now(dt.timezone.utc)+dt.timedelta(seconds=seconds+60)<base.DEADLINE,
            "child and closeout budget")
    name=f"{arm}-{kind}" if kind in MODES else f"{arm}-s{seed}"
    write_new(OUTPUT/f"{name}-idle.json",wait_idle())
    env=history.replay.child_environment()
    env["PYTHONHASHSEED"]=str((491 if kind=="smoke" else 499) if kind in MODES else seed)
    with (OUTPUT/f"{name}.log").open("x") as log:
        result=subprocess.run([sys.executable,"-m","mjlab_microduck.foundation_yaw_experiment",
            "--source",source,"--child",kind,"--arm",arm,"--seed",str(seed)],
            stdout=log,stderr=subprocess.STDOUT,timeout=seconds,env=env)
    require(result.returncode==0,name+" child failed")


def campaign(source):
    verify_source(source)
    now=dt.datetime.now(dt.timezone.utc)
    require(now<base.LAST_START and now+dt.timedelta(seconds=CAMPAIGN_SECONDS+90)<base.DEADLINE,"full F1-Y budget before07:00")
    retained=verify_history();runtime=base.training.runtime_identity()
    OUTPUT.mkdir(parents=True,exist_ok=False);end=time.monotonic()+CAMPAIGN_SECONDS
    rows,pairs=[],[]
    decision=dict(protocol=PROTOCOL,source=source,decision="runtime-failure-stop",reports=rows,pairs=pairs,
        pilot_slot=3,policy_acceptance=False,physical_motion_authorized=False,obstacles_admitted=False,
        hop_validated=False,stabilization_retention_validated=False)
    fingerprints={}
    def evaluate(arm,seed):
        run_child("evaluate",arm,source,end,seed=seed)
        p=OUTPUT/f"{arm}-s{seed}.json";r=json.loads(p.read_text());_,expected=checkpoint(arm,source)
        base.validate_controller_trace(r,True,protocol=PROTOCOL,seeds=SEEDS,checkpoint_sha=expected)
        require(r["source"]==source,"fresh evaluation source")
        rows.append(dict(arm=arm,seed=seed,sha256=sha256(p)))
        state=json.loads((OUTPUT/f"{arm}-s{seed}-fingerprint.json").read_text())
        if arm=="parent":fingerprints[seed]=state
        else:require(canonical(state)==canonical(fingerprints[seed]),"matched evaluation process settings")
        for name,g in r["groups"].items():
            require(g["pre_reset_torque_p99"]<=.60,"absolute pre-reset torque: "+name)
        audit_motor_trace(r)
        return r
    try:
        write_new(OUTPUT/"launch.json",dict(protocol=PROTOCOL,source=source,retained=retained,runtime=runtime,
            idle=wait_idle(),neck_weights=WEIGHTS,yaw_ranges=YAW_RANGES,pilot_slot=3,training_seed=499,smoke_seed=491,
            evaluation_seeds=SEEDS,campaign_seconds=CAMPAIGN_SECONDS,deadline=base.DEADLINE.isoformat()))
        parents={}
        for seed in SEEDS:
            parents[seed]=evaluate("parent",seed)
            if parents[seed]["safety_failures"]:
                decision.update(decision="reference-safety-stop",failures=parents[seed]["safety_failures"]);break
        else:
            for mode in MODES:
                for arm in ARMS:
                    if mode=="pilot":
                        smoke=validate_training("smoke",arm,source)
                        require(smoke["wall_seconds"]*50*1.5<870,"measured smoke fits pilot cap")
                    run_child(mode,arm,source,end);validate_training(mode,arm,source)
                require(sha256(OUTPUT/f"control-{mode}/initial.pt")==sha256(OUTPUT/f"yaw-{mode}/initial.pt"),
                        "identical paired restored learned state")
                a=json.loads((OUTPUT/f"control-{mode}/fingerprint.json").read_text())
                b=json.loads((OUTPUT/f"yaw-{mode}/fingerprint.json").read_text())
                require(canonical(a)==canonical(b),"matched training process settings")
            for seed in SEEDS:
                control=evaluate("control",seed)
                if control["safety_failures"]:
                    decision.update(decision="reference-safety-stop",failures=control["safety_failures"]);break
                yaw=evaluate("yaw",seed)
                pair=paired_decision(parents[seed],control,yaw);pairs.append(pair)
                if pair["failures"]:
                    decision.update(decision="numerical-gate-stop",failures=pair["failures"]);break
            else:decision.update(decision="single-seed-diagnostic-support-only",failures=[])
        verify_history()
    except Exception as error:
        decision.update(decision="runtime-failure-stop",error=f"{type(error).__name__}: {error}");raise
    finally:
        write_new(OUTPUT/"decision.json",decision)
        files={str(p.relative_to(OUTPUT)):dict(sha256=sha256(p),bytes=p.stat().st_size)
               for p in sorted(OUTPUT.rglob("*")) if p.is_file()}
        write_new(OUTPUT/"manifest.json",dict(protocol=PROTOCOL,source=source,files=files))
    print(json.dumps(decision),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",required=True)
    parser.add_argument("--child",choices=(*MODES,"evaluate"))
    parser.add_argument("--arm",choices=(*ARMS,"parent"))
    parser.add_argument("--seed",type=int,choices=SEEDS,default=SEEDS[0])
    args=parser.parse_args()
    if args.child:
        launch=json.loads((OUTPUT/"launch.json").read_text())
        require(launch["source"]==args.source and not (OUTPUT/"decision.json").exists(),"open exact F1-Y campaign")
        child(args.child,args.arm,args.seed,args.source)
    else:
        require(args.arm is None,"arm is child-only");campaign(args.source)


if __name__=="__main__":main()
