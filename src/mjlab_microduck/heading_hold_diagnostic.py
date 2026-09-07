"""Frozen-gait heading-hold A/B, not obstacle or hop admission."""

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time

import torch

from mjlab_microduck.command_delivery import PROTOCOL as DELIVERY, prepare_actor_command_input
from mjlab_microduck.first_attempt_smoke import require, sha256, canonical, ACTOR_SHA256
from mjlab_microduck.foundation_evaluation import candidate_failures, validate_speed_classification
from mjlab_microduck.foundation_pilot import runtime_identity
from mjlab_microduck.foundation_trace_audit import audit_trace
from mjlab_microduck.recovery_ab import verify_source, write_new
from mjlab_microduck.rollout_repeatability import ROOT, ACTOR, check_host
from mjlab_microduck.speed_response_control import run_control

PROTOCOL = "heading-hold-frozen-narrow-s443-v1"
OUTPUT = ROOT / "artifacts/evaluations" / PROTOCOL
MODEL = ROOT / "artifacts/experiments/f1r-width-paired-s421-v1/narrow-pilot/model_8498.pt"
MODEL_SHA = "7ed703d6b5b8407da912f51755be8a8e57698340f62d0f5d3a80cf195ec1f80f"
HOP = ROOT / "logs/rsl_rl/hop_k3900_h1t/2026-09-05_15-47-06_h1t-k3900-117c881-s67-6000x256/model_5999.pt"
HOP_SHA = "454bd7db3da50896c2b00cebd72ea7eecf1fee3e9802a6ee03693e8fa858040a"
SEEDS = (443,449,457)
LAST_START = dt.datetime(2026,9,7,4,30,tzinfo=dt.timezone.utc)


class HeadingHold:
    """50Hz proportional yaw reference, clamped and slew limited; no joint access."""

    def __init__(self, enabled):
        require(type(enabled) is bool, "explicit controller mode")
        self.enabled, self.next_step, self.previous = enabled, 0, None

    def provenance(self):
        return dict(protocol=PROTOCOL, enabled=self.enabled, delivery=DELIVERY,
            heading_source="exact simulation heading relative to initial route; no perception model",
            kp=1., yaw_cap_rps=.35, yaw_slew_rps2=1., dt_s=.02, forward_mps=.3,
            joint_authority=False, policy_acceptance=False)

    def command(self, heading, step):
        require(type(step) is int and step == self.next_step and step < 400, "ordered bounded controller step")
        require(heading.ndim == 1 and heading.is_floating_point()
                and bool(torch.isfinite(heading).all()), "finite vector heading")
        previous = torch.zeros_like(heading) if self.previous is None else self.previous
        require(previous.shape == heading.shape and previous.device == heading.device
                and previous.dtype == heading.dtype, "stable controller batch")
        wrapped = torch.atan2(heading.sin(), heading.cos())
        target = (-wrapped).clamp(-.35,.35) if self.enabled else torch.zeros_like(wrapped)
        yaw = previous + (target-previous).clamp(-.02,.02)
        self.previous, self.next_step = yaw.detach().clone(), step+1
        return torch.stack((torch.full_like(yaw,.3), torch.zeros_like(yaw), yaw), -1)

    def prepare(self, observations, env, step, heading):
        target = self.command(heading,step)
        command = env.command_manager.get_command("twist")
        command.copy_(target)
        return prepare_actor_command_input(observations, command, env.observation_manager,
                                           protocol=DELIVERY, step=step)


def validate_controller_trace(report, enabled, *, protocol=PROTOCOL, seeds=SEEDS, checkpoint_sha=MODEL_SHA):
    canonical(report)
    require(report["protocol"] == protocol and report["seed"] in seeds
            and report["checkpoint_sha256"] == checkpoint_sha, "frozen heading diagnostic identity")
    controller = HeadingHold(enabled)
    require(report["command_adapter"] == controller.provenance(), "exact controller constants")
    audit_trace(report)
    trace = report["command_trace"]
    n = report["sample_steps"]
    require(trace["inference_and_simulation_steps"] == n, "executed command count")
    issued = torch.tensor(trace["issued"],dtype=torch.float32)
    consumed = torch.tensor(trace["actor_input"],dtype=torch.float32)
    cached = torch.tensor(trace["cached"],dtype=torch.float32)
    require(issued.shape == consumed.shape == cached.shape == (n,8,3)
            and bool(torch.isfinite(cached).all()), "command trace coverage")
    require(torch.equal(issued,consumed), "issued commands reached raw actor input")
    headings = torch.tensor(report["route_trace"]["velocity"],dtype=torch.float32)[:,:,3]
    expected = torch.stack([controller.command(h,i) for i,h in enumerate(headings)])
    require(bool(torch.allclose(expected,issued,atol=1e-6,rtol=0)), "controller rule reproduced from heading trace")
    validate_speed_classification(report)


def compare(off,on):
    validate_controller_trace(off,False); validate_controller_trace(on,True)
    require(off["seed"] == on["seed"] and not off["safety_failures"], "safe paired reference")
    failures = candidate_failures(on,off,protocol=PROTOCOL,seeds=SEEDS)
    if "settled" in on["groups"]:
        a,b = off["groups"]["settled"], on["groups"]["settled"]
        require(a["pre_reset_joint_p99"].keys() == b["pre_reset_joint_p99"].keys(), "named joint match")
        for joint in a["pre_reset_joint_p99"]:
            if b["pre_reset_joint_p99"][joint] > a["pre_reset_joint_p99"][joint]+.02:
                failures.append("joint-torque-nonregression:"+joint)
        for key in ("pre_reset_squared_utilization_mean","pre_reset_mechanical_abs_power_mean_w"):
            if b[key] > a[key]*1.05: failures.append(key+"-nonregression")
    return dict(failures=failures, policy_acceptance=False,
                hop_validated=False, stabilization_retention_validated=False, obstacles_admitted=False)


def preserved():
    values={"locomotion_parent":sha256(ACTOR),"narrow_diagnostic":sha256(MODEL),"hop_h1t_rejected":sha256(HOP)}
    require(values == dict(locomotion_parent=ACTOR_SHA256,narrow_diagnostic=MODEL_SHA,hop_h1t_rejected=HOP_SHA),
            "retained capability checkpoints unchanged")
    return values


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",required=True)
    parser.add_argument("--child",choices=("off","on"))
    parser.add_argument("--seed",type=int,choices=SEEDS,default=SEEDS[0])
    args=parser.parse_args()
    verify_source(args.source)
    runtime=runtime_identity(); host=check_host(); identities=preserved()
    if args.child:
        launch=json.loads((OUTPUT/"launch.json").read_text())
        require(launch["source"]==args.source and not (OUTPUT/"decision.json").exists(), "open exact-source experiment")
        report=run_control(checkpoint=MODEL,seed=args.seed,protocol=PROTOCOL,retain_route_trace=True,
                           command_adapter=HeadingHold(args.child=="on"))
        validate_controller_trace(report,args.child=="on")
        report.update(source=args.source,retained_checkpoint_hashes=preserved())
        write_new(OUTPUT/f"{args.child}-s{args.seed}.json",report)
        return
    require(dt.datetime.now(dt.timezone.utc)<LAST_START,"bounded start window")
    OUTPUT.mkdir(parents=True,exist_ok=False)
    write_new(OUTPUT/"launch.json",dict(protocol=PROTOCOL,source=args.source,runtime=runtime,host=host,
        retained_checkpoint_hashes=identities,seeds=SEEDS,controller=HeadingHold(True).provenance()))
    started=time.monotonic(); rows=[]
    decision=dict(protocol=PROTOCOL,source=args.source,decision="runtime-failure-stop",reports=rows,
        policy_acceptance=False,hop_validated=False,stabilization_retention_validated=False,
        obstacles_admitted=False,physical_motion_authorized=False)
    try:
        for seed in SEEDS:
            reports={}
            for arm in ("off","on"):
                require(time.monotonic()-started<750,"remaining child budget")
                check_host()
                with (OUTPUT/f"{arm}-s{seed}.log").open("x") as log:
                    child=subprocess.run([sys.executable,"-m","mjlab_microduck.heading_hold_diagnostic",
                        "--source",args.source,"--child",arm,"--seed",str(seed)],stdout=log,stderr=subprocess.STDOUT,
                        timeout=90,env={**os.environ,"CUDA_VISIBLE_DEVICES":"0","OMP_NUM_THREADS":"1"})
                require(child.returncode==0,f"{arm} child failed")
                path=OUTPUT/f"{arm}-s{seed}.json"
                reports[arm]=json.loads(path.read_text())
                rows.append(dict(arm=arm,seed=seed,sha256=sha256(path)))
                if arm=="off" and reports[arm]["safety_failures"]:
                    decision.update(decision="reference-safety-stop",failures=reports[arm]["safety_failures"])
                    break
            if decision["decision"]=="reference-safety-stop": break
            paired=compare(reports["off"],reports["on"])
            if paired["failures"]:
                decision.update(decision="numerical-gate-stop",**paired)
                break
        else: decision.update(decision="heading-diagnostic-support-only",failures=[])
        decision["retained_checkpoint_hashes"]=preserved()
    except Exception as error:
        decision.update(decision="runtime-failure-stop",error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_new(OUTPUT/"decision.json",decision)
        files={str(p.relative_to(OUTPUT)):dict(sha256=sha256(p),bytes=p.stat().st_size)
               for p in sorted(OUTPUT.rglob("*")) if p.is_file()}
        write_new(OUTPUT/"manifest.json",dict(protocol=PROTOCOL,source=args.source,files=files))
    print(json.dumps(decision),flush=True)


if __name__=="__main__": main()
