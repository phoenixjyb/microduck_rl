"""CPU command-domain diagnosis, not a causal claim or a new training gate."""

import json
from pathlib import Path

import torch
import yaml

from mjlab_microduck import foundation_neck_experiment as exp
from mjlab_microduck.first_attempt_smoke import canonical, require, sha256

MANIFEST_SHA = "8e6fcb53477ee6edb6508d55678073540b59d3ade61981bb9437a6ce636b0bcf"
DECISION_SHA = "ce37f29a36a078522bd09fc497d62d541b4fcaa3a58523fbb2349f13cd603196"
SOURCE = "4317802cb09a3464b2db805ce4bef801abecd825"
COLUMNS = ("lin_vel_x", "lin_vel_y", "ang_vel_z")


def command_coverage(consumed, ranges):
    """Compare in the actor's float32 command domain; no deadband or rounding."""
    canonical([consumed,ranges])
    require(set(ranges)==set(COLUMNS),"three body-command bounds")
    bounds=torch.tensor([ranges[k] for k in COLUMNS],dtype=torch.float32)
    values=torch.tensor(consumed,dtype=torch.float32)
    require(bounds.shape==(3,2) and values.shape==(400,8,3)
            and bool(torch.isfinite(bounds).all() and torch.isfinite(values).all())
            and bool((bounds[:,0]<=bounds[:,1]).all()),"finite ordered bounds and full command coverage")
    groups={}
    for group,start in (("all",0),("settled",100)):
        rows={}
        for i,key in enumerate(COLUMNS):
            v=values[start:,:,i];outside=(v<bounds[i,0])|(v>bounds[i,1])
            rows[key]=dict(minimum=float(v.min()),maximum=float(v.max()),
                absolute_mean=float(v.double().abs().mean()),outside_samples=int(outside.sum()),
                samples=v.numel(),outside_fraction=float(outside.double().mean()))
        groups[group]=rows
    return dict(groups=groups,training_bounds=ranges,
                comparison_domain="native-float32-actor-command; exact inclusive bounds",
                causal_failure_explained=False,policy_acceptance=False,training_admitted=False)


def audit_retained(root):
    """Read closed F1-N only; never launch simulation, mutate evidence or unpickle YAML."""
    root=Path(root)
    require(sha256(root/"manifest.json")==MANIFEST_SHA and sha256(root/"decision.json")==DECISION_SHA,
            "exact closed F1-N evidence")
    manifest=json.loads((root/"manifest.json").read_text())
    files=manifest["files"]
    require(manifest["source"]==SOURCE and set(files)=={str(p.relative_to(root)) for p in root.rglob("*")
            if p.is_file() and p.name!="manifest.json"},"complete original payload inventory")
    for name,row in files.items():
        p=root/name
        require(p.resolve().is_relative_to(root.resolve()) and p.stat().st_size==row["bytes"]
                and sha256(p)==row["sha256"],"unchanged payload "+name)
    bounds=[]
    for arm in exp.ARMS:
        # BaseLoader creates only strings/containers and never executes Python tags.
        cfg=yaml.load((root/f"{arm}-pilot/params/env.yaml").read_text(),Loader=yaml.BaseLoader)
        c=cfg["commands"]["twist"]
        require(c["heading_command"]=="false" and c["rel_heading_envs"]=="0.0"
                and c["rel_world_envs"]=="0.0","body-frame constant-command training")
        bounds.append({k:[float(x) for x in c["ranges"][k]] for k in COLUMNS})
    require(bounds[0]==bounds[1]==dict(lin_vel_x=[.3,.3],lin_vel_y=[0.,0.],ang_vel_z=[0.,0.]),
            "unchanged paired fine-tune domain")
    decision=json.loads((root/"decision.json").read_text());cases=[]
    for row in decision["reports"]:
        arm,seed=row["arm"],row["seed"]
        r=json.loads((root/f"{arm}-s{seed}.json").read_text())
        checkpoint=exp.base.MODEL_SHA if arm=="parent" else json.loads((root/f"{arm}-pilot/result.json").read_text())["final_sha256"]
        require(r["source"]==SOURCE,"closed evaluation source")
        exp.base.validate_controller_trace(r,True,protocol=exp.PROTOCOL,seeds=exp.SEEDS,checkpoint_sha=checkpoint)
        cases.append(dict(arm=arm,seed=seed,report_sha256=row["sha256"],
            comparison=command_coverage(r["command_trace"]["actor_input"],bounds[0])))
    return dict(protocol="f1n-retained-command-coverage-v1",manifest_sha256=MANIFEST_SHA,
        decision_sha256=DECISION_SHA,cases=cases,optimizer_updates=0,simulation_steps_executed=0,
        policy_acceptance=False,causal_failure_explained=False,third_pilot_admitted=False)
