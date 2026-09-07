"""One never-started F1-L treatment evaluation; immutable failed run retained."""

import argparse
import datetime as dt
import json
import os
import subprocess
import sys

from mjlab_microduck import foundation_lateral_experiment as exp
from mjlab_microduck.first_attempt_smoke import require,sha256,canonical
from mjlab_microduck.gpu_idle_gate import wait_idle
from mjlab_microduck.recovery_ab import verify_source,write_new

ORIGINAL=exp.OUTPUT
OUTPUT=ORIGINAL.parent/"f1l-lateral-s467-evaluation-closeout-v1"
ORIGINAL_SOURCE="e0cb4f4c44d5cd548fe5aee4458c59cda4510adf"
MANIFEST_SHA="44ce65ffec5252acd91bece8224d7e4487ef91f64a9a996ae1c69d00eecd553f"
DECISION_SHA="541cb79c4254f5116be939a17963529c017ceb39c0cdab48254baa9c0f83331c"
LAST_START=dt.datetime(2026,9,7,9,tzinfo=dt.timezone.utc)


def verify_retained():
    require(sha256(ORIGINAL/"manifest.json")==MANIFEST_SHA
            and sha256(ORIGINAL/"decision.json")==DECISION_SHA,"immutable original evidence")
    manifest=json.loads((ORIGINAL/"manifest.json").read_text())
    require(manifest["source"]==ORIGINAL_SOURCE and manifest["protocol"]==exp.PROTOCOL,"original source/protocol")
    for name,row in manifest["files"].items():
        path=ORIGINAL/name
        require(path.is_file() and path.stat().st_size==row["bytes"] and sha256(path)==row["sha256"],"retained payload: "+name)
    require(set(manifest["files"])=={str(p.relative_to(ORIGINAL)) for p in ORIGINAL.rglob("*")
                                    if p.is_file() and p.name!="manifest.json"},"original file coverage")
    d=json.loads((ORIGINAL/"decision.json").read_text())
    require(d["decision"]=="runtime-failure-stop" and d["error"]=="ValueError: idle/cool GPU required"
            and d["pairs"]==[] and [(r["arm"],r["seed"]) for r in d["reports"]]==
            [("parent",467),("parent",479),("parent",487),("control",467)],"exact interrupted boundary")
    require(not (ORIGINAL/"lateral-s467.json").exists() and not (ORIGINAL/"lateral-s467.log").exists(),
            "treatment evaluation never launched")
    exp.preserved()
    return manifest


def paired(report):
    parent=json.loads((ORIGINAL/"parent-s467.json").read_text())
    control=json.loads((ORIGINAL/"control-s467.json").read_text())
    for arm,r in (("parent",parent),("control",control),("lateral",report)):
        _,identity=exp.checkpoint(arm,ORIGINAL_SOURCE)
        exp.validate_controller_trace(r,True,protocol=exp.PROTOCOL,seeds=(467,),checkpoint_sha=identity)
    return exp.paired_decision(parent,control,report)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",required=True);parser.add_argument("--child",action="store_true")
    args=parser.parse_args()
    verify_source(args.source);exp.training.runtime_identity();verify_retained()
    if args.child:
        launch=json.loads((OUTPUT/"launch.json").read_text())
        require(launch["source"]==args.source and not (OUTPUT/"decision.json").exists(),"open exact-source closeout")
        exp.check_host()
        checkpoint,identity=exp.checkpoint("lateral",ORIGINAL_SOURCE)
        report=exp.run_control(checkpoint=checkpoint,seed=467,protocol=exp.PROTOCOL,
                               retain_route_trace=True,command_adapter=exp.HeadingHold(True))
        exp.validate_controller_trace(report,True,protocol=exp.PROTOCOL,seeds=(467,),checkpoint_sha=identity)
        report.update(source=args.source,training_source=ORIGINAL_SOURCE,retained_checkpoint_hashes=exp.preserved())
        write_new(OUTPUT/"lateral-s467.json",report)
        return
    require(dt.datetime.now(dt.timezone.utc)<LAST_START,"bounded closeout start")
    OUTPUT.mkdir(parents=True,exist_ok=False)
    decision=dict(protocol="f1l-first-pair-closeout-v1",source=args.source,training_source=ORIGINAL_SOURCE,
        decision="runtime-failure-stop",policy_acceptance=False,obstacles_admitted=False,
        hop_validated=False,stabilization_retention_validated=False,physical_motion_authorized=False)
    try:
        idle=wait_idle()
        write_new(OUTPUT/"launch.json",dict(source=args.source,original_source=ORIGINAL_SOURCE,
            original_manifest_sha256=MANIFEST_SHA,original_decision_sha256=DECISION_SHA,
            started_at=dt.datetime.now(dt.timezone.utc).isoformat(),idle=idle,seed=467,optimizer_updates=0))
        with (OUTPUT/"lateral-s467.log").open("x") as log:
            child=subprocess.run([sys.executable,"-m","mjlab_microduck.foundation_lateral_closeout",
                "--source",args.source,"--child"],stdout=log,stderr=subprocess.STDOUT,timeout=90,
                env={**os.environ,"CUDA_VISIBLE_DEVICES":"0","OMP_NUM_THREADS":"1","PYTHONUNBUFFERED":"1"})
        require(child.returncode==0,"single closeout child failed")
        report=json.loads((OUTPUT/"lateral-s467.json").read_text());canonical(report)
        pair=paired(report)
        decision.update(decision="numerical-gate-stop" if pair["failures"] else "paired-case-support-only",
                        pair=pair,report_sha256=sha256(OUTPUT/"lateral-s467.json"))
        verify_retained()
    except Exception as error:
        decision.update(decision="runtime-failure-stop",error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_new(OUTPUT/"decision.json",decision)
        files={str(p.relative_to(OUTPUT)):dict(sha256=sha256(p),bytes=p.stat().st_size)
               for p in sorted(OUTPUT.rglob("*")) if p.is_file()}
        write_new(OUTPUT/"manifest.json",dict(source=args.source,original_manifest_sha256=MANIFEST_SHA,files=files))
    print(json.dumps(decision),flush=True)


if __name__=="__main__":main()
