"""CPU-only reconciliation of the retained safety-stop report; no GPU retry."""

import argparse
import json
import os

from mjlab_microduck import foundation_lateral_closeout as closeout
from mjlab_microduck.first_attempt_smoke import require,sha256
from mjlab_microduck.recovery_ab import verify_source,write_new

RETAINED=closeout.OUTPUT
OUTPUT=RETAINED.parent/"f1l-lateral-s467-cpu-reconciliation-v1"
CLOSEOUT_SOURCE="22afcfa71ccd899da5c6cade69f9a56bb29e0754"
CLOSEOUT_MANIFEST_SHA="6212bd8d80d9fc4749ac681830f8c1c2e183b9200a212419458635c14d1a86fb"
REPORT_SHA="3761e3239f410c8d311ad1bbe7534789b524769e9c89540c19ff33ae5aba797d"


def read_report():
    require(sha256(RETAINED/"manifest.json")==CLOSEOUT_MANIFEST_SHA,"immutable closeout manifest")
    manifest=json.loads((RETAINED/"manifest.json").read_text())
    require(manifest["source"]==CLOSEOUT_SOURCE,"exact simulation source")
    for name,row in manifest["files"].items():
        p=RETAINED/name
        require(p.is_file() and p.stat().st_size==row["bytes"] and sha256(p)==row["sha256"],"closeout payload "+name)
    require(set(manifest["files"])=={p.name for p in RETAINED.iterdir() if p.is_file() and p.name!="manifest.json"},"exact closeout coverage")
    old=json.loads((RETAINED/"decision.json").read_text())
    require(old["decision"]=="runtime-failure-stop" and old["error"]=="KeyError: 'body_mean_in_band_all_envs'", "exact reporting failure")
    require(sha256(RETAINED/"lateral-s467.json")==REPORT_SHA,"retained report hash")
    report=json.loads((RETAINED/"lateral-s467.json").read_text())
    require(report["source"]==CLOSEOUT_SOURCE and report["training_source"]==closeout.ORIGINAL_SOURCE,
            "simulation/training provenance")
    return report


def reconcile(source):
    require(os.environ.get("CUDA_VISIBLE_DEVICES")=="","explicit CPU-only reconciliation")
    verify_source(source);closeout.exp.training.runtime_identity();closeout.verify_retained()
    report=read_report();pair=closeout.paired(report)
    decision=dict(protocol="f1l-retained-cpu-reconciliation-v1",source=source,
        training_source=closeout.ORIGINAL_SOURCE,simulation_source=CLOSEOUT_SOURCE,
        original_manifest_sha256=closeout.MANIFEST_SHA,closeout_manifest_sha256=CLOSEOUT_MANIFEST_SHA,
        report_sha256=REPORT_SHA,pair=pair,
        decision="numerical-gate-stop" if pair["failures"] else "paired-case-support-only",
        optimizer_updates=0,simulation_steps_executed=0,policy_acceptance=False,
        obstacles_admitted=False,hop_validated=False,stabilization_retention_validated=False,
        physical_motion_authorized=False)
    OUTPUT.mkdir(parents=True,exist_ok=False)
    write_new(OUTPUT/"decision.json",decision)
    write_new(OUTPUT/"manifest.json",dict(source=source,files={"decision.json":dict(
        sha256=sha256(OUTPUT/"decision.json"),bytes=(OUTPUT/"decision.json").stat().st_size)}))
    return decision


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--source",required=True)
    args=parser.parse_args();print(json.dumps(reconcile(args.source)),flush=True)


if __name__=="__main__":main()
