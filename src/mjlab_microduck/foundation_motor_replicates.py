"""Six fixed, fresh motor recordings; no optimizer, retry or policy promotion."""

import argparse
import datetime as dt
import json
import subprocess
import sys
import time

from mjlab_microduck import foundation_replay_control as replay
from mjlab_microduck import motor_measurement_contract as contract
from mjlab_microduck.first_attempt_smoke import canonical, require, sha256
from mjlab_microduck.gpu_idle_gate import wait_idle
from mjlab_microduck.motor_trace_audit import audit_motor_trace
from mjlab_microduck.recovery_ab import verify_source, write_new

exp = replay.exp
PROTOCOL = contract.PROTOCOL
OUTPUT = exp.OUTPUT.parent / PROTOCOL
CHILD_SECONDS, SERVICE_SECONDS, RESERVE_SECONDS = 120, 900, 60
HASH_MANIFEST = "e90ee62d458421bdb7c69398e25d4613727dd2cac236840fadd8d0bc5451f87c"
HASH_DECISION = "dad1bb5b099ccd12c7410655655484d3c09f57e2fa8b0f975a6f3ff08c30a4b8"


def verify_history():
    helpers = replay.verify_history()
    replay.verify_off_control()
    p = replay.HASH_OUTPUT
    require(sha256(p/"manifest.json") == HASH_MANIFEST and sha256(p/"decision.json") == HASH_DECISION,
            "closed startup-hash control identity")
    files = json.loads((p/"manifest.json").read_text())["files"]
    require(len(files) == 10, "complete startup-hash evidence")
    for name, row in files.items():
        f = p/name
        require(f.resolve().is_relative_to(p.resolve()) and f.is_file()
                and f.stat().st_size == row["bytes"] and sha256(f) == row["sha256"], "unchanged startup-hash payload")
    require(set(files) == {str(f.relative_to(p)) for f in p.rglob("*")
                          if f.is_file() and f.name != "manifest.json"}, "exact startup-hash coverage")
    for arm, expected in contract.CHECKPOINTS.items():
        _, actual = exp.checkpoint(arm, replay.timing.ORIGINAL_SOURCE)
        require(actual == expected, "fixed original checkpoint: "+arm)
    return dict(helpers=helpers, checkpoints=dict(contract.CHECKPOINTS),
                baseline_fingerprint=json.loads((p/"first-fingerprint.json").read_text()))


def run_child(case, source, end):
    require(case in contract.ORDER, "fixed recording case")
    require(time.monotonic()+CHILD_SECONDS+RESERVE_SECONDS < end, "bounded remaining recording budget")
    require(dt.datetime.now(dt.timezone.utc)+dt.timedelta(seconds=CHILD_SECONDS+RESERVE_SECONDS) < exp.DEADLINE,
            "recording closeout before07:00")
    write_new(OUTPUT/f"{case}-idle.json", wait_idle())
    with (OUTPUT/f"{case}.log").open("x") as log:
        result = subprocess.run([sys.executable,"-m","mjlab_microduck.foundation_motor_replicates",
            "--source",source,"--case",case], stdout=log, stderr=subprocess.STDOUT,
            timeout=CHILD_SECONDS, env=replay.child_environment(True))
    require(result.returncode == 0, case+" recording child failed")


def campaign(source):
    verify_source(source)
    now = dt.datetime.now(dt.timezone.utc)
    require(now < exp.LAST_START and now+dt.timedelta(seconds=SERVICE_SECONDS+90) < exp.DEADLINE,
            "no late recording start; full cap and closeout must fit")
    history, runtime = verify_history(), exp.training.runtime_identity()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    end = time.monotonic()+SERVICE_SECONDS
    decision = dict(protocol=PROTOCOL, source=source, decision="runtime-failure-stop", reports=[],
                    optimizer_updates=0, historical_rejection_reopened=False, **contract.CLAIMS)
    records = []
    try:
        write_new(OUTPUT/"launch.json",dict(protocol=PROTOCOL,source=source,history=history,runtime=runtime,
            cases=list(contract.ORDER),seed=503,idle=wait_idle(),child_seconds=CHILD_SECONDS,
            service_seconds=SERVICE_SECONDS,closeout_reserve_seconds=RESERVE_SECONDS,
            child_environment={k:replay.child_environment(True).get(k) for k in replay.ENV_KEYS},
            optimizer_updates=0,**contract.CLAIMS))
        for case in contract.ORDER:
            run_child(case,source,end)
            path = OUTPUT/f"{case}.json"
            report = json.loads(path.read_text())
            decision["reports"].append(dict(case=case,sha256=sha256(path)))
            require(report["source"] == source and canonical(report["measurement_identity"]) == canonical(contract.identity(case)),
                    "exact new recording identity")
            exp.validate_controller_trace(report,True,protocol=exp.PROTOCOL,seeds=(503,),
                checkpoint_sha=contract.CHECKPOINTS[case.split("-")[0]])
            analysis = audit_motor_trace(report)
            failures = list(report["safety_failures"])
            for name, group in report["groups"].items():
                if group["pre_reset_torque_p99"] > .60: failures.append(name+"-absolute-pre-reset-torque")
            if failures:
                write_new(OUTPUT/f"{case}-analysis.json",analysis)
                decision.update(decision="absolute-safety-stop",failed_case=case,failures=failures)
                break
            analysis = contract.measure_case(report,source=source,case=case)
            write_new(OUTPUT/f"{case}-analysis.json",analysis)
            states = json.loads((OUTPUT/f"{case}-fingerprint.json").read_text())
            require(canonical(states) == canonical(history["baseline_fingerprint"]), "unchanged baseline numerical settings")
            records.append(dict(case=case,report=report,fingerprint=states))
            print(json.dumps(dict(case=case,status="raw-accounting-verified-not-accepted",
                                  sample_steps=report["sample_steps"])),flush=True)
        else:
            decision.update(contract.describe_dataset(records,source=source))
        verify_history()
    except Exception as error:
        decision.update(decision="runtime-failure-stop",error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_new(OUTPUT/"decision.json",decision)
        files={str(p.relative_to(OUTPUT)):dict(sha256=sha256(p),bytes=p.stat().st_size)
               for p in sorted(OUTPUT.rglob("*")) if p.is_file()}
        write_new(OUTPUT/"manifest.json",dict(protocol=PROTOCOL,source=source,files=files))
    print(json.dumps(dict(decision=decision["decision"],cases=len(decision["reports"]),optimizer_updates=0)),flush=True)


def record_child(case, source):
    verify_source(source)
    history=verify_history(); exp.training.runtime_identity(); exp.check_host()
    launch=json.loads((OUTPUT/"launch.json").read_text())
    require(launch["source"]==source and launch["cases"]==list(contract.ORDER)
            and not (OUTPUT/"decision.json").exists(), "open exact-source recording campaign")
    for earlier in contract.ORDER[:contract.ORDER.index(case)]:
        require((OUTPUT/f"{earlier}-analysis.json").is_file(), "preceding case audited before next child")
    require(dt.datetime.now(dt.timezone.utc)+dt.timedelta(seconds=CHILD_SECONDS) < exp.DEADLINE, "child recording deadline")
    path, expected=exp.checkpoint(case.split("-")[0],replay.timing.ORIGINAL_SOURCE)
    before=replay.fingerprint()
    require(canonical(before)==canonical(history["baseline_fingerprint"]["before"]), "frozen entry fingerprint")
    report=exp.run_control(checkpoint=path,seed=503,protocol=exp.PROTOCOL,retain_route_trace=True,
                           command_adapter=exp.HeadingHold(True),retain_motor_trace=True)
    after=replay.fingerprint()
    report.update(source=source,retained_checkpoint_hashes=exp.preserved(),measurement_identity=contract.identity(case))
    # Always retain returned raw samples before any accounting assertion.
    write_new(OUTPUT/f"{case}.json",report)
    write_new(OUTPUT/f"{case}-fingerprint.json",dict(before=before,after=after))
    require(report["checkpoint_sha256"]==expected,"recorded fixed checkpoint")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",required=True)
    parser.add_argument("--case",choices=contract.ORDER)
    args=parser.parse_args()
    if args.case:record_child(args.case,args.source)
    else:campaign(args.source)


if __name__ == "__main__":
    main()
