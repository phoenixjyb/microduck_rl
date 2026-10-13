"""Keep both the original cautious and latest steady walking/idle bars."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat

from . import integrated_turn_foundation as base
from . import community_policy_inspection as inspection
from . import steady_steering_retention as previous

BASELINE_SOURCE = "7f55ded7c615e0adda4f859e6e05dde40c447e66"
BASELINE_CHECKPOINT_SHA = "d307531267971fc8276f158d42cc0ac9259b4a69efe1ccfcedd5b773e600b47f"
BASELINE_SUMMARY_SHA = "182e7fd99c6ec069f8e6b932ccc3008e0125fb6ba94a7e08f55e6d72edf7f155"
BASELINE_PATH = Path("/home/converge/.local/state/v8-steady-steering-7f55ded7-20261013-handoff/steady-steering-summary.json")


def load_baseline():
    info = BASELINE_PATH.lstat()
    base.require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid()
                 and info.st_size <= 128*1024, "owned compact sealed steady summary")
    payload = inspection._read_payload(BASELINE_PATH)
    base.require(hashlib.sha256(payload).hexdigest() == BASELINE_SUMMARY_SHA,
                 "exact sealed steady summary bytes")
    value = json.loads(payload)
    base.require(value["source"] == BASELINE_SOURCE
        and value["evaluations"]["final-eval"]["checkpoint_sha256"] == BASELINE_CHECKPOINT_SHA,
        "sealed steady source and final999 identity")
    # A new control must not silently replace/relax the existing cautious bars.
    previous.load_baseline()
    return value


def compare(rows, baseline):
    report = previous._compare_sealed_reference(rows, baseline,
        source=BASELINE_SOURCE, checkpoint_sha=BASELINE_CHECKPOINT_SHA,
        summary_sha=BASELINE_SUMMARY_SHA, protocol="v8-projected-heading-retention-v1")
    original = previous.compare(rows, previous.load_baseline())
    report["original_cautious_retention"] = original
    if original["failures"]:
        report["failures"].extend(["original-cautious", *failure] for failure in original["failures"])
        report["decision"] = "walking-idle-not-retained"
    return report
