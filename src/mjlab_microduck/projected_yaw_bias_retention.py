"""Preserve the authenticated cautious and steady speed/idle bars unchanged."""
from . import projected_heading_retention as reference

BASELINE_SOURCE = reference.BASELINE_SOURCE
BASELINE_CHECKPOINT_SHA = reference.BASELINE_CHECKPOINT_SHA
BASELINE_SUMMARY_SHA = reference.BASELINE_SUMMARY_SHA


def load_baseline():
    return reference.load_baseline()


def compare(rows, baseline):
    result = reference.compare(rows, baseline)
    result["protocol"] = "v8-projected-yaw-bias-retention-v1"
    return result
