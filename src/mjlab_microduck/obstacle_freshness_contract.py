"""Non-wired CPU metadata contract; not a sensor adapter or safe-stop policy.

Acquisition and first receipt timestamps must already share a receiver-owned
monotonic timebase. This module neither synchronizes clocks nor authenticates
publishers, binds payload bytes, updates receiver state, or controls the duck.
"""

from __future__ import annotations

import re


PROTOCOL = "external-obstacle-freshness-metadata-v1"
CONTEXT_FIELDS = ("source_id", "stream_epoch", "episode_id", "timebase_id")
PACKET_FIELDS = {*CONTEXT_FIELDS, "sequence", "captured_ns", "first_received_ns",
                 "estimate_status", "payload_sha256"}
PHASES = ("approach", "interaction", "recovery", "stance", "pre-takeoff", "airborne", "landing")
LIMIT_FIELDS = {"max_capture_age_ns", "max_receipt_age_ns"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def nonnegative_integer(value):
    return type(value) is int and value >= 0


def valid_context(value):
    return (type(value) is dict and set(value) == set(CONTEXT_FIELDS)
            and all(type(v) is str and 0 < len(v) <= 128 and v.strip() == v for v in value.values()))


def valid_packet(value):
    return (type(value) is dict and set(value) == PACKET_FIELDS
            and valid_context({key: value[key] for key in CONTEXT_FIELDS})
            and all(nonnegative_integer(value[key]) for key in
                    ("sequence", "captured_ns", "first_received_ns"))
            and type(value["estimate_status"]) is str
            and value["estimate_status"] in ("detected", "clear", "invalid")
            and type(value["payload_sha256"]) is str
            and re.fullmatch(r"[0-9a-f]{64}", value["payload_sha256"]) is not None)


def assess(packet, *, context, now_ns, limits, phase, previous=None, last_check_ns=None):
    """Classify declared metadata only, with explicit caller-owned history.

No default age limits are supplied. `previous` is the latest structurally valid,
ordered same-context envelope, including stale/invalid estimates; `last_check_ns`
is the last receiver check, including failed checks. Repeated polls must retain
original first receipt. Do not fall back to an older good packet after a newer
invalid estimate; the sequence watermark still advances for the latter.
Bad caller configuration/history raises; malformed current packets classify as
untrusted. Caller must retain its history; this pure function does not do so.
"""
    require(valid_context(context), "explicit receiver context required")
    require(nonnegative_integer(now_ns), "receiver monotonic integer time required")
    require(type(limits) is dict and set(limits) == LIMIT_FIELDS
            and all(type(v) is int and v > 0 for v in limits.values()),
            "two explicit positive integer age limits required; no defaults")
    require(type(phase) is str and phase in PHASES, "explicit supported phase required")
    require(last_check_ns is None or nonnegative_integer(last_check_ns), "invalid receiver history time")
    if previous is not None:
        require(valid_packet(previous) and all(previous[key] == context[key] for key in CONTEXT_FIELDS)
                and previous["captured_ns"] <= previous["first_received_ns"]
                and last_check_ns is not None and previous["first_received_ns"] <= last_check_ns,
                "structural watermark in the exact receiver context required")

    def result(status, reasons, ages=None, cached=False):
        fresh = status in ("fresh-detection-metadata", "fresh-clear-metadata")
        required_case = ("payload-and-producer-semantics-validation" if fresh else
                         "interruption-in-air-landing-before-stop" if phase == "airborne" else
                         "interruption-during-landing-and-recovery" if phase == "landing" else
                         "interruption-before-takeoff" if phase == "pre-takeoff" else
                         "grounded-sensor-loss-stop-and-reacquisition")
        return dict(protocol=PROTOCOL, status=status, reasons=reasons, phase=phase,
                    declared_ages_ns=ages, identical_cached_packet=cached,
                    required_case_family=required_case,
                    operational_limits_selected=False, clock_binding_verified=False,
                    payload_binding_verified=False, producer_semantics_verified=False,
                    receiver_state_advanced=False, runtime_wired=False,
                    safe_stop_verified=False, policy_acceptance=False,
                    motion_command=None, transition_authorized=False,
                    training_authorized=False, physical_motion_authorized=False)

    if last_check_ns is not None and now_ns < last_check_ns:
        return result("receiver-clock-regression", ["now precedes last receiver check"])
    if packet is None:
        return result("missing", ["no packet"])
    if not valid_packet(packet):
        return result("malformed", ["exact metadata schema and domains required"])
    if any(packet[key] != context[key] for key in CONTEXT_FIELDS):
        return result("identity-mismatch", ["source, epoch, episode or timebase differs"])
    if not packet["captured_ns"] <= packet["first_received_ns"] <= now_ns:
        return result("invalid-time-order", ["capture <= first receipt <= now required"])
    cached = False
    if previous is not None:
        if packet["sequence"] < previous["sequence"]:
            return result("out-of-order", ["sequence precedes accepted watermark"])
        if packet["sequence"] == previous["sequence"]:
            if packet != previous:
                return result("duplicate-conflict", ["same sequence changed immutable metadata"])
            cached = True
        elif (packet["captured_ns"] < previous["captured_ns"]
              or packet["first_received_ns"] < previous["first_received_ns"]):
            return result("out-of-order", ["new sequence moved capture or receipt backward"])
    ages = dict(capture=now_ns - packet["captured_ns"], receipt=now_ns - packet["first_received_ns"])
    reasons = [key + " age exceeded" for key in ("capture", "receipt")
               if ages[key] > limits[f"max_{key}_age_ns"]]
    if reasons:
        return result("stale", reasons, ages, cached)
    if packet["estimate_status"] == "invalid":
        return result("invalid-estimate", ["publisher marks estimate invalid"], ages, cached)
    status = "fresh-detection-metadata" if packet["estimate_status"] == "detected" else "fresh-clear-metadata"
    return result(status, [], ages, cached)
