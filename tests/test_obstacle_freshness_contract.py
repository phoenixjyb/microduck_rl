"""Synthetic nanosecond values are arithmetic probes, not deployment limits."""

from copy import deepcopy

import pytest

from mjlab_microduck.obstacle_freshness_contract import CONTEXT_FIELDS, PHASES, assess


CONTEXT = dict(source_id="synthetic-source", stream_epoch="boot-A", episode_id="episode-7",
               timebase_id="receiver-clock-A")
LIMITS = dict(max_capture_age_ns=100, max_receipt_age_ns=80)


def packet(**changes):
    return dict(CONTEXT, sequence=10, captured_ns=900, first_received_ns=920,
                estimate_status="detected", payload_sha256="a" * 64) | changes


def check(value, **changes):
    return assess(value, **(dict(context=CONTEXT, now_ns=1000, limits=LIMITS, phase="approach") | changes))


def no_authority(result):
    assert result["motion_command"] is None
    for key in ("operational_limits_selected", "clock_binding_verified", "payload_binding_verified",
                "producer_semantics_verified", "receiver_state_advanced", "runtime_wired",
                "safe_stop_verified", "policy_acceptance", "transition_authorized",
                "training_authorized", "physical_motion_authorized"):
        assert result[key] is False


@pytest.mark.parametrize("status,expected", [("detected", "fresh-detection-metadata"),
    ("clear", "fresh-clear-metadata"), ("invalid", "invalid-estimate")])
def test_boundary_age_and_producer_states_are_distinct(status, expected):
    result = check(packet(estimate_status=status))
    assert result["status"] == expected
    assert result["declared_ages_ns"] == dict(capture=100, receipt=80)
    no_authority(result)


@pytest.mark.parametrize("limits,reasons", [
    (dict(max_capture_age_ns=99, max_receipt_age_ns=80), ["capture age exceeded"]),
    (dict(max_capture_age_ns=100, max_receipt_age_ns=79), ["receipt age exceeded"]),
    (dict(max_capture_age_ns=99, max_receipt_age_ns=79), ["capture age exceeded", "receipt age exceeded"]),
])
def test_capture_and_receipt_ages_are_independent(limits, reasons):
    result = check(packet(), limits=limits)
    assert result["status"] == "stale" and result["reasons"] == reasons
    no_authority(result)


def test_receiving_an_old_capture_recently_does_not_make_it_fresh():
    assert check(packet(captured_ns=899, first_received_ns=1000))["status"] == "stale"


def test_cached_repoll_does_not_refresh_age_or_mutate_history():
    current = packet()
    before = deepcopy(current)
    result = check(current, previous=current, last_check_ns=1000, now_ns=1001)
    assert result["status"] == "stale" and result["identical_cached_packet"] is True
    assert current == before
    no_authority(result)


@pytest.mark.parametrize("change", [dict(first_received_ns=930), dict(captured_ns=901),
    dict(payload_sha256="b" * 64), dict(estimate_status="clear")])
def test_same_sequence_cannot_refresh_or_change_packet(change):
    result = check(packet(**change), previous=packet(), last_check_ns=950)
    assert result["status"] == "duplicate-conflict"


@pytest.mark.parametrize("change", [dict(sequence=9), dict(sequence=11, captured_ns=899),
                                   dict(sequence=11, first_received_ns=919)])
def test_ordered_watermark_rejects_backward_sequence_or_time(change):
    assert check(packet(**change), previous=packet(), last_check_ns=950)["status"] == "out-of-order"


def test_newer_invalid_estimate_prevents_fallback_to_old_good_packet():
    previous = packet(sequence=11, estimate_status="invalid")
    assert check(packet(), previous=previous, last_check_ns=950)["status"] == "out-of-order"


def test_newer_fresh_packet_after_invalid_is_metadata_only():
    result = check(packet(sequence=12, captured_ns=980, first_received_ns=990),
                   previous=packet(sequence=11, estimate_status="invalid"), last_check_ns=950)
    assert result["status"] == "fresh-detection-metadata"
    no_authority(result)


@pytest.mark.parametrize("key", CONTEXT_FIELDS)
def test_packet_cannot_change_trusted_receiver_identity(key):
    assert check(packet(**{key: "different"}))["status"] == "identity-mismatch"


@pytest.mark.parametrize("change", [dict(captured_ns=921), dict(first_received_ns=1001)])
def test_future_or_impossible_timestamps_are_untrusted(change):
    assert check(packet(**change))["status"] == "invalid-time-order"


def test_receiver_clock_rollback_is_not_a_fresh_packet():
    assert check(packet(), last_check_ns=1001)["status"] == "receiver-clock-regression"


@pytest.mark.parametrize("phase", PHASES)
def test_missing_sensor_in_every_phase_never_emits_motion(phase):
    result = check(None, phase=phase)
    assert result["status"] == "missing"
    if phase == "airborne":
        assert result["required_case_family"] == "interruption-in-air-landing-before-stop"
    elif phase == "landing":
        assert result["required_case_family"] == "interruption-during-landing-and-recovery"
    elif phase == "pre-takeoff":
        assert result["required_case_family"] == "interruption-before-takeoff"
    else:
        assert result["required_case_family"] == "grounded-sensor-loss-stop-and-reacquisition"
    no_authority(result)


@pytest.mark.parametrize("change", [dict(sequence=True), dict(captured_ns=-1), dict(first_received_ns=1.5),
    dict(captured_ns=float("nan")), dict(payload_sha256="wrong"), dict(estimate_status="no-detection"),
    dict(estimate_status=["detected"]), dict(extra=True)])
def test_malformed_current_packet_is_not_a_clear_observation(change):
    assert check(packet(**change))["status"] == "malformed"


@pytest.mark.parametrize("changes", [dict(limits={}), dict(limits=dict(max_capture_age_ns=True, max_receipt_age_ns=1)),
    dict(limits=dict(max_capture_age_ns=0, max_receipt_age_ns=1)), dict(now_ns=True),
    dict(context={}), dict(phase="unknown"), dict(previous=packet()), dict(last_check_ns=-1),
    dict(previous=packet(episode_id="old-episode"), last_check_ns=950)])
def test_bad_receiver_configuration_is_an_error_not_an_implicit_default(changes):
    with pytest.raises(ValueError):
        check(packet(), **changes)
