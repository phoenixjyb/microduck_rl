"""Test-only metadata bookkeeping; not a receiver, controller or resumption latch."""

from copy import deepcopy

import pytest

from mjlab_microduck.obstacle_freshness_contract import CONTEXT_FIELDS, PHASES, assess


CONTEXT = dict(source_id="synthetic", stream_epoch="boot-A", episode_id="episode-A",
               timebase_id="receiver-A")
LIMITS = dict(max_capture_age_ns=100, max_receipt_age_ns=80)  # toy arithmetic, not operational
ORDERED = {"fresh-detection-metadata", "fresh-clear-metadata", "invalid-estimate", "stale"}


def packet(seq=10, capture=900, receipt=920, status="detected", **context):
    return dict(CONTEXT | context, sequence=seq, captured_ns=capture, first_received_ns=receipt,
                estimate_status=status, payload_sha256="a" * 64)


class HistoryProbe:
    """Exercise documented caller watermarks only; never model motion/resumption."""

    def __init__(self):
        self.context = dict(CONTEXT)
        self.previous = None
        self.last_check = None

    def check(self, value, now, phase="approach"):
        before = deepcopy((value, self.context, self.previous))
        result = assess(value, context=self.context, now_ns=now, limits=LIMITS, phase=phase,
                        previous=self.previous, last_check_ns=self.last_check)
        assert (value, self.context, self.previous) == before  # classifier is pure
        assert result["motion_command"] is None
        for key in ("receiver_state_advanced", "runtime_wired", "policy_acceptance", "safe_stop_verified",
                    "transition_authorized", "training_authorized", "physical_motion_authorized"):
            assert result[key] is False
        if result["status"] in ORDERED:
            self.previous = deepcopy(value)
        if self.last_check is None or now >= self.last_check:
            self.last_check = now
        return result


def test_cached_poll_sequence_expires_without_receipt_refresh():
    probe, sample = HistoryProbe(), packet()
    assert probe.check(sample, 950)["status"] == "fresh-detection-metadata"
    assert probe.check(sample, 1000)["status"] == "fresh-detection-metadata"
    assert probe.check(sample, 1001)["status"] == "stale"
    assert probe.check(sample, 1050)["status"] == "stale"
    assert probe.previous == sample and probe.previous["first_received_ns"] == 920


@pytest.mark.parametrize("phase", PHASES)
def test_missing_then_new_fresh_metadata_never_resumes_a_policy(phase):
    probe = HistoryProbe()
    probe.check(packet(), 950, phase)
    assert probe.check(None, 960, phase)["status"] == "missing"
    assert probe.previous["sequence"] == 10
    result = probe.check(packet(11, 965, 970), 980, phase)
    assert result["status"] == "fresh-detection-metadata"
    assert result["required_case_family"] == "payload-and-producer-semantics-validation"
    assert result["producer_semantics_verified"] is False


def test_invalid_new_sample_blocks_older_good_and_conflicting_same_sequence():
    probe = HistoryProbe()
    probe.check(packet(), 950)
    assert probe.check(packet(11, 950, 960, "invalid"), 970)["status"] == "invalid-estimate"
    assert probe.check(packet(), 975)["status"] == "out-of-order"
    assert probe.check(packet(11, 950, 960, "clear"), 980)["status"] == "duplicate-conflict"
    assert probe.previous["sequence"] == 11 and probe.previous["estimate_status"] == "invalid"
    assert probe.check(packet(12, 981, 990, "clear"), 1000)["status"] == "fresh-clear-metadata"


def test_delayed_new_sample_keeps_sequence_watermark_without_becoming_fresh():
    probe = HistoryProbe()
    probe.check(packet(), 950)
    assert probe.check(packet(20, 950, 1090), 1100)["status"] == "stale"
    # Apparently recent times do not override an older sequence number.
    assert probe.check(packet(19, 1101, 1102), 1103)["status"] == "out-of-order"
    assert probe.previous["sequence"] == 20


@pytest.mark.parametrize("key", CONTEXT_FIELDS)
def test_unsolicited_context_change_does_not_evict_the_watermark(key):
    probe = HistoryProbe()
    probe.check(packet(), 950)
    assert probe.check(packet(1000, 950, 960, **{key: "untrusted-change"}), 970)["status"] == "identity-mismatch"
    assert probe.previous["sequence"] == 10
    assert probe.context == CONTEXT
    assert probe.check(packet(11, 980, 990), 1000)["status"] == "fresh-detection-metadata"


def test_explicit_test_context_reset_cannot_accept_old_episode_packets():
    probe = HistoryProbe()
    probe.check(packet(), 950)
    # Test fixture only: a production trusted rebind/handshake is NOT implemented.
    probe.context["episode_id"] = "episode-B"
    probe.previous = None
    assert probe.check(packet(), 960)["status"] == "identity-mismatch"
    assert probe.check(packet(0, 965, 970, episode_id="episode-B"), 980)["status"] == "fresh-detection-metadata"
    assert probe.last_check == 980


def test_malformed_input_advances_check_time_but_never_packet_watermark():
    probe = HistoryProbe()
    probe.check(packet(), 950)
    assert probe.check({"sequence": 1000}, 980)["status"] == "malformed"
    assert probe.previous["sequence"] == 10 and probe.last_check == 980
    assert probe.check(packet(11, 951, 952), 960)["status"] == "receiver-clock-regression"
    assert probe.last_check == 980 and probe.previous["sequence"] == 10


def test_future_timestamp_and_receive_refresh_conflicts_do_not_replace_history():
    probe = HistoryProbe()
    probe.check(packet(), 950)
    assert probe.check(packet(1000, 2000, 2010), 960)["status"] == "invalid-time-order"
    assert probe.check(packet(receipt=970), 980)["status"] == "duplicate-conflict"
    assert probe.previous == packet()
    assert probe.check(packet(), 1001)["status"] == "stale"
