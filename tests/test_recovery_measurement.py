import math

import pytest
import torch

from mjlab_microduck.recovery_measurement import RecoveryMeasurement


def sample(observer, phase=2, speed=.4, done=False):
    observer.begin(observer.next_step, torch.tensor([phase]), torch.tensor([speed], dtype=torch.float64))
    observer.finish(torch.tensor([done]))


def test_requires_half_second_span_not_half_second_sample_count():
    observer = RecoveryMeasurement(1, .4, .02)
    for _ in range(25): sample(observer)
    assert observer.report()["environments"][0]["status"] == "censored-before-window"
    sample(observer, done=True)
    row = observer.report()["environments"][0]
    assert row["status"] == "recovered-in-window"
    assert row["stable_recovery_latency_s"] == .5


def test_terminal_reset_cannot_manufacture_speed_recovery():
    observer = RecoveryMeasurement(1, .4, .02)
    sample(observer, speed=.1, done=True)
    for _ in range(100): sample(observer)
    row = observer.report()["environments"][0]
    assert row["status"] == "censored-before-window" and row["stable_recovery_latency_s"] is None


@pytest.mark.parametrize("interruption", ["speed", "phase"])
def test_contiguous_stability_and_deadline_are_required(interruption):
    observer = RecoveryMeasurement(1, .4, .02)
    for i in range(130):
        sample(observer, phase=1 if interruption == "phase" and i % 25 == 24 else 2,
               speed=.1 if interruption == "speed" and i % 25 == 24 else .4)
    assert observer.report()["environments"][0]["status"] == "window-missed"


def test_late_success_remains_failed_window():
    observer = RecoveryMeasurement(1, .4, .02)
    for _ in range(90): sample(observer, speed=.1)
    for _ in range(26): sample(observer)
    row = observer.report()["environments"][0]
    assert row["stable_recovery_latency_s"] == pytest.approx(2.3)
    assert row["status"] == "window-missed"


def test_no_recovery_is_not_success_and_rng_inputs_unchanged():
    observer = RecoveryMeasurement(1, .4, .02)
    state = torch.get_rng_state().clone()
    phase, speed = torch.tensor([0]), torch.tensor([.4])
    observer.begin(0, phase, speed)
    observer.finish(torch.tensor([True]))
    assert observer.report()["counts"]["not-observed"] == 1
    assert torch.equal(state, torch.get_rng_state())
    assert phase.item() == 0 and speed.item() == pytest.approx(.4)


def test_asynchronous_terminals_and_exact_deadline_are_per_environment():
    observer = RecoveryMeasurement(3, .4, .02)
    for step in range(101):
        speeds = torch.tensor([.4, .1, .4 if step >= 75 else .1], dtype=torch.float64)
        observer.begin(step, torch.tensor([2, 2, 2]), speeds)
        observer.finish(torch.tensor([step == 25, step == 80, step == 100]))
    rows = observer.report()["environments"]
    assert [r["status"] for r in rows] == [
        "recovered-in-window", "censored-before-window", "recovered-in-window"]
    assert rows[0]["stable_recovery_latency_s"] == .5
    assert rows[2]["stable_recovery_latency_s"] == 2.
    assert observer.report()["counts"]["recovered-in-window"] == 2


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nonfinite_active_speed_refused_without_progress(bad):
    observer = RecoveryMeasurement(1, .4, .02)
    with pytest.raises(ValueError): sample(observer, speed=bad)
    assert observer.next_step == 0 and not observer.pending and observer.entry == [None]


def test_lifecycle_refuses_skips_duplicates_and_unfinished_reports():
    observer = RecoveryMeasurement(1, .4, .02)
    with pytest.raises(ValueError): observer.finish(torch.tensor([False]))
    with pytest.raises(ValueError): observer.report()
    with pytest.raises(ValueError): observer.begin(1, torch.tensor([2]), torch.tensor([.4]))
    observer.begin(0, torch.tensor([2]), torch.tensor([.4]))
    with pytest.raises(ValueError): observer.report()
    with pytest.raises(ValueError): observer.begin(0, torch.tensor([2]), torch.tensor([.4]))
    with pytest.raises(ValueError): observer.finish(torch.tensor([1]))
    observer.finish(torch.tensor([False]))


@pytest.mark.parametrize("args", [(True, .4, .02), (65, .4, .02), (1, 0., .02),
                                  (1, math.inf, .02), (1, .4, True), (1, .4, .2)])
def test_invalid_configuration(args):
    with pytest.raises(ValueError): RecoveryMeasurement(*args)
