from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

from mjlab_microduck import gait_phase_heading_diagnosis as diagnosis

SPEC = importlib.util.spec_from_file_location("phase_cli", Path(__file__).parents[1] / "scripts/diagnose_gait_phase_heading.py")
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)


def test_group_means_preserve_world_sign_cancellation_and_empty_phases():
    reports, periods = [], {}
    for speed, yaw in ((0., 0.), (.3, 0.), (.2, -.2), (.2, .2)):
        periods[(speed, yaw)] = [] if speed == 0 else [.44, .46, .48]
        for seed in (839, 853, 857):
            worlds = []
            for w in range(8):
                sign = 1 if w < 4 else -1
                phases = {p: dict(interval_count=0, duration_s=0., signed_residual_rad=0.,
                                  absolute_residual_rad=0.) for p in diagnosis.PHASES}
                phases["left-only"] = dict(interval_count=179, duration_s=3.58,
                                           signed_residual_rad=.358 * sign, absolute_residual_rad=.358)
                worlds.append(dict(settled_duration_s=3.58, phases=phases,
                                   signed_endpoint_rate_rad_s=.1 * sign, absolute_endpoint_rate_rad_s=.1))
            reports.append(dict(seed=seed, speed=speed, yaw=yaw, worlds=worlds))
    before = deepcopy(reports)
    output = cli.group_reports(reports, periods)
    assert reports == before
    assert len(output) == 4
    for group in output:
        assert group["interval_count"] == 24 * 179
        assert group["mean_signed_endpoint_rate_rad_s"] == pytest.approx(0.)
        assert group["mean_absolute_endpoint_rate_rad_s"] == pytest.approx(.1)
        assert group["phases"]["left-only"]["fraction_of_measured_duration"] == pytest.approx(1.)
        assert group["phases"]["flight"]["signed_residual_rate_rad_s"] is None
        assert group["cycle_duration_s"] == (None if group["command"][0] == 0 else dict(median=.46, p10=.44, p90=.48))


def test_bound_reader_refuses_changed_bytes_and_symlink(tmp_path):
    file = tmp_path / "input.json"
    file.write_text('{}')
    digest = cli.sha(file)
    assert cli.read_bound(file, digest) == {}
    file.write_text('{"changed":1}')
    with pytest.raises(ValueError, match="hash mismatch"):
        cli.read_bound(file, digest)
    link = tmp_path / "link.json"
    link.symlink_to(file)
    with pytest.raises(ValueError, match="non-symlink"):
        cli.sha(link)


def test_reference_pins_are_exact_and_three_distinct_sources():
    assert set(cli.REFERENCES) == {"steady", "cumulative-heading", "projected-rate"}
    assert len({r['source'] for r in cli.REFERENCES.values()}) == 3
    for r in cli.REFERENCES.values():
        assert len(r['source']) == 40
        assert all(len(r[key]) == 64 for key in ('checkpoint', 'summary', 'manifest'))
