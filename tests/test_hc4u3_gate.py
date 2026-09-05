import json

import pytest

from mjlab_microduck.hc4u3_gate import STAGE, compare_hc4u3_prescreen
from test_hc4u1_gate import _reports


def reports(tmp_path, **kwargs):
    paths = _reports(tmp_path, candidate_stage=STAGE, candidate_sha256="a" * 64, seed=293, **kwargs)
    for path in paths:
        payload = json.loads(path.read_text())
        payload["obstacle_sensor_model"] = {field: 0.0 for field in (
            "range_noise_m", "bearing_noise_rad", "width_noise_m", "height_noise_m",
            "closing_rate_noise_mps", "dropout_probability",
        )}
        path.write_text(json.dumps(payload))
    return paths


def test_new_phase_candidate_keeps_local_timeout_gate(tmp_path):
    paths = reports(tmp_path, candidate_overrides={
        (.30, 1.15, 0.0): {"clean_pass_events": 63, "attempt_timeout_events": 1},
    })
    result = compare_hc4u3_prescreen(*paths, candidate_sha256="a" * 64, seed=293)
    assert result["decision"] == "stop"
    assert any(c["name"] == "per_cell_timeout_non_regression" and c["status"] == "fail" for c in result["checks"])


def test_new_phase_candidate_accepts_only_declared_identity_and_seed(tmp_path):
    paths = reports(tmp_path)
    assert compare_hc4u3_prescreen(*paths, candidate_sha256="a" * 64, seed=293)["decision"] == "continue_fresh_seeds"
    with pytest.raises(ValueError, match="predeclared"):
        compare_hc4u3_prescreen(*paths, candidate_sha256="a" * 64, seed=251)
    with pytest.raises(ValueError, match="supervisor checkpoint"):
        compare_hc4u3_prescreen(*paths, candidate_sha256="b" * 64, seed=293)


@pytest.mark.parametrize("changes", [{"clean_pass_events": 65}, {"recovery_route_speed_mps": float("nan")}])
def test_bad_evidence_never_passes(tmp_path, changes):
    paths = reports(tmp_path, candidate_overrides={(.30, 1.15, 0.0): changes})
    with pytest.raises(ValueError):
        compare_hc4u3_prescreen(*paths, candidate_sha256="a" * 64, seed=293)
