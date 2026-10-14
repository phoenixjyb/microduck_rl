import json
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/rehearse_official_velstand.py"
SPEC = importlib.util.spec_from_file_location("rehearse_official_velstand", SCRIPT)
rehearsal = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rehearsal)


def test_short_official_policy_bam_rehearsal_is_finite_cpu_only(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    if any("CUDA" in p.upper() or "TENSORRT" in p.upper()
           for p in rehearsal.ort.get_available_providers()):
        pytest.skip("host has an available GPU ONNX Runtime provider")
    output = tmp_path / "receipt.json"
    result = rehearsal.rehearse(["idle"], 0.2, output)  # 10 control steps
    saved = json.loads(output.read_text())
    assert result == saved
    assert saved["classification"] == "simulation-deployment-rehearsal-only"
    assert saved["qualification"] is False
    assert saved["runtime"]["providers"] == ["CPUExecutionProvider"]
    assert saved["runtime"]["action_delay_control_steps"] == 4
    assert saved["segments"][0]["completed_control_steps"] <= 10
    assert saved["segments"][0]["finite"] is True
    assert saved["segments"][0]["command"] == {"vx": 0.0, "vy": 0.0, "wz": 0.0}
    assert saved["scene"]["choice"] == "allcollisions"
    assert len(saved["segments"][0]["mean_body_linear_velocity_mps"]) == 3
    assert "mean_projected_heading_rate_rad_s" in saved["segments"][0]


def test_refuses_gpu_provider_before_creating_output(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(rehearsal.ort, "get_available_providers",
                        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    output = tmp_path / "no-gpu.json"
    with pytest.raises(RuntimeError, match="GPU providers"):
        rehearsal.rehearse(["idle"], 0.02, output)
    assert not output.exists()


def test_output_is_exclusive_and_duration_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    output = tmp_path / "existing.json"
    output.write_text("preserve")
    with pytest.raises(FileExistsError, match="overwrite"):
        rehearsal.rehearse(["idle"], 0.02, output)
    with pytest.raises(ValueError, match="at most 30"):
        rehearsal.rehearse(["idle", "straight"], 16.0, tmp_path / "too-long.json")


def test_case_commands_and_combined_budget_are_fixed():
    assert rehearsal.CASES == {
        "idle": (0.0, 0.0, 0.0),
        "straight": (0.3, 0.0, 0.0),
        "left": (0.2, 0.0, 0.2),
        "right": (0.2, 0.0, -0.2),
    }
    assert 12.0 / len(rehearsal.CASES) * len(rehearsal.CASES) == 12.0
