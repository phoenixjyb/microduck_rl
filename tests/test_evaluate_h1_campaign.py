import importlib.util
from pathlib import Path

import pytest


_SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_h1_campaign.py"
_SPEC = importlib.util.spec_from_file_location("evaluate_h1_campaign", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
campaign = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(campaign)


def test_parse_training_run_requires_expected_seed_and_existing_directory(tmp_path):
    seed, path = campaign.parse_training_run(f"47={tmp_path}")
    assert seed == 47
    assert path == tmp_path.resolve()
    with pytest.raises(ValueError, match="unexpected training seed"):
        campaign.parse_training_run(f"61={tmp_path}")
    with pytest.raises(ValueError, match="SEED="):
        campaign.parse_training_run(str(tmp_path))


def test_preflight_requires_every_common_checkpoint_before_gpu_work(tmp_path):
    runs = {}
    for seed in campaign.TRAINING_SEEDS:
        run = tmp_path / str(seed)
        run.mkdir()
        runs[seed] = run
    with pytest.raises(FileNotFoundError, match="model_500.pt"):
        campaign.preflight_checkpoints(runs)

    for run in runs.values():
        for iteration in campaign.CHECKPOINT_ITERATIONS:
            (run / f"model_{iteration}.pt").write_bytes(b"checkpoint")
    checkpoints = campaign.preflight_checkpoints(runs)
    assert len(checkpoints) == 3 * len(campaign.CHECKPOINT_ITERATIONS)
    assert checkpoints[(59, 7999)].name == "model_7999.pt"


def test_preflight_rejects_partial_training_seed_set(tmp_path):
    with pytest.raises(ValueError, match="exactly"):
        campaign.preflight_checkpoints({47: tmp_path, 53: tmp_path})
