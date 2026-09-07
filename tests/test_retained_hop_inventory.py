"""Historical rejection reconciliation, not a new simulator evaluation."""

from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest

from mjlab_microduck.retained_hop_inventory import (
    BASELINE, EVAL, HOST_ROOT, ITERATIONS, MANIFEST, TRAIN,
    audit, reconcile_causal,
)


REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def retained_root():
    for root in (REPO / "artifacts/retained/h1t-seed67-v1", REPO):
        if (root / TRAIN / "model_5999.pt").is_file():
            return root
    pytest.skip("selected historical H1-T artifacts not present")


def synthetic_comparison():
    return dict(
        baseline=dict(path=HOST_ROOT + BASELINE, checkpoint="unchanged/model_5999.pt",
                      sha256="a" * 64, gates=dict(falls=23)),
        candidate=dict(path=HOST_ROOT + EVAL + "/iteration-5999/hop-checkpoint-evaluation.json",
                       checkpoint="unchanged/model_5999.pt", sha256="b" * 64,
                       gates=dict(falls=4)),
        comparisons=[dict(name="episode_pass", status="fail", candidate=0.0)],
        decision="stop", physical_motion_authorized=False,
    )


def test_only_reader_locations_may_change():
    stored = synthetic_comparison()
    computed = deepcopy(stored)
    computed["baseline"]["path"] = "/mac/mirrored/baseline.json"
    computed["candidate"]["path"] = "/mac/mirrored/candidate.json"
    before = deepcopy(stored), deepcopy(computed)
    result = reconcile_causal(stored, computed)
    assert result["decision"] == "stop"
    assert result["baseline"]["path"] == BASELINE
    assert (stored, computed) == before


@pytest.mark.parametrize("change", ["metric", "status", "decision", "hash", "checkpoint", "extra", "nan"])
def test_location_reconciliation_never_hides_nonpath_changes(change):
    stored = synthetic_comparison()
    computed = deepcopy(stored)
    if change == "metric":
        computed["candidate"]["gates"]["falls"] = 3
    elif change == "status":
        computed["comparisons"][0]["status"] = "pass"
    elif change == "decision":
        computed["decision"] = "advance_to_multi_seed"
    elif change == "hash":
        computed["baseline"]["sha256"] = "0" * 64
    elif change == "checkpoint":
        computed["candidate"]["checkpoint"] = "/different/model_5999.pt"
    elif change == "extra":
        computed["extra"] = True
    else:
        computed["candidate"]["gates"]["falls"] = float("nan")
    with pytest.raises(ValueError):
        reconcile_causal(stored, computed)


def test_historical_location_is_not_rewritten_to_justify_a_match():
    stored = synthetic_comparison()
    stored["baseline"]["path"] = "/wrong/baseline.json"
    with pytest.raises(ValueError, match="historical report path changed"):
        reconcile_causal(stored, stored)


def test_a_matching_synthetic_pass_cannot_reopen_closed_hop():
    stored = synthetic_comparison()
    stored["decision"] = "advance_to_multi_seed"
    with pytest.raises(ValueError, match="closed H1-T rejection"):
        reconcile_causal(stored, stored)


def test_actual_selected_artifacts_reproduce_all_six_rejections(retained_root, monkeypatch):
    import torch
    initialized = torch.cuda.is_initialized()

    def forbidden(*args, **kwargs):
        raise AssertionError("checkpoint deserialization is outside this audit")

    monkeypatch.setattr(torch, "load", forbidden)
    result = audit(retained_root, REPO)
    assert torch.cuda.is_initialized() is initialized
    assert result["selected_payload_count"] == 11
    assert result["decision"] == "historical-hop-rejection-reproduced"
    assert [row["iteration"] for row in result["evaluations"]] == list(ITERATIONS)
    assert all(row["decision"] == "rejected" for row in result["evaluations"])
    assert result["descriptor"] is None
    assert result["declared_mechanics_group"] == "sprung-k3900-not-rigid-locomotion"
    assert all(value is False for key, value in result.items()
               if key.endswith(("_verified", "_authorized")) or key == "policy_acceptance")
    failed = [item["name"] for item in result["causal_comparison"]["comparisons"]
              if item["status"] == "fail"]
    assert failed == ["episode_pass_above_zero", "drift_below_0_30_m", "rated_speed_does_not_regress"]
    final_failures = [gate["name"] for gate in result["evaluations"][-1]["gates"]
                      if gate["status"] == "fail"]
    assert final_failures == ["episode_pass_fraction", "falls", "drift", "rated_speed_exceedance",
                              "near_stall_fraction"]


@pytest.mark.parametrize("target", [f"{TRAIN}/model_5999.pt", f"{TRAIN}/params/env.yaml",
    f"{TRAIN}/params/agent.yaml", BASELINE, f"{EVAL}/h1t-vs-h1p-causal-gate.json"] +
    [f"{EVAL}/iteration-{i}/hop-checkpoint-evaluation.json" for i in ITERATIONS])
def test_each_selected_payload_is_bound(retained_root, tmp_path, target):
    manifest = json.loads((REPO / MANIFEST).read_text())
    for name in manifest["payloads"]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(retained_root / name, path)
    changed = tmp_path / target
    changed.write_bytes(b"x" * changed.stat().st_size)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        audit(tmp_path, REPO)


def test_inventory_cannot_be_changed_to_bless_new_evidence(retained_root, tmp_path):
    path = tmp_path / MANIFEST
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    with pytest.raises(ValueError, match="pinned hop inventory"):
        audit(retained_root, tmp_path)
