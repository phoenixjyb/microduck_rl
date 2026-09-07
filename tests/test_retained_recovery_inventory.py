"""CPU evidence tests; eight clean passages are not recovery/stop admission."""

from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest

from mjlab_microduck.retained_recovery_inventory import (
    ACTOR, EVAL, FAR, MANIFEST, NEAR, SOURCE, audit, verify_receipts,
)


REPO = Path(__file__).resolve().parents[1]
PAYLOADS = json.loads((REPO / MANIFEST).read_text())["payloads"]


@pytest.fixture
def retained_root():
    for root in (REPO / "artifacts/retained/recovery-seed379-v1", REPO):
        if (root / EVAL / "decision.json").is_file() and (root / ACTOR).is_file():
            return root
    pytest.skip("selected recovery artifacts not present")


def copy_selected(root, target):
    for name in PAYLOADS:
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, destination)


def synthetic_receipts():
    models, command = {"actor": "synthetic"}, ["python", "baseline-only"]
    launch = dict(source=SOURCE, protocol="recovery-cap-specialist-s379-v1", model_sha256=models)
    child = dict(index=0, returncode=None, command=command)
    run = dict(index=0, returncode=0, command=command)
    runtime = dict(runs=[run])
    return deepcopy((launch, child, runtime, run, models, command))


def test_one_synthetic_baseline_receipt_is_not_changed():
    args = synthetic_receipts()
    before = deepcopy(args)
    verify_receipts(*args)
    assert args == before


@pytest.mark.parametrize("change", ["extra-child", "wrong-index", "crash", "cap", "model", "source"])
def test_receipts_cannot_claim_treatment_or_different_frozen_policy(change):
    args = synthetic_receipts()
    launch, child, runtime, run, models, command = args
    if change == "extra-child":
        runtime["runs"].append(deepcopy(run))
    elif change == "wrong-index":
        run["index"] = 1
    elif change == "crash":
        run["returncode"] = 1
    elif change == "cap":
        run["command"] = command + ["--recovery-acceleration-mps2", "0.2"]
    elif change == "model":
        launch["model_sha256"] = {"actor": "different"}
    else:
        launch["source"] = "0" * 40
    with pytest.raises(ValueError):
        verify_receipts(*args)


def test_real_baseline_failure_never_becomes_cap_or_stop_evidence(retained_root, monkeypatch):
    import torch
    from mjlab_microduck import recovery_ab
    initialized = torch.cuda.is_initialized()

    def forbidden(*args, **kwargs):
        raise AssertionError("no checkpoint load or experiment launcher allowed")

    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(recovery_ab, "main", forbidden)
    result = audit(retained_root, REPO)
    assert torch.cuda.is_initialized() is initialized
    assert result["selected_payload_count"] == 11 and result["closed_directory_exact"] is True
    assert result["model_roles"][NEAR] == "executed-near-supervisor"
    assert result["model_roles"][FAR] == "declared-reference-only-not-executed"
    decision = result["historical_decision"]
    assert decision["decision"] == "numerical-gate-stop"
    assert decision["failures"] == ["recovery-window"]
    assert decision["reports"][0]["outcomes"]["clean_pass_events"] == 8
    assert result["recovery_measurement"]["counts"]["window-missed"] == 8
    assert result["descriptor"] is None
    assert all(value is False for key, value in result.items()
               if key.endswith(("_verified", "_executed", "_evaluated", "_authorized"))
               or key == "policy_acceptance")


@pytest.mark.parametrize("name", list(PAYLOADS))
def test_every_selected_payload_is_hash_bound(retained_root, tmp_path, name):
    copy_selected(retained_root, tmp_path)
    path = tmp_path / name
    path.write_bytes(b"x" * path.stat().st_size)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        audit(tmp_path, REPO)


@pytest.mark.parametrize("extra", ["a2-report.json", "b-report.json", "symlink"])
def test_extra_or_symlinked_evidence_cannot_extend_closed_prefix(retained_root, tmp_path, extra):
    copy_selected(retained_root, tmp_path)
    path = tmp_path / EVAL / extra
    if extra == "symlink":
        path.symlink_to(tmp_path / EVAL / "decision.json")
    else:
        path.write_text("{}")
    with pytest.raises(ValueError, match="inventory|symlink"):
        audit(tmp_path, REPO)


def test_changed_inventory_cannot_bless_replaced_artifacts(retained_root, tmp_path):
    path = tmp_path / MANIFEST
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    with pytest.raises(ValueError, match="pinned recovery inventory"):
        audit(retained_root, tmp_path)
