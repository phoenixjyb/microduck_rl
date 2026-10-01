"""Pinned release source-gap reporting; mocks graph inspection and never infer."""

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from mjlab_microduck import community_source_gaps as gaps
from mjlab_microduck.skill_compatibility import STATIC_FIELDS


EXPECTED_CARD_HASHES = {
    "walk": {
        "README.md": "98b45ea81164d1e1a1dd82255207053b15cd6c69d922a1c5cf3387ce604d4b74",
        "manifest.json": "622048c2c23ea58942023f66fd16b189a875fd169e88d85beb16ebbe63b20c94",
    },
    "hop": {
        "README.md": "108c65ce78acfdeee87ca115e6bf3e638fbb18b4bcbf42ba058c89da2a128198",
        "manifest.json": "c3e351d46a0401d437916aa7520127505b50065cc0cd3225fc8408357f345089",
    },
}


def test_card_hashes_are_exactly_pinned():
    assert gaps.CARD_HASHES == EXPECTED_CARD_HASHES


@pytest.mark.parametrize("payload,match", [
    (b'{"a":1,"a":2}', "duplicate"),
    (b'{"value":NaN}', "nonfinite"),
    (b'{"value":Infinity}', "nonfinite"),
    (b'{"value":1e400}', "Out of range float values"),
    (b"{", "Expecting|invalid manifest JSON"),
    (b"\xff", "invalid manifest JSON"),
    (b"[]", "manifest object"),
    (b"x" * (128 * 1024 + 1), "bounded"),
])
def test_manifest_parser_rejects_duplicate_nonfinite_overflow_and_malformed(payload, match):
    with pytest.raises(ValueError, match=match):
        gaps.parse_card(payload)


def _mock_static_inspection(monkeypatch):
    def verified(_root, record, _interface):
        return b"mocked-hash-bound-policy", {
            "artifact": {"sha256": record["sha256"], "bytes": record["bytes"]},
            "interface_classification": "feedforward-microduck-api1",
            "runtime_inputs": [{"name": "obs", "dtype": "float32", "shape": [1, 61]}],
            "outputs": [{"name": "actions", "dtype": "float32", "shape": [1, 14]}],
            "onnx": {"ir_version": 8, "opsets": [{"domain": "", "version": 18}]},
            "metadata": {"fixture": "static-mock-only"},
        }

    monkeypatch.setattr(gaps, "verified_policy", verified)
    monkeypatch.setattr(gaps, "inspect_affine_prefix", lambda *_: {
        "decision": "recognized-static-affine-prefix-only",
        "formula": "(obs - mean) / denominator",
        "affine_values": {"mean": [0.0] * 61, "denominator": [1.0] * 61},
        "actor_forward_executed": False,
    })


def _fixture_source_root(monkeypatch, tmp_path):
    root = tmp_path / "fixture-repo"
    cards = {}
    manifests = {
        "walk": {
            "schema_version": 2, "model_api": 1, "obs_len": 61, "action_len": 14,
            "policies": [{"file": "velstand.onnx", "entry_pose": "standing",
                          "training": {"branch": "protective_fall", "source_file": "velstand_best.onnx"}}],
        },
        "hop": {
            "schema_version": 2, "model_api": 1, "obs_len": 61, "action_len": 14,
            "name": "happy-hop", "training": {"upstream_base": "repo@commit", "run": "run-name"},
        },
    }
    policy_records = {}
    for kind in ("walk", "hop"):
        directory = root / "intake" / kind
        directory.mkdir(parents=True)
        readme = (kind + " fixture readme\n").encode()
        manifest = (json.dumps(manifests[kind], sort_keys=True) + "\n").encode()
        (directory / "README.md").write_bytes(readme)
        (directory / "manifest.json").write_bytes(manifest)
        (directory / "policy.onnx").write_bytes(b"not parsed by mocked verifier")
        cards[kind] = {
            "README.md": hashlib.sha256(readme).hexdigest(),
            "manifest.json": hashlib.sha256(manifest).hexdigest(),
        }
        policy_records[kind] = {
            "repo": "fixture/" + kind, "revision": "a" * 40,
            "local_path": f"intake/{kind}/policy.onnx",
            "sha256": ("b" if kind == "walk" else "c") * 64,
            "bytes": len(b"not parsed by mocked verifier"),
        }

    declaration = {
        "schema": "fixture-declaration-v1",
        "experiment_id": "fixture-source-gaps",
        "policies": policy_records,
        "interface": {"input": {"name": "obs", "dtype": "float32", "shape": [1, 61]},
                      "output": {"name": "actions", "dtype": "float32", "shape": [1, 14]}},
    }
    declaration_bytes = (json.dumps(declaration, sort_keys=True) + "\n").encode()
    monkeypatch.setattr(gaps, "CARD_HASHES", cards)
    monkeypatch.setattr(gaps, "load_declaration", lambda _root: (declaration, declaration_bytes))
    _mock_static_inspection(monkeypatch)
    return root, declaration, cards


def test_fixed_report_preserves_all_static_gaps_without_synthesized_descriptor(monkeypatch, tmp_path):
    root, _, _ = _fixture_source_root(monkeypatch, tmp_path)
    report = gaps.build_gap_report(root)
    assert report["protocol"] == "community-c1-source-gap-inventory-v1"
    assert report["decision"] == "author-source-handoff-required-no-policy-trial"
    assert set(report["candidates"]) == {"walk", "hop"}
    assert report["policy_inferences"] == report["simulation_steps"] == report["optimizer_updates"] == 0
    for candidate in report["candidates"].values():
        assert [gap["field"] for gap in candidate["gaps"]] == list(STATIC_FIELDS)
        assert all(gap["status"] == "effective-author-receipt-missing" for gap in candidate["gaps"])
        assert candidate["descriptor"] is None
        assert candidate["training_source_binding_verified"] is False
        assert len(candidate["supporting_cards"]) == 2
    assert all(value is False for key, value in report.items()
               if key.endswith(("verified", "authorized", "acceptance")))
    assert report["complete_skill_descriptor_available"] is False


def test_fixture_card_bytes_are_hashed_and_missing_or_changed_cards_fail(monkeypatch, tmp_path):
    root, _, fixture_hashes = _fixture_source_root(monkeypatch, tmp_path)
    report = gaps.build_gap_report(root)
    for kind, pinned in fixture_hashes.items():
        actual = report["candidates"][kind]["supporting_cards"]
        assert {name: item["sha256"] for name, item in actual.items()} == pinned

    (root / "intake" / "walk" / "README.md").write_bytes(b"changed fixture readme\n")
    with pytest.raises(ValueError, match="supporting card SHA256 mismatch"):
        gaps.build_gap_report(root)

    (root / "intake" / "walk" / "README.md").unlink()
    with pytest.raises(ValueError):
        gaps.build_gap_report(root)


def test_missing_or_wrong_candidate_is_not_guessed(monkeypatch):
    with pytest.raises(ValueError, match="known pinned candidate"):
        gaps.published_claims("other", {})
    with pytest.raises(ValueError, match="one official velstand entry"):
        gaps.published_claims("walk", {"schema_version": 2, "model_api": 1,
            "obs_len": 61, "action_len": 14, "policies": [{"file": "walking.onnx"}]})
    with pytest.raises(ValueError, match="published Happy Hop claims"):
        gaps.published_claims("hop", {"schema_version": 2, "model_api": 1,
            "obs_len": 61, "action_len": 14, "name": "other", "training": {}})


@pytest.mark.parametrize("field,value", [
    ("schema_version", 2.0), ("model_api", 1.0), ("model_api", True),
    ("obs_len", 61.0), ("action_len", 14.0),
])
def test_published_api_dimensions_require_exact_integer_types(field, value):
    manifest = {"schema_version": 2, "model_api": 1, "obs_len": 61, "action_len": 14,
                "name": "happy-hop", "training": {}}
    manifest[field] = value
    with pytest.raises(ValueError, match="published API1 declaration"):
        gaps.published_claims("hop", manifest)


@pytest.mark.parametrize("damage", ["missing", "wrong"])
def test_report_rejects_missing_or_wrong_pinned_candidate(monkeypatch, tmp_path, damage):
    root, fixture_declaration, _ = _fixture_source_root(monkeypatch, tmp_path)
    declaration = copy.deepcopy(fixture_declaration)
    payload = (json.dumps(declaration, sort_keys=True) + "\n").encode()
    if damage == "missing":
        del declaration["policies"]["hop"]
    else:
        declaration["policies"]["walk"]["local_path"] = (
            "intake/hop/policy.onnx")
    monkeypatch.setattr(gaps, "load_declaration", lambda _root: (declaration, payload))
    with pytest.raises((KeyError, ValueError)):
        gaps.build_gap_report(root)


def test_published_walker_entry_and_hop_upstream_are_not_source_identity():
    walk = gaps.published_claims("walk", {
        "schema_version": 2, "model_api": 1, "obs_len": 61, "action_len": 14,
        "policies": [{"file": "velstand.onnx", "entry_pose": "standing",
                      "training": {"branch": "protective_fall", "source_file": "velstand_best.onnx"}}],
    })
    hop = gaps.published_claims("hop", {
        "schema_version": 2, "model_api": 1, "obs_len": 61, "action_len": 14,
        "name": "happy-hop", "training": {"upstream_base": "repo@commit", "run": "run-name"},
    })
    assert walk["entry_pose"] == "standing"
    assert walk["training"]["branch"] == "protective_fall"
    assert walk["exact_training_commit"] is None
    assert hop["training"]["upstream_base"] == "repo@commit"
    assert hop["exact_training_commit"] is None


def test_handoff_states_official_walker_difference_and_missing_overlay(monkeypatch, tmp_path):
    root, _, _ = _fixture_source_root(monkeypatch, tmp_path)
    report = gaps.build_gap_report(root)
    walker = report["candidates"]["walk"]["published"]
    assert walker["exact_training_commit"] is None
    assert walker["entry_pose"] == "standing"
    overlay = next(item for item in report["handoff"] if item["field"] == "training_overlay")
    assert overlay["published_upstream_base"] == "d424a0c899f6b33cbd3daeb279913134349c0b63"
    assert overlay["published_run"] == "2026-08-31_20-39-21_happy_hop_clearance_35mm_stage1"
    assert "upstream base is not the missing overlay" in overlay["needed"]
    entry = next(item for item in report["handoff"] if item["field"] == "entry_policy")
    assert entry["published_filename"] == "walking_backlash_model_5000.onnx"
    assert "official velstand is not this identity" in entry["needed"]


def test_cli_reports_errors_as_json_and_nonzero(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(sys, "argv", ["community-source-gaps", str(tmp_path)])
    monkeypatch.setattr(gaps, "build_gap_report", lambda _root: (_ for _ in ()).throw(ValueError("fixture failure")))
    with pytest.raises(SystemExit) as exc:
        gaps.main()
    assert exc.value.code == 2
    assert json.loads(capsys.readouterr().out) == {"error": "fixture failure"}


def test_isolated_import_uses_only_standard_library_modules():
    module_names = "numpy,onnx,onnxruntime,mujoco,torch,warp,bam"
    command = (
        "import importlib,sys; importlib.import_module('mjlab_microduck.community_source_gaps'); "
        f"print(','.join(sorted(set(sys.modules) & set({module_names!r}.split(',')))))"
    )
    result = subprocess.run([sys.executable, "-c", command], text=True,
                            capture_output=True, check=True,
                            cwd=Path(__file__).resolve().parents[1])
    assert result.stdout.strip() == ""
