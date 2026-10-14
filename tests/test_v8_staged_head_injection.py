"""Historical compatibility and explicit argument dispatch, CPU only."""
from types import SimpleNamespace
import sys

from mjlab_microduck import v8_gait_scale_contrast as driver
from mjlab_microduck import v8_staged_head_goals as pilot


def test_driver_preflight_accepts_explicit_argv_without_changing_process_arguments(monkeypatch):
    names = ("ARMS MODES PROTOCOL CAPS PARENT_SHA verify_source _read_parent "
             "recipe_match prepare_config frozen_profile_report Observer initialize "
             "validate_training_receipt validate_evaluation_receipt").split()
    recipe = SimpleNamespace(**{name: getattr(pilot, name) for name in names})
    seen = []
    recipe.verify_source = lambda args, **kwargs: seen.append((args.source, kwargs)) or {}
    recipe._read_parent = lambda: {
        "actor_state_dict": {}, "critic_state_dict": {}, "optimizer_state_dict": {}}
    recipe.recipe_match = lambda mode: {"mode": mode}
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(driver.torch.cuda, "is_initialized", lambda: False)
    external_args = ["unrelated-entrypoint", "--not-a-duck-argument"]
    monkeypatch.setattr(sys, "argv", external_args)
    report = driver._main_impl(recipe, argv=["--source", "a" * 40,
        "--run-id", "explicit-preflight", "--preflight"])
    assert report["source"] == "a" * 40
    assert seen == [("a" * 40, {"training": False})]
    assert sys.argv is external_args


def test_pilot_explicit_dispatch_preserves_injected_recipe_and_historical_globals(monkeypatch):
    seen = []
    recipe = SimpleNamespace(MODES=pilot.MODES, ROOT=pilot.ROOT)
    monkeypatch.setattr(driver, "main", lambda **kwargs: seen.append(kwargs) or "delegated")
    external_args = ["another-caller", "--different"]
    monkeypatch.setattr(sys, "argv", external_args)
    args = ["--source", "b" * 40, "--run-id", "explicit", "--preflight"]
    assert pilot.main(argv=args, experiment=recipe) == "delegated"
    assert seen == [{"experiment": recipe, "argv": args}]
    assert sys.argv is external_args
    assert pilot.MODES["foundation"] == (256, 1000, 967)


def test_historical_recipe_remains_json_equal_to_sealed_pilot():
    import json
    import os
    from pathlib import Path

    # Reuse only the small retained summary, never duplicate remote checkpoints.
    default = Path(__file__).resolve().parents[1] / (
        "artifacts/training/v8-staged-head-goals-bb223f66-20261014-closeout/head-goals-summary.json")
    packet = Path(os.environ.get("MICRODUCK_PILOT_SUMMARY_FOR_TEST", str(default)))
    if not packet.is_file():
        import pytest
        pytest.skip("small historical seal is optional on fresh checkouts")
    sealed = json.loads(packet.read_text())
    for mode in pilot.MODES:
        assert pilot.recipe_match(mode) == sealed["training"]["control"][mode]["recipe_match"]


def test_replication_run_identity_cannot_use_another_seed_prefix():
    import pytest
    from mjlab_microduck import v8_staged_head_replication as replication

    recipe = replication.Experiment(977)
    identity = SimpleNamespace(source="a" * 40,
        run_id="v8-staged-head-replication-s983-aaaaaaaa-20261014")
    with pytest.raises(ValueError, match="exact source-bound"):
        recipe.verify_source(identity, training=False)


def test_all_replication_stage_receipts_bind_prospective_and_actual_stage_seeds():
    import json
    from mjlab_microduck import v8_staged_head_replication as replication

    recipe = replication.Experiment(977)
    for mode, (_worlds, _updates, seed) in recipe.MODES.items():
        proof = recipe.recipe_match(mode)
        assert proof["learning_seed"] == 977 and proof["stage_seed"] == seed
        assert json.loads(json.dumps(proof)) == proof
        profile = recipe.frozen_profile_report("b" * 40, "staged", mode)
        assert profile["learning_seed"] == 977 and profile["stage_seed"] == seed
        assert profile["stage_spec"] == list(recipe.MODES[mode])
