"""Evaluate and select every predeclared common checkpoint in the H1 campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from mjlab_microduck.hop_checkpoint_sweep import summarize_h1_checkpoint_sweep
from mjlab_microduck.hop_evaluation import H1_PROTOCOL


TRAINING_SEEDS = (47, 53, 59)
CHECKPOINT_ITERATIONS = (*range(500, 8000, 500), 7999)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_training_run(value: str) -> tuple[int, Path]:
    seed_text, separator, path_text = value.partition("=")
    if not separator or not seed_text or not path_text:
        raise ValueError("training run must use SEED=/absolute/run/directory")
    seed = int(seed_text)
    if seed not in TRAINING_SEEDS:
        raise ValueError(f"unexpected training seed {seed}")
    path = Path(path_text).expanduser().resolve(strict=True)
    if not path.is_dir():
        raise ValueError(f"training run is not a directory: {path}")
    return seed, path


def preflight_checkpoints(runs: dict[int, Path]) -> dict[tuple[int, int], Path]:
    if sorted(runs) != list(TRAINING_SEEDS):
        raise ValueError(f"training runs must contain exactly {list(TRAINING_SEEDS)}")
    checkpoints = {}
    for seed in TRAINING_SEEDS:
        for iteration in CHECKPOINT_ITERATIONS:
            checkpoint = runs[seed] / f"model_{iteration}.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            checkpoints[(seed, iteration)] = checkpoint.resolve()
    return checkpoints


def _evaluation_matches(path: Path, checkpoint: Path) -> bool:
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return (
        value.get("protocol") == H1_PROTOCOL
        and value.get("checkpoint_sha256") == _sha256(checkpoint)
        and value.get("seeds") == [211, 223, 227]
        and value.get("num_envs") == 128
        and value.get("cycles") == 6
    )


def evaluate_campaign(runs: dict[int, Path], output_dir: Path, *, device: str) -> dict:
    checkpoints = preflight_checkpoints(runs)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluator = Path(__file__).with_name("evaluate_hop_checkpoint.py").resolve(strict=True)
    candidates = []

    # Iteration-major order yields a useful partial campaign if execution is
    # interrupted: every completed block contains all three training seeds.
    for iteration in CHECKPOINT_ITERATIONS:
        for seed in TRAINING_SEEDS:
            checkpoint = checkpoints[(seed, iteration)]
            case_dir = output_dir / f"iteration-{iteration}" / f"training-seed-{seed}"
            evaluation = case_dir / "hop-checkpoint-evaluation.json"
            if not _evaluation_matches(evaluation, checkpoint):
                command = [
                    sys.executable,
                    str(evaluator),
                    str(checkpoint),
                    "--output-dir",
                    str(case_dir),
                    "--device",
                    device,
                ]
                subprocess.run(command, check=True)
            candidates.append(
                {
                    "training_seed": seed,
                    "checkpoint_iteration": iteration,
                    "hop_evaluation": str(evaluation),
                }
            )

    manifest = {
        "schema_version": 1,
        "campaign_id": "h1-k3900-6c4e470-seeds47-53-59",
        "protocol": H1_PROTOCOL,
        "training_source_commit": "6c4e470e658968d63ad246644f4c5ba02e89bd74",
        "training_seeds": list(TRAINING_SEEDS),
        "checkpoint_iterations": list(CHECKPOINT_ITERATIONS),
        "candidates": candidates,
        "physical_motion_authorized": False,
    }
    manifest_path = output_dir / "h1-campaign-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    summary = summarize_h1_checkpoint_sweep(manifest_path)
    summary_path = output_dir / "h1-checkpoint-sweep.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"h1_campaign_decision={summary['decision']}")
    print(f"h1_campaign_manifest={manifest_path}")
    print(f"h1_campaign_summary={summary_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-run",
        action="append",
        required=True,
        help="repeat exactly three times as SEED=/absolute/run/directory",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    parsed = [parse_training_run(value) for value in args.training_run]
    runs = dict(parsed)
    if len(runs) != len(parsed):
        parser.error("training seeds must not be repeated")
    evaluate_campaign(runs, args.output_dir, device=args.device)


if __name__ == "__main__":
    main()
