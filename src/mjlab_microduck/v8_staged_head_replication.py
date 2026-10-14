"""Matched prospective learning-seed replications of staged head goals.

Each :class:`Experiment` is an immutable recipe binding. The rejected pilot
remains a source of schedule/configuration only; its learner endpoint is never
used as a parent. This module does not itself authorize policy acceptance or
physical motion.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from . import integrated_turn_foundation as base
from . import v8_head_goal_schedule as schedule
from . import v8_staged_head_goals as pilot


PROTOCOL = "v8-staged-head-replication-v1"
LEARNING_SEEDS = (977, 983)
AUTHORIZED_LEARNING_SEED = 977
PILOT_SOURCE = "bb223f668c556de2dc651ef40a381ac5fcd5a0ad"
PILOT_SUMMARY_SHA256 = "a81efd4ea694dd0d5147bd45c575544c273f910e7f5470de6c186a4bf1fc2b15"
PILOT_MANIFEST_SHA256 = "dc420efcea50e0f355d715f29a88c77c005f1062ba515db794762ddaf4871f6b"
PILOT_HANDOFF = Path("/home/converge/.local/state/v8-staged-head-goals-bb223f66-20261014-handoff")
PILOT_SUMMARY = PILOT_HANDOFF / "head-goals-summary.json"
PILOT_MANIFEST = PILOT_HANDOFF / "head-goals-manifest.json"

ARMS = pilot.ARMS
PARENT = pilot.PARENT
PARENT_SHA = pilot.PARENT_SHA
PARENT_ITERATION = pilot.PARENT_ITERATION
PARENT_STEPS = pilot.PARENT_STEPS
PARENT_ADAM_STEP = pilot.PARENT_ADAM_STEP
PARENT_CRITIC_COUNT = pilot.PARENT_CRITIC_COUNT
TAUS = pilot.TAUS
WEIGHT = pilot.WEIGHT
CAPS = {"smoke": 300, "benchmark": 300, "foundation": 2100, "final-eval": 600}
ROOT = pilot.ROOT


def _require(condition: bool, message: str) -> None:
    base.require(condition, message)


@dataclass(frozen=True)
class Experiment:
    """One fixed prospective learning-seed binding for the shared pilot recipe."""

    learning_seed: int

    def __post_init__(self) -> None:
        _require(type(self.learning_seed) is int and self.learning_seed in LEARNING_SEEDS,
                 "learning seed is one of the two predeclared prospective seeds")

    @property
    def PROTOCOL(self) -> str:
        return PROTOCOL

    @property
    def RUN_PREFIX(self) -> str:
        return f"v8-staged-head-replication-s{self.learning_seed}"

    @property
    def ARMS(self):
        return ARMS

    @property
    def MODES(self):
        return {"smoke": (64, 5, 947), "benchmark": (256, 10, 953),
                "foundation": (256, 1000, self.learning_seed)}

    @property
    def STAGE_SPECS(self):
        # Use the parent's ordinary mode matrix, while binding only foundation
        # to this immutable experiment's prospective learning seed.
        return None

    @property
    def CAPS(self):
        return dict(CAPS)

    @property
    def PARENT(self):
        return PARENT

    @property
    def PARENT_SHA(self):
        return PARENT_SHA

    @property
    def PARENT_ITERATION(self):
        return PARENT_ITERATION

    @property
    def PARENT_STEPS(self):
        return PARENT_STEPS

    @property
    def PARENT_ADAM_STEP(self):
        return PARENT_ADAM_STEP

    @property
    def PARENT_CRITIC_COUNT(self):
        return PARENT_CRITIC_COUNT

    @property
    def TAUS(self):
        return dict(TAUS)

    @property
    def WEIGHT(self):
        return WEIGHT

    @property
    def ROOT(self):
        return ROOT

    @property
    def DECISION_USEFUL(self):
        return "staged-head-replication-useful-not-qualified"

    @property
    def DECISION_REJECTED(self):
        return "staged-head-replication-rejected"

    @property
    def DECISION_FILENAME(self):
        return "staged-head-replication-decision.json"

    @property
    def COMPARISON_METADATA(self):
        return {"learning_seed": self.learning_seed,
                "pilot_summary_sha256": PILOT_SUMMARY_SHA256}

    @property
    def KL_PROFILE(self):
        return "staged-head-replication"

    def require_admission(self, *, training: bool) -> None:
        _require(type(training) is bool, "explicit admission purpose")
        if training:
            _require(self.learning_seed == AUTHORIZED_LEARNING_SEED,
                     "seed 983 requires separately reviewed seed 977 sealed decision")

    def verify_source(self, args, *, training: bool):
        self.require_admission(training=training)
        return pilot.driver.verify_source(args, training=training, run_prefix=self.RUN_PREFIX)

    def _read_parent(self):
        """Authenticate the rejected pilot seal before the original parent."""
        summary = json.loads(pilot.headroom._owned_payload(
            PILOT_SUMMARY, PILOT_SUMMARY_SHA256))
        manifest = json.loads(pilot.headroom._owned_payload(
            PILOT_MANIFEST, PILOT_MANIFEST_SHA256))
        comparison = summary.get("comparison", {})
        groups = comparison.get("groups", {})
        failures = comparison.get("failures")
        decision_file = "staged-head-goals-decision.json"
        decision_sha = summary.get("comparison_sha256")
        inventory = manifest.get("files", {})
        decision_entry = inventory.get(decision_file, {})
        total_bytes = manifest.get("total_bytes")
        _require(summary.get("source") == PILOT_SOURCE
                 and summary.get("protocol") == pilot.PROTOCOL
                 and summary.get("status") == "verified-evidence-not-policy-acceptance"
                 and comparison.get("decision") == "staged-head-goals-lesson-rejected"
                 and failures == [["control", "original_parent",
                                   "walking-idle-retention-failure"]]
                 and summary.get("parent_checkpoint_sha256") == PARENT_SHA
                 and manifest.get("source") == PILOT_SOURCE
                 and decision_entry.get("sha256") == decision_sha
                 and type(total_bytes) is int and 1 <= total_bytes <= 1_000_000_000
                 and groups.get("control", {}).get("complete_moving_route_worlds") == 7
                 and groups.get("staged", {}).get("complete_moving_route_worlds") == 13
                 and groups.get("control", {}).get("moving_speed_passes") == 67
                 and groups.get("staged", {}).get("moving_speed_passes") == 72
                 and summary.get("policy_acceptance") is False
                 and summary.get("physical_motion_authorized") is False,
                 "authenticated rejected staged-head pilot and exact original-parent retention failure")
        # The pilot final endpoints remain diagnostic evidence only. The
        # inherited reader resolves and authenticates the original headroom
        # checkpoint after this seal has passed.
        return pilot._read_parent()

    def prepare_config(self, mode, arm):
        _require(mode in self.MODES and arm in self.ARMS, "declared replication mode and arm")
        cfg, agent = pilot.prepare_config(mode, arm)
        seed = self.MODES[mode][2]
        cfg.seed = seed
        agent.seed = seed
        agent.experiment_name = self.PROTOCOL
        agent.run_name = mode
        return cfg, agent

    def _canonical(self, mode, arm, *, mask):
        return pilot._canonical(mode, arm, mask=mask, experiment=self)

    def recipe_match(self, mode):
        value = pilot.recipe_match(mode, experiment=self)
        value["learning_seed"] = self.learning_seed
        value["stage_seed"] = self.MODES[mode][2]
        value["protocol"] = self.PROTOCOL + ":recipe-v1"
        return value

    def frozen_profile_report(self, source, arm, mode):
        value = pilot.frozen_profile_report(source, arm, mode, experiment=self)
        value["learning_seed"] = self.learning_seed
        value["stage_seed"] = self.MODES[mode][2]
        value["pilot_summary_sha256"] = PILOT_SUMMARY_SHA256
        return value

    def initialize(self, runner, env, output):
        self.require_admission(training=True)
        return pilot.initialize(runner, env, output, experiment=self)

    def validate_training_receipt(self, root, arm, mode, source):
        return pilot.validate_training_receipt(root, arm, mode, source, experiment=self)

    @staticmethod
    def validate_head_contract(env):
        return pilot.validate_head_contract(env)

    @staticmethod
    def capture_head_reward_snapshot(env, expected_step=None):
        return pilot.capture_head_reward_snapshot(env, expected_step=expected_step)

    def validate_evaluation_receipt(self, root, arm, source, checkpoint_sha):
        return pilot.validate_evaluation_receipt(root, arm, source, checkpoint_sha,
                                                 experiment=self)

    def diagnostic_comparison(self, control, staged):
        return pilot.diagnostic_comparison(control, staged, experiment=self)

    def compare_retained(self, source, run_id):
        self.require_admission(training=True)
        return pilot.compare_retained(source, run_id, experiment=self)

    def _boundary_smoke(self, argv):
        return pilot._boundary_smoke(argv, experiment=self)

    def _require_boundary_receipts(self, root, source):
        return pilot._require_boundary_receipts(root, source, experiment=self)

    def main(self, argv=None):
        argv = list(sys.argv[1:] if argv is None else argv)
        if argv and argv[0] in ("compare", "boundary-smoke"):
            command = argv.pop(0)
        else:
            command = None
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--learning-seed", type=int, required=True)
        args, remaining = parser.parse_known_args(argv)
        _require(args.learning_seed == self.learning_seed,
                 "CLI learning seed matches immutable experiment binding")
        if command == "compare":
            self.require_admission(training=True)
            return pilot.main(argv=["compare", *remaining], experiment=self)
        if command == "boundary-smoke":
            self.require_admission(training=True)
            return self._boundary_smoke(remaining)
        # A preflight is CPU-only and does not launch a learner. Seed 983 can
        # be configured and preflighted, but its training modes remain gated.
        if "--preflight" not in remaining:
            self.require_admission(training=True)
        return pilot.main(argv=remaining, experiment=self)


class Observer(pilot.Observer):
    """Retain pilot telemetry while binding reports to replication protocol."""

    def __init__(self, mode, arm):
        super().__init__(mode, arm)
        self.protocol = PROTOCOL
        self.receipt_profile = "staged-head-replication"


# The injected driver resolves this class from each immutable Experiment.
Experiment.Observer = Observer


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--learning-seed", type=int, required=True)
    known, _rest = parser.parse_known_args(args[1:] if args and args[0] in
                                           ("compare", "boundary-smoke") else args)
    experiment = Experiment(known.learning_seed)
    return experiment.main(args)


if __name__ == "__main__":  # pragma: no cover - exercised through module CLI
    main()
