"""Profile identity and matched-initialization checks for cautious v8 learning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
from collections.abc import Mapping
import stat
from typing import Any

import torch

from . import integrated_turn_foundation as base
from . import official_native_reference as pinned
from . import community_policy_inspection as inspection


CAUTIOUS = "cautious"
STOCK = "stock"
CAUTIOUS_PROTOCOL = "v8-cautious-initial-lr-v1"
CAUTIOUS_RUN_PREFIX = "v8-cautious-lr"
CAUTIOUS_INITIAL_LR = 1e-4
STOCK_INITIAL_LR = 1e-3
FIRST_KL_MEAN_MAX = .05
FIRST_KL_MINIBATCH_MEAN_MAX = .10

CAUTIOUS_MODES = {
    "smoke": (64, 5, 947),
    "benchmark": (256, 10, 953),
    "foundation": (256, 1000, 967),
}
CAUTIOUS_CAPS = {
    "smoke": 300, "benchmark": 300, "initial-eval": 600,
    "foundation": 2100, "final-eval": 600,
}


@dataclass(frozen=True)
class LearningProfile:
    name: str
    protocol: str
    run_prefix: str
    initial_lr: float
    modes: dict[str, tuple[int, int, int]]
    caps: dict[str, int]
    matched_start: bool = False
    first_update_kl_guard: bool = False


def profile(name: str) -> LearningProfile:
    if name == STOCK:
        # Imported lazily to keep stock constants single-sourced and unchanged.
        from . import v8_actor_learning as learning
        return LearningProfile(STOCK, learning.PROTOCOL, "v8-actor-foundation",
                               STOCK_INITIAL_LR, learning.MODES, learning.CAPS)
    if name == CAUTIOUS:
        return LearningProfile(CAUTIOUS, CAUTIOUS_PROTOCOL, CAUTIOUS_RUN_PREFIX,
                               CAUTIOUS_INITIAL_LR, CAUTIOUS_MODES, CAUTIOUS_CAPS,
                               matched_start=True, first_update_kl_guard=True)
    raise ValueError("profile must be exactly 'stock' or 'cautious'")


@dataclass(frozen=True)
class ControlInitial:
    tag: str
    source: str
    checkpoint_sha256: str
    path: Path
    seed: int


def control_initial(mode: str) -> ControlInitial:
    root = base.ROOT / "artifacts/training"
    controls = {
        "smoke": ControlInitial(
            "stock-v8-smoke-c178fbcc", "c178fbcc53c9d7e16d9bd09b5b21896fecf18645",
            "1f4c8b3c1538ce88c1cb0409658c998ce0d6454b960214775e422015d898b2e3",
            root / "v8-actor-smoke-c178fbcc-20261013" / "initial.pt", 947),
        "benchmark": ControlInitial(
            "stock-v8-foundation-benchmark-722e7a6a", "722e7a6ac6f9cf2995cbab37b7b13aa14e7c218a",
            "80084814096f99a56a11f78d6e0526c99261b1d0eaae3a57da79517733b61163",
            root / "v8-actor-foundation-722e7a6a-20261013" / "benchmark" / "initial.pt", 953),
        "foundation": ControlInitial(
            "stock-v8-foundation-foundation-722e7a6a", "722e7a6ac6f9cf2995cbab37b7b13aa14e7c218a",
            "29d3ea8491c84d57db12ea37ffad9d586fa8ab582be621e3028e5f27055ba974",
            root / "v8-actor-foundation-722e7a6a-20261013" / "foundation" / "initial.pt", 967),
    }
    if mode not in controls:
        raise ValueError("matched initialization exists only for cautious learning stages")
    return controls[mode]


def _canonical(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        return {"tensor_dtype": str(tensor.dtype), "tensor_shape": list(tensor.shape),
                "tensor_sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest()}
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, slice):
        return {"slice": [_canonical(value.start), _canonical(value.stop), _canonical(value.step)]}
    if isinstance(value, Path):
        return str(value)
    if callable(value):
        return f"{value.__module__}:{value.__qualname__}"
    if value is None or type(value) in (str, int, float, bool):
        return value
    if hasattr(value, "value") and type(getattr(value, "value")) in (str, int, float):
        return value.value
    raise ValueError(f"unsupported config value in matched profile: {type(value).__name__}")


def _config_json(cfg: Any, agent: Any, *, mask_allowed: bool) -> str:
    cfg_value, agent_value = asdict(cfg), asdict(agent)
    if mask_allowed:
        agent_value["experiment_name"] = "<profile-protocol>"
        agent_value["run_name"] = "<stage>"
        algorithm = agent_value.get("algorithm")
        base.require(isinstance(algorithm, dict) and "learning_rate" in algorithm,
                     "algorithm learning-rate profile field")
        algorithm["learning_rate"] = "<profile-initial-lr>"
    return json.dumps(_canonical({"environment": cfg_value, "agent": agent_value}),
                      sort_keys=True, separators=(",", ":"), allow_nan=False)


def require_stock_recipe_match(mode: str) -> dict[str, str]:
    """Prove cautious differs from stock only in protocol identity and initial LR."""
    from . import v8_actor_learning as learning

    if mode in learning.MODES:
        stock_cfg, stock_agent = learning.prepare_config(mode, profile=STOCK)
    elif mode == "smoke":
        from . import v8_actor_smoke as smoke
        stock_cfg, stock_agent = smoke.prepare_config()
    else:
        raise ValueError("unknown cautious learning stage")
    cautious_cfg, cautious_agent = learning.prepare_config(mode, profile=CAUTIOUS)
    stock = _config_json(stock_cfg, stock_agent, mask_allowed=True)
    cautious = _config_json(cautious_cfg, cautious_agent, mask_allowed=True)
    base.require(stock == cautious,
                 "cautious recipe must match stock except profile identity and initial LR")
    return dict(stock_config_sha256=hashlib.sha256(stock.encode()).hexdigest(),
                cautious_config_sha256=hashlib.sha256(cautious.encode()).hexdigest(),
                profile_config_match=True)


def _state_digest(state: Any, label: str) -> str:
    base.require(isinstance(state, Mapping) and bool(state), f"nonempty {label} state mapping")
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        base.require(type(name) is str and isinstance(tensor, torch.Tensor)
                     and tensor.device.type == "cpu", f"CPU tensor in {label} state")
        value = tensor.detach().contiguous()
        descriptor = json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":"))
        digest.update(descriptor.encode())
        digest.update(b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _assert_state_equal(candidate: Any, control: Any, label: str) -> str:
    base.require(isinstance(candidate, Mapping) and isinstance(control, Mapping)
                 and set(candidate) == set(control), f"exact {label} state keys")
    for name in candidate:
        left, right = candidate[name], control[name]
        base.require(isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)
                     and left.device.type == right.device.type == "cpu"
                     and left.dtype == right.dtype and left.shape == right.shape
                     and torch.equal(left, right), f"matched {label} tensor {name}")
    candidate_sha = _state_digest(candidate, label)
    control_sha = _state_digest(control, f"control {label}")
    base.require(candidate_sha == control_sha, f"byte-identical {label} state hash")
    return candidate_sha


def _optimizer_groups(optimizer_state: Any, label: str) -> list[dict[str, Any]]:
    base.require(isinstance(optimizer_state, dict)
                 and isinstance(optimizer_state.get("state"), dict)
                 and not optimizer_state["state"], f"empty {label} Adam state")
    groups = optimizer_state.get("param_groups")
    base.require(isinstance(groups, list) and groups, f"{label} Adam parameter groups")
    identifiers = []
    result = []
    for group in groups:
        base.require(isinstance(group, dict) and "lr" in group
                     and isinstance(group.get("params"), list), f"{label} optimizer group layout")
        params = group["params"]
        base.require(all(type(parameter) is int for parameter in params),
                     f"serialized {label} parameter identifiers")
        identifiers.extend(params)
        result.append(group)
    base.require(len(identifiers) == 17 and len(set(identifiers)) == 17,
                 f"{label} optimizer must own 17 ordered unique parameters")
    return result


def _assert_only_lr_differs(candidate_optimizer: Any, control_optimizer: Any) -> str:
    candidate = _optimizer_groups(candidate_optimizer, "candidate")
    control = _optimizer_groups(control_optimizer, "control")
    base.require(len(candidate) == len(control), "matched optimizer group count")
    signature = []
    for new, old in zip(candidate, control, strict=True):
        base.require(set(new) == set(old), "identical optimizer group fields")
        base.require(new["params"] == old["params"], "identical ordered parameter identifiers")
        base.require(new["lr"] == CAUTIOUS_INITIAL_LR and old["lr"] == STOCK_INITIAL_LR,
                     "only declared optimizer LR values differ")
        base.require(all(new[key] == old[key] for key in new if key != "lr"),
                     "all optimizer fields except LR match")
        signature.append({key: _canonical(value) for key, value in new.items() if key != "lr"})
    encoded = json.dumps(signature, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _control_import_receipt(control: ControlInitial) -> dict[str, Any]:
    report_path = control.path.parent / "actor-import.json"
    try:
        report = json.loads(report_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("stock control actor-import receipt is unavailable") from exc
    base.require(report.get("artifact_sha256") == pinned.V8.sha256
                 and report.get("ppo_resume") is False
                 and report.get("actor_class") == "mjlab_microduck.v8_actor_seed:FrozenV8Actor"
                 and report.get("initial_learning_rate") == STOCK_INITIAL_LR
                 and report.get("target_iteration") == 0
                 and report.get("target_common_step_counter") == 0
                 and report.get("adam_state_entries") == 0
                 and report.get("critic_normalizer_count") == 0,
                 "stock control was saved at fresh zero clocks with empty Adam")
    return report


def matched_initialization(runner: Any, env: Any, mode: str, source: str) -> dict[str, Any]:
    """Authenticate a stock initial checkpoint and compare its full initial PPO state."""
    control = control_initial(mode)
    base.require(mode in CAUTIOUS_MODES and CAUTIOUS_MODES[mode][2] == control.seed,
                 "matched control and cautious run use the same declared seed")
    recipe = require_stock_recipe_match(mode)
    try:
        control_info = control.path.lstat()
    except OSError as exc:
        raise ValueError("retained stock control checkpoint is unavailable") from exc
    base.require(stat.S_ISREG(control_info.st_mode) and control_info.st_uid == os.getuid(),
                 "stock control initial checkpoint is an owned regular file")
    raw = inspection._read_payload(control.path)
    digest = hashlib.sha256(raw).hexdigest()
    base.require(digest == control.checkpoint_sha256, "exact retained stock control initial checkpoint")
    try:
        saved = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError("stock control initial checkpoint must be weights-only compatible") from exc
    base.require(isinstance(saved, dict)
                 and set(saved) == {"actor_state_dict", "critic_state_dict", "optimizer_state_dict", "iter", "infos"}
                 and type(saved["iter"]) is int and saved["iter"] == 0
                 and isinstance(saved["infos"], dict) and set(saved["infos"]) == {"env_state"}
                 and isinstance(saved["infos"]["env_state"], dict)
                 and set(saved["infos"]["env_state"]) == {"common_step_counter"}
                 and type(saved["infos"]["env_state"]["common_step_counter"]) is int
                 and saved["infos"]["env_state"]["common_step_counter"] == 0,
                 "stock control checkpoint has exact initial runner schema and iteration zero")
    _control_import_receipt(control)

    alg = runner.alg
    base.require(type(alg.actor).__name__ == "FrozenV8Actor"
                 and type(runner.current_learning_iteration) is int
                 and runner.current_learning_iteration == 0
                 and type(env.common_step_counter) is int and env.common_step_counter == 0,
                 "candidate exact actor class and fresh zero clocks")
    expected_seed_digest = torch.tensor(list(bytes.fromhex(pinned.V8.sha256)), dtype=torch.uint8)
    base.require(hasattr(alg.actor, "v8_seed_digest")
                 and torch.equal(alg.actor.v8_seed_digest.detach().cpu(), expected_seed_digest),
                 "candidate actor retains the exact pinned v8 seed identity")
    base.require(type(alg.optimizer) is torch.optim.Adam and not alg.optimizer.state,
                 "candidate has fresh empty Adam")
    base.require(type(alg.learning_rate) in (int, float)
                 and alg.learning_rate == CAUTIOUS_INITIAL_LR
                 and alg.schedule == "adaptive" and alg.desired_kl == .01
                 and alg.num_learning_epochs == 5 and alg.num_mini_batches == 4,
                 "cautious candidate keeps target PPO settings and initial LR")
    from .integrated_fresh_foundation import normalizer_count
    base.require(normalizer_count(alg.actor) == normalizer_count(alg.critic) == 0
                 and alg.actor.obs_normalizer.eps == 0.0 and alg.actor.obs_normalizer.until == 0,
                 "candidate actor is frozen and both model counters are empty")

    candidate_actor = {key: value.detach().cpu().clone() for key, value in alg.actor.state_dict().items()}
    candidate_critic = {key: value.detach().cpu().clone() for key, value in alg.critic.state_dict().items()}
    actor_sha = _assert_state_equal(candidate_actor, saved["actor_state_dict"], "actor")
    critic_sha = _assert_state_equal(candidate_critic, saved["critic_state_dict"], "critic")
    candidate_optimizer = alg.optimizer.state_dict()
    control_optimizer = saved["optimizer_state_dict"]
    optimizer_group_sha = _assert_only_lr_differs(candidate_optimizer, control_optimizer)
    return dict(protocol="v8-matched-initialization-v1", profile=CAUTIOUS,
                source=source, mode=mode, seed=control.seed, control_model_tag=control.tag,
                control_source=control.source, control_checkpoint_sha256=digest,
                stock_config_sha256=recipe["stock_config_sha256"],
                cautious_config_sha256=recipe["cautious_config_sha256"],
                actor_state_sha256=actor_sha, critic_state_sha256=critic_sha,
                optimizer_non_lr_groups_sha256=optimizer_group_sha,
                actor_state_equal=True, critic_state_equal=True,
                optimizer_state_empty=True, parameter_groups_match_except_lr=True,
                control_learning_rate=STOCK_INITIAL_LR,
                candidate_learning_rate=CAUTIOUS_INITIAL_LR,
                candidate_iteration=0, candidate_common_step_counter=0,
                control_iteration=0, control_common_step_counter=0,
                actor_normalizer_count=0, critic_normalizer_count=0,
                ppo_resume=False, tensors_in_report=False,
                trajectory_identity_claim=False, policy_acceptance=False,
                physical_motion_authorized=False)


def first_update_kl_report(mode: str, source: str, summary: dict[str, Any], *, steps: int) -> dict[str, Any]:
    mean_kl = summary["kl_mean_mean"]
    max_batch_mean = summary["kl_mean_max"]
    return dict(protocol="v8-first-update-kl-guard-v1", profile=CAUTIOUS,
                mode=mode, source=source, update=1, common_step_counter=steps,
                desired_kl=.01, initial_learning_rate=CAUTIOUS_INITIAL_LR,
                optimizer_minibatches=summary["optimizer_minibatches"],
                mean_raw_kl=mean_kl, maximum_minibatch_mean_raw_kl=max_batch_mean,
                mean_raw_kl_limit=FIRST_KL_MEAN_MAX,
                maximum_minibatch_mean_raw_kl_limit=FIRST_KL_MINIBATCH_MEAN_MAX,
                passed=(steps == 24 and summary["optimizer_minibatches"] == 20
                        and type(mean_kl) in (int, float) and type(max_batch_mean) in (int, float)
                        and math.isfinite(mean_kl) and math.isfinite(max_batch_mean)
                        and mean_kl <= FIRST_KL_MEAN_MAX
                        and max_batch_mean <= FIRST_KL_MINIBATCH_MEAN_MAX),
                policy_acceptance=False, physical_motion_authorized=False)


def validate_first_update_kl(root: str | Path, mode: str, source: str) -> dict[str, Any]:
    path = Path(root) / mode / "first-update-kl.json"
    try:
        report = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("cautious first-update KL receipt unavailable") from exc
    expected_keys = {
        "protocol", "profile", "mode", "source", "update", "common_step_counter",
        "desired_kl", "initial_learning_rate", "optimizer_minibatches", "mean_raw_kl",
        "maximum_minibatch_mean_raw_kl", "mean_raw_kl_limit",
        "maximum_minibatch_mean_raw_kl_limit", "passed", "policy_acceptance",
        "physical_motion_authorized",
    }
    base.require(isinstance(report, dict) and set(report) == expected_keys
                 and report.get("protocol") == "v8-first-update-kl-guard-v1"
                 and report.get("profile") == CAUTIOUS
                 and report.get("mode") == mode and report.get("source") == source
                 and report.get("update") == 1 and report.get("common_step_counter") == 24
                 and report.get("desired_kl") == .01
                 and report.get("initial_learning_rate") == CAUTIOUS_INITIAL_LR
                 and report.get("optimizer_minibatches") == 20
                 and report.get("mean_raw_kl_limit") == FIRST_KL_MEAN_MAX
                 and report.get("maximum_minibatch_mean_raw_kl_limit") == FIRST_KL_MINIBATCH_MEAN_MAX
                 and type(report.get("mean_raw_kl")) in (int, float)
                 and math.isfinite(report["mean_raw_kl"])
                 and type(report.get("maximum_minibatch_mean_raw_kl")) in (int, float)
                 and math.isfinite(report["maximum_minibatch_mean_raw_kl"])
                 and report.get("mean_raw_kl", float("inf")) <= FIRST_KL_MEAN_MAX
                 and report.get("maximum_minibatch_mean_raw_kl", float("inf")) <= FIRST_KL_MINIBATCH_MEAN_MAX
                 and report.get("passed") is True
                 and report.get("policy_acceptance") is False
                 and report.get("physical_motion_authorized") is False,
                 "source-bound cautious first-update KL guard")
    return report
