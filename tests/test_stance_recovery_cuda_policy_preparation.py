"""Source-only guards for CUDA recovery policy preparation.

All positive receipts below are synthetic contract fixtures. These tests never
claim native CPU provenance, CUDA allocation, learner preparation or training.
"""

from hashlib import sha256

import pytest
import torch

from mjlab_microduck import stance_cpu_replay_profile as cpu_profile
from mjlab_microduck import stance_cuda_probe as host
from mjlab_microduck import stance_execution_profile as execution
from mjlab_microduck import stance_ppo, stance_training_smoke
from mjlab_microduck import stance_recovery_contract as baseline
from mjlab_microduck import stance_recovery_cuda_policy_preparation as preparation
from mjlab_microduck import stance_recovery_parent as parent

SOURCE = "a" * 40
SYNTHETIC_RAW = b"synthetic bytes; not the selected historical checkpoint"


def _synthetic_cpu_receipt():
    return dict(
        protocol=parent.PROTOCOL,
        parent_checkpoint_sha256=baseline.CHECKPOINT_SHA256,
        parent_identity=parent.expected_identity(),
        parent_state_sha256=parent.PARENT_STATE_SHA256,
        cpu_math_profile=cpu_profile.expected_receipt(),
        strict_actor_restore=True,
        strict_critic_restore=True,
        models_trainable=True,
        weights_only=True,
        fresh_optimizer_required=True,
        optimizer_created=False,
        optimizer_restored=False,
        simulator_restored=False,
        storage_restored=False,
        rng_restored=False,
        normalization_restored=False,
        historical_archive_authenticated_by_loader=False,
        installed_in_learner=False,
        execution_admitted=False,
        **baseline.FALSE_FLAGS,
    )


def _bind(receipt, *, source=SOURCE):
    digest = preparation._cpu_parent_receipt_digest(receipt)
    return digest, preparation.cpu_parent_binding(source, digest)


def _install_source_only_environment(monkeypatch, calls):
    monkeypatch.setattr(
        execution, "PROFILE", dict(execution.PROFILE, name=execution.WSL)
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(
        stance_training_smoke, "inherited_lease", lambda fd: calls.append(("lease", fd))
    )
    monkeypatch.setattr(
        host,
        "identity",
        lambda source: (
            calls.append(("identity", source))
            or {
                "source": source,
                "execution_profile": execution.PROFILE,
                "synthetic_test_fixture": True,
            }
        ),
    )


def test_synthetic_cpu_parent_receipt_checks_exact_identity_and_source_binding():
    receipt = _synthetic_cpu_receipt()
    digest, binding = _bind(receipt)

    assert (
        preparation.validate_cpu_parent_receipt(receipt, digest, binding, source=SOURCE)
        == binding
    )
    assert all(value is False for value in baseline.FALSE_FLAGS.values())


@pytest.mark.parametrize(
    "mutation",
    [
        "protocol",
        "checkpoint",
        "identity",
        "state",
        "actor",
        "critic",
        "profile",
        "archive-auth",
        "installed",
        "admission",
        "capability",
        "extra",
    ],
)
def test_cpu_parent_receipt_rejects_wrong_purpose_profile_and_authority(mutation):
    receipt = _synthetic_cpu_receipt()
    if mutation == "protocol":
        receipt["protocol"] = "other"
    elif mutation == "checkpoint":
        receipt["parent_checkpoint_sha256"] = "0" * 64
    elif mutation == "identity":
        receipt["parent_identity"]["purpose"] = "pilot"
    elif mutation == "state":
        receipt["parent_state_sha256"] = "0" * 64
    elif mutation == "actor":
        receipt["strict_actor_restore"] = False
    elif mutation == "critic":
        receipt["strict_critic_restore"] = False
    elif mutation == "profile":
        receipt["cpu_math_profile"]["torch_version"] = "unverified"
    elif mutation == "archive-auth":
        receipt["historical_archive_authenticated_by_loader"] = True
    elif mutation == "installed":
        receipt["installed_in_learner"] = True
    elif mutation == "admission":
        receipt["execution_admitted"] = True
    elif mutation == "capability":
        receipt[next(iter(baseline.FALSE_FLAGS))] = True
    elif mutation == "extra":
        receipt["caller_override"] = True
    digest, binding = _bind(receipt)

    with pytest.raises(ValueError):
        preparation.validate_cpu_parent_receipt(receipt, digest, binding, source=SOURCE)


def test_cpu_parent_receipt_hash_covers_the_whole_canonical_receipt():
    receipt = _synthetic_cpu_receipt()
    digest, binding = _bind(receipt)
    receipt["installed_in_learner"] = True

    with pytest.raises(ValueError, match="whole canonical"):
        preparation.validate_cpu_parent_receipt(receipt, digest, binding, source=SOURCE)


def test_caller_binding_is_exactly_source_and_receipt_hash_bound():
    receipt = _synthetic_cpu_receipt()
    digest, binding = _bind(receipt)
    binding["source"] = "b" * 40

    with pytest.raises(ValueError, match="caller binding"):
        preparation.validate_cpu_parent_receipt(receipt, digest, binding, source=SOURCE)


def test_parent_byte_hash_refuses_before_host_lookup_or_decode(monkeypatch):
    calls = []
    _install_source_only_environment(monkeypatch, calls)
    receipt = _synthetic_cpu_receipt()
    digest, binding = _bind(receipt)
    monkeypatch.setattr(
        preparation.checkpoint,
        "validate_identity",
        lambda *args, **kwargs: pytest.fail("decode before exact parent hash"),
    )

    with pytest.raises(ValueError, match="exact D1 parent bytes"):
        preparation.prepare_policy(
            b"not parent bytes",
            source=SOURCE,
            lease_fd=17,
            cpu_parent_receipt=receipt,
            cpu_parent_receipt_sha256=digest,
            caller_binding=binding,
            seed=653,
            worlds=64,
        )
    assert calls == [("lease", 17)]


def test_cuda_unavailable_refuses_before_parent_decode_or_device_allocation(
    monkeypatch,
):
    calls = []
    _install_source_only_environment(monkeypatch, calls)
    monkeypatch.setattr(
        baseline, "CHECKPOINT_SHA256", sha256(SYNTHETIC_RAW).hexdigest()
    )
    receipt = _synthetic_cpu_receipt()
    digest, binding = _bind(receipt)

    def synthetic_identity(source):
        calls.append(("identity", source))
        return {
            "source": source,
            "execution_profile": execution.PROFILE,
            "synthetic_test_fixture": True,
        }

    monkeypatch.setattr(preparation.host, "identity", synthetic_identity)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        torch.cuda,
        "device_count",
        lambda: pytest.fail("device count after unavailable CUDA"),
    )
    monkeypatch.setattr(
        preparation.checkpoint,
        "validate_identity",
        lambda *args, **kwargs: pytest.fail("parent construction before CUDA gate"),
    )
    monkeypatch.setattr(
        torch,
        "load",
        lambda *args, **kwargs: pytest.fail("deserialize before CUDA gate"),
    )

    with pytest.raises(ValueError, match="single visible CUDA device"):
        preparation.prepare_policy(
            SYNTHETIC_RAW,
            source=SOURCE,
            lease_fd=17,
            cpu_parent_receipt=receipt,
            cpu_parent_receipt_sha256=digest,
            caller_binding=binding,
            seed=653,
            worlds=64,
        )
    assert calls == [("lease", 17), ("identity", SOURCE)]


@pytest.mark.parametrize("seed,worlds", [(True, 64), (653, 2), (659, 65), (657, 64)])
def test_bad_seed_or_world_count_refuses_after_lease_before_host_or_cuda(
    monkeypatch, seed, worlds
):
    calls = []
    _install_source_only_environment(monkeypatch, calls)
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: pytest.fail("CUDA query before seed/world contract"),
    )

    with pytest.raises(ValueError):
        preparation.prepare_policy(
            SYNTHETIC_RAW,
            source=SOURCE,
            lease_fd=23,
            cpu_parent_receipt={},
            cpu_parent_receipt_sha256="0" * 64,
            caller_binding={},
            seed=seed,
            worlds=worlds,
        )
    assert calls == [("lease", 23)]


def test_missing_shared_lease_refuses_before_source_or_cuda_checks(monkeypatch):
    calls = []
    monkeypatch.setattr(
        stance_training_smoke,
        "inherited_lease",
        lambda _fd: (_ for _ in ()).throw(ValueError("lease absent")),
    )
    monkeypatch.setattr(host, "identity", lambda _source: calls.append("identity"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: calls.append("cuda"))

    with pytest.raises(ValueError, match="lease absent"):
        preparation.prepare_policy(
            SYNTHETIC_RAW,
            source=SOURCE,
            lease_fd=23,
            cpu_parent_receipt={},
            cpu_parent_receipt_sha256="0" * 64,
            caller_binding={},
            seed=653,
            worlds=64,
        )
    assert calls == []


def test_non_10098_profile_refuses_before_host_or_cuda_checks(monkeypatch):
    calls = []
    _install_source_only_environment(monkeypatch, calls)
    monkeypatch.setattr(
        execution, "PROFILE", dict(execution.PROFILE, name=execution.DEFAULT)
    )
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: pytest.fail("CUDA query before exact host profile"),
    )

    with pytest.raises(ValueError, match="10098"):
        preparation.prepare_policy(
            SYNTHETIC_RAW,
            source=SOURCE,
            lease_fd=23,
            cpu_parent_receipt={},
            cpu_parent_receipt_sha256="0" * 64,
            caller_binding={},
            seed=653,
            worlds=64,
        )
    assert calls == [("lease", 23)]


def test_non_cuda0_visibility_refuses_before_cuda_availability_query(monkeypatch):
    calls = []
    _install_source_only_environment(monkeypatch, calls)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(
        baseline, "CHECKPOINT_SHA256", sha256(SYNTHETIC_RAW).hexdigest()
    )
    receipt = _synthetic_cpu_receipt()
    digest, binding = _bind(receipt)
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: pytest.fail("availability query before CUDA0 visibility"),
    )

    with pytest.raises(ValueError, match="only visible device"):
        preparation.prepare_policy(
            SYNTHETIC_RAW,
            source=SOURCE,
            lease_fd=23,
            cpu_parent_receipt=receipt,
            cpu_parent_receipt_sha256=digest,
            caller_binding=binding,
            seed=653,
            worlds=64,
        )
    assert calls == [("lease", 23), ("identity", SOURCE)]


def test_installed_rsl_source_pins_are_actually_checked():
    preparation._check_rsl_sources()


def _real_cpu_storage():
    obs = preparation.TensorDict(
        {"actor": torch.zeros(64, 44), "critic": torch.zeros(64, 50)}, [64]
    )
    return preparation.RolloutStorage("rl", 64, 28, obs, (10,), device="cpu")


def test_real_empty_cpu_storage_is_not_cuda_preparation():
    storage = _real_cpu_storage()
    assert storage.step == 0 and storage.distribution_params is None
    assert storage.actions.shape == (28, 64, 10)
    assert storage.dones.dtype == torch.uint8
    with pytest.raises(ValueError, match="fresh empty 64-world CUDA"):
        preparation._check_storage(storage)


def test_cuda_label_does_not_qualify_real_cpu_storage_tensors():
    storage = _real_cpu_storage()
    storage.device = preparation.DEVICE  # Deliberate synthetic metadata lie.
    with pytest.raises(ValueError, match="fresh finite zero CUDA observation"):
        preparation._check_storage(storage)


def test_stock_cpu_ppo_layout_cannot_be_relabelled_as_cuda():
    actor, critic = preparation.checkpoint.fresh_models(577)
    storage = _real_cpu_storage()
    algorithm = preparation.PPO(
        actor, critic, storage, **stance_ppo.CONFIG, device="cpu"
    )
    assert algorithm.rnd is None and algorithm.symmetry is None
    assert algorithm.rnd_optimizer is None and not algorithm.optimizer.state
    assert all(
        getattr(algorithm, key) == value
        for key, value in stance_ppo.CONFIG.items()
        if key != "optimizer"
    )
    algorithm.device = preparation.DEVICE  # Still actual CPU parameters.
    with pytest.raises(ValueError, match="finite trainable CUDA0 actor parameter"):
        preparation._check_policy(actor, critic, algorithm, storage)


@pytest.mark.parametrize("flag", tuple(baseline.FALSE_FLAGS))
def test_every_capability_flag_is_refused_before_provenance_claim(flag):
    receipt = _synthetic_cpu_receipt()
    receipt[flag] = True
    digest, binding = _bind(receipt)
    with pytest.raises(ValueError, match="capability flags remain false"):
        preparation.validate_cpu_parent_receipt(receipt, digest, binding, source=SOURCE)
