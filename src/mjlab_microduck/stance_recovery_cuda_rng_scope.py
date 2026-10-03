"""Private CUDA0 default-stream RNG scope for a later source-only PPO bridge.

This small sibling does not call PPO, construct an environment, allocate model
or optimizer state, or qualify CUDA sampling. RSL's pinned ``PPO.act`` has no
generator argument, so callers may scope its default-stream draw through this
object while preserving the caller's CPU and CUDA0 RNG streams.
"""

from contextlib import contextmanager
from hashlib import sha256
import os
from pathlib import Path
import re
from threading import Lock, active_count, current_thread, main_thread

import torch

from mjlab_microduck import stance_cuda_probe as host
from mjlab_microduck import stance_execution_profile as execution
from mjlab_microduck import stance_recovery_cuda_policy_preparation as preparation
from mjlab_microduck import stance_training_smoke as training_smoke
from mjlab_microduck.first_attempt_smoke import canonical, require

PROTOCOL = "football-b1d-private-cuda0-rng-scope-v1"
SEEDS = (653, 659)
DEVICE = "cuda:0"
FALSE_FLAGS = dict(preparation.FALSE_FLAGS)
# Torch's CUDA default generator is process-global. This excludes other users
# of this helper; it does not make unrelated Torch RNG calls thread-safe.
_PROCESS_CUDA0_SCOPE_LOCK = Lock()


def _digest_state(state):
    return sha256(state.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _require_isolated_main_thread():
    require(
        current_thread() is main_thread() and active_count() == 1,
        "private CUDA RNG scope requires the isolated main Python thread",
    )


class CudaPrivateRngScope:
    """Temporarily bind one retained private CUDA0 RNG state to the default stream.

    The constructor requires an already initialized CUDA0 runtime and an
    inherited shared-lease descriptor. ``state`` is expected to be the state
    of a private ``torch.Generator(device="cuda:0").manual_seed(seed)``; the
    class verifies this binding without changing any global RNG.
    """

    def __init__(self, seed, state, *, lease_fd):
        # Validate scalar arguments before asking the inherited-lease helper to
        # inspect the descriptor; both checks precede any CUDA access.
        require(
            type(seed) is int and seed in SEEDS, "one fixed private CUDA learner seed"
        )
        require(
            type(lease_fd) is int and lease_fd >= 0,
            "inherited shared GPU lease descriptor",
        )
        training_smoke.inherited_lease(lease_fd)
        self._seed = seed
        self._lease_fd = lease_fd
        self._lock = Lock()
        self._active = False
        self._faulted = False
        self._scope_count = 0
        self._caller_cpu_preserved = False
        self._caller_cuda_preserved = False
        self._source, self._host_identity = self._checked_context()
        _require_isolated_main_thread()

        # is_initialized() is deliberately checked before any API that can
        # lazily initialize CUDA. No CUDA availability probe is used here.
        require(
            torch.cuda.is_initialized() is True,
            "CUDA runtime must already be initialized",
        )
        require(
            torch.cuda.device_count() == 1 and torch.cuda.current_device() == 0,
            "exact single visible CUDA0 device",
        )

        require(
            torch.is_tensor(state)
            and state.device.type == "cpu"
            and state.dtype == torch.uint8
            and state.ndim == 1
            and state.numel() > 0,
            "private CUDA state is a nonempty one-dimensional CPU uint8 tensor",
        )
        caller_cpu = torch.random.get_rng_state().detach().clone()
        caller_cuda = torch.cuda.get_rng_state(0).detach().cpu().clone()
        try:
            private_generator = torch.Generator(device=DEVICE)
            private_generator.manual_seed(seed)
            expected_state = private_generator.get_state().detach().cpu().contiguous()
            actual_default_layout = (
                torch.cuda.get_rng_state(0).detach().cpu().contiguous()
            )
            cpu_unchanged = torch.equal(caller_cpu, torch.random.get_rng_state())
            cuda_unchanged = torch.equal(
                caller_cuda, torch.cuda.get_rng_state(0).detach().cpu()
            )
            require(
                cpu_unchanged and cuda_unchanged,
                "private generator construction preserves caller RNG streams",
            )
        except BaseException:
            # Restore via exact states, never reseed. This also contains a
            # misbehaving constructor before refusing it.
            torch.random.set_rng_state(caller_cpu)
            torch.cuda.set_rng_state(caller_cuda, device=0)
            raise
        supplied = state.detach().cpu().contiguous()
        require(
            supplied.shape == expected_state.shape == actual_default_layout.shape
            and supplied.dtype == expected_state.dtype == actual_default_layout.dtype
            and torch.equal(supplied, expected_state),
            "private state is exact seed-bound CUDA0 generator state layout",
        )
        self._state = supplied.clone()
        self._initial_state_sha256 = _digest_state(self._state)

    @property
    def seed(self):
        return self._seed

    @property
    def state(self):
        """Return a detached clone; callers cannot mutate the retained state."""
        return self._state.detach().clone()

    @property
    def state_sha256(self):
        return _digest_state(self._state)

    @property
    def receipt(self):
        """Return a fresh non-admitting receipt for the current retained state."""
        return {
            "protocol": PROTOCOL,
            "source": self._source,
            "seed": self._seed,
            "device": DEVICE,
            "state_bytes": int(self._state.numel()),
            "initial_state_sha256": self._initial_state_sha256,
            "private_state_sha256": self.state_sha256,
            "scope_count": self._scope_count,
            "caller_cpu_rng_preserved": self._caller_cpu_preserved,
            "caller_cuda_rng_preserved": self._caller_cuda_preserved,
            "scope_active": self._active,
            "faulted": self._faulted,
            "native_cuda_rng_scope_qualified": False,
            "native_cuda_sampling_qualified": False,
            **FALSE_FLAGS,
        }

    def _checked_context(self):
        training_smoke.inherited_lease(self._lease_fd)
        require(
            os.environ.get("CUDA_VISIBLE_DEVICES") == "0",
            "CUDA0 must be the only visible device from process start",
        )
        require(
            execution.PROFILE == execution.select(execution.WSL),
            "exact frozen WSL10098 execution profile",
        )
        require(
            Path.cwd().resolve() == execution.ROOT.resolve(),
            "current directory is the exact frozen WSL worktree",
        )
        source = host.read("git", "rev-parse", "HEAD")
        require(
            type(source) is str and re.fullmatch(r"[0-9a-f]{40}", source) is not None,
            "exact current source revision",
        )
        identity = host.identity(source)
        require(
            identity.get("source") == source
            and identity.get("execution_profile") == execution.PROFILE,
            "exact current source, host, runtime and WSL profile identity",
        )
        return source, identity

    def _verify_context(self):
        _require_isolated_main_thread()
        source, identity = self._checked_context()
        require(
            source == self._source
            and canonical(identity) == canonical(self._host_identity),
            "source and host identity unchanged during private RNG scope",
        )
        require(
            torch.cuda.is_initialized() is True
            and torch.cuda.device_count() == 1
            and torch.cuda.current_device() == 0,
            "initialized single-device CUDA0 remains current",
        )

    @contextmanager
    def scope(self):
        """Advance only this object's CUDA0 RNG state around the caller's action.

        The body should contain the stock stochastic call (for example
        ``algorithm.act(cuda_observations)``). The pinned RSL sampler consumes
        the default CUDA generator, not a passed ``torch.Generator``.
        """
        require(not self._faulted, "faulted private CUDA RNG scope cannot be reused")
        try:
            _require_isolated_main_thread()
        except BaseException:
            self._faulted = True
            raise
        if not self._lock.acquire(blocking=False):
            self._faulted = True
            raise ValueError("private CUDA RNG scope is not reentrant or concurrent")
        process_lock_acquired = False
        if not _PROCESS_CUDA0_SCOPE_LOCK.acquire(blocking=False):
            self._faulted = True
            self._lock.release()
            raise ValueError("another private CUDA0 default-stream scope is active")
        process_lock_acquired = True
        self._active = True
        try:
            self._verify_context()
            caller_cpu = torch.random.get_rng_state().detach().clone()
            caller_cuda = torch.cuda.get_rng_state(0).detach().cpu().clone()
            with torch.random.fork_rng(devices=[0]):
                self._verify_context()
                torch.cuda.set_rng_state(self._state.detach().clone(), device=0)
                try:
                    yield
                finally:
                    advanced = torch.cuda.get_rng_state(0).detach().cpu().contiguous()
                    require(
                        advanced.shape == self._state.shape
                        and advanced.dtype == torch.uint8,
                        "advanced private CUDA0 RNG state layout unchanged",
                    )
                    self._state = advanced.clone()
            require(
                not self._faulted,
                "scope was faulted by a swallowed reentry or nested refusal",
            )
            cpu_restored = torch.random.get_rng_state()
            cuda_restored = torch.cuda.get_rng_state(0).detach().cpu()
            self._caller_cpu_preserved = torch.equal(caller_cpu, cpu_restored)
            self._caller_cuda_preserved = torch.equal(caller_cuda, cuda_restored)
            require(
                self._caller_cpu_preserved and self._caller_cuda_preserved,
                "caller CPU and CUDA0 default RNG states restored",
            )
            self._verify_context()
            self._scope_count += 1
        except BaseException:
            self._faulted = True
            raise
        finally:
            self._active = False
            if process_lock_acquired:
                _PROCESS_CUDA0_SCOPE_LOCK.release()
            self._lock.release()
