"""One audited runtime construction on private default-generator streams.

BAM's fixed-range startup draws still consume randomness. This separate scope
accounts for those draws without changing BAM, physics or the learner stream.
The CPU checker verifies retained evidence only, never CUDA RNG re-execution.
"""

from copy import deepcopy
from hashlib import sha256
import os
from pathlib import Path
import re
from threading import Lock

import torch

from mjlab_microduck import stance_recovery_cuda_rng_scope as sampler
from mjlab_microduck import stance_recovery_schedule as schedule
from mjlab_microduck import stance_training_smoke as training
from mjlab_microduck.stance_recovery_schedule_runtime import ScheduledRecoveryRuntime
from mjlab_microduck.first_attempt_smoke import canonical, require

PROTOCOL = "football-b1d-constructor-private-rng-v1"
CPU_SEED, CUDA_SEED, WORLDS = 977, 983, 64
NOMINAL = dict(
    vin_tensor=7.5, vin_drop_gain=0.1, kp_scale=1.0, kd_scale=1.0, friction_scale=1.0
)
CALLER_KEYS = {"cpu_before", "cpu_after", "cuda_before", "cuda_after"}
PRIVATE_KEYS = {"cpu_start", "cpu_end", "cuda_start", "cuda_end"}
FLAGS = {
    **sampler.FALSE_FLAGS,
    "constructor_rng_qualified": False,
    "cuda_constructor_reexecuted": False,
    "training_admitted": False,
}
KEYS = {
    "protocol",
    "source",
    "cpu_seed",
    "cuda_seed",
    "worlds",
    "status",
    "constructor_calls",
    "faulted",
    "caller_states",
    "private_states",
    "state_sha256",
    "nominal_parameters",
    "caller_cpu_preserved",
    "caller_cuda_preserved",
    "error_type",
    "error",
    *FLAGS,
}


def _state(value, label):
    require(
        torch.is_tensor(value)
        and value.device.type == "cpu"
        and value.dtype == torch.uint8
        and value.layout == torch.strided
        and value.ndim == 1
        and value.is_contiguous()
        and 0 < value.numel() <= 16384,
        "bounded raw constructor state: " + label,
    )
    return value


def _digest(value):
    return sha256(_state(value, "digest").numpy().tobytes()).hexdigest()


def _nominal(value):
    require(
        type(value) is dict and set(value) == set(NOMINAL),
        "exact constructor nominal parameter fields",
    )
    for key, expected in NOMINAL.items():
        tensor = value[key]
        require(
            torch.is_tensor(tensor)
            and tensor.device.type == "cpu"
            and tensor.dtype == torch.float32
            and tensor.shape == (WORLDS, 1)
            and tensor.layout == torch.strided
            and torch.isfinite(tensor).all()
            and (tensor == expected).all(),
            "fixed constructor parameter " + key,
        )


class CudaConstructorRng:
    """One-shot constructor, not an arbitrary callback/default-RNG sandbox."""

    def __init__(self, source, *, lease_fd):
        require(
            type(source) is str and re.fullmatch(r"[0-9a-f]{40}", source),
            "exact constructor source",
        )
        require(type(lease_fd) is int and lease_fd >= 0, "inherited constructor lease")
        training.inherited_lease(lease_fd)  # Before CUDA queries or state creation.
        self._source, self._lease_fd = source, lease_fd
        self._lock = Lock()
        self._used = False
        self._faulted = False
        self._identity = self._context()
        require(
            not sampler._PROCESS_CUDA0_SCOPE_LOCK.locked(),
            "constructor initialization cannot borrow an active sampler scope",
        )
        before_cpu = torch.random.get_rng_state().clone()
        before_cuda = torch.cuda.get_rng_state(0).cpu().clone()
        cpu = torch.Generator(device="cpu").manual_seed(CPU_SEED).get_state().clone()
        cuda = (
            torch.Generator(device="cuda:0")
            .manual_seed(CUDA_SEED)
            .get_state()
            .cpu()
            .clone()
        )
        require(
            torch.equal(before_cpu, torch.random.get_rng_state())
            and torch.equal(before_cuda, torch.cuda.get_rng_state(0).cpu()),
            "dedicated constructor generators preserve caller streams",
        )
        _state(cpu, "CPU start")
        _state(cuda, "CUDA start")
        require(
            cpu.shape == before_cpu.shape and cuda.shape == before_cuda.shape,
            "constructor private and caller layouts match",
        )
        self._starts = dict(cpu_start=cpu, cuda_start=cuda)
        self._receipt = dict(
            protocol=PROTOCOL,
            source=source,
            cpu_seed=CPU_SEED,
            cuda_seed=CUDA_SEED,
            worlds=WORLDS,
            status="not-started",
            constructor_calls=0,
            faulted=False,
            caller_states={},
            private_states={},
            state_sha256={},
            nominal_parameters={},
            caller_cpu_preserved=False,
            caller_cuda_preserved=False,
            error_type=None,
            error=None,
            **FLAGS,
        )

    def _context(self):
        training.inherited_lease(self._lease_fd)
        sampler._require_isolated_main_thread()
        require(
            os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
            and sampler.execution.PROFILE
            == sampler.execution.select(sampler.execution.WSL)
            and Path.cwd().resolve() == sampler.execution.ROOT.resolve(),
            "isolated exact WSL constructor context",
        )
        require(
            torch.cuda.is_initialized() is True
            and torch.cuda.device_count() == 1
            and torch.cuda.current_device() == 0,
            "already initialized single visible CUDA0 constructor",
        )
        return sampler.host.identity(self._source)

    @property
    def receipt(self):
        # Never expose aliases to caller/private stream state or physical fields.
        result = deepcopy(self._receipt)
        result["faulted"] = self._faulted
        return result

    def construct(self, declaration):
        if self._used or self._faulted or not self._lock.acquire(blocking=False):
            self._faulted = True
            self._receipt.update(
                status="faulted",
                faulted=True,
                error_type="ValueError",
                error="constructor scope is one-shot and nonreentrant",
            )
            raise ValueError("constructor scope is one-shot and nonreentrant")
        process_lock = False
        attempted_scope = False
        self._used = True
        try:
            require(
                canonical(self._context()) == canonical(self._identity),
                "unchanged constructor source and context",
            )
            checked = schedule.checked(declaration)
            require(
                checked["source"] == self._source and checked["worlds"] == WORLDS,
                "exact source-bound CUDA64 constructor schedule",
            )
            process_lock = sampler._PROCESS_CUDA0_SCOPE_LOCK.acquire(blocking=False)
            require(process_lock, "another CUDA default-stream scope is active")
            caller = dict(
                cpu_before=torch.random.get_rng_state().clone(),
                cuda_before=torch.cuda.get_rng_state(0).cpu().clone(),
            )
            self._receipt.update(
                status="running",
                caller_states=caller,
                private_states=deepcopy(self._starts),
            )
            attempted_scope = True
            try:
                with torch.random.fork_rng(devices=[0]):
                    torch.random.set_rng_state(self._starts["cpu_start"].clone())
                    torch.cuda.set_rng_state(
                        self._starts["cuda_start"].clone(), device=0
                    )
                    try:
                        self._receipt["constructor_calls"] = 1
                        env = ScheduledRecoveryRuntime(
                            declaration, device="cuda:0", solved_field_check="packed"
                        )
                        require(
                            type(env) is ScheduledRecoveryRuntime,
                            "exact constructed scheduled runtime",
                        )
                        self._receipt["nominal_parameters"] = {
                            key: tensor.detach().cpu().clone()
                            for key, tensor in env.motor.nominal_parameters.items()
                        }
                        _nominal(self._receipt["nominal_parameters"])
                    finally:
                        self._receipt["private_states"].update(
                            cpu_end=torch.random.get_rng_state().clone(),
                            cuda_end=torch.cuda.get_rng_state(0).cpu().clone(),
                        )
            finally:
                caller.update(
                    cpu_after=torch.random.get_rng_state().clone(),
                    cuda_after=torch.cuda.get_rng_state(0).cpu().clone(),
                )
                self._receipt["caller_cpu_preserved"] = torch.equal(
                    caller["cpu_before"], caller["cpu_after"]
                )
                self._receipt["caller_cuda_preserved"] = torch.equal(
                    caller["cuda_before"], caller["cuda_after"]
                )
                self._receipt["state_sha256"] = {
                    key: _digest(value)
                    for group in (caller, self._receipt["private_states"])
                    for key, value in group.items()
                }
                require(
                    self._receipt["caller_cpu_preserved"]
                    and self._receipt["caller_cuda_preserved"],
                    "fork scope restores constructor caller streams exactly",
                )
            require(not self._faulted, "constructor fault cannot be swallowed")
            require(
                canonical(self._context()) == canonical(self._identity),
                "unchanged closed constructor context",
            )
            self._receipt["status"] = "success"
            return env
        except BaseException as error:
            self._faulted = True
            self._receipt.update(
                status="faulted",
                faulted=True,
                error_type=type(error).__name__,
                error=str(error),
            )
            raise
        finally:
            if not attempted_scope and self._faulted:
                self._receipt["status"] = "faulted"
            if process_lock:
                sampler._PROCESS_CUDA0_SCOPE_LOCK.release()
            self._lock.release()


def check(receipt, *, source, prepared_caller):
    """Strict CPU consistency reader; raw CUDA endpoints are not resimulated."""
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        and not torch.cuda.is_initialized(),
        "CUDA-hidden constructor receipt reader",
    )
    require(
        type(receipt) is dict
        and set(receipt) == KEYS
        and receipt["protocol"] == PROTOCOL
        and receipt["source"] == source
        and receipt["cpu_seed"] == CPU_SEED
        and type(receipt["cpu_seed"]) is int
        and receipt["cuda_seed"] == CUDA_SEED
        and type(receipt["cuda_seed"]) is int
        and receipt["worlds"] == WORLDS
        and type(receipt["worlds"]) is int
        and receipt["status"] == "success"
        and receipt["faulted"] is False
        and receipt["constructor_calls"] == 1
        and type(receipt["constructor_calls"]) is int
        and receipt["error_type"] is None
        and receipt["error"] is None
        and receipt["caller_cpu_preserved"] is True
        and receipt["caller_cuda_preserved"] is True
        and all(receipt[key] is False for key in FLAGS),
        "exact successful non-admitting constructor receipt",
    )
    require(
        type(source) is str and re.fullmatch(r"[0-9a-f]{40}", source),
        "constructor reader source",
    )
    caller, private = receipt["caller_states"], receipt["private_states"]
    require(
        type(caller) is dict
        and set(caller) == CALLER_KEYS
        and type(private) is dict
        and set(private) == PRIVATE_KEYS
        and type(prepared_caller) is dict
        and set(prepared_caller) == CALLER_KEYS,
        "complete caller and private constructor endpoints",
    )
    hashes = {}
    for group in (caller, private):
        for key, value in group.items():
            hashes[key] = _digest(value)
    require(receipt["state_sha256"] == hashes, "constructor raw-state hashes")
    cpu_initial = torch.Generator(device="cpu").manual_seed(CPU_SEED).get_state()
    require(
        torch.equal(private["cpu_start"], cpu_initial),
        "seed-bound constructor CPU start",
    )
    for device in ("cpu", "cuda"):
        before, after = caller[device + "_before"], caller[device + "_after"]
        _state(prepared_caller[device + "_before"], "prepared before")
        _state(prepared_caller[device + "_after"], "prepared after")
        require(
            torch.equal(before, after)
            and torch.equal(before, prepared_caller[device + "_before"])
            and torch.equal(after, prepared_caller[device + "_after"]),
            "constructor preserves exact prepared caller " + device,
        )
        require(
            private[device + "_start"].shape
            == private[device + "_end"].shape
            == before.shape,
            "fixed constructor " + device + " state layouts",
        )
    _nominal(receipt["nominal_parameters"])
    return dict(
        protocol=PROTOCOL + ":score",
        source=source,
        constructor_calls=1,
        state_sha256=hashes,
        caller_streams_preserved=True,
        nominal_fields_exact=True,
        private_cpu_advanced=not torch.equal(private["cpu_start"], private["cpu_end"]),
        private_cuda_advanced=not torch.equal(
            private["cuda_start"], private["cuda_end"]
        ),
        **FLAGS,
    )
