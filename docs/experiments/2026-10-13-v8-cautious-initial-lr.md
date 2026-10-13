# Cautious v8 actor adaptation — matched initial-rate contrast

The [stock initial-rate round](2026-10-13-v8-actor-foundation-learning.md)
improves forward-speed passes4→65/72 but loses steering/one idle world. Its first
update has mean raw KL3.788118 and maximum minibatch mean4.282735 at a target.01;
adaptive LR briefly rises to.0015. Later updates are mostly near.01–.02. This
is a measured early-retention risk, not proof that initial LR caused the loss.

## One declared learning change

Profile `cautious`, protocol `v8-cautious-initial-lr-v1`, run prefix
`v8-cautious-lr-<source8>-20261013`. Start from the ORIGINAL pinned v8 actor,
not the rejected stock final999. The sole learning change is initial LR
`1e-3 → 1e-4`; keep adaptive KL.01 and its existing rate adjustment/floor,
fresh scalar Gaussian std.3, fresh critic/Adam/clocks, frozen actor affine,
all rewards, sampler, curricula, plant, noise, delays, timestep and old gates.
Profile identity/extra admission telemetry do not alter policy inputs or losses.

Reuse the retained stock control rather than spend GPU on a duplicate:

| Stage | Worlds × updates; seed | Exact control initial PT SHA256 |
| --- | --- | --- |
| Smoke |64 ×5;947 | `1f4c8b3c1538ce88c1cb0409658c998ce0d6454b960214775e422015d898b2e3` |
| Benchmark |256 ×10;953 | `80084814096f99a56a11f78d6e0526c99261b1d0eaae3a57da79517733b61163` |
| Foundation |256 ×1000;967 | `29d3ea8491c84d57db12ea37ffad9d586fa8ab582be621e3028e5f27055ba974` |

Before each cautious update sequence, authenticate the owned non-symlink control
bytes and require identical actor/critic tensor states (including normalizers,
seed digest and Gaussian std), empty initial Adam, zero iteration/environment
clocks and the same optimizer group settings/order except LR. Retain the exact
control hash and a create-exclusive matched-initialization receipt. These prove
model/optimizer initialization matching, not byte-identical simulated trajectories
or complete single-cause attribution across different source executions.

## Small-run and early-update admission

Run64-world/five-update smoke first; require complete finite checkpoint and the
regular normalized export. Then the256-world/10-update benchmark. The ONLY new
process guard for this profile checks the first update's20 raw minibatches:
mean of minibatch KL means≤.05 and maximum minibatch mean≤.10. Retain the
first-update KL record BEFORE checking it, including a refusal. This additional
prospective retention guard does not retrospectively reclassify the stock job
or change skill qualification. Do not alter the KL tensor, loss, clipping,
optimizer updates or adaptive schedule to make the guard pass.

If it fails, stop the sequence, retain initial/available explicitly labeled
aborted-state checkpoint and exact failure evidence, restore only the authorized
original workload after proven Duck exit, and diagnose read-only. No automatic
retry, smaller rate, extra budget or suppressed failure. The smoke and benchmark
must both pass before any foundation launch; foundation's first update must also
pass. A passing early guard is not a motion or teacher acceptance.

## Unchanged learning/evaluation path

After benchmark, evaluate its original-mean initial actor with the same12 fixed
held-out cases and baseline first-episode/motor/range safety gate. Foundation
must fit the unchanged conservative time projection (<2070s), then receives
1000 fresh updates (6,144,000 transitions). Save0/250/500/750/999; select ONLY
final999. Same seeds839/853/857,8 worlds,240×.02 s,startup60 and source-bound
actor-only checkpoint inference. Compare retained stock and cautious final
cases, and both against the original v8 baseline, without a checkpoint search.

All absolute speed/yaw, first-terminal, rated-speed, force-utilization, named
motor-hinge and right-yaw-stop gates stay unchanged. Keep left-hip-yaw margin
in the report: it reached.023102 rad in the stock final, although no range gate
failed. Never clip low-kp targets or assume it is the old right-hip problem.
The owner approved a separate gait-aware path/heading qualification on October13.
It will have its own prospective contract; it does not replace this round's
instantaneous yaw bar or alter previously sealed rejected results.

Observe finite reward/models/losses/Adam, falls/NaNs, frozen actor count0,
learned critic counts,20 raw-KL minibatches/update, actual command dwell/live
curricula, force/power/thermal/soft/rated exposure and durable checkpoints.
Smoke/benchmark caps300s each, evaluations600s each, foundation2100s; one
sequential user service capped4500s,8 GiB RAM/200% CPU/Torch20% GPU. Same owned
exclusive lease, native runtime/UUID/occupancy/temperature guards, DINO preserved.
Pause/restore ONLY originally active user vLLM. No unrelated workload mutation,
policy promotion, harder-skill run, video, raw perception or physical motion.

Source/focused checks:130 affected CPU tests pass (two legacy ONNX exporter
deprecation warnings), including real actor/PPO construction and the exact
initial-checkpoint schema, UID, tensor-byte and result-receipt regressions.
Read-only native inspection confirms all three retained control initial PTs
carry `infos.env_state.common_step_counter=0`, iteration0 and fresh Adam.
Exact native affected checks, guarded service preflight and actual candidate
matched-control validation still precede learning admission. Results pending.
