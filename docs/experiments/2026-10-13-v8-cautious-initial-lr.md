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

## Native closeout — completed, legacy gate rejected

Tested/pushed source `934bdff47a63d458f571189fe5db155fcd4fd8d0` ran as
`recomo-duck-v8-cautious-lr-934bdff4-20261013.service`, inactive/success0 after
the five sequential stages. Native affected tests:1436 pass,18 warnings. Full
packet remains only on100.100 at
`artifacts/training/v8-cautious-lr-934bdff4-20261013`; Mac retains just the
independently checked summary/inventory, not duplicate checkpoints/traces.

All three actual candidate actor/critic state digests match the retained stock
controls before updating. Both normalizer counts0, clocks0, empty Adam and all
17 ordered optimizer parameters/settings match except the declared initial LR.
The masked full recipe hashes match too. These are verified initial model-state
matches, not a claim of byte-identical ensuing physics trajectories.

| Stage | Complete budget | Learner elapsed | First-update mean / max minibatch KL |
| --- | --- | --- | --- |
| Smoke |64×5,7680 transitions |4.154s |.014070 / .026477 |
| Benchmark |256×10,61440 transitions |7.256s |.010639 / .022693 |
| Foundation |256×1000,6144000 transitions |556.117s |.010451 / .020339 |

All prospective first-update guards pass. Conservative benchmark projection
1148.362s fits2070s. Final checkpoint0/250/500/750/999 durable; final999 hash
`820c86e28b3d05d8bd292b63777b78cfa0b4657d69cc028ad17f8afe1ab91738`.
Actor count0/affine unchanged; critic count6144000;17 Adam states each20000
steps;20 captured raw minibatches/update;0 NaN terminations. Training falls1131
(stock1321), a descriptive count, not held-out or physical stability acceptance.
The foundation initial PT differs in whole bytes because LR differs; its model
tensor digests match. Updated Gaussian std remains our PPO state, not v8 history.

All12 final held-out cases complete240 steps with no terminals or motor/range
gate failure. Moving speed passes66/72 (stock65, original4); raw instantaneous
yaw passes0/72. Idle speed/yaw each24/24 (stock idle yaw23/24). The original
legacy decision is still `foundation-not-ready`: all9 moving cases rejected.

| Command vx / yaw | Original v8 mean vx / body wz | Stock final999 | Cautious final999 |
| --- | --- | --- | --- |
| .30 / 0 | .12479 / +.02925 | .22194 / +.11808 | .23196 / +.13181 |
| .20 / −.20 | .08233 / −.18850 | .14885 / −.07043 | .17006 / −.04381 |
| .20 / +.20 | .08118 / +.17443 | .15129 / +.20184 | .17063 / +.15871 |

Cautious final mean speed MAE .07413/.04091/.03973 and raw yaw MAE
.32133/.29671/.28994 for these three commands. Forward tracking improves, but
the substantial steering degradation persists despite removing the first-update
KL shock. Therefore initial LR alone is not an adequate steering-retention fix;
do not promote the candidate or infer an obstacle/hop/football skill. The next
[gait-aware route measurement](2026-10-13-gait-aware-route-qualification.md)
tests actual path/heading separately, without rewriting this rejection.

Final held-out maximum force utilization p99 .593614 (raw .356169 Nm), zero
rated-speed exceedance, minimum motor-hinge margin .037974rad. GPU maximum
1854MiB, minimum free14340MiB, maximum67°C. DINO PID2029 preserved; originally
active user vLLM restored and owns GPU again; both system scopes and user subject
worker remain inactive. The certificate verifies all checkpoints, finite updates,
norm/Adam clocks, exports, per-case files and recomputed legacy decision.

| Sealed artifact | SHA256 |
| --- | --- |
| Foundation result | `81fb85083254d641a1e7e9a623c39e8aae4e4aec27386195be7cef1507019beb` |
| Final normalized ONNX | `4d35c804dbeb9c6a0eb9ba58e94fbc358ce276cbfe312aa3302fc2cc77d1a5af` |
| Final held-out result | `39bb68c9149a3f47f8d6b604fa825ed9669362a1790ed0034eeb7d81f25beeb8` |
| Summary | `5c8542012dd31a42e6482300290de20ea1a90b68735a720f94f7c7b2f20f0b32` |
| Full packet inventory | `2a902da7e38b6faad9ce5a79fa87ac8dc5e44e13ed9bb58d2b6470f03c42e27e` |
| Scoped restoration | `166558503ffa57fbf0e70495671c5ca23d25e3e5f3592bced3f3c7bc60ab75ce` |

No automatic retry, checkpoint search, extra budget, raw perception, video,
policy publication, harder-skill promotion or physical motion occurred.
