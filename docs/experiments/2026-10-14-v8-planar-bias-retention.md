# Persistent planar-speed retention lesson

Status: **completed; planar-bias lesson rejected; GPU lease closed and vLLM restored**.
Branch `integration/upstream-2026-10-10`; protocol `v8-planar-bias-retention-v1`.

## Question and prior evidence

The [seed977 staged-head pair](2026-10-14-staged-head-replication.md) is rejected
and closed: complete moving routes15→8/72, speed67/72 in both arms below the
original-parent69 bar, and negative-turn heading regression+.04336rad/s. Its
head ranges really changed; fewer falls did not yield better held-out routes.
Seed983 remains locked. Do not reuse either rejected endpoint as a teacher.

The older [gentle-turn support contraction](2026-10-13-quiet-head-gentle-turn-lesson.md)
already doubled target-band exposure but worsened speed/steering. Do not repeat
a yaw-support or instantaneous angular-width sweep. The current sampler has
substantial real straight and both-sign turn exposure; it is not a dead input.

New question: does pricing **persistent planar command error**, in addition to
the existing persistent yaw error, preserve walking speed and improve routes
without suppressing gait oscillation or worsening turning? The Gaussian linear
tracker currently has weight2/std√.1; a weak persistent-speed gradient is a
hypothesis, not a causal conclusion from those weights. The new EMA cost is
tested for DC response and alternating-residual cancellation before training.

## One factor and frozen behavior

Both arms begin independently from ORIGINAL provisional parent
`dddd37576fa7d2cd17bb4201826a7fa8aec0404b7b5af1339b79d492e4a9355a`,
`v8-motor-headroom-06c2b709-20261014/control/foundation/model_999.pt`.
Authenticate the exact rejected977 summary
`a344c638a7da8f398cc1832d4da41680f8cdeb5c0dc251366b3d8b9f64c82634`
and inventory
`66081b90b72ed98a4ae01cda7198b09085ed83167166d4027e486ea61c8cdfaa`
before resolving that parent. Preserve actor/critic/17 Adam histories and
LR.00011390625000000005; reset only local clocks/episode state. Initial parent
Adam60000, critic18432000, actor normalizer count0/frozen affine; no RNG or
simulator-state import and no bitwise-resume claim.

Shared added diagnostic slot `planar_bias`: weight0 control, weight−1 treatment.
Its positive cost is `abs(EMA(vx_cmd−vx_body)) + abs(EMA(vy_cmd−vy_body))`, with
tau.5s and `alpha=min(1,step_dt/tau)`=.04 at50Hz. Components are body-frame m/s;
fixed reference1m/s, no command-relative normalization or division by zero.
State resets per world on episode start or ANY twist change, including yaw;
counter rollback resets all history. Same-step metric reads cannot advance it;
skipped steps, stale/mutated inputs, nonfinite values and incompatible layouts
refuse. This causal EMA attenuates oscillations; it is not an exact complete-cycle
average. Zero arm skips the reward callable in reward computation and receives
exactly zero weighted reward; its metric computes the same diagnostic cache
without changing actions, commands, RNG or observations.

The scalar cost weight is the ONLY paired recipe difference. Both head ranges
stay at the parent's±.39/±.39/±.49/±.11; no staged widening. Existing yaw-bias−1,
tau.5, head tracking2/std.5, head bias1/tau1, action-rate−.4, selected motor cost0,
seven collapsed parent curricula, DR/noise/delay/BAM/solver and PPO stay fixed.
Twist support remains vx[.2,.4],vy0,wz[−.5,.5], dwell3–5s, standing.10,
forward-only.25, no turn-in-place. Actor61→14 is unchanged, unfiltered/unclipped,
with no raw perception or new actor/critic inputs.

Foundation learning seed is prospectively fixed991 for BOTH arms. It belongs
to this new objective test, not the locked983 replication or a seed reroll of
the head lesson. One pair only; no seed search or automatic follow-on queue.

## Bounded admission and telemetry

| Phase per arm | Worlds × updates | Seed | Child cap |
| --- | ---: | ---: | ---: |
| Smoke | 64×5 | 947 | 300s |
| Benchmark | 256×10 | 953 | 300s |
| Foundation | 256×1000 | 991 | 2100s |
| Final999 held-out matrix | 8×240 per case | 839/853/857 | 600s |

Run focused CPU contracts on Mac and exact clean native revision, then authenticate
parent/runtime preflight. Both smokes and both benchmarks precede either full arm;
each conservative benchmark projection must fit2070s. Retain first-update KL
mean≤.05/max minibatch mean≤.10, finite reward/optimizer telemetry, actual command
dwell, motor-envelope/range/thermal costs and live frozen profile throughout.

Snapshot new planar residual/EMA/cost and ACTUAL weighted reward before autoreset;
bind the command to the original pre-action snapshot. Retain24 observations per
update,256×24 samples, signed forward/lateral means, raw absolute errors,
absolute EMA components, summed cost, nonpositive reward and activity counts.
No recomputation from post-reset data and no silently inactive treatment.

Save initial and0/250/500/750/999; evaluate only final999, no favorable checkpoint
selection. Each arm6,144,000 transitions; final local env24000/Adam80000/
critic24576000, actor count0. Operational success remains separate from skill.

## Unchanged decisions and ownership

Held-out12 cases/arm retain exact seeds839/853/857×idle/(.3,0)/(.2,−.2)/(.2,+.2),
8 worlds,240 first-episode steps, startup60, full trace/contact timing and every
complete-cycle AND full-trajectory endpoint/maximum route criterion. Keep legacy
instantaneous tracking diagnostic, all motor/range/idle gates and no terminal
substitution. Useful lesson still requires≥6 additional complete moving worlds,
no paired speed-count loss, heading growth≤.01rad/s and XY growth≤.02m/s per
moving command, safety and ALL THREE walking references in BOTH arms. Original
parent speed bar69/72 and per-command MAE growth≤.02m/s remain unchanged.
No post-result gate change; a speed gain bought by poorer left turns is rejected.

Even a useful diagnostic is not route/policy/teacher acceptance: full moving
qualification and independent confirmation are separate. If rejected, stop this
pair, seal/diagnose, do not reroll, unlock983 or start obstacle/hop/football/video.

100.100 only, same owned GPU lease and UUID/DINO checks. One sequential retained
nonenabled user service, outer7800s/MemoryMax8GiB/CPU200%/Torch20%GPU; child cap
sum6600s plus comparison300s fits. Pause only originally active USER vLLM,
preserve DINO2029 and unrelated workloads; prove Duck exit and DINO-only idle
before scoped restoration. No100.98/NX/FilmBrain, packages/drivers/reboot,
policy publication, perception training or physical motion. Raw assets stay
native; keep only small seals on the Mac.

## Exact source and first native admissions

Executed source `03ede0e67a4291ab144b30e22a3aa92b77d098c5` passed136 focused
CPU tests on Mac in5.72s and on100.100 in6.16s, including the inherited yaw,
continuation and closed head-replication contracts. Owner review caught and
tested the observer seam where the parent consumes its command: the planar
observer retains that pre-action command before the inherited step clears it.
Luna's separate reward tests and driver/gate tests were reviewed before admission.
Compilation, shell syntax, unit verification and diff checks passed.

Native CPU preflight authenticated the rejected977 seals, ORIGINAL parent
actor/critic/Adam hashes, all matched recipes and unchanged runtime pins:
torch2.9.1, Warp1.12.0, MuJoCo3.10.0, MuJoCo-Warp3.8.1, mjlab1.3.0,
rsl-rl-lib5.0.1, ONNX1.22.0 and ONNX Runtime1.24.4. No install or driver change.

Service `recomo-duck-v8-planar-bias-s991-03ede0e6-20261014.service` started
2026-10-14 20:27:48 Shanghai. Native assets:
`/home/converge/work/microduck_rl-upstream-20261010/artifacts/training/v8-planar-bias-s991-03ede0e6-20261014`.
The original USER vLLM-only restoration context is retained separately at
`/home/converge/.local/state/v8-planar-bias-s991-03ede0e6-20261014-handoff`.

| Admission | Control | Planar |
| --- | --- | --- |
| Smoke64×5 seconds | 4.400 | 4.451 |
| Smoke KL mean / maximum minibatch mean | .018230 / .032449 | .018680 / .031811 |
| Benchmark256×10 seconds | 8.012 | 8.177 |
| Benchmark KL mean / maximum minibatch mean | .016612 / .021898 | .016355 / .021766 |
| Conservative1000-update projection | 1261.80s | 1286.53s |

All four short runs passed before the full pair, projections<2070s. Both ordinary
smokes had zero falls/NaNs. Actual smoke-final planar cost/reward is
.075478/0 control versus.073521/−.073521 treatment, with24 observed steps/update.
This proves the treatment is active and the control inert, not improved skill.
Short result SHA256 control/planar:

- Smoke `f1caf40091db9e1eecee53f7fc28666090e6ace3fb2096284404233f1d3f4e77` /
  `07afaa01bd2ec87b000a8f320864b317ee2d04b77188628fcb856f842520d123`.
- Benchmark `d65f22d41df752de1f585b5ba6307572f4724ca40f3077ea77a6d030a6980c61` /
  `2acc7d5f4ad8c0362d77cf85844fb383bddb561b79616f71c72fd069d493ee56`.

At the first retained full check, control reached update27/common648 with zero
NaN terminations. GPU used1710MiB/free14484MiB at59C; owners were only preserved
DINO2029 and the leased Duck child. All four protected scopes were inactive.
Full training, both final999 evaluations, unchanged comparison and post-exit
restoration remain pending. Do not advance the active native checkout from03ede0e6.

Executed context/restorer/handoff/unit SHA256 respectively:
`632c91c1b2ede83cbfb0e9da951a61f52c6e5d38657b2d7caafb866a91bbc6c9`,
`d4fcc6b28f5dcb9ba3c29065b8e2db89ab3c6d688cdc25a5051834d55a36b96e`,
`daf227216f3ed4cf45b454de05b814f1d6dc5d7a3510df1e63a844f27f7703f6`,
`77fe61a53261420da33d4b766ccd45bd2232b371b80beb308477b5389d41393b`.

## Final closeout (2026-10-14)

Both arms completed 1000 updates / 6,144,000 transitions and retained initial,
0/250/500/750/999 checkpoints. Control/treatment runtime was 651.823/661.531s;
training falls were 601/626, with zero NaN terminations. Final999 optimizer and
normalizer tensors were finite. Both 12-case evaluations completed, and all
three walking references in each arm remained `walking-idle-retained`.

The unchanged decision is **`planar-bias-lesson-rejected`**: complete moving
worlds fell **13→5**, paired speed passes **71→69/72**, and negative-turn heading
regressed. Neither endpoint is a promoted policy or a new teacher. No reroll,
obstacle/hop/football/video job follows this pair.

Final control/treatment checkpoint SHA256:
`abe5d4db20055a118b61b3ef9e6dfa77657937772338708a31f4d5d0ed69dbae` /
`705ad52df15452cc0c3b3e60760a59038d6f3c03dc090f2fb3b283aed14023b7`.
Decision SHA256:
`2c85d1cfae57ea731cb1c0c7c5c4d175ef03d743e3948beca97869722ffbd7eb`.
Large receipts/checkpoints remain in the native run directory above.

Independent import-mode validation found a metadata defect in all six training
receipts: the original `python -m` invocation serialized the observer callable
as `__main__:capture_planar_metric`; imported recomputation used
`mjlab_microduck.v8_planar_bias:capture_planar_metric`. Changing only that field
in a read-only recomputation reproduces all saved full/masked recipe hashes.
Original CLI validations and both evaluation validations passed; independent
training-receipt validation did not. Historical receipt/decision bytes are
preserved. The future canonical serializer now fixes the callable name, with
a regression test; this is not a retrospective acceptance or a reason to
repeat rejected learning.

Native service finished inactive/success/exit0. At the post-exit check,
original USER vLLM was active (PID2208422,9692MiB), DINO2029 remained alive
(946MiB), GPU50C, and no Duck GPU process remained. The next development slice
restores the pinned upstream v8 recipe and exercises its deployment simulator;
it does not extend the custom reward campaign.
