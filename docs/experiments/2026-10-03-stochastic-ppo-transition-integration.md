# Native stochastic PPO transition integration

This is a bounded integration diagnostic, not a recovery training job. The
unchanged seed-577 iteration-255 parent must retain byte SHA
`2d36df17b17ff5da7d75414254db5535b7aa699b197899ad45902f7e432800b5`.
Its actor and critic are restored into the reviewed private-RNG bridge without
changing their observations, Gaussian scale, weights or optimizer state.

## Predeclared capture and independent replay

Use a fresh CPU `ScheduledRecoveryRuntime` with exactly two fixed training rows:
zero wrench and +x 2 N for 40 ms at physics step 250. The learner and owned
initialization seed are 653; no held-out seed enters a training rollout.
Collect exactly 28 stochastic policy transitions / 280 Euler steps unless a
real first terminal, exception or wall limit stops the attempt. The complete
nonzero pulse fits this horizon; the zero-row onset at step 500 does not, and
its later phase window is not claimed to have executed.

Retain original sampled actions before runtime clipping/slew, pre-action actor
and critic inputs, values, log probabilities, Gaussian parameters, rewards,
done masks, private/caller RNG states, ordinary rollout storage tensors,
pre-reset terminal observations and bridge receipts. Full-batch force arrays
and forced-pre/integrated/unforced-post phases are retained alongside the
physical/control trace. Different stochastic row actions do not constitute a
matched zero/push trajectory pair; frozen-parent first-attempt comparisons own
that separate causal check.

No returns, advantages, optimizer update, storage clear, counter relocation,
retry, policy retuning or student export is allowed. An unexpected terminal
retains its actual pre-reset evidence and partial storage, then rejects this
short integration attempt. It is not silently replaced with a fresh rollout.

One CUDA-hidden CPU user service is capped at 180 seconds, 2 GiB, 200% CPU,
Nice 10 and control-group ownership. Collection is at most 120 seconds with
retention room reserved inside the service cap; raw bytes are capped at 64 MiB.
A separate 120-second, 2 GiB CPU service rehashes the exact five-file inventory,
strictly reloads the parent, reproduces the private seed-653 stochastic sample
chain, validates storage/physical/control/force evidence, and writes its own
decision. A wall-expired or terminal prefix can be scored as rejected; it cannot
pass the 28-transition gate. Capture links are retained before postchecks; a
later failure remains explicit and cannot qualify even when its trace replays.
Broken current source/profile/service guards are not bypassed for that replay.
The run's exact service-cap receipt and finite in-cap elapsed time must also
pass before qualification. Launch needs 180 + 120 + 60 = 360 seconds before
the unchanged renewed 20:00 Shanghai cutoff.

The already closed CPU timing and scheduled CUDA prerequisites are rehashed;
their earlier source and deadlines remain unchanged. This run never overlaps
another owned Duck service or changes host source while a native job is active.
FilmBrain and unrelated workloads remain untouched, protected services remain
inactive, and the frozen driver/package/math profile is preserved.

## Remaining gates

A qualified result establishes only this short native CPU2 transition path.
Full-episode timeout/selective-reset behavior, finite optimizer updates,
GPU learner throughput, repeatable measured deficits and held-out promotion
remain separate gates. Neither hopping, obstacle avoidance, rolling-football
balance nor physical readiness is accepted by this diagnostic.

Reviewed implementation, focused contract checks (**46 passed in 8.40 s**)
and broader adjacent regressions (**147 passed in 17.19 s**) passed locally
with CUDA hidden. These are source/synthetic checks, not native evidence.
No native capture has started. The broad screen's independent closeout exposed
a duplicate false-flag keyword in its final receipt constructor. Read-only
diagnosis found the same latent constructor issue in this unrun probe. It is
corrected before launch, with direct constructor regressions for both admissible
and inadmissible capture receipts; no integration gate or acceptance flag changes.
The WSL worktree was kept at the original broad-capture source until the
separately predeclared repair audit was ready and all owned native jobs were
terminal. It is now clean at evaluator source
`b1878b8715efbb7ea6e166bc7d345efecf5285de`. The repair-v2 audit closed with
receipt SHA `adba134d905c8e07c9b871cce85b26a5704335944b47b6001ddfa9ecd0c2fcac`:
25 complete cases, no candidate deficit, and all 79 original files unchanged.
Both earlier failed audit services remain failed and were not restarted.
The final combined repair/PPO/schedule source regression suite passed 145 tests
in 18.95 seconds with CUDA explicitly hidden, including both actual constructor
paths. Native capture, reset and optimizer qualification remain pending.
Exact reviewed source, service invocations and artifact hashes will be appended
after execution and independent closeout.

The final repair-v2/PPO/schedule suite also passed on the frozen WSL profile:
**147 in 26.23 seconds**, test invocation
`4dcb7df82d1b414fa224e2f52bbf64a2`, receipt SHA
`747dcdf6fb61d9bc4aee3a367736f3d02e73a417a1acf98c1602a18fb1e229ce`.
This test service reached its 2-GiB memory cap and 2.1-GiB swap peak; that is
test-resource evidence, not native collection or learner-throughput qualification.
The native probe retained the already declared 180/120-second run/closeout
limits and source above; neither a longer episode nor an optimizer step was added.

## Native collection and failed independent closeout

`microduck-cpu-ppo-run-b1878b8715ef.service`, invocation
`c94ac07ae4fd4ab5868815d8c3637716`, completed successfully. It retained all
**28 transitions / 280 Euler steps**, with no terminal or optimizer update.
Collection took **3.5569308400154114 seconds**; the in-process run took
6.891804591985419 seconds. The service used 15.050 seconds CPU, a 1.5-GiB
memory peak and no swap. These are one warm-cache observation, not a measured
p95 throughput or full-episode/reset qualification.

The five original files remain in
`artifacts/evaluations/stance-wsl-cpu-stochastic-ppo-transition-b1878b8715ef`.
The launch SHA is
`0dbe88f4b580507b26eac1bf17ca5d20647cb016103c6a8a43c170f54f79a528`;
the report SHA is
`20bfa6222ab737c5f19a299d5a15f864078e6e3c49eb8ef1c4d55702aefedebf`.
The original capture is **6,889,991 bytes**, SHA
`180675277204d08b54fdd4daa605d8c7fc2f3a074b2f65c0c15c3ce3c76bfac3`.
The external first-launch admission receipt binds the closed broad audit and
tests, SHA `1b517a9f919221f1f0359c6d4dab137227b527da2cff5910c08c4e9bb1b20fe8`.

Independent closeout invocation `ccd6921149844ace9fa9e12cca4bd816` failed in
`microduck-cpu-ppo-closeout-b1878b8715ef.service` before physical/stochastic replay:
`ValueError: action, bridge receipt, and terminal observation binding`.
That unit remains failed, PID 0, zero restarts and exit status 1; no closeout
receipt was written. The rollout is **not independently qualified**.

Read-only diagnosis checked every conjunct on all 28 actual retained ticks.
Only the reward comparison failed: bridge receipts have CPU float32 shape
`(2,)`, while RSL storage and policy rows have `(2,1)`. Every value matched
exactly after selecting the column; all other binding conjuncts passed. The
strict fix validates both original shapes/dtypes/finiteness, then compares
the vector to `stored_reward[:, 0]` without broadcasting or tolerance.
It does not alter the capture, bridge, parent, sampler, storage, gates or caps.
The focused CUDA-hidden trace/probe/bridge suite passed **57 in 6.76 seconds**,
including the valid distinct layouts and eight malformed-value/layout cases.

The separate immutable failure evidence is retained at
`artifacts/tools/stochastic-ppo-closeout-failure-b1878b8715ef`:

| Evidence | SHA256 |
| --- | --- |
| Failure receipt | `7c96ad502b9f38778965fcade4d9c5b6d15e98dfcbc893a34e7421466a1ce9b6` |
| Exact invocation journal | `5e155e8284847062e7af0f0a78d6685f3f24767d1f5e79f21528d79b0168793f` |
| Original trace source file | `1d7d9b5e066b14092c1c73c0f65a82d5a67c7931248cdc47ff69b7bf73483384` |

The Mac mirror at
`artifacts/retained/stochastic-ppo-failed-b1878b8715ef.7ZfWmn` retains the
original five-file capture, failure receipt/journal and first-launch receipt.
Do not restart the failed service or collect a replacement rollout. A separately
predeclared bounded saved-capture audit must bind distinct artifact/evaluator
sources, the exact failure evidence and unchanged dependencies, then perform
fresh full replay before the transition gate can pass.

## Saved-capture audit predeclaration

Protocol `football-b1d-cpu-stochastic-ppo-receipt-repair-v1` must read only that
immutable five-file capture and pinned failure evidence. Every transitive old
in-package replay dependency must remain byte-identical, except the trace file's
reviewed strict reward-layout correction, pinned at SHA
`5c5aee58d4e384cf28b61a5377c50065c27619b4003631a8f5dc663513f8580a`.
The old and evaluator contexts must match every field except the explicitly
distinct source revision. Retain the same parent, seed, schedule, original
collection caps and numerical/no-update gates. Do not restart the failed unit,
overwrite its directory, move a storage cursor, retune sampling or recollect.

After focused tests and clean exact-source WSL preflight, one new evaluator-source
user service may independently rehash and score all 28 transitions, private RNG,
stored raw policy samples, endpoint observations, storage, physical/control
evidence and all 280 force substeps. It is capped at **120 seconds, 2 GiB, 200%
CPU, Nice 10 and control-group ownership**; reserve **180 seconds** before the
unchanged 20:00 cutoff. No other owned Duck job may overlap it. It must write only
the new three-file directory `stance-wsl-cpu-ppo-receipt-repair-<evaluator12>`
containing `launch.json`, `source-inventory.json` and `receipt.json`, distinguishing
the original artifact source from the current evaluator source. Rehash all five
old files and the failure evidence before and after scoring, keep the failed
service unchanged, and recheck current source/profile/services and caps afterward.
A failed or partial replay is retained as failure, never promoted to qualification.
Fresh independent qualification remains pending; no optimizer job is admitted
by this source preparation or by the locally verified reward-layout helper.

## Subsequent optimizer-evidence preparation

The separate optimizer-trace module is source preparation only: no CLI, job
admission, policy export, resume or recovery acceptance. It requires an
authenticated complete native CPU2 trace, then permits exactly one actual bridge
update (20 Adam steps) on a fresh full rollout. It retains the actual post-step
endpoint, GAE, model schema/state, named Adam moments/scalar steps, metrics,
finite-gradient/moment hook counts and private/caller RNG states. Independent
optimizer replay reconstructs storage through public `add_transition()` and
repeats the same math without creating an environment or resimulating physics.
Synthetic tests remain explicitly non-native. Native update, timeout/reset,
throughput and capability gates remain unrun. A future supervisor still needs
its own tested limits, failure/partial-update retention and retained predeclaration
after the saved-capture transition audit passes.

The final reviewed reward-layout/audit/optimizer and adjacent schedule suites
passed **193 tests in 24.19 seconds** locally with CUDA hidden. The real mirrored
failure receipt/journal passed the audit's semantic readers, and the 89-file
old replay-source inventory allowed only the pinned reward-layout leaf change.
These checks do not substitute for the fresh capped native saved-capture audit
or authorize an optimizer job.

The final host-path review confirmed that native failure evidence lives under
`artifacts/tools/stochastic-ppo-closeout-failure-b1878b8715ef`; the audit reads
that path, not the Mac mirror. After separating native and test-mirror paths,
the same **193 tests passed in 22.39 seconds**, with no skips on the Mac campaign
checkout. The frozen WSL test/preflight must exercise the actual native reader
before launching the audit.
