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
No native capture has started; the WSL worktree remains frozen for the active 25-case CPU screen.
Exact reviewed source, service invocations and artifact hashes will be appended
after execution and independent closeout.
