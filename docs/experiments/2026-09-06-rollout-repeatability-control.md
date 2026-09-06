# Recording-disabled repeatability control

Date: 2026-09-06. Status: closed invalid at first-child motor gate; no pair comparison. Diagnostic only.

The original overnight window ended at **07:00 Asia/Shanghai** before this
control was ready. No seed-367 control was launched in that window. Source
commit `3e630e84abff8cfbea23f46085aed06197f8d1ee` retained the expired 06:25
guard. CPU/source validation did not reopen GPU authority.

### Single-control authorization amendment, before execution

After closeout the user explicitly approved one bounded ten-minute diagnostic
("yes go on"). This amendment permits exactly the predeclared two sequential
seed-367 recording-disabled processes, with no retries or follow-on job.
Launch **before 2026-09-06 09:05 Asia/Shanghai** (01:05 UTC), with the existing
600-second service cap and 15-second cleanup limit. Thus the workload must
finish by 09:15, with at most 15 seconds for enforced cleanup. This is a new
single-control window, not a renewed overnight campaign. Only the runner start
deadline changes; all cases, artifacts, simulation behavior and analysis remain
frozen. Commit, push and test the exact amended source before launch. Leave
protected services and 100.98 unchanged, and stop after retaining the result.

## Question and limits

The seed-359 recorder off/on smoke stopped at exact-report equality despite
four clean first attempts in both runs. Its first observed representative
divergence was at 0.1 s; no off/off baseline existed. Test the minimal missing
control: can the same frozen rollout, with recording **disabled in both
processes**, reproduce its retained report exactly? No causal claim about the
recorder follows from either result. The failed seed-359 smoke remains failed,
and U3/U4/O3a remain rejected. No new policy or dataset is produced.

## Predeclared execution

Use the clean feature-branch commit containing this document and
`mjlab_microduck.rollout_repeatability`, and retain its exact full Git identity
before launch. The rollout, recorder, existing smoke comparator and policy
implementations are unchanged from source `0aac755d3b3a7620aba7dfd978e4e750c7e429df`.

- Actor: `logs/rsl_rl/run_motor_aware/2026-09-02_22-45-55_stage2-motor-aware-4096x3000-36667ee/model_7998.pt`,
  SHA-256 `080f98ae4d5ce731d143c733181bb89d504cb4b51ff39532efccd0b5fdc09c54`.
- Supervisor: `artifacts/checkpoints/hc4u4-bc60b2c-s42/supervisor.pt`, SHA-256
  `29855a51df8fe885d6ffed7fedf028093a8449a68b10b4b0e8a4bde7069bcf5b`.
- Fresh physics seed **367**, four environments, 700-step ceiling, single
  0.40 m/s / 0.90 m forward / 0.00 m lateral case. First terminal attempt only,
  unchanged 12-second timeout, exact compact geometry, no recording and no
  data collection. Do not reuse seed 359 or any acceptance seed.
- Two separate foreground processes, strictly sequential, named `first` and
  `second`. Both have identical behavioral arguments and environment; only
  their output directories differ. Do not add deterministic-backend settings
  or alter physics, seeding, numerical precision, rewards, sensors or policy.
- Parent output: `artifacts/evaluations/rollout-repeatability-disabled-s367-v1`.
  Refuse existing output. Retain source, checkpoint hashes, package versions,
  relevant nonsecret numerical environment values, exact commands, logs,
  process durations/exit codes, both reports and a full deterministic difference
  list in `decision.json`. Never retry or choose among repeated controls.
- One retained user service, `microduck-rl-repeatability-s367-v1.service`,
  total `RuntimeMaxSec=600`, child timeout 240 seconds, control-group cleanup
  and 15-second stop timeout. At most these two processes; no chained jobs.

Before launch: focused tests locally and remotely on the exact committed/pushed
source; clean exact feature branch; matching artifacts; idle/cool GPU with no
compute process; both protected AI Mission services inactive. Recheck occupancy
before the second process. The original 06:25 start / 07:00 stop is superseded
only for this pair by the explicit single-control amendment above. Preserve
unrelated workloads and 100.98/FilmBrain. No H1/H2,
Locked, video, raw perception, actor-observation expansion or physical motion.

## Frozen analysis and terminal decisions

Validate report/case/actor/supervisor/window identities, finite JSON, no recorder
or dataset fields, all four completed first attempts, reconciled integer outcome
counts and nonempty representative trace. Falls, NaN/nonfinite, hard/other
failures, unresolved attempts, rated-speed exceedance or torque p99 >0.60 close
the control as invalid. Validate each child report before starting another
process; an invalid first report stops the pair immediately. Matching collisions or timeouts remain descriptive
controller outcomes, not policy acceptance.

For two valid reports, compare **all** canonical JSON with exact equality;
strip no field and introduce no tolerance. Save every typed leaf/length
difference in deterministic order, including outcomes, metrics and traces.
Serialized signed-zero differences are differences too; JSON key order is not.
Identical reports yield `same-seed-reports-match-in-control`; nonidentical
reports yield `same-seed-reports-diverge-with-recording-disabled` and deliberate
exit 2. Missing artifacts, runtime errors/timeouts, invalid identity or numerical
failure yield `invalid-control-stop`. No decision admits another GPU job.

Divergence would demonstrate lack of exact retained-output repeatability in
this uninstrumented control, not establish its mechanism, recorder innocence,
or the cause of an earlier collision. Equality would show only that this tiny
pair matched; it would not explain the seed-359 off/on failure. Report either
result without retries, policy changes, post-hoc tolerances or retroactive
acceptance. Diagnose read-only, retain hashes/backups, and commit evidence.

## Source validation and overnight closeout

The CPU-only control suite passed **63 tests**; the combined recorder, rollout,
U4 collection/contract and U3/U1 gate regression suite passed **263 tests** with
`CUDA_VISIBLE_DEVICES=''` and `OMP_NUM_THREADS=1`. Tests cover exact and signed-zero
differences, malformed/unsafe reports, unchanged artifacts, refusal of dirty or
wrong source and expired deadlines, sequential commands with recording disabled
in both children, immediate stop on an invalid first report, timeouts, occupied
GPU and inactive-service requirements. All GPU subprocesses in the new runner
tests are mocked. Passing source tests is not a live repeatability result.

At **07:36:23 Asia/Shanghai**, read-only checks on 100.100 found no compute
process, GPU 0% / 12 MiB / 44 C, and both protected services inactive with
MainPID 0. The actor and supervisor matched the hashes above; the retained
seed-359 stop decision still hashed to
`292024b847ee6aa2b74c084181fd9c985055b8838d8bb68a8e15a38c124ea788`.
The seed-367 output directory did not exist. The expired overnight automation
was deleted through the app; checkpoints, evidence and services were untouched.
This closes the overnight window, not the scientific repeatability question.

## Retained single-control result: motor stop, no repeatability verdict

Amended source `e658562058237af4923fa5f610e9eb56c28c7c1a` was committed and
pushed before launch. All **264** focused tests passed locally (26.82 s) and
on 100.100 (5.34 s) with CUDA hidden. Exact clean source, artifact hashes, an
idle GPU (0%, 12 MiB, 44 C), inactive protected services and absent output/unit
were verified before starting the retained service.

`microduck-rl-repeatability-s367-v1.service` ran **08:54:56--08:55:11 Shanghai**,
with RuntimeMaxSec 600, TimeoutStopSec 15, KillMode control-group and Restart no.
Its first child exited **0**, taking **15.235372732859105 s**. The parent then
deliberately exited **2** with `ValueError: runtime motor check` and decision
`invalid-control-stop`. This was not a child crash, timeout or comparison
mismatch. Per the predeclared early-stop rule, the second child was **never
started**; no second directory/report exists. There was no retry.

The first report records four clean completed first attempts and zero
collisions, timeouts, falls, NaN/nonfinite steps, hard/other failures, unresolved
attempts or rated-speed exceedance. It stopped after 407 simulation steps.
These clean obstacle outcomes do not override the separate motor gate.

| First-child reported metric | Value |
| --- | ---: |
| Motor torque utilization p99 (dimensionless) | 0.6125551462173462 |
| Frozen maximum torque utilization p99 | 0.60 |
| Absolute excess | 0.012555146217346214 |
| Motor speed utilization p99 | 0.3969431221485138 |
| Motor near-stall fraction | 0.00009536524885334074 |
| Motor thermal load proxy mean | 0.04175148159265518 |
| Mean passage time (s) | 7.4899996519088745 |
| Mean lateral excursion (m) | 0.3671746104955673 |
| Recovery route speed (m/s) | 0.30098631978034973 |

Read-only source inspection confirms that the reported torque statistic is
the 0.99 quantile of pooled active-environment absolute simulated actuator
forces, normalized by the model's 0.60 Nm stall reference. The control threshold
is **0.60 utilization**, not 0.60 Nm. The reported statistic exceeds that
threshold by about 2.09%; rated-speed exposure is zero, so it is the torque
term that rejects this report. This is a simulation-envelope diagnostic, not
physical motor-temperature or hardware-safety evidence. The retained report
does not contain per-joint force histories, so it cannot identify the joint or
phase causing the tail, nor independently reconstruct the underlying quantile.

CPU validation of the unchanged report reproduces the exact exception. The
runtime receipt confirms one successful child only; all five backup hashes
match remote originals, and every policy/recorder/data/motion/further-GPU
authorization remains false. The GPU was idle again at **08:55:19** (0%, 12 MiB,
46 C), no compute process, both protected services inactive with MainPID 0.

Remote retention: `artifacts/evaluations/rollout-repeatability-disabled-s367-v1`.
Local backup: `artifacts/overnight-20260906-u4/rollout-repeatability-disabled-s367-v1`.

| Artifact | SHA-256 |
| --- | --- |
| `decision.json` | `4523d18c0296912bcd85bf7d53e2ef265d33ca0fd0226204e622494ed3959295` |
| First report | `5d75a781d21ad4a302ac35ffb1e6b6294e961da8c30b0952d927817a821ffe26` |
| `first.log` | `7cf60ee3c48fccdf085086d39ac7c74f0f2cf24eab9f698e6380233a38619eac` |
| `launch.json` | `660c585a2b7baf07e19a960f5f3a727b673aaa2f8283ae739c4d0d64c4d71c26` |
| `runtime.json` | `6d8d1e2b4ffd814a195aeafc90407a385c522334feb19ce92fa849de368b3b5c` |

**No repeatability conclusion is available** because there is no pair. This
neither explains nor clears the failed seed-359 off/on smoke; U3/U4/O3a remain
rejected. No case, seed, threshold, recorder, policy or simulator change was
made after the stop. Two CPU regression cases retain the observed torque value
and verify that an otherwise successful first child prevents a second launch.
The expanded **266-test** focused suite passed locally with CUDA hidden (7.22 s)
before the evidence commit; runtime code is unchanged from the launch source.
The single-control authorization is spent. Stop here: no replacement seed,
new GPU diagnostic, refit, training, promotion, video or physical motion.
