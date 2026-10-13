# Steering precision inside the small-head lesson, predeclared

Protocol `integrated-quiet-head-yaw-focus-v1`, branch
`integration/upstream-2026-10-10`; implementation
`src/mjlab_microduck/integrated_quiet_head_yaw_focus.py`.

## Evidence and narrowly scoped question

The [head lesson](2026-10-13-motor-range-head-lesson.md) improved all moving
group speed/yaw means and every paired case motor/range aggregate; speed48→64/72,
falls1404→976. It still rejects yaw0/72, with worst straight-world heading drift
43.41° over3.58s and worst-world windowed body-wz error.213995rad/s. This is a
real remaining steering problem, not permission to average away drift or change
the old gate. Keep the simpler head configuration for **both** new arms, not its
rejected endpoint as teacher.

Test the yaw-axis width of the positive angular Gaussian: variance.5→.1 with
coefficient2 and roll/pitch variance.5 unchanged. Control is the exact stock
XYZ function `2*exp(-(yaw_error²+roll_pitch_rate²)/.5)`; treatment reuses the
already tested `mdp.factorized_body_angular_tracking` product
`2*exp(-yaw_error²/.1-roll_pitch_rate²/.5)`. No new MDP duplicate, additive
aggregation, weight boost, rotation tax, reward history/EMA or observer filter.
Both still price commanded nonzero turns, not absolute yaw motion. The wider
roll/pitch kernel and separate-.05 trunk-angular/-.02 whole-body momentum costs
remain. This may still tax unavoidable yaw oscillation; unchanged held-out
tracking and motor/retention gates determine whether it helps rather than merely
inflating/deflating reward logs. It is not a demonstrated cause or accepted skill.

Earlier precision work used another parent, wider head goals and other objective
conditions. Do not pool those arms with this new matched control. No automatic
old-knob repeat or longer run is authorized by a previous result.

## Fixed reference and contracts

Both warm-load original unaccepted936a, never b291/e90 or an old precision arm:
`artifacts/training/motor-range-forward-3940e89f-20261012/control/model_999.pt`,
SHA `936a19973df11fba53614d6da0bc826117461f1f43f700c9152943cde6100bc6`.
Exact parent models/normalizers43008000/17 Adam140000/LR restored, original
iter999/env24000 verified, learner/env clocks restarted0. Materialize seven
final source curricula before setting the common exact beginner head ranges
±.05/±.05/±.07/±.015rad; all four input slots stay nonzero. Same head tracking+2/
bias+3, resampling2–5s, zero-head probability0; full locomotion eight-bin coverage,
standing.25, body goals, DR/noise/delays, BAM plant/solver and PPO are unchanged.

Both keep stock XYZ linear2/stdsqrt(.1), action-rate-1, motor-1 (.60Nm/.70soft/
overload4), bilateral named hip-yaw qpos-4/.15rad and zero forward-error term.
Actor raw61→raw14 is normalized, unfiltered/unclipped; mouth excluded. Official
export bakes the normalizer. No raw perception, new actor/critic input, controller
switch or physical robot motion.

|Phase per arm|Worlds × updates|Matched seed|Child cap|
|---|---:|---:|---:|
|ONNX smoke|64×5|1637|600s|
|Measured benchmark|256×10|1657|600s|
|Control/yawfocus|256×1000|1663|2100s|
|Final999 full matrices|8×240 per case|839/853/857|900s combined|

Full native scoped CPU regression and both smokes/benchmarks precede either
learner; `benchmark_elapsed*100*1.5<2040s`, no cap widening. Each6144000
transitions/20000 minibatches, final counts49152000/Adam160000; save initial and
0/250/500/750/999. No stage calendar or reward schedule. Run control→yawfocus→
both final matrices serially. Evaluate only final999, never selected early saves.

## Direct telemetry and unchanged decisions

Existing raw yaw-error-square, roll/pitch-rate-square and actual weighted angular
snapshot is captured before reset. Verify live callable/params/weight every step,
exact expected positive product, finite raw errors and signed sample counts;
retain actual yaw variance.5/.1, XY.5, coefficient2, aggregate label and24×world
counts in every row. Head command/reward exposure, actual linear/action terms,
optimizer, motor, range and fall telemetry remain; native and portable closed-row
validators check the complete declared contracts. Metrics add no RNG/reward or
physics forward/sense operation.

The same12 cases per arm, seeds839/853/857 ×idle/(.3,0)/(.2,-.2)/(.2,.2),
first-episode8worlds×240steps, startup60 omission for tracking only. All original
per-world speed≤.03 idle/≤.10 moving and yaw≤.10 MAE gates remain, no terminal/
rated-speed exceedance, case pooled torque p99≤.60Nm, all14 hard margins≥-.005rad
and right-hip stop≤1%. Keep the original combined decision and fixed denominator.

Both new matrices additionally retain the existing **read-only** body quaternion,
body/world angular rate and gait trace for every command, not only straight.
Capture uses the already tested pre-action API, with its truthful field timing;
no sensor refresh, action/observation mutation or new commands. This is telemetry
only, not a new gate. It allows later per-world heading-vs-rate diagnosis for
arcs and idle; never compare absent old arc paths as if they had been retained.
Any replay numerical mismatch stays explicit and does not rewrite old results.

Report all groups' means/world passes, vx/signed yaw and individual regressions;
every paired torque/soft-limit/power/thermal/range/stop result; training falls
total/first100/last100; actual angular/head/linear/action mass. Descriptive
tracking benefit requires all three moving speed means lower, no yaw mean
regression and no paired held-out motor/range/stability regression. It waives
no hard gate. Even a numerical pass requires independent-seed confirmation,
retention and transfer checks. Policy/simulator/physical acceptance remain false.
No automatic longer run, endpoint adoption, head-range expansion, transition/
obstacle/hop/football promotion, MP4 or publication.

## Native admission and shared-host boundary

Use exact clean tested/pushed source on100.100 at
`/home/converge/work/microduck_rl-upstream-20261010`, source frozen while active.
Fresh UUID/DINO2029/exe/uid and original protected-service context must pass.
Pause/restore only originally active user `recomo-ai-mission-vllm.service`,
other three scopes inactive; preserve DINO/unrelated workloads and100.98/NX.
Owned lease, used≤6144MiB/free≥8192MiB/temp<75°C, Torch20%, CPUthreads1,
8GiB/CPU200%; nonenabled preflight45min/matched95min/handoff150min with15s child
grace. Restore only after no Duck child/GPU and leasefree/DINO-only proof, then
prove vLLM GPU cgroup ownership. Unknown entrants cause refusal, not concurrent
work or unrelated stops. No driver/package changes. Source, native admission,
learner, numerical outcomes and physical motion are separate acceptance levels.

Tests, native admission and results are pending at this declaration.

## Source preparation evidence

CUDA-hidden single-thread122 focused checks passed7.55s: matched configs/parent/
seeds/gates, stock/factorized parity, independent XY width, live angular drift,
actual pre-reset product,24-step action/linear/head/yaw telemetry, row faults and
all-command trace forwarding; existing gait and precision tests included.
Two initial fixture failures were corrected (normalizing copies rather than
mutating returned configs; corrupting the captured reward rather than its later
live matrix). Production behavior and gates were not changed to make tests pass.

Independent source/declaration/API review cleared; installed primary angular
reward source confirms z-command error plus zero-command XY rates. Trace captures
only read/clone cached fields; no RNG/sense/step mutation. Twelve ops templates
compile in a no-write dry run;2 portable angular arithmetic tests pass including
variance/label/count/sign/nonfinite failures. Documentation links and whitespace
checks pass. Full native scoped regression, both smokes and projected budgets
remain mandatory before either new learner; native results are still pending.

## Native admission passed; new matched learners active

Exact clean tested/pushed native source
`154516d8211e1f8e451b9212289adea462353ab0`;1199 scoped regression checks
passed37.00s with16 existing warnings. Normalized61→14 ONNX smokes passed
4.712/4.643s,0/0 falls/NaNs; benchmarks8.084/8.150s,2/4 falls, zero NaNs,
projections1212.595/1222.458s below2040s. Both actual initial checkpoints hash
`d7dd1a99d9ca1f950d6ee2f3603a65661acbb9fff3c8848494cac5eb276f4235`,
with exact original936a models/normalizers/17 Adam/LR restored.

All30 closed rows pass action/linear/head/yaw validators. Both head ranges are
the small nonzero lesson; coefficient2 and XY variance.5 stay fixed, actual yaw
variance.5/.1 and product labels differ as declared. Final benchmark angular
reward.631537→.350205 decreases by construction under the narrower kernel;
raw yaw-error square.525011→.539585 is not an improvement claim. Judge only
the declared final matrix, not this short timing benchmark.

Admission SHA `81af7c83db2368b8ecb735abae8d9fa8763a40e6f5adc18fa2ce3ef245561c5e`;
original context `7a9ee84d9b6ce1ad41558af7e4004aa084eb539eae0c55de85352e2ab7bc475f`.
Closed JSON-only Mac packet `artifacts/evaluations/quiet-yawfocus-154516d8-20261013-preflight`;
owner compared both hashes to fresh native reads and repeated all30 four-contract
row validations. Independent generated-ops review, shell/Python/systemd checks,
3 mock restore tests with7 refusals and2 portable angular arithmetic tests pass.

Run `artifacts/training/quiet-yawfocus-154516d8-20261013`, managed by
`microduck-quiet-yawfocus-154516d8-matched-eval.service` and
`recomo-duck-quiet-yawfocus-154516d8-handoff.service`, control→yawfocus→both
final999 matrices. First retained audit control265/1000,284 training falls,
zero NaNs, finite metrics,250 checkpoint durable; GPU1710MiB/65°C, DINO2029
preserved, all four protected scopes inactive during scoped user-vLLM pause.
Native source154516d8 is frozen while active. Final numerical results, extended
trace coverage, checkpoint seal and original-service restoration are pending;
no qualified motion, video, harder-skill promotion or physical authority follows.
