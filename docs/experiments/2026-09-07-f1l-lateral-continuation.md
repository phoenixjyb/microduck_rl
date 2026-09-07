# F1-L: paired motor-aware lateral-velocity continuation

Status: **training complete; evaluation orchestration stopped**. Predeclared
source `e0cb4f4c44d5cd548fe5aee4458c59cda4510adf`. User authorized the next gait-training
step after the frozen heading-hold diagnostic. Simulation only; this does not
admit obstacle training, hopping, or an integrated controller.

## Hypothesis and one changed term

Heading hold reduced seed-443 worst heading from 0.577876 to 0.157293rad and
worst sampled cross-route displacement from 0.506726 to 0.199146m, without
failing the paired motor margins. It left yaw-aligned lateral absolute speed
around 0.072–0.080m/s, two slow environments, and only 2/8 stable speed windows.
This separates a heading-control improvement from an unresolved low-level gait
problem. Residual lateral motion is not reclassified as harmless sway.

Continue the retained, **unaccepted** F1-R narrow model8498 in two matched arms:

- Model SHA256 `7ed703d6b5b8407da912f51755be8a8e57698340f62d0f5d3a80cf195ec1f80f`.
- Restore actor, critic, normalizers, Adam and saved common step204000. Synchronize
  Python learning rate to restored Adam; first new update8499, final8998 after500.
- Both arms retain linear tracking weight4.5, variance0.05; motor-load weight-2,
  action-rate weight-1, existing completed curricula, rigid feet, randomized
  training events, physics, terminations, 61D actor, observation noise/delay and
  PPO settings. Fixed training command(0.30,0,0), no heading controller in training.
- Add one privileged simulation reward: squared yaw-aligned lateral base speed
  divided by(0.1m/s)^2. Control weight0; treatment weight-0.5. At lateral0.1m/s
  this contributes-0.5 reward-rate, or-0.01 at dt0.02. It is sign-symmetric,
  world-yaw-rotation invariant, unsaturated and independent of vertical speed.
  Corrupt raw lateral state raises before reward-manager NaN sanitization.
- No gain search, speed-controller change, motor-weight change, observation
  expansion, raw perception, joint-action override, hop update or mechanics switch.

This mild penalty is a testable hypothesis, not a claim it will suppress lateral
motion without hurting balance or speed. The matched control distinguishes its
effect from merely spending another500 updates. No file is overwritten.

## Fixed budget and execution order

Exact branch `feat/athletics-obstacle-curriculum`, clean pushed source, host100.100
only, frozen runtime/dependency hashes. Original walking parent and rejected
H1-T checkpoint sentinels remain unchanged before/after the campaign.

1. Evaluate the retained parent with fixed heading hold ON, seeds467/479/487,
   sequentially. Stop immediately on reference safety/coverage failure, before
   spending any optimizer work. Parent speed failures remain descriptive.
2. Ten-update smoke for each arm, seed461, each from the same parent. Require
   successful source-matched smokes, identical restored-state hashes and a
   conservative smoke-time estimate fitting the900s pilot budget.
3. Two500-update pilots, seed463, 256environments,24steps/update,50-update save
   cadence; both independently resume the parent, not their smoke checkpoint.
   Require identical initial checkpoint hashes. Expected final model8998 and
   common step216000. Final only; no best-checkpoint selection.
4. Evaluate control then treatment at each held-out seed467/479/487. Stop at
   the first unsafe reference or failed treatment pair; do not continue a seed
   sweep after a failed gate. At most9evaluation cases/72first episodes.

Evaluation uses the unchanged400step/8env/.02s first-attempt protocol with
100startup and300settled samples. Both arms and parent use the same fixed
heading controller and fresh command adapter from the prior diagnostic. Keep
its constants and raw command/route trace audit unchanged; each report must
match its own checkpoint hash and this new experiment protocol/seed.

One sequential user service, hard3150seconds,180seconds/smoke,900seconds/pilot,
90seconds/evaluation child,30second closeout reserve. Start before September7
14:00 Shanghai; refuse late launch, reused output or replay after decision.
Expected pilot cost is around ten minutes total from prior measured throughput,
but the service/runtime guards, not this estimate, enforce the bound.
No concurrent GPU workload; require idle/cool GPU and both named protected
system services inactive before each child. Preserve unrelated workloads.

Output: `artifacts/experiments/f1l-lateral-paired-s463-v1`.

## Unchanged gates and capability boundaries

Training retains finite-action/optimizer checks, reset-safe named motor stream,
gross fall-fraction>0.50, pooled torque p99>0.85 and rated-speed exposure>1% stop
guards, plus sampled GPU<80°C and protected-service checks. These stochastic
training guards are not relaxed evaluation or physical safety thresholds.
Retain all reward/optimizer TensorBoard tags and per-update motor/fall JSONL;
verify the new lateral reward is active in treatment and zero in control.

Evaluation retains all F1 absolute safety, per-env speed, heading<=0.25rad,
absolute cross-route speed<=0.05m/s and stable-speed-window gates. Compare
treatment with both frozen parent and matched control using existing speed,
heading, lateral and pooled torque nonregression margins. Also require each
named joint torque p99<=reference+0.02, squared load and mean absolute mechanical
power<=reference*1.05 against both references. State body, route and stable-window
failures independently instead of hiding later failures behind a first label.

All three clean numerical pairs yield at most
`single-seed-diagnostic-support-only`, never policy acceptance. Failure closes
the experiment: diagnose read-only, retain hashes, test and commit/push evidence;
no automatic retry, second training revision, larger obstacle stage or video.
Standing/stopping/disturbance recovery and composed-controller retention remain
separate required future matrices. H1-T is still rejected; hopping is not yet
a validated capability to claim preserved. No physical motion is authorized.

## Prelaunch checks

Local focused CPU regression: **698 passed**, no skips, two existing
actuator/site-pattern warnings,16.47s. Tests cover reward frame/sign/scale and
nonfinite/overflow refusal; paired config equality except the one weight;
explicit resumed iteration/common-step/Adam state; unchanged historical defaults;
fresh heading-command trace identity; absolute and per-joint paired motor gates;
reference-before-training, initial-state mismatch, first numerical failure,
runtime failure, no retry and exact retained manifest coverage.

## Retained training and interrupted evaluation

Remote regression also passed698 tests (12.08s, same two warnings, no skips).
Service `microduck-rl-f1l-e0cb4f4-s463.service` ran September7
13:06:51–13:18:35 Shanghai. All three parent references completed400steps
without safety failures, followed by both smokes and both500-update pilots.

| Arm | Updates | Training seconds | Fall windows | Maximum fall fraction/window | Final checkpoint SHA256 |
| --- | ---: | ---: | ---: | ---: | --- |
| Control | 500 | 289.362 | 8 | 1/256 | `d01385604edd6b8ff3d053062536aa1c3126997dc3110d0e01df232b37aea2f7` |
| Lateral | 500 | 291.659 | 8 | 1/256 | `95717faf177fd83da446433d22091a71a1e7f78c148e83645b9b1ef1fffe9ef4` |

Both final checkpoints are model8998 with common step216000. All26 saved
post-update checkpoints plus4initial snapshots were verified finite in the
local mirror, including exact update labels and saved common steps. All four
initial snapshots were also checked against the original parent: actor, critic,
normalizers and Adam match exactly, with only the declared next-iteration label.
Control/treatment smoke durations were6.552/6.654s. Each arm retained75 finite
TensorBoard scalar tags; the treatment lateral penalty was active (final pilot
episode-rate metric-0.260035), while control was identically zero. No training
gross guard fired. Maximum sampled training GPU temperatures were58/56°C;
these are sampled values, not continuous peaks. Training falls and small rated
speed exposure are not deterministic-evaluation acceptance.

Parent467/479/487 and control467 reports were retained. The service then failed
**before launching treatment467**, at the between-child `check_host()` call:

`ValueError: idle/cool GPU required`

The preceding no-compute-PID and protected-service checks passed. That error
combines utilization/temperature/memory conditions; the raw rejected telemetry
was not retained, so its precise triggering value is unknown. Subsequent
read-only checks found0% GPU use,45–46°C,12MiB and no compute PID. Transient
utilization telemetry after child exit is plausible, not proven. Neither
`lateral-s467.log` nor its JSON exists in the original run; the treatment
evaluation was not attempted. The retained decision remains
**runtime-failure-stop**, with no pairs and no causal policy decision.

Original evidence:72payload files,162,237,955bytes plus manifest, mirrored under
`artifacts/diagnostics/f1l-lateral-paired-s463-v1` locally; all local payload
hashes match the remote-generated manifest. The original remote directory and
all its files remain immutable.

- Original manifest SHA256: `44ce65ffec5252acd91bece8224d7e4487ef91f64a9a996ae1c69d00eecd553f`.
- Original decision SHA256: `541cb79c4254f5116be939a17963529c017ceb39c0cdab48254baa9c0f83331c`.
- Control467 report SHA256: `89a5f592615879b351d9d94fccf4dab6500da34d8623dd3b2f3dab784ba067ca`.

## Separately predeclared evaluation-only closeout

Do not rerun the failed campaign, any training, any existing evaluation, or
alter its manifest/decision. Complete only the **never-started treatment467**
as a separate retained user service/output, using the unchanged model8998,
controller, seeds, first-attempt protocol, raw trace audit and numerical gates.
Reference parent467/control467 are reused by exact hashes; no cherry-picking.
This finishes the first numerical pair, not the remaining479/487 matrix.

- New directory: `artifacts/experiments/f1l-lateral-s467-evaluation-closeout-v1`.
- New source must be clean and pushed. The closeout verifies every original
  payload hash, exact interrupted boundary and absence of an earlier treatment
  attempt. It reads but cannot append to the original output.
- Before launch, require two consecutive zero-utilization GPU samples one
  second apart, with a10second total probe window. Retain all samples. Any
  compute PID, protected service active, memory>=100MiB or temperature>=80°C
  fails immediately: never wait through or stop an unrelated workload. Only
  nonzero utilization with no compute PID and otherwise safe state can settle.
  Each read subprocess has a timeout capped by the remaining probe budget.
- The child still runs the original strict idle-host check. One90second child,
  one180second user service, start before September7 17:00 Shanghai. The earlier
  unexecuted14:30 draft window expired; the user's16:20 follow-up renews this
  bounded closeout only, with no numerical or checkpoint changes. No retry,
  optimizer step, gain change or additional seed. If readiness fails again,
  retain the failure and stop.
- Numerical failure remains numerical-gate-stop; a clean pair is at most
  paired-case-support-only. All obstacle, hop, integrated-stability and physical
  acceptance flags remain false. Verify hashes and publish both the failed
  original evidence and separate closeout result without relabeling either.

Closeout prelaunch local regression: **711 passed**,11.62s, no skips, same two
existing warnings. New checks cover transient telemetry, two consecutive idle
samples, immediate refusal of another workload/protected service/high memory or
temperature, per-command deadline bounds, one child only, immutable original
manifest refusal, no replay, and failure after comparison reverting the overall
decision to runtime-failure-stop.
