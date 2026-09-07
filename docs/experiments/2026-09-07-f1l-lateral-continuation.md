# F1-L: paired motor-aware lateral-velocity continuation

Status: **predeclared, not executed**. User authorized the next gait-training
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
