# October 1 Duck development window and C1-S result

The user authorized Duck development through **2026-10-01 18:00 Asia/Shanghai
(10:00 UTC)** and the proposed walker-first CPU diagnostic. This is a new
window, not an extension of an expired September campaign. Use
`feat/athletics-obstacle-curriculum`; no physical Duck is available.

## Closed first attempt

The [C1-S declaration](2026-10-01-community-hop-c1s-predeclaration.md) and
[runner](2026-10-01-community-hop-c1s-runner.md) were exercised from clean
`e095ffa179feae232d1282c9793b63a45bfc5edb`:

```sh
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m mjlab_microduck.community_hop_diagnostic --execute-approved
```

Retained, ignored directory (no raw policy weights committed):

```text
artifacts/community/c1s-velstand-substitution-v1/20261001T033238Z-7202669610a8423b999c6a53218db34b
```

Decision: **`walk-only-rejected-transfer-not-run`**. The 12 s walker-only
control completed 600 CPU policy inferences, 2,400 physics advances and 2,401
measured states. Campaign wall time was 4.09304 s; the case child exited normally
after 2.52875 s. No fatal event, runtime error, timeout, reset or missing journal
record occurred. Happy Hop was **not invoked**. This consumed attempt cannot
be retried automatically or reinterpreted as a failed author hop experiment.

| Evidence | Result | Unchanged indicator |
| --- | --- | --- |
| Approach forward-speed MAE, 2–3 s | 0.20009136 m/s | Failed 0.08 m/s limit against 0.20 m/s command |
| Resume forward-speed MAE, 11–12 s | 0.20001691 m/s | Failed same limit |
| Stable entry and entire 8–9 s zero-command window | Both passed | Required before transfer |
| Qualified airborne episodes, 4–9 s | Zero | Passed control requirement, not a hop achievement |
| Hop/settle planar drift | 0.00080010 m | Passed 0.10 m limit |
| Peak modeled current | 0.55529024 A | Passed 1.75 A limit |
| Peak modeled torque | 0.20324372 Nm | Descriptive, not hardware measurement |
| Peak absolute summed mechanical power | 0.23997199 W | Descriptive, not thermal calibration |
| Soft-limit-exposed substeps | Zero | Passed |

The positive command reached observation index 48 and the declared initial
one-control-step action delay was preserved. The supported symptom is nearly
stationary behavior under the walking command. Its cause remains unresolved.
The walker manifest supplies a branch but no exact training commit; ONNX
`run_path` is `None`. Our lag/plant choices remain surrogate settings, not
recovered author training/export semantics.

## Immutable receipts

- Manifest SHA256: `d9893807e4f7dcf1e70fde8e7a7aae7da3ac0ec40b53cf40f084c88e4ddbfa21`.
- Case report SHA256: `7c1c254e41562b4f8442000794a589076a5b511720a91d7a5752504c840a4145`.
- Journal SHA256: `83ce5761de0c4870c588fa577ff9f0bed3176d9b1a74d8c23d891d656abe55e3`.
- Trace SHA256: `8f9e983d4cf046d4f2d5216ade13465c6079470c2ff58555fc765cab47f708d2`.
- Compiled MJB SHA256: `23c4b58780681c4d7645abc8d3f5f27b80467ee94ef9fa8e75aaf09a5dcc6fc5`.
- Declaration SHA256: `236d7ca25e1a0508e6a913944f822a44244fb032362bd4ee3a38e5315c9eb700`.
- Frozen scorer SHA256: `019ade48269ef5bbcf59abd9f8160dbf87caa1432d834fcbb6e6bcf5744b3ef8`.

Copied declarations, source receipts, compiled plants, journal, recovered trace,
report, child records and decision remain in the original directory. Store
offline analysis elsewhere; never overwrite its manifest. Hash consistency is
not independent simulator authentication. Author replication, skill acceptance,
training/promotion, impact/thermal calibration, deployment and physical-motion
flags remain false.

## Bounded sequence and current handoff

1. Integrate an offline manifest/journal/scorer audit: actual actor-input
   command/history checks and descriptive phase statistics, no new inference
   or physics. Test, review, commit and push only the feature branch.
2. Inspect exported normalization and local source conventions statically.
   Both pinned exports contain `obs -> Sub -> Div -> Gemm` and nonzero
   first-layer velocity-command columns. This excludes absent affine
   preprocessing or disconnected first-layer columns, **not** effective
   downstream sensitivity or training parity. Keep `normalizer_verified` false.
3. Prepare a source-grounded contract/gap inventory for the smallest next
   diagnostic. Change one stated axis, use a new identity and fixed budget,
   preserve measurements, and distinguish delivery from policy/plant effects.
   Do not guess the missing author walker or relabel a stock policy as it.
4. Continue fixture-tested compatibility/curriculum work within the window.
   Walking/stop stability precedes hop handoff. Avoidance may slow in the
   interaction zone, but must recover speed before/after it. Preserve separate
   stock-hop, sprung-H1, stance and football tracks and historical results.

The consumed C1-S attempt is closed; transfer remains blocked. Offline audit
and affine-prefix tooling below are implemented and tested. The next code
slice is the source-grounded contract/gap inventory, not another rollout. Never run its
`--execute-approved` entry point again under the same declaration. Subsequent
handoffs must name actual source commits, retained evidence, checks and the
next bounded executable step. Missing author provenance does not justify
weakening the walker baseline.

No blind training restart, gate/contact/motor relaxation, raw perception,
observation expansion, MP4, physical motion, deployment, dependency/driver
upgrade or unrelated service change. A new policy execution needs its own
complete predeclaration and scoped authority. A long GPU job additionally
needs fresh host/runtime qualification, an idle GPU and a budget fitting the
remaining window; this CPU probe qualifies neither remote host.

## Scheduled continuation and hard stop

Same-chat continuation through 18:00 was requested. Its actual creation/status
is the app scheduling record, not a claim made by this document. On each run,
verify the clock, branch/worktree and latest handoff. Preserve dirty/external
work; never change an active diagnostic's source or overlap workloads. Report
completed milestones or real failures, not repetitive unchanged status.

At or after 18:00 Shanghai today, start **no new Duck work**. Confirm owned
Duck children are safely complete/stopped and evidence/checkpoints are durable.
Leave unrelated services unchanged, delete the matching continuation and report
the retained state. Do not restore protected AI services without a new request.

## Offline evidence tooling

The [campaign auditor](../../src/mjlab_microduck/community_hop_evidence_audit.py)
is now implemented. It checks every manifest entry, declaration/scorer bytes,
setup/case identities, compiled-plant hashes, CPU worker/child receipts, exact
journal event order, observation command/raw-action history, carried delay and
held actions, frozen measurements and sequential decisions. It reads only
complete quiescent evidence; incomplete or malformed records fail closed.
File reads are bounded and reject symlinks/nonregular files without blocking
on a FIFO. Large compiled models are hash-streamed. An optional `--output`
is exclusive and must be outside the original campaign.

```sh
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m mjlab_microduck.community_hop_evidence_audit \
  artifacts/community/c1s-velstand-substitution-v1/20261001T033238Z-7202669610a8423b999c6a53218db34b
```

The [static affine-prefix inspector](../../src/mjlab_microduck/community_policy_normalizer.py)
recognizes only fixed float32 `Sub -> Div -> Gemm` wiring on the metadata-bound
API1 layout, rejects bypasses/overridable constants, and retains all 61 mean
and positive denominator values. It never executes the actor. For the actual
walker, velocity index 48 has mean 0.04126639 and denominator 0.20790750;
all 512 first-layer coefficients in that column are nonzero. Happy Hop's
corresponding values are 0.05079065 and 0.12259971, also 512 nonzero entries.
These values describe the pinned graph, not a recovered training specification.
`normalizer_verified` and effective command-sensitivity verification stay false.

The auditor re-scores the real retained trace as
`walk-only-rejected-transfer-not-run`, with all 18 manifest files consistent.
Initial walking-window mean forward velocity was 0.00263521 m/s and displacement
0.00790563 m; double-foot contact fraction was 0.99166667. Resume-window mean
velocity was 0.00235844 m/s. These are descriptive statistics, not new gates.
The protocol's `hop` phase label in a walker-only report means its zero-command
time window, **not** hop-policy execution.

Read-only Luna reviews checked the trace/command mapping and normalizer
recognition. A bounded Luna implementation supplied the audit and fixtures;
owner integration added nonblocking special-file reads and strict runtime
integer-type checks. Synthetic test runs do not consume another public attempt.

Validation: **384 scoped tests passed**, including 47 new offline audit and
static-prefix checks plus the unchanged C1/H1/static-retention selection.
`git diff --check`, Markdown rendering and all changed documents' local links
passed. Full repository tests, new public-policy attempts, GPU host/runtime
qualification, training, video and physical tests were not run. The immutable
declaration and frozen scorer SHA256 values above were rechecked unchanged.
