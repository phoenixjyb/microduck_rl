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

The consumed C1-S attempt is closed; transfer remains blocked. Offline audit,
affine-prefix tooling and the source-grounded gap inventory below are implemented.
The next slice is the separate WSL finite-check performance diagnosis, not
another community-policy rollout. Never run its
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

Continuation through 18:00 was requested. The user clarified that this session
runs in the Codex CLI. App scheduling calls did not return a confirmation or
register a task; do not claim a background automation exists. Continue in the
active CLI session. On each continuation,
verify the clock, branch/worktree and latest handoff. Preserve dirty/external
work; never change an active diagnostic's source or overlap workloads. Report
completed milestones or real failures, not repetitive unchanged status.

At or after 18:00 Shanghai today, start **no new Duck work**. Confirm owned
Duck children are safely complete/stopped and evidence/checkpoints are durable.
Leave unrelated services unchanged, delete the matching continuation and report
the retained state if a matching automation actually exists. Do not restore
protected AI services without a new request.

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

### Retained offline receipt and executable handoff

Offline analysis from clean `a9d1a0d47c12de4a4b76958382dd22c4d18d1dbc` is
retained separately in:

```text
artifacts/community/c1s-offline-audit-v1/20261001T035849Z-3cd581a01d024ae2b20aa7ab739e1dcf
```

Its manifest SHA256 is
`94d297d7d96bd043fb752cfdf1edf1037970fc566a824ed24f1600eb33f5aa56`;
audit SHA256 is
`092712c71e77eb2c7822693c5dab0c5dd276defddb024ad0a54d741da29ee179`.
Four manifested JSON records contain the audit, both exact static-prefix
descriptions and a source/hash receipt. Analysis executed zero actor forwards,
policy inferences or physics advances. The [compact result index](2026-10-01-community-hop-c1s-result-index.json)
is committed so the fork retains identifiers even though large local traces,
compiled models and public weights remain ignored and unmodified.

The [source-gap inventory](../../src/mjlab_microduck/community_source_gaps.py)
is now implemented. It pins both releases' policy/card bytes, retains literal
graph metadata and hashes the affine constants, and uses the existing six-field
static-skill vocabulary. It reports all effective author receipts as missing,
plus the exact entry walker, task/run overlay and transition-state handoff gaps.
It has no receipt-admission or policy-execution switch and never synthesizes a
skill descriptor. The producing commit remains unknown; a branch, export date,
`run_path=None`, upstream base or named checkpoint is not substituted for it.

```sh
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m mjlab_microduck.community_source_gaps .
```

The actual report is 12,914 bytes, SHA256
`f33438574cba501d852ef76912027342979bee2a62615be23ccd856a0004a615`, with twelve
per-candidate effective-receipt gaps plus three shared handoff gaps. It loaded
no Torch, ONNX Runtime, MuJoCo, Warp or BAM module and executed zero actor
forwards, inference or physics. Fixture tests use temporary inert intake files,
not ignored local policy/card dependencies.
The report is retained unchanged at
`artifacts/community/source-gaps-v1/20261001-pinned-c1-source-gaps.json`.
Its 23 focused tests passed; the following broader selection passed 408 CPU
checks, with the 110 new-tool checks separately repeated after final integration.

Missing source/overlay/entry identity must still be reported as missing;
matching 61/14 shapes or these static affine constants cannot fill it. Reuse
the existing skill compatibility and retained-binding contracts rather than
creating another unvalidated switch path. Do not run a new policy while that
contract is incomplete. Remote host/runtime qualification remains a separate
read-only step, not a result of these local CPU tools.

The next separately predeclared [WSL finite-check probe](2026-10-01-wsl-finite-check-probe.md)
tests a fresh packed finite predicate on identical tensors. It leaves the old
runtime and every physical check unchanged and does not admit training or a
community-policy retry.

That probe has now completed successfully with source `80f1749bb1eb`, all 33
fault errors agreeing and unchanged input bits. Fresh-packing times were
0.104, 0.113 and 0.110 times the paired legacy times. WSL and Mac CPU evidence
replay passed; the runtime remained at clean `987b452dbfdb`, and FilmBrain was
preserved. The linked experiment retains exact launch/report/tool/input hashes.
This is a checker-only result, **not** learned capability or end-to-end training
qualification. Next review a minimal opt-in runtime integration, then obtain
fresh source-bound device and full collection/optimizer evidence without
widening the existing WSL budgets.
