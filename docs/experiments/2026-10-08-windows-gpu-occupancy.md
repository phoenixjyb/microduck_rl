# Windows-side GPU observation, 2026-10-08

This is a read-only capacity observation, not native Duck qualification or a
permission to start training. Windows and WSL share the physical RTX PRO 4000
Blackwell; an empty WSL compute-process list does not establish Windows idleness.

Use `scripts/inspect_windows_gpu.ps1` from Windows PowerShell, or pass its exact
UTF-16LE encoded contents through WSL interoperability. It samples Windows'
NVIDIA telemetry, CIM dedicated-memory records resolved to process names, and
active engine records. Default: three samples, two seconds apart. No process,
service, driver, environment or cache is changed. Counts and intervals are
bounded. CIM memory records and engine instances can include other adapters;
retain their LUID/physical-instance names and do not blindly sum utilization.
Samples are point-in-time observations, not a continuous lease.

At 13:06–13:08 Asia/Shanghai, Windows NVIDIA telemetry reported 3,384–3,416 MiB
allocated out of 24,467 MiB, with 0% sampled utilization. WDDM process records
included Explorer, ShellHost, SearchHost, TextInputHost, Copilot and WebView2.
Windows CIM exposed a much larger dedicated allocation for PID 5744, `ugraf`,
about 3,089,944,576 bytes (2,946.80 MiB). This allocation was not identifiable
from the empty WSL compute-process query, and Windows NVIDIA's own WDDM process
table was also incomplete relative to CIM. Do not interpret missing process
rows as proof that a consumer has released its memory.

The existing FilmBrain user services stayed active at PIDs 521 and 298048;
protected AI-mission user services stayed inactive. No Duck GPU job started.

## Delivery checks

Windows parser validation, four out-of-range parameter refusal tests and the
static read-only command-profile check passed. The probe reports
`idle_gpu_proven=false`, `training_authorized=false` and
`workloads_changed=false` irrespective of instantaneous utilization.

### Retained exact-source execution

Collector source `bc162cdf3227df8487afbe63b886edbd5105d1be`; whole PowerShell
script SHA256 `599e5e8a657d51557b159c3ed926e12f012d5ddef3ba8f4d46f9412fe2d4c351`.
The Mac owner checked the clean exact branch, whole committed script bytes and
Windows machine `DESKTOP-HNKBDR1`, and ran those exact encoded bytes via WSL
interop. Source and script remained unchanged before/after. No remote checkout
was used as a claim of executed-source identity.

The first owner guard rejected the script's existing `Set-StrictMode` command
before execution because the owner allowlist omitted it. Read-only AST inspection
identified the omission. Only that local allowlist was repaired; the collector
script bytes did not change. The failed evidence directory was retained, and
the successful run used a fresh `-guard-fix` directory. No unrelated command or
mutation was allowed as a workaround.

Three observations were retained at 13:34:23, 13:34:26 and 13:34:31 Shanghai:

| Sample | NVIDIA used MiB | NVIDIA utilization snapshot | Temperature | CIM `ugraf` dedicated bytes |
| --- | ---: | ---: | ---: | ---: |
| 1 | 3,530 | 0% | 45 C | 3,709,870,080 |
| 2 | 3,794 | 0% | 45 C | 3,709,870,080 |
| 3 | 7,561 | 0% | 46 C | 8,347,607,040 |

The last CIM engine sample recorded `ugraf` PID 5744 / LUID `0x0001224A`
at **14% 3D-engine utilization**. Other active engine instances used LUID
`0x00011E90`; Windows exposes both NVIDIA and Intel graphics, so percentages
must not be summed or blindly assigned to the NVIDIA adapter. NVIDIA and CIM
queries occur sequentially and have different refresh windows: their counts
and utilization snapshots are not simultaneous and need not numerically match.
In particular, NVIDIA's earlier 0% snapshot did **not** establish an idle lease.

Windows NVIDIA now listed PID 5744 but could not provide its process name or
memory usage (`Insufficient Permissions` / `N/A`). CIM/Get-Process resolved
the name `ugraf`; executable-path/version access remained unavailable. No
privilege escalation or guessed program identity was used. No CUDA training
process was positively identified; that does not exclude unidentifiable or
intermittent work. Memory use is dynamic and substantial, so do not launch a
Duck GPU job from these observations.

Evidence root:
`artifacts/tools/windows-gpu-snapshot-bc162cdf3227-guard-fix`.

| Whole retained file | Bytes | SHA256 |
| --- | ---: | --- |
| stdout JSON | 9,266 | `5557b2ee1c2b1037680371f8af649a7a2fd828511e7b3f23a9a30f125d8da445` |
| stderr | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| owner proof | 576 | `8f1b0629978e70d1fd1042ad97acfc7de047f4947bca4bf0e25fe02f36db5d1b` |

Proof records parser/profile/parameter checks, exact source and script hash,
whole output inventory, timing, and false GPU-workload/mutation flags.
FilmBrain remained active at its original PIDs, protected services remained
inactive in user/system scopes, and the WSL environment alias was unchanged.

## Consequence for the next development step

Continue CPU-only preparation for the reviewed
[literal solver-target binding](2026-10-08-solver-binary-scope.md). Before a native
probe, review both Windows and WSL telemetry immediately before and throughout
the bounded job. Do not stop `ugraf`, desktop applications or FilmBrain to obtain
capacity. A sustained idle sample, application memory allocation or apparent free memory
alone is not numerical qualification or a long-training reservation.
