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

Pending Windows parser validation and retained three-sample execution. The
probe reports `idle_gpu_proven=false`, `training_authorized=false` and
`workloads_changed=false` irrespective of instantaneous utilization.

## Consequence for the next development step

Continue CPU-only preparation for the reviewed
[literal solver-target binding](2026-10-08-solver-binary-scope.md). Before a native
probe, review both Windows and WSL telemetry immediately before and throughout
the bounded job. Do not stop `ugraf`, desktop applications or FilmBrain to obtain
capacity. A sustained idle sample, application memory allocation or apparent free memory
alone is not numerical qualification or a long-training reservation.
