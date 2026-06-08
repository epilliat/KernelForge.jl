# Matvec autotune — relaunch plan (RTX1000)

## Goal

Produce `data/tuning/RTX1000.json` with every cell of the pow-2/pow-8 grid
tuned for `KernelForge.matvec` on Float32. The lean grid has **94 cells
total** for the default `safety=4`. Resume picks up from whatever is
already in the JSON.

## What works (do NOT redo these)

Four real bugs/false-trails were fixed across iterations. Don't re-investigate them:

1. **Precompilation bug** in KernelForge. `oem_warp.jl` was double-included
   via `oem_shared.jl` (which had `include("oem_warp.jl")` at line 26) AND
   via `sample_sort.jl` (which had `include("oem_shared.jl")` at line 36) →
   `cmpex` defined twice → `ERROR: Method overwriting is not permitted
   during Module precompilation` → KernelForge loaded **uncached every
   process** (~10 min JIT each time). **Fixed:** nested includes removed,
   all `sort/*` includes centralized in
   [src/KernelForge.jl:55-67](../../../src/KernelForge.jl#L55) in dependency
   order. `using KernelForge` now precompiles in 8 s.

2. **LLVM compile blowup** at large `Nitem × chunksz × cld(wg, warpsz)`. The
   matvec kernel allocates `@localmem NTuple{Nitem,H}` of length
   `chunksz·cld(workgroup,warpsz)`
   ([matvec_kernel.jl:21](../../../src/mapreduce/2D/matvec_kernel.jl#L21)).
   LLVM/GPUCompiler scalarizes that aggregate → with `Nitem=16, chunksz=256,
   wg=512` (element count 32768) a single kernel compilation blew up to
   **15+ GB RSS** and crashed the machine (OOM, took VS Code with it).
   **Fixed** in [autotune.jl](autotune.jl) `param_candidates`:
   - `Nitem ≤ cld(32, sizeof(T))` (F32 → max 8)
   - `Nitem × chunksz ≤ 1024`
   - `Nitem × chunksz × cld(workgroup, warpsz) ≤ 4096` (element count)
   - `... × sizeof(H) ≤ 48 KB` (static-shared launch ceiling)

3. **The "hang" was a misdiagnosis.** Initial runs appeared "stuck at
   JSON=40 for 45 min with GPU 100%" — looked like a hung kernel. False.
   Two repros ([repro_hang.jl](repro_hang.jl), once with async+poll and once
   with `CUDA.@profile`) tested **3367 candidates with `@profile`** across
   cells 41-48, **0 hangs**. Two real causes of the appearance:
   - Old checkpoint cadence was every 10 cells, so JSON stayed at 40 until
     cell 50 done → looked frozen during cells 41-49.
   - Small-n cells (e.g. cell 48 = `n=1, p=2097152`) had **855
     candidates** in the old grid, most of them over-provisioned (lanes
     idle) and redundant → each cell legitimately took 10-30 min.

   **Fixed:**
   - Checkpoint every cell (not every 10) — JSON tracks progress live.
   - `Nitem × chunksz ≤ n` constraint in `param_candidates` — a workgroup
     covers `Nitem*chunksz` rows, so never provision more than `n`. Effect:
     `n=1` → 855 → **27** candidates; `n=4` → 855 → 162; large cells
     unchanged.

4. **StaticSize ndrange recompile** (the real RAM hog). Both kernel launches
   in [src/mapreduce/2D/matvec.jl](../../../src/mapreduce/2D/matvec.jl)
   used the positional 3-arg form `matvec_kernel!(backend, workgroup,
   ndrange)(...)` → KA bakes `ndrange` as `StaticSize` in the kernel type
   → GPUCompiler creates a **fresh specialization for every distinct
   `(n, p)` shape**, even when `(Nitem, chunksz, Nblocks, workgroup)` are
   the same. With 94 shapes × ~500 candidates each, the in-process
   GPUCompiler cache (no public mid-process release API) ballooned to
   17.89 GB → VS Code OOM-crashed; then to 14 GB after the orchestrator
   ceiling was lowered → cleaner kills but still messy.
   **Fixed**: ndrange moved to kwarg position at
   [matvec.jl:441-444](../../../src/mapreduce/2D/matvec.jl#L441) and
   [matvec.jl:471-475](../../../src/mapreduce/2D/matvec.jl#L471):
   `matvec_kernel!(backend, workgroup)(args...; ndrange = ndrange)`.
   Verified: a second call with new `(n, p)` but same params runs in
   **0.19 ms** (pure cache hit) instead of ~11 s (recompile).
   This is also a production speedup for any caller that benches matvec
   on multiple shapes in one session.

## Multi-pass exit (belt-and-braces)

The cumulative `(Nitem, chunksz, Nblocks, workgroup)` cache still grows
slowly even after fix #4. To avoid ever hitting the orchestrator's hard
RAM kill, autotune.jl now supports `max_cells_per_run` (kwarg) /
`AUTOTUNE_MAX_CELLS` (env var, defaults to 0 = unbounded). After tuning
that many fresh cells the process prints `PARTIAL_RUN_EXIT: …` and
`exit(0)` cleanly; the orchestrator detects the sentinel in the log and
relaunches, resuming from the checkpoint JSON.

The orchestrator exports `AUTOTUNE_MAX_CELLS=30` by default — keeps peak
RSS under ~8 GB per process. Adjust at the top of
`/tmp/autotune_orchestrator.sh` if needed.

## State of the runtime tooling

- [autotune.jl](autotune.jl) — the autotuner. Supports `resume=true`
  (loads existing JSON, skips done cells), skip-list (`/tmp/autotune_skip.txt`,
  cells the orchestrator flagged as stalling), per-cell marker file
  (`/tmp/autotune_cell.txt`, for stall detection), checkpoint after every
  cell. Backend-agnostic (CUDA / AMDGPU via 5 hooks defined inside `@eval
  begin…end` to defer macro expansion).
- `/tmp/autotune_orchestrator.sh` — hang-tolerant orchestrator: launches
  the autotune, monitors **VmHWM** (kill at 22 GB — protects the 30 GB
  machine) and **per-cell stall** (same cell > 10 min → kill, append to
  skip-list, relaunch). Loops until "Done." appears in the log.
- `/tmp/autotune_watchdog.sh` — legacy RAM-only watchdog; the orchestrator
  subsumes it.
- [diag_hang.jl](diag_hang.jl), [repro_hang.jl](repro_hang.jl) — kept for
  posterity; the hang investigation is closed.

## Cleanup before relaunch

If a previous run is somehow still alive:

```bash
kill -9 $(cat /tmp/autotune_pid 2>/dev/null) 2>/dev/null
opid=$(ps -eo pid,args | awk '/autotune_orchestrator\.sh/ && !/awk/{print $1}')
[ -n "$opid" ] && kill -9 $opid
```

Then verify: `pgrep -af 'autotune\.jl|autotune_orchestrator'` should be empty
and `nvidia-smi --query-gpu=memory.used --format=csv,noheader` should be ~10 MiB.

The skip-list should be empty for a clean run:
```bash
: > /tmp/autotune_skip.txt
```

## Relaunch — recommended (hang-tolerant orchestrator)

```bash
cd /home/emmanuel/Packages/KernelForge.jl
setsid nohup bash /tmp/autotune_orchestrator.sh > /dev/null 2>&1 < /dev/null & disown
```

`setsid` reparents the orchestrator off the VS Code/harness shell tree
(PPID becomes init/systemd) so a VS Code crash does **not** kill it. The
orchestrator itself launches julia as its child. Verify after ~5 s:

```bash
pgrep -af 'autotune_orchestrator|autotune\.jl'
```

You should see the orchestrator bash + a julia child. The julia's PPID
should be the orchestrator's PID, NOT `256641` (the harness shell PID at
the time of writing — adapt).

## Relaunch — minimal (no orchestrator, just detached julia)

If the orchestrator is overkill (post-fix, the hangs were spurious so the
stall-kill is mostly insurance):

```bash
cd /home/emmanuel/Packages/KernelForge.jl
setsid nohup julia --project=perfs/envs/benchenv/cuda \
    data/tuning/matvec/autotune.jl \
    > /tmp/autotune_rtx1000_f32.log 2>&1 < /dev/null & disown
sleep 5
ps -eo pid,comm,args | awk '$2=="julia" && /autotune\.jl/{print $1}' > /tmp/autotune_pid
```

## Monitoring

```bash
# Cell count in the JSON (advances per cell now):
julia -e 'using JSON3; println(length(JSON3.read(read("/home/emmanuel/Packages/KernelForge.jl/data/tuning/RTX1000.json",String))["matvec"]["size_4"])," cells")'

# Peak RSS recorded by the orchestrator/watchdog:
awk '{printf "%.2f GB\n", $1/1048576}' /tmp/autotune_peak_rss.kb

# Live process state:
ps -eo pid,etime,pcpu,args -p "$(cat /tmp/autotune_pid)" 2>/dev/null
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader

# Orchestrator log (events: launch, RAM kill, stall, Done):
cat /tmp/autotune_orchestrator.log
```

## Expected timing

With the lean grid + precompilation fixed:

- Module load: ~30 s-1 min (KernelForge cached, CUDA ext warm).
- First cell on a fresh process: ~3-5 min (compile ~all distinct matvec
  GPU kernel specs).
- Subsequent cells: ~10-60 s each (reuse compiled kernels). Small-n
  cells are now ~27-470 candidates, not 855.
- **Total for ~69 remaining cells: ~30-60 min.** Peak RSS ~6-9 GB (well
  under the 22 GB watchdog limit).

## Verification when done

- `/tmp/autotune_rtx1000_f32.log` ends with `Done. N cells tuned, total
  wall = …`.
- `/tmp/autotune_orchestrator.log` has `AUTOTUNE DONE`.
- `data/tuning/RTX1000.json` has ~105-109 cells in `matvec.size_4`
  (less than 109 if some cells were skip-listed; those will use the
  in-source heuristic at lookup time).
- The skip-list `/tmp/autotune_skip.txt` is empty or contains only a
  handful of (n, p) pairs.

## File map

| path | role |
|---|---|
| [data/tuning/matvec/autotune.jl](autotune.jl) | autotuner (the main script) |
| [data/tuning/matvec/repro_hang.jl](repro_hang.jl) | hang repro (CUDA.@profile path) — investigation closed |
| [data/tuning/matvec/diag_hang.jl](diag_hang.jl) | hang diag (async+poll path) — investigation closed |
| `data/tuning/RTX1000.json` | output (40 cells currently, target ~109) |
| `data/tuning/RTX1000_dev.json` | dev smoke output (ignore) |
| `/tmp/autotune_orchestrator.sh` | hang-tolerant orchestrator |
| `/tmp/autotune_watchdog.sh` | legacy RAM watchdog |
| `/tmp/autotune_pid` | julia PID for orchestrator/scripts to read |
| `/tmp/autotune_cell.txt` | per-cell marker (stall detection input) |
| `/tmp/autotune_skip.txt` | cells the orchestrator declared stalling |
| `/tmp/autotune_peak_rss.kb` | running peak VmHWM in KB |
| `/tmp/autotune_rtx1000_f32.log` | autotune stdout/stderr |
| `/tmp/autotune_orchestrator.log` | orchestrator events |

## Wider context (for a fresh agent)

The full project reference is [`AGENTS.md`](../../AGENTS.md) at the repo
root. Skim sections:
- *Repo layout* — `src/` structure.
- *Public API signatures* — `matvec`, `matvec!`, the allocation contract.
- *Architecture types* — `CUDAArch` / `AMDArch` hierarchy, `detect_arch`.
- *Running benchmarks* — for the bench protocol the autotune uses.

The matvec source the autotune drives:
- [src/mapreduce/2D/matvec.jl](../../../src/mapreduce/2D/matvec.jl) —
  public API, `resolve_parameters` (the validity asserts mirrored in
  `param_candidates`).
- [src/mapreduce/2D/matvec_kernel.jl](../../../src/mapreduce/2D/matvec_kernel.jl)
  — the `@kernel`. Note the `@localmem NTuple{Nitem,H} ...` at line ~21 —
  that aggregate is what blows LLVM up at large `Nitem·chunksz·cld(wg,warpsz)`.
