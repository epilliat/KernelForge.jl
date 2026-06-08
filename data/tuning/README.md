# KernelForge tuning artifacts

Per-arch autotune outputs consumed at load-time by
[`src/tuning/loader.jl`](../../src/tuning/loader.jl). Each
architecture ships three files:

| File                  | Format        | Role                                          |
|-----------------------|---------------|-----------------------------------------------|
| `<Arch>.json`         | JSON          | Canonical, human-readable record of the sweep. Audited and diff-able. |
| `<Arch>.jl`           | Julia literal | Runtime artifact loaded by `load_tunings_for!` — populates `TUNING_TABLE` so `lookup_matvec` / `lookup_vecmat` can fill the knobs `resolve_parameters` doesn't get from the caller. |
| `<Arch>_inline.jl`    | Julia source  | Auto-generated `@inline` method overrides for `default_workgroup` / `default_blocks` / `default_nitem` on `MapReduce1D` and `Scan1D`. Loaded once per session for precompile-friendly dispatch. |

The loader degrades gracefully — when no file is present for the
detected arch, `lookup_*` return `nothing` and `resolve_parameters`
falls through to the heuristic `default_*` methods in
[`src/algorithms.jl`](../../src/algorithms.jl),
[`src/scan/scan.jl`](../../src/scan/scan.jl), and
[`src/mapreduce/1D/mapreduce1d.jl`](../../src/mapreduce/1D/mapreduce1d.jl).
Shipping a tuning is therefore pure upside for users on matching
hardware; users on un-tuned arches lose nothing.

## What ships

Currently only **RTX1000** (Turing TU117). Add tunings for your arch
by following the contribution flow below.

## Contributing a new arch

1. Clone KernelForge.jl and check out `main` (or the release branch
   you want to attach the tuning to).
2. Confirm the backend extension picks up your device:
   ```
   julia --project -e 'using KernelForge, CUDA; @show CUDA.functional()'
   ```
   For AMDGPU substitute `using AMDGPU` and `AMDGPU.functional()`.
3. Run the full sweep (≈ 55–60 min wall on a single GPU; see
   [`autotune_all.jl`](autotune_all.jl) header for individual
   per-op times). Smoke first with `KF_AUTOTUNE_DEV=1` (~30 s):
   ```
   KF_AUTOTUNE_DEV=1 julia --project data/tuning/autotune_all.jl    # smoke
   julia --project data/tuning/autotune_all.jl                       # real
   ```
   Outputs land in `data/tuning/<Arch>.{json,jl,_inline.jl}` —
   diagnostics (`*_diag.csv`, intermediate `.json.*` / `.jl.*`
   snapshots) are gitignored automatically.
4. Sanity-check that the canonical files load:
   ```
   julia --project -e '
     using KernelForge
     KernelForge.load_tunings_for!(:<Arch>)
     @show KernelForge.lookup_matvec(KF.<Arch>(), Float32, 1024, 1024)
   '
   ```
   You should see a `NamedTuple` of resolved knobs, not `nothing`.
5. PR the three new files (`<Arch>.json`, `<Arch>.jl`,
   `<Arch>_inline.jl`) plus a one-line entry above under "What ships".

The schema (`schema_version` field in the canonical `.json`) is
versioned — if you bump it, update the loader's compatibility check
in `src/tuning/loader.jl` in the same PR.

## Per-op driver files

Individual sweeps can be invoked standalone — useful when iterating
on a single op without re-running the others:

- [matvec/autotune.jl](matvec/autotune.jl) — see also
  [matvec/RELAUNCH.md](matvec/RELAUNCH.md) for the historical
  bug-fix narratives (do not undo those caps).
- [vecmat/autotune.jl](vecmat/autotune.jl)
- [mapreduce1d/autotune.jl](mapreduce1d/autotune.jl)
- [scan/autotune.jl](scan/autotune.jl)
- [sort_columns/autotune.jl](sort_columns/autotune.jl)

The driver scripts are backend-agnostic (CUDA + AMDGPU) and use the
`compile_kernel_only` hook from
[`src/tuning/precompile.jl`](../../src/tuning/precompile.jl) to warm
the GPUCompiler cache before measurement.
