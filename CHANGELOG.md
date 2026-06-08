# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [SemVer](https://semver.org/spec/v2.0.0.html). Pre-1.0:
minor bumps may include breaking changes; they will always be called out
under **Breaking changes** below.

## [0.2.0]

### Breaking changes
- `KernelForge.sort1d!` is renamed to `KernelForge.sort!`. The
  `helpers.jl` internal `sort!` call has been qualified as
  `Base.sort!` so the meaning doesn't drift; downstream callers
  using `KF.sort1d!` must update to `KF.sort!`. No deprecation shim
  is shipped — per the package's minimal-shim policy. See
  commit `df6937e`.

### Added
- `KernelForge.Random` submodule — Philox4x32-10 counter-based GPU
  RNG with Uniform / Normal / Exponential / Bernoulli distributions,
  plus `randperm!` built on the new `sortperm!`. Requires Random123
  for matching CPU-side reference streams in tests.
- `KernelForge.sortperm!` — returns the permutation index vector
  that sorts a 1D source array. Built on the keyval one-sweep radix.
- `KernelForge.sample_sort!` — GPU sample sort accepting an
  arbitrary `lt` comparator, for bitstypes outside the radix path.
- `KernelForge.sort_columns!` — batched per-column sort for K × M
  matrices (OEM + batched-radix dispatch).
- `KernelForge.sort!(src; keys=...)` — keyval radix sort variant.
- `KernelForge.sort!` in-place form (parity-aware ping-pong, no
  extra copy when the radix pass count is odd).
- Per-arch tuning table — `data/tuning/<Arch>.{json,jl,_inline.jl}`
  loaded by `KernelForge.load_tunings_for!`. Ships RTX1000 (Turing
  TU117) tunings; matvec / vecmat resolve knobs via `lookup_matvec`
  / `lookup_vecmat` ahead of the heuristic `default_*` methods.
  Falls through cleanly on un-tuned arches.
- Autotune drivers under `data/tuning/{matvec,vecmat,mapreduce1d,
  scan,sort_columns}/autotune.jl` + master `autotune_all.jl`. See
  `data/tuning/README.md` for the contribution flow.
- New benchmark scripts: `sort_perf_comparison.jl`,
  `sort_columns_perf_comparison.jl`, `random_perf_comparison.jl`,
  `randperm_perf_comparison.jl`. CUB + Thrust C++ benchmarks under
  `perfs/cuda_cpp/cub_nvcc/` (Makefile, two `.cu` files).
- `KernelForge.num_sms(::Backend)` interface + per-backend
  implementations.
- `KernelForge.compile_kernel_only(kobj; ndrange)` — compile-only
  path for the autotune harness; mirrors KA's CUDA / ROC functor
  prelude up to `@cuda/@roc launch=false`.

### Changed
- Benchmark result CSVs moved from in-tree `perfs/julia/results/` to
  the sibling repo
  [KernelForge-benchmarks](https://github.com/epilliat/KernelForge-benchmarks).
  Bench / plot scripts resolve the results root via:
  `KF_RESULTS_ROOT` env → `../KernelForge-benchmarks/results` →
  in-tree fallback (gitignored). See commit `48c3bb4`.
- Scan / matvec / vecmat launch sites switched from positional
  `ndrange` (which baked `ndrange` into the kernel as a `StaticSize`
  type parameter and triggered a recompile per distinct value) to
  the kwarg form `kernel!(be, wg)(...; ndrange=…)`.

### Fixed
- `@allocate` macro now preserves the `Expr(:parameters, ...)`
  kwargs block at position 2 of the rewritten call. Previously the
  tag landed before the kwargs block, yielding a malformed call.
  Unblocks `@allocate sort!(src; keys=...)` callsites.
- `matvec_kernel` tail predicate uses `lid` (workgroup-local index)
  instead of `lane` (warp-local). Same value when workgroup ≡ one
  warp, but wrong for multi-warp workgroups where multiple warps
  would each believe they owned the last row chunk.

### Added (deps)
- `Adapt` 4 — used by the tuning loader for backend-portable
  storage of the tuning tables.

## [0.1.4]

See `git log v0.1.4` for the v0.1.x history.
