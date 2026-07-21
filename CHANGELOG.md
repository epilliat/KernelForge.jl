# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [SemVer](https://semver.org/spec/v2.0.0.html). Pre-1.0:
minor bumps may include breaking changes; they will always be called out
under **Breaking changes** below.

## [Unreleased]

### Added
- `KernelForge.gemm` / `gemm!` — generalized matrix–matrix product
  `C[m,n] = g(op_k f(A[m,k], B[k,n]))`. Arbitrary isbits element
  types (custom structs) and arbitrary operators: the accumulator is
  seeded from a real `f(A[m,1], B[1,n])` and there is no split-K, so
  `op` needs neither an identity element nor associativity or
  commutativity. All four transpose states via `tA`/`tB` ∈ `{:N, :T}`,
  an `accT` accumulation-type override, and per-family tile knobs.
  `Int8 × Int8` accumulates in — and returns — `Int32`.
- `gemm(...; family=:mma)` — opt-in tensor-core (WMMA / MFMA) family
  for plain (`*`, `+`) products whose (compute, accumulate) pair the
  device announces: `Float16`/`BFloat16` → `Float32`, plus `Float64`
  and `Int8` → `Int32` where the hardware exposes them. Auto never
  selects it (it is slower than the generic kernel at small sizes);
  forcing it on an unsupported configuration errors rather than
  falling back silently.

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
- `KernelForge.sample_sort` — GPU sample sort accepting an
  arbitrary `lt` comparator, for bitstypes outside the radix path.
  Allocates its own workspace internally (no bang form — pass a
  `SampleSortWorkspace` for the pre-allocated variant).
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
- `sample_sort` leaf cap is an autotunable per-arch hook
  (`default_leaf_max(arch, T)`) with a caller override (`leaf_max=`)
  and a new `data/tuning/sample_sort/autotune.jl` driver. The cap is
  clamped to `[LEAF_MAX, leaf_max_cap(T)]` (the static-`@localmem`
  ceiling: 8192 for ≤4-byte keys, 4096 for 8-byte).

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
- `mapreduce` now coalesces loads of small isbits-**struct** eltypes.
  For a 1/2/4/8/16-byte non-primitive isbits `T`, the 1-src path
  reinterprets to the same-size unsigned (host-side, so the load
  widens under the LLVM vectorizer) and reconstructs `T` per value
  via a GPU-safe field decomposition. Fixes a ~1/16-bandwidth cliff:
  A100 `UnitFloat8→Float32` at 1e9 went 3123→663 µs (4.71×), closing
  the gap to CUB from 5.14× to ~1.09×.
- `sort!` on the sample path now threads a caller-supplied
  `SampleSortWorkspace` through the `tmp` argument (allocation
  contract): the radix / keyval paths reject a `SampleSortWorkspace`
  and the sample path rejects a plain `KernelBuffer`, each with a
  clear error, so a mis-typed pre-allocation fails loudly instead of
  silently re-allocating.

### Added (deps)
- `Adapt` 4 — used by the tuning loader for backend-portable
  storage of the tuning tables.

## [0.1.4]

See `git log v0.1.4` for the v0.1.x history.
