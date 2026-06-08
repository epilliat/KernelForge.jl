# API Reference

## Copy
```@docs
KernelForge.vcopy!
KernelForge.setvalue!
```

## Map-Reduce
```@docs
KernelForge.mapreduce
KernelForge.mapreduce!
KernelForge.mapreduce2d
KernelForge.mapreduce2d!
KernelForge.mapreduce1d
KernelForge.mapreduce1d!
KernelForge.mapreducedims
KernelForge.mapreducedims!
```

## Scan
```@docs
KernelForge.scan
KernelForge.scan!
```

## Search
```@docs
KernelForge.findfirst
KernelForge.findlast
KernelForge.argmax1d
KernelForge.argmax1d!
KernelForge.argmax
KernelForge.argmin
```

## Matrix-Vector
```@docs
KernelForge.matvec
KernelForge.matvec!
KernelForge.vecmat
KernelForge.vecmat!
```

## Sort
1D radix sort, permutation sort, general-comparator sample sort, and
batched per-column sort. The `sort1d!` name from v0.1.x has been
renamed to `sort!` — see the CHANGELOG for migration.

```@docs
KernelForge.sort
KernelForge.sort!
KernelForge.sortperm
KernelForge.sortperm!
KernelForge.sample_sort
KernelForge.sort_columns
KernelForge.sort_columns!
```

## Random
Philox4x32-10 counter-based GPU RNG with parameterised distributions.
Deterministic for a given seed across CPU and GPU.

```@docs
KernelForge.Random
KernelForge.Random.rand!
KernelForge.Random.randn!
KernelForge.Random.randperm!
KernelForge.Random.Uniform
KernelForge.Random.Normal
KernelForge.Random.Exponential
KernelForge.Random.Bernoulli
KernelForge.Random.Categorical
```

## Backend introspection
```@docs
KernelForge.num_sms
```

## Utilities
```@docs
KernelForge.get_allocation
```