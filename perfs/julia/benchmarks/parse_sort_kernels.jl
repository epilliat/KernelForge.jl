# Collate per-kernel profiler stats (rocprofv3 kernel_stats.csv on AMD, or nsys
# gpukernsum.csv on NVIDIA) into a uniform sort_kernels.csv.
#
# Input: KF_STATS_DIR holding files named `<method>__<type>__<N>__stats.csv`
#        (method in KernelForge|rocPRIM|CUB; type in UInt32|UInt64|Float32|Float64).
# Output: KF_OUT (default sort_kernels.csv), schema:
#   n, type, method, kernel, calls_per_sort, per_sort_us, min_dispatch_us, pct
# where per_sort_us is that kernel's total GPU time contribution per ONE sort, and
# pct its share of the sort's GPU time. n_sorts is inferred per-file from the
# onesweep call count / #passes (4 for 4-byte keys, 8 for 8-byte) so we don't need
# to know the driver's warmup/iteration counts.
#
# Kernel-name shortening (both vendors + KF) to a stable short tag.
using Printf

const _NPASSES = Dict("UInt32"=>4, "Float32"=>4, "UInt64"=>8, "Float64"=>8)

shorten(name) =
    # KF (Julia) + rocPRIM (hipCUB) + CUB (NVIDIA) onesweep / histogram / scan kernels.
    occursin("onesweep_kernel", name) || occursin("onesweep_iteration", name) ||
        occursin("Onesweep", name) ? "onesweep" :
    occursin("bucket_histogram_count", name)   ? "hist_count" :
    occursin("bucket_histogram_combine", name) ? "hist_combine" :
    occursin("scan_histogram", name) || occursin("onesweep_scan", name) ||
        occursin("ExclusiveSum", name) ? "scan_hist" :
    occursin("bucket_histogram", name) || occursin("onesweep_histograms", name) ||
        occursin("RadixSortHistogram", name) ? "hist" :  # single-stage (A100) + vendors
    occursin("radix_sort_", name) && occursin("Ontile", name) ? "onesweep" :
    # H2D reset copy (`copyto!(s, h)` re-prep in the CUDA.@profile driver) is a
    # profiling artifact, NOT part of the sort — tagged "reset" and dropped below.
    # rocprofv3 --kernel-trace never sees it (memcpy isn't a kernel), CUDA.@profile does.
    occursin("pageable to device", name) || occursin("host to device", name) ||
        occursin("HtoD", name) ? "reset" :
    occursin("fill", name) || occursin("Fill", name) || occursin("memset", name) ||
        occursin("Memset", name) || occursin("set device memory", name) ? "fill" :
    occursin("copy", name) || occursin("Copy", name) || occursin("DtoD", name) ? "copy" :
    "other"

# Manual quote-aware CSV line split: the first field (kernel Name) is quoted and
# may contain commas; the rest are plain numbers.
function split_stat_line(line)
    if startswith(line, '"')
        j = findnext('"', line, 2)
        name = line[2:j-1]
        rest = split(line[j+2:end], ',')
        return name, rest
    else
        parts = split(line, ',')
        return parts[1], parts[2:end]
    end
end

# rocprofv3 kernel_stats.csv columns:
#   Name, Calls, TotalDurationNs, AverageNs, Percentage, MinNs, MaxNs, StdDev
function parse_rocprof(path)
    rows = Tuple{String,Float64,Float64,Float64,Float64}[]  # name, calls, totalNs, minNs, pct
    for (i, line) in enumerate(eachline(path))
        i == 1 && continue
        isempty(strip(line)) && continue
        name, f = split_stat_line(line)
        length(f) < 6 && continue
        calls = parse(Float64, f[1]); totalNs = parse(Float64, f[2])
        pct = parse(Float64, f[4]); minNs = parse(Float64, f[5])
        push!(rows, (name, calls, totalNs, minNs, pct))
    end
    return rows
end

# nsys `cuda_gpu_kern_sum --format csv` columns (header-detected, Name is LAST and
# may be quoted with internal commas): Time (%), Total Time (ns), Instances, Avg,
# Med, Min (ns), Max, StdDev, Name.
function parse_nsys(path)
    rows = Tuple{String,Float64,Float64,Float64,Float64}[]
    lines = collect(eachline(path))
    hi = findfirst(l -> occursin("Total Time", l) && occursin("Name", l), lines)
    hi === nothing && return rows
    norm(s) = strip(replace(s, '"' => ""))
    cols = norm.(split(lines[hi], ','))
    ci_pct  = findfirst(c -> occursin("Time (%)", c), cols)
    ci_tot  = findfirst(c -> occursin("Total Time", c), cols)
    ci_ins  = findfirst(c -> occursin("Instances", c), cols)
    ci_min  = findfirst(c -> occursin("Min (ns)", c), cols)
    ci_name = findfirst(==("Name"), cols)
    (ci_pct === nothing || ci_name === nothing) && return rows
    for line in lines[hi+1:end]
        isempty(strip(line)) && continue
        f = split(line, ',')
        length(f) < ci_name && continue
        g(i) = parse(Float64, norm(f[i]))
        push!(rows, (join(f[ci_name:end], ","), g(ci_ins), g(ci_tot), g(ci_min), g(ci_pct)))
    end
    return rows
end

# Auto-detect the profiler CSV format per file (a run may mix rocprofv3-format KF
# stats with nsys-format CUB stats): nsys has "Total Time (ns)" ... "Name" (Name last);
# rocprofv3 / CUDA.@profile-emitted stats start with a quoted "Name" column first.
function detect_and_parse(path)
    # nsys `stats` output has preamble lines ("Generating SQLite…", "Processing…")
    # BEFORE the header, so scan for whichever header appears: nsys ("Total Time …
    # Name") vs rocprofv3 / CUDA.@profile (a leading quoted "Name" column).
    for l in eachline(path)
        s = strip(l)
        startswith(s, "\"Name\"") && return parse_rocprof(path)
        occursin("Total Time", s) && occursin("Name", s) && return parse_nsys(path)
    end
    return parse_rocprof(path)
end

function main()
    dir = get(ENV, "KF_STATS_DIR", ".")
    out = get(ENV, "KF_OUT", "sort_kernels.csv")
    files = filter(f -> endswith(f, "__stats.csv"), readdir(dir; join=true))
    open(out, "w") do io
        println(io, "n,type,method,kernel,calls_per_sort,per_sort_us,min_dispatch_us,pct")
        for path in sort(files)
            base = basename(path)
            m = match(r"^(.*?)__(.*?)__(\d+)__stats\.csv$", base)
            m === nothing && (@warn "skip unparseable name" base; continue)
            method, typ, nstr = m.captures[1], m.captures[2], m.captures[3]
            rows = detect_and_parse(path)
            isempty(rows) && continue
            npass = get(_NPASSES, typ, 4)
            # Aggregate by short tag FIRST (several raw kernels map to one tag —
            # rocPRIM's onesweep has 2 template variants: pass-1-const + passes-2..k).
            agg = Dict{String,Vector{Float64}}()  # tag => [calls, totalNs, pct, minNs(min)]
            for (name, calls, totalNs, minNs, pct) in rows
                tag = shorten(name)
                a = get!(agg, tag, [0.0, 0.0, 0.0, Inf])
                a[1] += calls; a[2] += totalNs; a[3] += pct; a[4] = min(a[4], minNs)
            end
            # n_sorts (number of full sorts traced): the scan-histogram kernel runs
            # EXACTLY once per sort in both KF and rocPRIM onesweep — the robust key.
            # Fallbacks: onesweep total / #passes; else the busiest kernel's calls
            # (rocPRIM switches to a single-kernel path at small N — no scan/onesweep).
            n_sorts = haskey(agg, "scan_hist") ? agg["scan_hist"][1] :
                      haskey(agg, "onesweep")  ? agg["onesweep"][1] / npass :
                      maximum(a[1] for a in values(agg))
            delete!(agg, "reset")   # drop the profiling driver's H2D reset copy
            # pct = share of the SORT's total GPU-kernel time (recomputed from
            # per-sort µs, not the profiler's raw % which can include memcpy) —
            # uniform across rocprofv3 / nsys / CUDA.@profile.
            total_us = sum(a[2] for a in values(agg); init = 0.0) / n_sorts / 1000
            for (tag, a) in sort(collect(agg); by = kv -> -kv[2][2])
                calls_per_sort = a[1] / n_sorts
                per_sort_us    = a[2] / n_sorts / 1000
                min_us         = a[4] / 1000
                pct            = total_us > 0 ? 100 * per_sort_us / total_us : 0.0
                @printf(io, "%s,%s,%s,%s,%.3f,%.3f,%.3f,%.2f\n",
                        nstr, typ, method, tag, calls_per_sort, per_sort_us, min_us, pct)
            end
        end
    end
    println(">>> wrote $out")
end

main()
