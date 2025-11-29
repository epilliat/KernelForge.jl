function create_kernel_stacked_barplot(df;
    algo="Dot",
    N=nothing,
    kernel_colors=[:blue, :red, :green, :orange, :purple, :cyan, :magenta, :yellow],
    overhead_color=:gray,
    overhead_alpha=0.6,
    kernel_labels=nothing,
    names=nothing,
    size_anotation=12,
    time_unit=:us)  # New parameter: :us (microseconds) or :ms (milliseconds)
    # Filter for the specific algorithm and N
    if isnothing(N)
        df_filtered = filter(row -> row.algo == algo, df)
    else
        df_filtered = filter(row -> row.algo == algo && row.datalength == N, df)
    end

    if nrow(df_filtered) == 0
        if isnothing(N)
            error("No data found for algo='$algo'")
        else
            error("No data found for algo='$algo' and N=$N")
        end
    end

    # Get N value for display
    N_val = isnothing(N) ? df_filtered.datalength[1] : N
    N_str = N_val >= 1e9 ? "N=$(round(N_val/1e9, digits=1))e9" :
            N_val >= 1e6 ? "N=$(round(Int, N_val/1e6))e6" :
            N_val >= 1e3 ? "N=$(round(Int, N_val/1e3))e3" : "N=$N_val"

    names = isnothing(names) ? sort(unique(df_filtered.name)) : names
    datatypes = [Float64, Float32, UInt8, Luma.UnitFloat8, QuaternionF32]

    # Time conversion factor and label
    time_factor = time_unit == :ms ? 0.001 : 1.0  # Convert μs to ms if needed
    time_label = time_unit == :ms ? "Duration (ms)" : "Duration (μs)"
    digits = time_unit == :ms ? 2 : 1
    # Initialize empty plot with N in title
    p = Plots.plot(
        legend=:topright,
        legend_background_color=RGBA(1, 1, 1, 0.9),  # Semi-transparent white background
        title="$algo: GPU Kernel Performance ($N_str)",
        ylabel=time_label,
        xlabel="",
        grid=true,
        framestyle=:box,
        bottom_margin=10 * Plots.Measures.pt,
        top_margin=5 * Plots.Measures.pt  # Add top margin to prevent clipping
    )

    bar_width = 0.35
    bar_positions = Float64[]
    bar_labels = String[]
    max_height = 0.0

    # Dynamically find all kernel columns
    kernel_nums = Int[]
    for col in propertynames(df_filtered)
        col_str = String(col)
        if occursin(r"^kernel(\d+)$", col_str)
            m = match(r"^kernel(\d+)$", col_str)
            push!(kernel_nums, parse(Int, m.captures[1]))
        end
    end
    sort!(kernel_nums)
    max_kernels = isempty(kernel_nums) ? 0 : maximum(kernel_nums)

    # Check which kernels actually have data
    has_kernel = Dict{Int,Bool}()
    has_overhead = false

    # Try to get kernel names from the dataframe if available
    default_kernel_labels = Dict{Int,String}()
    for k in 1:max_kernels
        has_kernel[k] = false
        kernel_name_col = Symbol("kernel$(k)_name")

        # Check if there's a name column for this kernel
        if hasproperty(df_filtered, kernel_name_col)
            for row in eachrow(df_filtered)
                if !ismissing(row[kernel_name_col])
                    default_kernel_labels[k] = row[kernel_name_col]
                    break
                end
            end
        end
    end

    # Use custom labels if provided, otherwise use defaults or "Kernel N"
    final_kernel_labels = Dict{Int,String}()
    for k in 1:max_kernels
        final_kernel_labels[k] = "Kernel $k"
        #end
    end

    # First pass to check what components exist
    for name in names
        for dtype in intersect(datatypes, unique(df.datatype))
            subset = filter(row -> row.name == name && row.datatype == dtype, df_filtered)
            if nrow(subset) > 0
                row = subset[1, :]

                for k in 1:max_kernels
                    kernel_col = Symbol("kernel$k")
                    if hasproperty(row, kernel_col) && !ismissing(row[kernel_col]) && row[kernel_col] > 0
                        has_kernel[k] = true
                    end
                end

                overhead = (coalesce(row.mean_duration_pipeline, 0.0) - coalesce(row.mean_duration_gpu, 0.0)) * time_factor
                has_overhead = has_overhead || (overhead > 0)
            end
        end
    end

    # Track if we've added labels - initialize with all needed labels
    added_labels = Dict{String,Bool}()
    added_labels["CPU Overhead"] = false
    for k in 1:max_kernels
        added_labels[final_kernel_labels[k]] = false
    end

    # First pass: Add CPU Overhead to legend (but draw nothing yet)
    # This ensures it appears first in the legend
    dummy_rectangle = Shape([0, 0, 0, 0], [0, 0, 0, 0])
    if has_overhead
        Plots.plot!(p, dummy_rectangle, fillcolor=overhead_color, fillalpha=overhead_alpha,
            label="CPU Overhead", linecolor=overhead_color, linewidth=0)
        added_labels["CPU Overhead"] = true
    end

    # Second pass: Add kernel labels to legend (but draw nothing yet)
    has_cub = "Cub" in names

    # Always add "Main Kernel" label for kernel 1 if it exists
    for k in 1:max_kernels
        if has_kernel[k]
            color = kernel_colors[min(k, length(kernel_colors))]
            # Only show "Main Kernel" label for kernel 1, no labels for others
            if k == 1
                Plots.plot!(p, dummy_rectangle, fillcolor=color, fillalpha=1.0,
                    label="Main Kernel", linecolor=color, linewidth=0)
                added_labels["Main Kernel"] = true
            else
                # Still add to legend system but with empty label (colors will still show)
                Plots.plot!(p, dummy_rectangle, fillcolor=color, fillalpha=1.0,
                    label="", linecolor=color, linewidth=0)
            end
        end
    end

    # Add special "nvcc benchmark" label for Cub if it exists
    if has_cub
        Plots.plot!(p, dummy_rectangle, fillcolor=:darkblue, fillalpha=1.0,
            label="nvcc benchmark", linecolor=:darkblue, linewidth=0)
    end

    # Now actually draw the bars
    position = 1.0
    for (group_idx, name) in enumerate(names)
        for dtype in intersect(datatypes, df.datatype)
            push!(bar_positions, position)
            # For Cub, show U8 instead of UF8 (since Cub doesn't support Luma.UnitFloat8)
            if name == "Cub"
                push!(bar_labels, dtype == Float32 ? "F32" : (dtype == Float64 ? "F64" : "U8"))
            else
                push!(bar_labels, dtype == Float32 ? "F32" : (dtype == Float64 ? "F64" : dtype == Luma.UnitFloat8 ? "UF8" : "U8"))
            end

            subset = filter(row -> row.name == name && row.datatype == dtype, df_filtered)

            if nrow(subset) > 0
                row = subset[1, :]

                # Build stack from bottom to top
                y_bottom = 0.0

                # Add all kernels dynamically
                for k in 1:max_kernels
                    kernel_col = Symbol("kernel$k")

                    if has_kernel[k] && hasproperty(row, kernel_col) && !ismissing(row[kernel_col]) && row[kernel_col] > 0
                        kernel_height = row[kernel_col] * time_factor

                        # Choose color from the provided array
                        # Special color for Cub to match the legend
                        if name == "Cub"
                            color = :darkblue
                        else
                            color = kernel_colors[min(k, length(kernel_colors))]
                        end

                        rectangle = Shape([position - bar_width, position - bar_width,
                                position + bar_width, position + bar_width],
                            [y_bottom, y_bottom + kernel_height,
                                y_bottom + kernel_height, y_bottom])

                        # Don't add labels here - we already added them above
                        Plots.plot!(p, rectangle, fillcolor=color, fillalpha=1.0,
                            label="", linecolor=color, linewidth=0.5)

                        y_bottom += kernel_height
                    end
                end

                # Overhead (top layer)
                overhead = (coalesce(row.mean_duration_pipeline, 0.0) - coalesce(row.mean_duration_gpu, 0.0)) * time_factor
                if overhead > 0
                    rectangle = Shape([position - bar_width, position - bar_width,
                            position + bar_width, position + bar_width],
                        [y_bottom, y_bottom + overhead,
                            y_bottom + overhead, y_bottom])
                    # Don't add label since we already added it above
                    Plots.plot!(p, rectangle, fillcolor=overhead_color, fillalpha=overhead_alpha,
                        label="", linecolor=overhead_color, linewidth=0.5)
                    y_bottom += overhead
                end

                max_height = max(max_height, y_bottom)

            end

            position += 1.0
        end
        position += 0.5  # Gap between groups
    end

    # Set axis limits and ticks
    Plots.plot!(p,
        xticks=(bar_positions, bar_labels),
        xlims=(0.2, maximum(bar_positions) + 0.8),
        ylims=(0, max_height * 1.2))  # Increased from 1.1 to 1.15 for more space above

    # Add group labels centered on each group
    for (i, name) in enumerate(names)
        # Count how many datatypes are actually present
        num_types = length(intersect(datatypes, unique(df.datatype)))

        # Calculate the center position of the group
        if i == 1
            label_x = (1.0 + num_types) / 2.0  # Center of first group
        else
            label_x = (1.0 + num_types) / 2.0 + (i - 1) * (num_types + 0.5)
        end

        annotate!(p, label_x, -max_height * 0.1,
            text(name, 10, :center))
    end

    # Add text annotations above the bars
    position = 1.0
    for (group_idx, name) in enumerate(names)
        for dtype in intersect(datatypes, df.datatype)
            subset = filter(row -> row.name == name && row.datatype == dtype, df_filtered)

            if nrow(subset) > 0
                row = subset[1, :]

                # Recalculate y_bottom (total bar height)
                y_bottom = 0.0

                # Add all kernels
                for k in 1:max_kernels
                    kernel_col = Symbol("kernel$k")
                    if has_kernel[k] && hasproperty(row, kernel_col) && !ismissing(row[kernel_col]) && row[kernel_col] > 0
                        y_bottom += row[kernel_col] * time_factor
                    end
                end

                # Add overhead
                overhead = (coalesce(row.mean_duration_pipeline, 0.0) - coalesce(row.mean_duration_gpu, 0.0)) * time_factor
                if overhead > 0
                    y_bottom += overhead
                end

                gpu_time = row.mean_duration_gpu * time_factor
                total_time = y_bottom

                # Total time (not bold, slightly higher)
                if name ∉ ("Cub",)
                    annotate!(p, position, total_time + ylims(p)[2] * 0.07,
                        text("\$$(string(round(total_time, digits=digits)))\$", size_anotation, :center, :bottom))
                end
                # GPU time (bold, just below total time)
                annotate!(p, position, total_time + ylims(p)[2] * 0.01,
                    text("\$\\mathbf{$(string(round(gpu_time, digits=digits)))}\$", size_anotation, :center, :bottom))
            end

            position += 1.0
        end
        position += 0.5  # Gap between groups
    end
    return p
end