using Pkg
Pkg.activate("docs")
using Documenter
using Luma


# Copy benchmark figures
src_dir = joinpath(@__DIR__, "..", "perfs", "cuda", "figures", "benchmark")
dst_dir = joinpath(@__DIR__, "src", "assets")

for file in readdir(src_dir)
    if endswith(file, ".png")
        cp(joinpath(src_dir, file), joinpath(dst_dir, file), force=true)
    end
end
makedocs(
    sitename="Luma.jl",
    modules=[Luma],
    format=Documenter.HTML(
        prettyurls=get(ENV, "CI", nothing) == "true",
        canonical="https://yourusername.github.io/Luma.jl/stable/",
        assets=["assets/custom.css"],
    ),
    pages=[
        "Home" => "index.md",
        "Performance" => "performances.md",
        "Examples" => "examples.md",
        "API Reference" => "api.md",
    ],
    checkdocs=:exports,
)

# Deploy to GitHub Pages (optional)
deploydocs(
    repo="github.com/yourusername/Luma.jl.git",
    devbranch="main",
)