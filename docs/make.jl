using Pkg
Pkg.activate("docs")

using Documenter
using KernelForge

makedocs(
    sitename="KernelForge.jl",
    modules=[KernelForge],
    format=Documenter.HTML(
        prettyurls=get(ENV, "CI", nothing) == "true",
        canonical="https://epilliat.github.io/KernelForge.jl/stable/",
        assets=["assets/custom.css"],
    ),
    pages=[
        "Home" => "index.md",
        "Performance" => "performances.md",
        "Examples" => "examples.md",
        "API Reference" => "api.md",
    ],
    checkdocs=:none,
)

# Deploy to GitHub Pages (optional)
deploydocs(
    repo="github.com/epilliat/KernelForge.jl.git",
    devbranch="main",
)