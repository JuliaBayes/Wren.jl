using Pkg: Pkg
Pkg.develop(Pkg.PackageSpec(; path=dirname(@__DIR__)))

using Wren
using Documenter
using DocumenterCodeBlocks

makedocs(;
    sitename="Wren.jl",
    format=Documenter.HTML(),
    modules=[Wren],
    pages=[
        "index.md",
        "syntax.md",
        "models.md",
        "submodels.md",
        "motivation.md",
        "api.md",
    ],
    # TODO(penelopeysm): Reenable!
    # checkdocs=:exports,
    # warnonly=false,
    warnonly=true,
    doctest=false,
)

Documenter.deploydocs(;
    repo="github.com/JuliaBayes/Wren.jl",
    target="build",
    push_preview=true,
)
