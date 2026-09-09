using Pkg: Pkg

using Wren
using WrenCore
using Documenter
using DocumenterCodeBlocks
using DocumenterInterLinks

links = InterLinks(
    "DynamicHMC" => "https://www.tamaspapp.eu/DynamicHMC.jl/dev/",
    "Plaice" => "https://juliabayes.org/Plaice.jl/dev/",
    "VarNames" => "https://juliabayes.org/VarNames.jl/stable/",
)

makedocs(;
    sitename="Wren.jl",
    # Explicitly pass an empty `inventory_version` to silence warnings when building docs.
    # This has no effect on deployed docs (i.e. on GitHub Pages) since Documenter will
    # automatically detect the inventory version from the git tag. See
    # https://github.com/JuliaDocs/Documenter.jl/pull/2449
    format=Documenter.HTML(; inventory_version=""),
    modules=[Wren, WrenCore],
    pages=[
        "index.md",
        "tutorials/syntax.md",
        "tutorials/models.md",
        "tutorials/submodels.md",
        "tutorials/mcmc.md",
        "explanation/motivation.md",
        "api.md",
    ],
    checkdocs=:exports,
    warnonly=false,
    doctest=false,
    plugins=[CodeBlocks(), links],
)

Documenter.deploydocs(;
    repo="github.com/JuliaBayes/Wren.jl",
    target="build",
    push_preview=true,
)
