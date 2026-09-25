# Accept PalmerPenguins download
ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

using Pkg: Pkg

using Wren
using WrenCore
using WrenDists
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
    modules=[
        Wren,
        WrenCore,
        WrenDists,
        Base.get_extension(WrenCore, :WrenCoreFlexiChainsExt),
    ],
    pages=[
        "index.md",
        "Tutorials" => [
            "tutorials/syntax.md",
            "tutorials/models.md",
            "tutorials/distributions.md",
            "tutorials/submodels.md",
            "tutorials/mcmc.md",
        ],
        "Explanations" => [
            "explanations/enzyme.md",
            "explanations/batch.md",
            "explanations/motivation.md",
        ],
        "Reference" => ["reference/distributions.md", "reference/api.md"],
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
