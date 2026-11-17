# Accept PalmerPenguins download
ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

using Wren
using WrenCore
using WrenDists
import Pathfinder
using Documenter
using DocumenterCodeBlocks
using DocumenterInterLinks
using DocumenterMermaid

# To get docstrings from extensions, we need to load their triggers.
using PosteriorDB

links = InterLinks(
    "DynamicHMC" => "https://www.tamaspapp.eu/DynamicHMC.jl/dev/",
    "FlexiChains" => "https://juliabayes.org/FlexiChains.jl/stable/",
    "Pathfinder" => "https://mlcolab.github.io/Pathfinder.jl/stable/",
    "Plaice" => "https://juliabayes.org/Plaice.jl/stable/",
    "VarNames" => "https://juliabayes.org/VarNames.jl/stable/",
)

MAX_HTML_SIZE = 400 * 1024 # 400 KB

makedocs(;
    sitename="Wren.jl",
    # Explicitly pass an empty `inventory_version` to silence warnings when building docs.
    # This has no effect on deployed docs (i.e. on GitHub Pages) since Documenter will
    # automatically detect the inventory version from the git tag. See
    # https://github.com/JuliaDocs/Documenter.jl/pull/2449
    format=Documenter.HTML(;
        inventory_version="",
        assets=["assets/custom.css"],
        size_threshold=MAX_HTML_SIZE,
        size_threshold_warning=MAX_HTML_SIZE,
    ),
    modules=[
        Wren,
        WrenCore,
        WrenDists,
        Base.get_extension(Wren, :WrenPathfinderExt),
        Base.get_extension(WrenCore, :WrenCoreFlexiChainsExt),
        Base.get_extension(WrenCore, :WrenCorePosteriorDBExt),
    ],
    pages=[
        "index.md",
        "Tutorials" => [
            "tutorials/beginner.md",
            "tutorials/defining_models.md",
            "tutorials/using_models.md",
            "tutorials/distributions.md",
            "tutorials/submodels.md",
            "tutorials/mcmc.md",
            "tutorials/workflow.md",
        ],
        "How-tos" => [
            "howtos/vecmodel.md"
            "howtos/benchmarking.md"
            "howtos/optimization.md"
            "howtos/pathfinder.md"
            "howtos/posteriordb.md"
            "howtos/custom-distributions.md"
        ],
        "Explanations" => [
            "explanations/enzyme.md",
            "explanations/reactant.md",
            "explanations/batch.md",
            "explanations/compilation.md",
        ],
        "Reference" =>
            ["reference/syntax.md", "reference/distributions.md", "reference/api.md"],
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
