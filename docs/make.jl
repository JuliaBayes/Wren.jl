using Pkg: Pkg

# Because our `docs` folder is not placed under Wren or WrenCore, we need to manually read
# in the version of Wren so that we have something to use for the version picker.
#
# In actual deployment settings, this should be ignored by Documenter, in favour of the
# version number inferred from the git tag:
# https://github.com/JuliaDocs/Documenter.jl/pull/2449
using TOML: TOML
const WREN_VERSION = TOML.parsefile(joinpath(@__DIR__, "..", "Wren", "Project.toml"))["version"]

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
    format=Documenter.HTML(; inventory_version=WREN_VERSION),
    modules=[Wren, WrenCore],
    pages=[
        "index.md",
        "syntax.md",
        "models.md",
        "submodels.md",
        "mcmc.md",
        "motivation.md",
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
