module Wren

using DensityInterface: DensityInterface, logdensityof
using DocStringExtensions: TYPEDFIELDS
using Random: AbstractRNG, default_rng
using VarNames: VarNames, varname, VarName, VarNamedTuple

include("expr.jl")
using .Expressions:
    AbstractWrenExpr,
    Observe,
    OtherStatement,
    ReturnStatement,
    Sample,
    WrenExpr,
    _show_wren_exprs,
    to_wren_exprs

include("accs/accs.jl")
include("accs/logp.jl")
include("accs/vnt.jl")
include("accs/raw_values.jl")
include("accs/accessors.jl")
include("latents/latents.jl")
include("model.jl")
include("interface.jl")

export @model,
    @observe,
    AbstractLatentSource,
    FromParams,
    FromPrior,
    condition,
    conditioned,
    evaluate,
    get_latent,
    get_raw_values,
    getlogjoint,
    getlogjoint_transformed,
    getloglikelihood,
    getlogprior

# Reexports
using VarNames: @vn, @vnt, VarNamedTuple, VarName
export @vn, @vnt, VarNamedTuple, VarName

end # module Wren
