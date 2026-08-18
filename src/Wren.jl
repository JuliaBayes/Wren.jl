module Wren

using ChangesOfVariables: ChangesOfVariables
using DensityInterface: DensityInterface, logdensityof
using DocStringExtensions: TYPEDFIELDS
using InverseFunctions: inverse
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
include("space.jl")
include("accs/vector_values.jl")
include("accs/accessors.jl")
include("latents/latents.jl")
include("model.jl")
include("interface.jl")

export @model,
    @observe,
    AbstractLatentSource,
    AbstractTransform,
    AbstractParameterSpace,
    AsUnconstrainedVec,
    AsVec,
    FromParams,
    FromPrior,
    FixedTransform,
    UnconstrainedSpace,
    AsOriginal,
    TransformedValue,
    VectorValueAccumulator,
    OriginalSpace,
    apply_parameter_space,
    condition,
    conditioned,
    evaluate,
    get_latent,
    get_raw_values,
    get_vector_values,
    getlogjac,
    getlogjoint,
    getlogjoint_transformed,
    getloglikelihood,
    getlogprior,
    internal_values_as_vector,
    target_transform

# Reexports
using VarNames: @vn, @vnt, VarNamedTuple, VarName
export @vn, @vnt, VarNamedTuple, VarName

end # module Wren
