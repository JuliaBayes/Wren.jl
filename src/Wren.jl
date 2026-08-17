module Wren

using DensityInterface: DensityInterface
using DocStringExtensions: TYPEDFIELDS
using Random: AbstractRNG, default_rng
using VarNames: VarNames, varname, VarName, VarNamedTuple

include("accs/accs.jl")
include("accs/logp.jl")
include("accs/vnt.jl")
include("accs/raw_values.jl")
include("accs/accessors.jl")

function default_accumulators(::Type{FloatT}=LogProbType) where {FloatT}
    return AccumulatorTuple(
        LogPriorAccumulator{FloatT}(),
        LogJacobianAccumulator{FloatT}(),
        LogLikelihoodAccumulator{FloatT}(),
        RawValueAccumulator(),
    )
end

include("latents/latents.jl")
include("model.jl")

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

end # module Wren
