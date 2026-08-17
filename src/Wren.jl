module Wren

using DensityInterface: DensityInterface
using DocStringExtensions: TYPEDFIELDS
using VarNames: VarNames, varname, VarName, VarNamedTuple

include("accs/accs.jl")
include("accs/logp.jl")
include("accs/vnt.jl")
include("accs/raw_values.jl")

function default_accumulators(::Type{FloatT}=LogProbType) where {FloatT}
    return AccumulatorTuple(
        LogPriorAccumulator{FloatT}(),
        LogJacobianAccumulator{FloatT}(),
        LogLikelihoodAccumulator{FloatT}(),
        RawValueAccumulator(),
    )
end

include("model.jl")

end # module Wren
