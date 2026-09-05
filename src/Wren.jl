module Wren

using DensityInterface: DensityInterface, logdensityof
import DifferentiationInterface as DI
using DocStringExtensions: TYPEDFIELDS
using LogDensityProblems: LogDensityProblems
using Plaice: Plaice
using Random: AbstractRNG, default_rng
using VarNames: VarNames, varname, VarName, VarNamedTuple, vnt_pretty_print
using StatsAPI: StatsAPI

function _issue_error(str)
    error(
        "$str. Please report this issue at https://github.com/JuliaBayes/Wren.jl/issues. Thank you!",
    )
end

# Parameter spaces
include("space.jl")

# Latent sources
include("latents/latents.jl")
include("latents/prior.jl")
include("latents/params.jl")

# Accumulators
include("accs/accs.jl")
include("accs/logp.jl")
include("accs/vnt.jl")
include("accs/params.jl")
include("accs/sitename_tracker.jl")
include("accs/accessors.jl")

# Model definition
include("model/expr.jl")
using .Expressions:
    AbstractWrenExpr,
    AddLogLikelihood,
    Observe,
    OtherStatement,
    Plate,
    ReturnStatement,
    Sample,
    WrenExpr,
    _show_wren_exprs,
    _wren_expr_to_model_expr,
    to_wren_exprs

include("model/model.jl")
include("model/submodel.jl")
include("model/emit.jl")

# Vectorisation
include("latents/vector.jl")
include("accs/vectorised_params.jl")
include("accs/vec_sample.jl")
include("vecmodel.jl")

# User-facing interface
include("interface.jl")
include("draw.jl")
include("test_ad.jl")

export @wren,
    @addloglikelihood,
    @observe,
    @plate,
    AbstractLatentSource,
    AbstractTransform,
    AbstractParameterSpace,
    AbstractADCorrectnessTestSetting,
    AbstractAccumulator,
    AccumulatorTuple,
    LogPriorAccumulator,
    LogLikelihoodAccumulator,
    LogJacobianAccumulator,
    ParamsAccumulator,
    AsUnconstrainedVec,
    AsVec,
    ADIncorrectException,
    ModelRunResult,
    FromParams,
    FromPrior,
    FromVector,
    FixedTransform,
    NoTest,
    UnconstrainedSpace,
    AsOriginal,
    TransformedValue,
    VecModel,
    VectorisedParamsAccumulator,
    vectorise_latents,
    WithBackend,
    WithExpectedResult,
    OriginalSpace,
    apply_parameter_space,
    condition,
    conditioned,
    iid,
    independent,
    rand_with_logp,
    evaluate,
    get_latent,
    vectorised_params,
    logjac,
    logjoint,
    logjoint_transformed,
    logprior,
    internal_values_as_vector,
    lp,
    lp_and_grad,
    lp_and_grad!,
    @bmodel,
    benchmark_ad,
    benchmark_and_test,
    test_ad,
    target_transform

# Reexports
using VarNames: @vn, @vnt, VarNamedTuple, VarName, Draw, parameters, extras
export @vn, @vnt, VarNamedTuple, VarName, Draw, parameters, extras

using DensityInterface: logdensityof
export logdensityof

using StatsAPI: params, loglikelihood
export params, loglikelihood

using DifferentiationInterface: AutoForwardDiff, AutoReverseDiff, AutoMooncake, AutoEnzyme
export AutoForwardDiff, AutoReverseDiff, AutoMooncake, AutoEnzyme

using EnzymeCore: Forward, Reverse
export Forward, Reverse

using LinearAlgebra: I, Diagonal, Symmetric
export I, Diagonal, Symmetric

# Sampling (could in principle be a different package)
module Samplers
using FlexiChains: FlexiChains
const CHAINTYPE = FlexiChains.VNChain

include("samplers/prior.jl")
include("samplers/nuts.jl")
export prior, nuts

# FlexiChains reexports
using FlexiChains:
    Wide,
    Long,
    At,
    Near,
    Contains,
    (..),
    Touches,
    Where,
    Not,
    DimArray,
    Begin,
    End,
    mean,
    median,
    std,
    var,
    quantile,
    summarystats,
    mad,
    ess,
    rhat,
    mcse,
    gelmandiag,
    gelmandiag_multivariate,
    discretediag,
    bfmi
export Wide,
    Long,
    At,
    Near,
    Contains,
    (..),
    Touches,
    Where,
    Not,
    DimArray,
    Begin,
    End,
    mean,
    median,
    std,
    var,
    quantile,
    summarystats,
    mad,
    ess,
    rhat,
    mcse,
    gelmandiag,
    gelmandiag_multivariate,
    discretediag,
    bfmi
end

end # module Wren
