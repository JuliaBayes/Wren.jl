module Wren

using WrenCore

# Samplers
include("samplers/prior.jl")
include("samplers/nuts.jl")
export prior, nuts

include("optimization.jl")
export optimize, MAP, MLE

# Distributions
module Dists
using Reexport: @reexport
@reexport using WrenDists
end

# Re-exports
using WrenCore:
    WrenCore,
    # DSL
    @wren,
    @addloglikelihood,
    @iid,
    @observe,
    @batch,
    # Models
    Model,
    condition,
    decondition,
    simulate,
    return_value,
    logjoint,
    logprior,
    # Vectorisation
    VecModel,
    logdensity,
    vectorise_latents,
    lp,
    lp_and_grad,
    lp_and_grad!,
    DEFAULT_AD,
    reduce_copies,
    # Benchmark and test
    @bmodel,
    benchmark_model,
    test_ad,
    WithBackend,
    WithExpectedResult,
    # Initialisations for MCMC
    FromPrior,
    FromValues,
    # PosteriorDB
    posteriordb_model,
    available_posteriordb_models

export WrenCore,
    # DSL
    @wren,
    @addloglikelihood,
    @iid,
    @observe,
    @batch,
    # Models
    Model,
    condition,
    decondition,
    simulate,
    return_value,
    logjoint,
    logprior,
    # Vectorisation
    VecModel,
    logdensity,
    vectorise_latents,
    lp,
    lp_and_grad,
    lp_and_grad!,
    DEFAULT_AD,
    reduce_copies,
    # Benchmark and test
    @bmodel,
    benchmark_model,
    test_ad,
    WithBackend,
    WithExpectedResult,
    # Initialisations for MCMC
    FromPrior,
    FromValues,
    # PosteriorDB
    posteriordb_model,
    available_posteriordb_models

using VarNames: @vn, @vnt, VarNamedTuple, VarName, Draw, parameters, extras
export @vn, @vnt, VarNamedTuple, VarName, Draw, parameters, extras

using DensityInterface: logdensityof
export logdensityof

using StatsAPI: loglikelihood, predict
export loglikelihood, predict

using DifferentiationInterface: AutoForwardDiff, AutoReverseDiff, AutoMooncake, AutoEnzyme
export AutoForwardDiff, AutoReverseDiff, AutoMooncake, AutoEnzyme

using EnzymeCore: Forward, Reverse
export Forward, Reverse

using LinearAlgebra: I, Diagonal, Symmetric
export I, Diagonal, Symmetric

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

end # module Wren
