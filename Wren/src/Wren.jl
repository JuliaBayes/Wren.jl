module Wren

using WrenCore

# Samplers
include("samplers/prior.jl")
include("samplers/nuts.jl")
export prior, nuts

# Re-exports
using WrenCore: @wren,
    # DSL
    @addloglikelihood,
    @iid,
    @observe,
    @plate,
    # Models
    Model,
    condition,
    rand_with_logp,
    logjoint,
    logprior,
    # Vectorisation
    VecModel,
    vectorise_latents,
    lp,
    lp_and_grad,
    lp_and_grad!,
    # Benchmark and test
    @bmodel,
    benchmark_ad,
    test_ad,
    benchmark_and_test,
    WithBackend,
    WithExpectedResult,
    NoTest,
    # Initialisations for MCMC
    FromPrior,
    FromParams

export @wren,
    # DSL
    @addloglikelihood,
    @iid,
    @observe,
    @plate,
    # Models
    Model,
    condition,
    rand_with_logp,
    logjoint,
    logprior,
    # Vectorisation
    VecModel,
    vectorise_latents,
    lp,
    lp_and_grad,
    lp_and_grad!,
    # Benchmark and test
    @bmodel,
    benchmark_ad,
    test_ad,
    benchmark_and_test,
    WithBackend,
    WithExpectedResult,
    NoTest,
    # Initialisations for MCMC
    FromPrior,
    FromParams

using VarNames: @vn, @vnt, VarNamedTuple, VarName, Draw, parameters, extras
export @vn, @vnt, VarNamedTuple, VarName, Draw, parameters, extras

using DensityInterface: logdensityof
export logdensityof

using StatsAPI: loglikelihood
export loglikelihood

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
