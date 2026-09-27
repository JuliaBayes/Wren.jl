using WrenDists
using WrenDists: logdensityof, test_dist, ADTestCase
using FillArrays
import DifferentiationInterface as DI
using Distributions
using ForwardDiff
using ReverseDiff
using Mooncake
using Enzyme
using LinearAlgebra
using Plaice
using Random
using Test

const RUN_REACTANT = !Sys.iswindows()

if RUN_REACTANT
    include("reactant.jl")
end

const UNIVARIATE_PLAICE = (from_vec, from_unconstrained_vec)
const MULTIVARIATE_PLAICE = (to_vec, from_vec, to_unconstrained_vec, from_unconstrained_vec)
const DIRICHLET_PLAICE = (to_vec, from_vec)

@testset "WrenDists" begin
    # Continuous unconstrained
    include("univariate/normal.jl")
    include("univariate/laplace.jl")
    include("univariate/cauchy.jl")
    include("univariate/tdist.jl")

    # Continuous positive
    include("univariate/lognormal.jl")
    include("univariate/exponential.jl")
    include("univariate/weibull.jl")
    include("univariate/gamma.jl")
    include("univariate/halfnormal.jl")
    include("univariate/halfcauchy.jl")
    include("univariate/halftdist.jl")

    # Continuous bounded
    include("univariate/beta.jl")
    include("univariate/uniform.jl")

    # Discrete univariate
    include("univariate/bernoulli.jl")
    include("univariate/categorical.jl")
    include("univariate/binomial.jl")
    include("univariate/poisson.jl")
    include("univariate/geometric.jl")
    include("univariate/negativebinomial.jl")

    # Multivariate
    include("multivariate/mvnormalscale.jl")
    include("multivariate/dirichlet.jl")
    include("multivariate/multinomial.jl")

    include("independent.jl")
end
