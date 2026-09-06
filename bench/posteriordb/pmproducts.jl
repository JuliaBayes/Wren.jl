"""
    PMProducts

Products of independent ProbabilityMeasures observations.

ProbabilityMeasures has no product measure, and the observation blocks of the
PosteriorDB translations are products of a univariate family over a vector of
parameters. Each measure here evaluates its density as array arithmetic followed
by a reduction, with no scalar indexing, so Reactant can trace it.

Latent variables need no measure from this module: every vector-valued prior in
the catalog is isotropic, and `ProbabilityMeasures.MvNormal` with a
`UniformScaling` factor already traces cleanly.

Terms that depend only on the observations are computed once by `make_model` and
stored in the measure, which keeps the traced density a pure reduction.
"""
module PMProducts

using DensityInterface: DensityInterface
using LinearAlgebra: LowerTriangular, Symmetric, cholesky, diag
using LogExpFunctions: log1pexp
using SpecialFunctions: loggamma

export ProductNormal, ProductPoisson, ProductBinomialLogit, DenseMvNormal

const LOG2PI = log(2 * pi)

# Scale terms accept either one shared value or one value per observation.
_sumlog(scale::Number, n::Integer) = n * log(scale)
_sumlog(scale::AbstractArray, ::Integer) = sum(log, scale)

"""
    ProductNormal(mu, sigma)

Independent normal observations with means `mu` and standard deviations `sigma`.
Either argument may be a scalar shared across all observations.
"""
struct ProductNormal{M,S}
    mu::M
    sigma::S
end

function DensityInterface.logdensityof(d::ProductNormal, x::AbstractVector{<:Number})
    z = (x .- d.mu) ./ d.sigma
    n = length(x)
    return -(sum(abs2, z) + n * LOG2PI) / 2 - _sumlog(d.sigma, n)
end

"""
    DenseMvNormal(mu, covariance)

Correlated normal observations with mean `mu` and a dense positive-definite
`covariance`, which the density factorises.

`ProbabilityMeasures.MvNormal` takes a Cholesky factor directly, but reads its
determinant one diagonal entry at a time, which Reactant rejects.

Reactant cannot differentiate this density either: its Enzyme pipeline has no
adjoint for `stablehlo.cholesky`. A model that uses this measure is
gradient-benchmarked under Enzyme alone.
"""
struct DenseMvNormal{M,C}
    mu::M
    covariance::C
end

# Reactant's factorization object exposes `factors` and `uplo`, but no `L`.
# `uplo` is an ordinary `Char`, so the branch resolves before anything is traced.
function _lower_factor(factorization)
    factors = factorization.factors
    return LowerTriangular(factorization.uplo == 'U' ? permutedims(factors) : factors)
end

function DensityInterface.logdensityof(d::DenseMvNormal, x::AbstractVector{<:Number})
    lower = _lower_factor(cholesky(Symmetric(d.covariance)))
    z = lower \ (x .- d.mu)
    n = length(x)
    return -(sum(abs2, z) + n * LOG2PI) / 2 - sum(log, diag(lower))
end

"""
    ProductPoisson(lambda)

Independent Poisson observations with rates `lambda`.
"""
struct ProductPoisson{L}
    lambda::L
end

function DensityInterface.logdensityof(d::ProductPoisson, x::AbstractVector{<:Number})
    return sum(x .* log.(d.lambda) .- d.lambda .- loggamma.(x .+ 1))
end

"""
    ProductBinomialLogit(trials, logit_p)

Independent binomial observations with `trials` trials and success probabilities
`logistic.(logit_p)`. `trials` may be a scalar shared across all observations.

Working on the logit scale keeps the density finite for the large linear
predictors that vague regression priors admit.
"""
struct ProductBinomialLogit{N,P}
    trials::N
    logit_p::P
end

function DensityInterface.logdensityof(d::ProductBinomialLogit, x::AbstractVector{<:Number})
    return sum(
        x .* d.logit_p .- d.trials .* log1pexp.(d.logit_p) .+ loggamma.(d.trials .+ 1) .-
        loggamma.(x .+ 1) .- loggamma.(d.trials .- x .+ 1),
    )
end

end # module PMProducts
