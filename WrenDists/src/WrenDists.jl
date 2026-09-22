module WrenDists

using DensityInterface: DensityInterface, logdensityof
using EnzymeCore: EnzymeCore, EnzymeRules, Annotation, Const
using IrrationalConstants: invsqrt2, log2π, logπ, logtwo, sqrt2
using LogExpFunctions: log1mexp, log1pexp, logistic, xlogy
using LinearAlgebra: Diagonal, LowerTriangular, UniformScaling, diag
using Random: Random, AbstractRNG
using ReactantCore: within_compile
using SpecialFunctions: erfc, erfcinv, gamma_inc, logbeta, logerfc, loggamma
using Plaice: Plaice
using VarNames: VarNames
include("common.jl")
include("gammainc.jl")
include("betainc.jl")

# Abstract types exist solely for Plaice dispatch purposes, nothing else!
abstract type UnivariateDist <: WrenDist end
abstract type UnconstrainedUnivariateDist <: UnivariateDist end
abstract type PositiveUnivariateDist <: UnivariateDist end
abstract type DiscreteUnivariateDist <: UnivariateDist end
abstract type MultivariateDist <: WrenDist end
abstract type DiscreteMultivariateDist <: MultivariateDist end

# Independent products (needs abstract types above)
include("independent.jl")
include("univariate/batched.jl")

# Univariate continuous
include("univariate/normal.jl")
include("univariate/lognormal.jl")
include("univariate/exponential.jl")
include("univariate/weibull.jl")
include("univariate/uniform.jl")
include("univariate/laplace.jl")
include("univariate/cauchy.jl")
include("univariate/gamma.jl")
include("univariate/beta.jl")
include("univariate/halfnormal.jl")
include("univariate/halfcauchy.jl")
include("univariate/halftdist.jl")

# Univariate discrete
include("univariate/bernoulli.jl")
include("univariate/bernoullilogit.jl")
include("univariate/binomial.jl")
include("univariate/binomiallogit.jl")
include("univariate/categorical.jl")
include("univariate/geometric.jl")
include("univariate/poisson.jl")
include("univariate/logpoisson.jl")

# Plaice interface for univariate distributions
include("univariate/plaice.jl")

# Multivariate
include("multivariate/mvnormalscale.jl")
include("multivariate/dirichlet.jl")
include("multivariate/multinomial.jl")

# Plaice interface for multivariate distributions
include("multivariate/plaice.jl")

export logdensityof

export Normal, LogNormal, Exponential, Weibull, Uniform, Laplace, Cauchy, Gamma, Beta
export HalfNormal, HalfCauchy, HalfTDist
export Bernoulli, BernoulliLogit, Binomial, BinomialLogit, Categorical, Geometric
export Poisson, LogPoisson
export MvNormalScale, Dirichlet, Multinomial

end # module WrenDists
