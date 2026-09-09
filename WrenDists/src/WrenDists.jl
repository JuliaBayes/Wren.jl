module WrenDists

using DensityInterface: DensityInterface, logdensityof
using IrrationalConstants: log2π, logπ, logtwo, sqrt2
using LinearAlgebra: Diagonal, LowerTriangular, UniformScaling
using Random: Random, AbstractRNG
using SpecialFunctions: erfcinv, gamma_inc, loggamma
using Plaice: Plaice
using VarNames: VarNames

include("common.jl")
include("gammainc.jl")

# Univariate continuous

# Abstract types exist solely for Plaice dispatch purposes, nothing else!
abstract type UnivariateDist <: WrenDist end
abstract type UnconstrainedUnivariateDist <: UnivariateDist end
abstract type PositiveUnivariateDist <: UnivariateDist end
include("univariate/normal.jl")
include("univariate/lognormal.jl")
include("univariate/exponential.jl")
include("univariate/weibull.jl")
include("univariate/uniform.jl")
include("univariate/laplace.jl")
include("univariate/cauchy.jl")
include("univariate/gamma.jl")
include("univariate/beta.jl")

# Univariate discrete
abstract type DiscreteUnivariateDist <: UnivariateDist end
include("univariate/bernoulli.jl")
include("univariate/binomial.jl")
include("univariate/categorical.jl")
include("univariate/geometric.jl")
include("univariate/poisson.jl")

# Plaice interface for univariate distributions
include("univariate/plaice.jl")

# Multivariate
include("multivariate/mvnormal.jl")
include("multivariate/dirichlet.jl")
include("multivariate/multinomial.jl")

export logdensityof

export Normal, LogNormal, Exponential, Weibull, Uniform, Laplace, Cauchy, Gamma, Beta
export Bernoulli, Binomial, Categorical, Geometric, Poisson
export MvNormal, Dirichlet, Multinomial

end # module WrenDists
