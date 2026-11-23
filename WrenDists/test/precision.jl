using WrenDists: WrenDists, logdensityof
using Plaice: Plaice
using Random: Xoshiro
using LinearAlgebra: Diagonal, I
using FillArrays: Fill
using Test

@testset "Float32 precision" begin
    μ = Float32[0, 1]
    σ = Float32[1, 2]
    α = Float32[2, 3]
    p = Float32[0.25, 0.75]
    counts = [3, 4]
    continuous_dists = (
        WrenDists.Normal(1.0f0),
        WrenDists.LogNormal(1.0f0),
        WrenDists.Exponential(1.0f0),
        WrenDists.Weibull(2.0f0, 1.0f0),
        WrenDists.Uniform(0.0f0, 1.0f0),
        WrenDists.Laplace(1.0f0),
        WrenDists.Logistic(1.0f0),
        WrenDists.Cauchy(1.0f0),
        WrenDists.Flat{Float32}(),
        WrenDists.PositiveFlat{Float32}(),
        WrenDists.TruncatedFlat(0.0f0, 1.0f0),
        WrenDists.Gamma(2.0f0),
        WrenDists.InverseGamma(2.0f0),
        WrenDists.TDist(3.0f0),
        WrenDists.Beta(2.0f0, 3.0f0),
        WrenDists.HalfNormal(0.0f0, 1.0f0),
        WrenDists.HalfCauchy(0.0f0, 1.0f0),
        WrenDists.BatchedNormal(μ, σ),
        WrenDists.BatchedNormal(μ, Fill(1.0f0, 2)),
        WrenDists.BatchedLogNormal(μ, σ),
        WrenDists.BatchedLogNormal(μ, Fill(1.0f0, 2)),
        WrenDists.BatchedExponential(σ),
        WrenDists.BatchedWeibull(α, σ),
        WrenDists.BatchedGamma(α, σ),
        WrenDists.BatchedInverseGamma(α, σ),
        WrenDists.BatchedBeta(α, σ),
        WrenDists.BatchedTDist(α, μ, σ),
        WrenDists.BatchedTDist(Fill(3.0f0, 2), μ, σ),
        WrenDists.MvNormalScale(μ, Float32[1 0; 0.5 1]),
        WrenDists.MvNormalScale(μ, Diagonal(σ)),
        WrenDists.MvNormalScale(μ, 1.0f0 * I),
        WrenDists.Dirichlet(α),
        WrenDists.Independent([WrenDists.Normal(m, 1.0f0) for m in μ]),
        WrenDists.Independent([WrenDists.Gamma(a) for a in α]),
        WrenDists.Independent(WrenDists.Repeated(WrenDists.Normal(1.0f0), (2,))),
        WrenDists.Independent(WrenDists.Repeated(WrenDists.LogNormal(1.0f0), (2,))),
        WrenDists.IIDTDist((2,), 3.0f0, 0.0f0, 1.0f0),
    )

    @testset "$(typeof(d))" for d in (
        continuous_dists...,
        WrenDists.Bernoulli(0.5f0),
        WrenDists.BernoulliLogit(1.0f0),
        WrenDists.Binomial(3, 0.5f0),
        WrenDists.BinomialLogit(3, 1.0f0),
        WrenDists.NegativeBinomial(2.0f0),
        WrenDists.Poisson(2.0f0),
        WrenDists.PoissonLog(1.0f0),
        WrenDists.Categorical(p),
        WrenDists.Geometric(0.5f0),
        WrenDists.BatchedBernoulli(p),
        WrenDists.BatchedBernoulliLogit(μ),
        WrenDists.BatchedBinomial(counts, p),
        WrenDists.BatchedBinomialLogit(counts, μ),
        WrenDists.BatchedNegativeBinomial(α, p),
        WrenDists.BatchedPoisson(α),
        WrenDists.BatchedPoissonLog(μ),
        WrenDists.BatchedCategorical(reshape(p, 2, 1)),
        WrenDists.Multinomial(3, p),
    )
        x = @inferred rand(Xoshiro(8675309), d)
        # Discrete samplers may return integer counts.
        @test eltype(x) <: Union{Float32,Integer}
        @test (@inferred logdensityof(d, x)) isa Float32
    end

    # The transforms to and from unconstrained space, which a `VecModel` uses, should not
    # promote either.
    @testset "unconstrained transforms of $(typeof(d))" for d in continuous_dists
        x = rand(Xoshiro(8675309), d)
        z, ladj = Plaice.with_logabsdet_jacobian(Plaice.to_unconstrained_vec(d), x)
        @test eltype(z) == Float32
        @test ladj isa Float32
        x2, inv_ladj = Plaice.with_logabsdet_jacobian(Plaice.from_unconstrained_vec(d), z)
        @test eltype(x2) == Float32
        @test inv_ladj isa Float32
        @test x2 ≈ x
    end
end
