using WrenDists
using WrenDists: logdensityof
using Distributions
using LinearAlgebra
using Random
using Test

include("plaice.jl")

const RUN_REACTANT = !Sys.iswindows()

if RUN_REACTANT
    include("reactant.jl")
end

function test_dist(name, d_wd, d_ref; plaice_zero_allocs=())
    @testset "$name" begin
        rng = Xoshiro(468)
        @testset "logdensityof consistent with Distributions" begin
            for _ in 1:100
                x = rand(rng, d_ref)
                @test logdensityof(d_wd, x) ≈ logpdf(d_ref, x)
            end
        end
        @testset "rand obeys rng" begin
            @test rand(Xoshiro(468), d_wd) == rand(Xoshiro(468), d_wd)
            @test rand(Xoshiro(468), d_wd) != rand(Xoshiro(469), d_wd)
        end
        @testset "Plaice" begin
            Plaice.test_all(d_wd; expected_zero_allocs=plaice_zero_allocs)
        end
        if RUN_REACTANT
            @testset "Reactant" begin
                points = [rand(rng, d_ref) for _ in 1:10]
                test_reactant(d_wd, points)
            end
        end
    end
end

const UNIVARIATE_PLAICE = (from_vec, from_unconstrained_vec)
const MULTIVARIATE_PLAICE = (to_vec, from_vec, to_unconstrained_vec, from_unconstrained_vec)
const DIRICHLET_PLAICE = (to_vec, from_vec)

@testset "WrenDists" begin
    # Continuous unconstrained
    test_dist(
        "Normal",
        WrenDists.Normal(1.0, 2.0),
        Distributions.Normal(1.0, 2.0);
        plaice_zero_allocs=UNIVARIATE_PLAICE,
    )
    test_dist(
        "Laplace",
        WrenDists.Laplace(1.0, 2.0),
        Distributions.Laplace(1.0, 2.0);
        plaice_zero_allocs=UNIVARIATE_PLAICE,
    )
    test_dist(
        "Cauchy",
        WrenDists.Cauchy(1.0, 2.0),
        Distributions.Cauchy(1.0, 2.0);
        plaice_zero_allocs=UNIVARIATE_PLAICE,
    )

    # Continuous positive
    test_dist(
        "LogNormal",
        WrenDists.LogNormal(0.5, 0.3),
        Distributions.LogNormal(0.5, 0.3);
        plaice_zero_allocs=UNIVARIATE_PLAICE,
    )
    test_dist(
        "Exponential",
        WrenDists.Exponential(3.0),
        Distributions.Exponential(3.0);
        plaice_zero_allocs=UNIVARIATE_PLAICE,
    )
    test_dist(
        "Weibull",
        WrenDists.Weibull(1.5, 2.0),
        Distributions.Weibull(1.5, 2.0);
        plaice_zero_allocs=UNIVARIATE_PLAICE,
    )
    test_dist(
        "Gamma",
        WrenDists.Gamma(2.0, 1.0),
        Distributions.Gamma(2.0, 1.0);
        plaice_zero_allocs=UNIVARIATE_PLAICE,
    )

    # Continuous bounded
    test_dist(
        "Beta",
        WrenDists.Beta(2.0, 3.0),
        Distributions.Beta(2.0, 3.0);
        plaice_zero_allocs=UNIVARIATE_PLAICE,
    )
    test_dist(
        "Uniform",
        WrenDists.Uniform(1.0, 3.0),
        Distributions.Uniform(1.0, 3.0);
        plaice_zero_allocs=UNIVARIATE_PLAICE,
    )

    # Discrete univariate
    test_dist(
        "Bernoulli",
        WrenDists.Bernoulli(0.3),
        Distributions.Bernoulli(0.3);
        plaice_zero_allocs=UNIVARIATE_PLAICE,
    )
    test_dist(
        "Categorical",
        WrenDists.Categorical([0.2, 0.3, 0.5]),
        Distributions.Categorical([0.2, 0.3, 0.5]);
        plaice_zero_allocs=UNIVARIATE_PLAICE,
    )
    test_dist(
        "Binomial",
        WrenDists.Binomial(20, 0.4),
        Distributions.Binomial(20, 0.4);
        plaice_zero_allocs=UNIVARIATE_PLAICE,
    )
    test_dist(
        "Poisson",
        WrenDists.Poisson(7.0),
        Distributions.Poisson(7.0);
        plaice_zero_allocs=UNIVARIATE_PLAICE,
    )
    test_dist(
        "Geometric",
        WrenDists.Geometric(0.3),
        Distributions.Geometric(0.3);
        plaice_zero_allocs=UNIVARIATE_PLAICE,
    )

    # Additional parameter combos
    test_dist("Gamma(0.5,3)", WrenDists.Gamma(0.5, 3.0), Distributions.Gamma(0.5, 3.0))
    test_dist("Gamma(1,1)", WrenDists.Gamma(1.0, 1.0), Distributions.Gamma(1.0, 1.0))
    test_dist("Beta(0.5,0.5)", WrenDists.Beta(0.5, 0.5), Distributions.Beta(0.5, 0.5))
    test_dist("Beta(1,1)", WrenDists.Beta(1.0, 1.0), Distributions.Beta(1.0, 1.0))

    # Multivariate
    μ = [1.0, 2.0]
    L = [1.0 0.0; 0.5 1.0]
    Σ = L * L'
    test_dist(
        "MvNormal",
        WrenDists.MvNormal(μ, L),
        Distributions.MvNormal(μ, Σ);
        plaice_zero_allocs=MULTIVARIATE_PLAICE,
    )
    σ = [1.0, 2.0]
    test_dist(
        "MvNormal (diagonal)",
        WrenDists.MvNormal(μ, Diagonal(σ)),
        Distributions.MvNormal(μ, Diagonal(σ .^ 2));
        plaice_zero_allocs=MULTIVARIATE_PLAICE,
    )
    test_dist(
        "MvNormal (isotropic)",
        WrenDists.MvNormal(μ, 2.0 * I),
        Distributions.MvNormal(μ, 4.0 * I);
        plaice_zero_allocs=MULTIVARIATE_PLAICE,
    )
    test_dist(
        "Dirichlet",
        WrenDists.Dirichlet([2.0, 3.0, 1.0]),
        Distributions.Dirichlet([2.0, 3.0, 1.0]);
        plaice_zero_allocs=DIRICHLET_PLAICE,
    )
    test_dist(
        "Multinomial",
        WrenDists.Multinomial(10, [0.2, 0.3, 0.5]),
        Distributions.Multinomial(10, [0.2, 0.3, 0.5]);
        plaice_zero_allocs=MULTIVARIATE_PLAICE,
    )
end
