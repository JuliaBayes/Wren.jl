using WrenDists
using WrenDists: logdensityof
using WrenCore: WrenCore
import DifferentiationInterface as DI
using Distributions
using ForwardDiff
using ReverseDiff
using Mooncake
using Enzyme
using LinearAlgebra
using Random
using Test

include("plaice.jl")

const TESTED_ADTYPES = (
    "ForwardDiff" => DI.AutoForwardDiff(),
    "ReverseDiff" => DI.AutoReverseDiff(),
    "ReverseDiffCpl" => DI.AutoReverseDiff(; compile=true),
    "MooncakeFwd" => DI.AutoMooncakeForward(),
    "MooncakeRvs" => DI.AutoMooncake(),
    "EnzymeFwd" => DI.AutoEnzyme(; mode=Enzyme.Forward),
    "EnzymeRvs" => DI.AutoEnzyme(; mode=Enzyme.Reverse),
)

function logp_from_uncons(y, dist)
    finv = Plaice.from_unconstrained_vec(dist)
    x = finv(y)
    # Just checking that the logdensity is differentiable, so no need to include the
    # Jacobian
    return logdensityof(dist, x)
end

const RUN_REACTANT = !Sys.iswindows()

if RUN_REACTANT
    include("reactant.jl")
end

function test_dist(
    name,
    d_wd,
    d_ref;
    plaice_zero_allocs=(),
    test_batched=true,
    skipped_adtypes=(),
)
    @testset "$name" begin
        rng = Xoshiro(468)

        @testset "logdensityof consistent with Distributions" begin
            for _ in 1:100
                x = rand(rng, d_ref)
                @test logdensityof(d_wd, x) ≈ logpdf(d_ref, x)
            end
        end

        @testset "logdensityof is differentiable wrt sample" begin
            if (
                d_wd isa WrenDists.DiscreteUnivariateDist ||
                d_wd isa WrenDists.DiscreteMultivariateDist
            )
                ref_adtype = DI.AutoForwardDiff()
                y = randn(rng, Plaice.unconstrained_vec_length(d_wd))
                ref_prep =
                    DI.prepare_gradient(logp_from_uncons, ref_adtype, y, DI.Constant(d_wd))
                @testset "$adtype_name" for (adtype_name, adtype) in TESTED_ADTYPES
                    adtype == ref_adtype || adtype_name in skipped_adtypes && continue
                    prep =
                        DI.prepare_gradient(logp_from_uncons, adtype, y, DI.Constant(d_wd))
                    for _ in 1:10
                        y = rand(rng, Plaice.unconstrained_vec_length(d_wd))
                        g_ref = DI.gradient(
                            logp_from_uncons,
                            ref_prep,
                            ref_adtype,
                            y,
                            DI.Constant(d_wd),
                        )
                        g = DI.gradient(
                            logp_from_uncons,
                            prep,
                            adtype,
                            y,
                            DI.Constant(d_wd),
                        )
                        @test isapprox(g, g_ref; rtol=1e-6, atol=1e-6)
                    end
                end
            end
        end

        @testset "rand obeys rng" begin
            @test rand(Xoshiro(468), d_wd) == rand(Xoshiro(468), d_wd)
            @test rand(Xoshiro(468), d_wd) != rand(Xoshiro(469), d_wd)
        end

        @testset "type stability" begin
            x = @inferred rand(rng, d_wd)
            @inferred logdensityof(d_wd, x)
        end

        @testset "Plaice" begin
            Plaice.test_all(d_wd; expected_zero_allocs=plaice_zero_allocs)
        end

        @testset "Reactant" begin
            if RUN_REACTANT
                points = [rand(rng, d_ref) for _ in 1:10]
                test_reactant(d_wd, points)
            end
        end

        if test_batched
            @testset "iid" begin
                is_univariate = d_wd isa WrenDists.UnivariateDist
                dims = is_univariate ? (3, 4) : (3,)
                m = WrenCore.iid(d_wd, dims)
                @test size(m) == (size(d_wd)..., dims...)
                @test length(m) == prod(size(m))
                x = rand(rng, m)
                @test size(x) == size(m)
                if is_univariate
                    @test logdensityof(m, x) ≈ sum(logdensityof(d_wd, xi) for xi in x)
                else
                    n = prod(size(d_wd))
                    slices = eachcol(reshape(x, n, :))
                    @test logdensityof(m, x) ≈ sum(logdensityof(d_wd, c) for c in slices)
                end
            end

            @testset "independent" begin
                if d_wd isa WrenDists.UnivariateDist
                    dists = [d_wd, d_wd, d_wd]
                    m = WrenCore.independent(dists)
                    x = rand(rng, m)
                    @test size(x) == (3,)
                    @test logdensityof(m, x) ≈ sum(logdensityof.(dists, x))
                end
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
        "MvNormal (full rank)",
        WrenDists.MvNormal(μ, L),
        Distributions.MvNormal(μ, Σ);
        plaice_zero_allocs=MULTIVARIATE_PLAICE,
        # https://github.com/EnzymeAD/Enzyme.jl/issues/1991
        skipped_adtypes=("EnzymeFwd"),
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
        # https://github.com/JuliaDiff/ReverseDiff.jl/issues/289
        skipped_adtypes=("ReverseDiff", "ReverseDiffCpl"),
    )
    test_dist(
        "Multinomial",
        WrenDists.Multinomial(10, [0.2, 0.3, 0.5]),
        Distributions.Multinomial(10, [0.2, 0.3, 0.5]);
        plaice_zero_allocs=MULTIVARIATE_PLAICE,
        # https://github.com/JuliaDiff/ReverseDiff.jl/issues/289
        skipped_adtypes=("ReverseDiff", "ReverseDiffCpl"),
    )
end
