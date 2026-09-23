using WrenDists
using WrenDists: logdensityof
using WrenCore: WrenCore
using FillArrays
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
const REF_ADTYPE = DI.AutoForwardDiff()

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

# TODO(penelopeysm): use DIT.Scenario for this when the following is fixed:
# https://github.com/JuliaDiff/DifferentiationInterface.jl/issues/1069
struct ADTestCase{F,X,C,S}
    f::F
    x::X
    contexts::C
    skipped_adtypes::S
end
function test_ad(t::ADTestCase)
    ref_prep = DI.prepare_gradient(t.f, REF_ADTYPE, t.x, t.contexts...)
    g_ref = DI.gradient(t.f, ref_prep, REF_ADTYPE, t.x, t.contexts...)
    @testset "$adtype_name" for (adtype_name, adtype) in TESTED_ADTYPES
        adtype == REF_ADTYPE || adtype_name in t.skipped_adtypes && continue
        prep = DI.prepare_gradient(t.f, adtype, t.x, t.contexts...)
        g = DI.gradient(t.f, prep, adtype, t.x, t.contexts...)
        @test isapprox(g, g_ref; rtol=1e-6, atol=1e-6)
    end
end

function test_dist(
    name,
    d_wd,
    d_ref;
    plaice_zero_allocs=(),
    test_batched=true,
    test_reactant=true,
    skipped_adtypes=(),
    other_ad_testcases=(),
)
    @testset "$name" begin
        @info "Testing $name"

        rng = Xoshiro(468)

        @testset "logdensityof consistent with Distributions" begin
            for _ in 1:100
                x = rand(rng, d_ref)
                @test logdensityof(d_wd, x) ≈ logpdf(d_ref, x)
            end
        end

        @testset "logdensityof is differentiable wrt sample" begin
            if !(
                d_wd isa WrenDists.DiscreteUnivariateDist ||
                d_wd isa WrenDists.DiscreteMultivariateDist
            )
                f = logp_from_uncons
                x = rand(rng, Plaice.unconstrained_vec_length(d_wd))
                test_ad(ADTestCase(f, x, (DI.Constant(d_wd),), skipped_adtypes))
            end
        end

        @testset "logdensityof is differentiable wrt params" begin
            for t in other_ad_testcases
                test_ad(t)
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
            if !test_reactant
                # Marker so that disabled Reactant tests show up as broken rather than
                # silently passing.
                @test_broken false
            elseif RUN_REACTANT
                points = [rand(rng, d_ref) for _ in 1:10]
                test_dist_reactant(d_wd, points)
            end
        end

        if test_batched
            @testset "iid" begin
                is_univariate = d_wd isa WrenDists.UnivariateDist
                dims = is_univariate ? (3, 4) : (3,)
                m = WrenCore.iid(d_wd, dims)
                @test size(m) == (size(d_wd)..., dims...)
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
        other_ad_testcases=(let f(x) = logdensityof(WrenDists.Normal(x[1], x[2]), 1.0)
            ADTestCase(f, [1.0, 2.0], (), ())
        end,),
    )
    test_dist(
        "Laplace",
        WrenDists.Laplace(1.0, 2.0),
        Distributions.Laplace(1.0, 2.0);
        plaice_zero_allocs=UNIVARIATE_PLAICE,
        # Sample point != μ to avoid abs(0) subgradient disagreement across AD backends
        other_ad_testcases=(let f(x) = logdensityof(WrenDists.Laplace(x[1], x[2]), 1.5)
            ADTestCase(f, [1.0, 2.0], (), ())
        end,),
    )
    test_dist(
        "Cauchy",
        WrenDists.Cauchy(1.0, 2.0),
        Distributions.Cauchy(1.0, 2.0);
        plaice_zero_allocs=UNIVARIATE_PLAICE,
        other_ad_testcases=(let f(x) = logdensityof(WrenDists.Cauchy(x[1], x[2]), 1.0)
            ADTestCase(f, [1.0, 2.0], (), ())
        end,),
    )

    test_dist(
        "TDist",
        WrenDists.TDist(4.0, 1.0, 2.0),
        1.0 + 2.0 * Distributions.TDist(4.0);
        plaice_zero_allocs=UNIVARIATE_PLAICE,
        other_ad_testcases=(
            let f(x) = logdensityof(WrenDists.TDist(x[1], x[2], x[3]), 1.0)
                ADTestCase(f, [4.0, 1.0, 2.0], (), ())
            end,
        ),
    )

    # Continuous positive
    test_dist(
        "LogNormal",
        WrenDists.LogNormal(0.5, 0.3),
        Distributions.LogNormal(0.5, 0.3);
        plaice_zero_allocs=UNIVARIATE_PLAICE,
        other_ad_testcases=(let f(x) = logdensityof(WrenDists.LogNormal(x[1], x[2]), 1.5)
            ADTestCase(f, [0.5, 0.3], (), ())
        end,),
    )
    test_dist(
        "Exponential",
        WrenDists.Exponential(3.0),
        Distributions.Exponential(3.0);
        plaice_zero_allocs=UNIVARIATE_PLAICE,
        other_ad_testcases=(let f(x) = logdensityof(WrenDists.Exponential(x[1]), 1.0)
            ADTestCase(f, [3.0], (), ())
        end,),
    )
    test_dist(
        "Weibull",
        WrenDists.Weibull(1.5, 2.0),
        Distributions.Weibull(1.5, 2.0);
        plaice_zero_allocs=UNIVARIATE_PLAICE,
        other_ad_testcases=(let f(x) = logdensityof(WrenDists.Weibull(x[1], x[2]), 1.0)
            ADTestCase(f, [1.5, 2.0], (), ())
        end,),
    )
    test_dist(
        "Gamma",
        WrenDists.Gamma(2.0, 1.0),
        Distributions.Gamma(2.0, 1.0);
        plaice_zero_allocs=UNIVARIATE_PLAICE,
        other_ad_testcases=(let f(x) = logdensityof(WrenDists.Gamma(x[1], x[2]), 1.0)
            ADTestCase(f, [2.0, 1.0], (), ())
        end,),
    )

    # Continuous positive, truncated at zero
    for (μ, σ) in ((0.0, 2.0), (-1.5, 0.7), (1.0, 2.0))
        test_dist(
            "HalfNormal($μ,$σ)",
            WrenDists.HalfNormal(μ, σ),
            truncated(Distributions.Normal(μ, σ); lower=0);
            plaice_zero_allocs=UNIVARIATE_PLAICE,
            other_ad_testcases=(
                let f(x) = logdensityof(WrenDists.HalfNormal(x[1], x[2]), 1.0)
                    ADTestCase(f, [μ, σ], (), ())
                end,
            ),
        )
        test_dist(
            "HalfCauchy($μ,$σ)",
            WrenDists.HalfCauchy(μ, σ),
            truncated(Distributions.Cauchy(μ, σ); lower=0);
            plaice_zero_allocs=UNIVARIATE_PLAICE,
            other_ad_testcases=(
                let f(x) = logdensityof(WrenDists.HalfCauchy(x[1], x[2]), 1.0)
                    ADTestCase(f, [μ, σ], (), ())
                end,
            ),
        )
    end
    for (ν, μ, σ) in ((3.0, 0.0, 2.0), (3.5, -1.5, 0.7), (40.0, 1.0, 2.0))
        test_dist(
            "HalfTDist($ν,$μ,$σ)",
            WrenDists.HalfTDist(ν, μ, σ),
            truncated(μ + σ * Distributions.TDist(ν); lower=0);
            plaice_zero_allocs=UNIVARIATE_PLAICE,
            other_ad_testcases=(
                let f(x) = logdensityof(WrenDists.HalfTDist(x[1], x[2], x[3]), 1.0)
                    ADTestCase(f, [ν, μ, σ], (), ())
                end,
            ),
        )
    end

    @testset "t distribution function" begin
        # At `t = -1000` Distributions' log CDF is off by about 3e-11 against a `BigFloat`
        # evaluation, so the far tail is only checked between the two paths below.
        ts = (-3.0, -0.8, -1e-3, 0.0, 1e-3, 0.8, 3.0, 50.0)
        # `loggamma(ν/2) - loggamma((ν + 1)/2)` cancels, costing digits as ν grows.
        for (ν, rtol) in
            ((1e-2, 1e-12), (1.0, 1e-12), (3.0, 1e-12), (1e2, 1e-12), (1e4, 1e-11))
            for t in ts
                reference = logcdf(Distributions.TDist(ν), t)
                @test WrenDists.tdistlogcdf(ν, t, WrenDists.tdist_logbeta(ν)) ≈ reference rtol =
                    rtol
            end
        end
        for ν in (1e-2, 1.0, 3.0, 1e2, 1e4, 1e6), t in (-1e3, ts...)
            logbeta_half = WrenDists.tdist_logbeta(ν)
            @test WrenDists.tdistlogcdf_fixed(ν, t, logbeta_half) ≈
                  WrenDists.tdistlogcdf(ν, t, logbeta_half) rtol = 1e-10
        end
    end

    @testset "t distribution quantile" begin
        #=
          The quantile is checked by feeding it back through the distribution function.
          A `ν` below one reaches `1e154`, where `t^2` overflows, a few nats down, so its
          targets stay shallow.
        =#
        shallow = (-log(2), log(0.3), log(0.1))
        deep = (shallow..., log(1e-3), -20.0, -100.0, -300.0)
        cases = (
            (1e-2, shallow, 1e-12),
            (1.0, deep, 1e-12),
            (3.0, deep, 1e-12),
            (1e2, deep, 1e-12),
            (1e4, deep, 1e-10),
        )
        for (ν, logps, rtol) in cases
            logbeta_half = WrenDists.tdist_logbeta(ν)
            for logp in logps
                t = WrenDists.tdistlowerquantile(ν, logp, logbeta_half)
                @test t <= 0
                @test WrenDists.tdistlogcdf(ν, t, logbeta_half) ≈ logp rtol = rtol
                @test WrenDists.tdistlowerquantile_fixed(ν, logp, logbeta_half) ≈ t rtol =
                    1e-10
            end
        end
    end

    @testset "HalfTDist sampling" begin
        us = (0.01, 0.1, 0.3, 0.5, 0.9, 0.99)
        for (ν, μ, σ) in (
            (3.0, 0.0, 2.0),
            (3.5, -1.5, 0.7),
            (40.0, 1.0, 2.0),
            (1.0, 0.0, 1.0),
            (3.0, -8.0, 1.0),
        )
            d = WrenDists.HalfTDist(ν, μ, σ)
            mass = ccdf(Distributions.TDist(ν), -μ / σ)
            xs = [WrenDists._halft_upper_quantile(d, u) for u in us]
            for (u, x) in zip(us, xs)
                @test x >= 0
                @test ccdf(Distributions.TDist(ν), (x - μ) / σ) / mass ≈ u rtol = 1e-8
            end
            @test issorted(xs; rev=true)
            # The whole mass lies above the draw at `u = 1`, which rounding can place
            # either side of zero.
            @test 0 <= WrenDists._halft_upper_quantile(d, 1.0) <= 1e-12 * σ
            if RUN_REACTANT
                test_halft_quantile_reactant(d, us)
            end
        end
    end

    # Continuous bounded
    test_dist(
        "Beta",
        WrenDists.Beta(2.0, 3.0),
        Distributions.Beta(2.0, 3.0);
        plaice_zero_allocs=UNIVARIATE_PLAICE,
        other_ad_testcases=(let f(x) = logdensityof(WrenDists.Beta(x[1], x[2]), 0.5)
            ADTestCase(f, [2.0, 3.0], (), ())
        end,),
    )
    test_dist(
        "Uniform",
        WrenDists.Uniform(1.0, 3.0),
        Distributions.Uniform(1.0, 3.0);
        plaice_zero_allocs=UNIVARIATE_PLAICE,
        other_ad_testcases=(let f(x) = logdensityof(WrenDists.Uniform(x[1], x[2]), 2.0)
            ADTestCase(f, [1.0, 3.0], (), ())
        end,),
    )

    # Discrete univariate
    test_dist(
        "Bernoulli",
        WrenDists.Bernoulli(0.3),
        Distributions.Bernoulli(0.3);
        plaice_zero_allocs=UNIVARIATE_PLAICE,
        other_ad_testcases=(let f(x) = logdensityof(WrenDists.Bernoulli(x[1]), 1)
            ADTestCase(f, [0.3], (), ())
        end,),
    )
    test_dist(
        "Categorical",
        WrenDists.Categorical([0.2, 0.3, 0.5]),
        Distributions.Categorical([0.2, 0.3, 0.5]);
        plaice_zero_allocs=UNIVARIATE_PLAICE,
        other_ad_testcases=(let f(x) = logdensityof(WrenDists.Categorical(x), 2)
            ADTestCase(f, [0.2, 0.3, 0.5], (), ())
        end,),
    )
    test_dist(
        "Binomial",
        WrenDists.Binomial(20, 0.4),
        Distributions.Binomial(20, 0.4);
        plaice_zero_allocs=UNIVARIATE_PLAICE,
        other_ad_testcases=(let f(x) = logdensityof(WrenDists.Binomial(20, x[1]), 8)
            ADTestCase(f, [0.4], (), ())
        end,),
    )
    test_dist(
        "Poisson",
        WrenDists.Poisson(7.0),
        Distributions.Poisson(7.0);
        plaice_zero_allocs=UNIVARIATE_PLAICE,
        other_ad_testcases=(let f(x) = logdensityof(WrenDists.Poisson(x[1]), 7)
            ADTestCase(f, [7.0], (), ())
        end,),
    )
    test_dist(
        "Geometric",
        WrenDists.Geometric(0.3),
        Distributions.Geometric(0.3);
        plaice_zero_allocs=UNIVARIATE_PLAICE,
        other_ad_testcases=(let f(x) = logdensityof(WrenDists.Geometric(x[1]), 3)
            ADTestCase(f, [0.3], (), ())
        end,),
    )
    test_dist(
        "BernoulliLogit",
        WrenDists.BernoulliLogit(-0.4),
        Distributions.BernoulliLogit(-0.4);
        plaice_zero_allocs=UNIVARIATE_PLAICE,
        other_ad_testcases=(let f(x) = logdensityof(WrenDists.BernoulliLogit(x[1]), 1)
            ADTestCase(f, [-0.4], (), ())
        end,),
    )
    test_dist(
        "BinomialLogit",
        WrenDists.BinomialLogit(20, -0.4),
        Distributions.Binomial(20, inv(1 + exp(0.4)));
        plaice_zero_allocs=UNIVARIATE_PLAICE,
        other_ad_testcases=(let f(x) = logdensityof(WrenDists.BinomialLogit(20, x[1]), 8)
            ADTestCase(f, [-0.4], (), ())
        end,),
    )
    test_dist(
        "LogPoisson",
        WrenDists.LogPoisson(log(7.0)),
        Distributions.Poisson(7.0);
        plaice_zero_allocs=UNIVARIATE_PLAICE,
        other_ad_testcases=(let f(x) = logdensityof(WrenDists.LogPoisson(x[1]), 7)
            ADTestCase(f, [log(7.0)], (), ())
        end,),
    )

    # Additional parameter combos
    test_dist(
        "Gamma(0.5,3)",
        WrenDists.Gamma(0.5, 3.0),
        Distributions.Gamma(0.5, 3.0);
        other_ad_testcases=(let f(x) = logdensityof(WrenDists.Gamma(x[1], x[2]), 1.0)
            ADTestCase(f, [0.5, 3.0], (), ())
        end,),
    )
    test_dist(
        "Gamma(1,1)",
        WrenDists.Gamma(1.0, 1.0),
        Distributions.Gamma(1.0, 1.0);
        other_ad_testcases=(let f(x) = logdensityof(WrenDists.Gamma(x[1], x[2]), 1.0)
            ADTestCase(f, [1.0, 1.0], (), ())
        end,),
    )
    test_dist(
        "Beta(0.5,0.5)",
        WrenDists.Beta(0.5, 0.5),
        Distributions.Beta(0.5, 0.5);
        other_ad_testcases=(let f(x) = logdensityof(WrenDists.Beta(x[1], x[2]), 0.5)
            ADTestCase(f, [0.5, 0.5], (), ())
        end,),
    )
    test_dist(
        "Beta(1,1)",
        WrenDists.Beta(1.0, 1.0),
        Distributions.Beta(1.0, 1.0);
        # α != 1 to avoid xlogy(0, x) gradient disagreement across AD backends
        other_ad_testcases=(let f(x) = logdensityof(WrenDists.Beta(x[1], x[2]), 0.5)
            ADTestCase(f, [1.5, 1.5], (), ())
        end,),
    )

    # Multivariate
    μ = [1.0, 2.0]
    L = [1.0 0.0; 0.5 1.0]
    Σ = L * L'
    test_dist(
        "MvNormalScale (full rank)",
        WrenDists.MvNormalScale(μ, L),
        Distributions.MvNormal(μ, Σ);
        plaice_zero_allocs=MULTIVARIATE_PLAICE,
        # https://github.com/EnzymeAD/Enzyme.jl/issues/1991
        skipped_adtypes=["EnzymeFwd"],
        other_ad_testcases=(
            let f(x) = logdensityof(
                    WrenDists.MvNormalScale(x[1:2], [x[3] 0.0; x[4] x[5]]),
                    [0.5, 0.5],
                )
                ADTestCase(f, [1.0, 2.0, 1.0, 0.5, 1.0], (), ("EnzymeFwd",))
            end,
        ),
    )
    σ = [1.0, 2.0]
    test_dist(
        "MvNormalScale (diagonal)",
        WrenDists.MvNormalScale(μ, Diagonal(σ)),
        Distributions.MvNormal(μ, Diagonal(σ .^ 2));
        plaice_zero_allocs=MULTIVARIATE_PLAICE,
        other_ad_testcases=(
            let f(x) = logdensityof(
                    WrenDists.MvNormalScale(x[1:2], Diagonal(x[3:4])),
                    [0.5, 0.5],
                )
                ADTestCase(f, [1.0, 2.0, 1.0, 2.0], (), ())
            end,
        ),
    )
    test_dist(
        "MvNormalScale (isotropic)",
        WrenDists.MvNormalScale(μ, 2.0 * I),
        Distributions.MvNormal(μ, 4.0 * I);
        plaice_zero_allocs=MULTIVARIATE_PLAICE,
        other_ad_testcases=(
            let f(x) = logdensityof(WrenDists.MvNormalScale(x[1:2], x[3] * I), [0.5, 0.5])
                ADTestCase(f, [1.0, 2.0, 2.0], (), ())
            end,
        ),
    )
    test_dist(
        "Dirichlet",
        WrenDists.Dirichlet([2.0, 3.0, 1.0]),
        Distributions.Dirichlet([2.0, 3.0, 1.0]);
        plaice_zero_allocs=DIRICHLET_PLAICE,
        # https://github.com/JuliaDiff/ReverseDiff.jl/issues/289
        skipped_adtypes=["ReverseDiff", "ReverseDiffCpl"],
        # α != 1 to avoid xlogy(0, x) gradient disagreement across AD backends
        other_ad_testcases=(
            let f(x) = logdensityof(WrenDists.Dirichlet(x), [0.3, 0.5, 0.2])
                ADTestCase(f, [2.0, 3.0, 1.5], (), ("ReverseDiff", "ReverseDiffCpl"))
            end,
        ),
    )
    test_dist(
        "Multinomial",
        WrenDists.Multinomial(10, [0.2, 0.3, 0.5]),
        Distributions.Multinomial(10, [0.2, 0.3, 0.5]);
        plaice_zero_allocs=MULTIVARIATE_PLAICE,
        # https://github.com/JuliaDiff/ReverseDiff.jl/issues/289
        skipped_adtypes=["ReverseDiff", "ReverseDiffCpl"],
        other_ad_testcases=(
            let f(x) = logdensityof(WrenDists.Multinomial(10, x), [2, 3, 5])
                ADTestCase(f, [0.2, 0.3, 0.5], (), ("ReverseDiff", "ReverseDiffCpl"))
            end,
        ),
    )

    # --- Batched distributions ---

    # Independent
    test_dist(
        "Independent single type",
        WrenCore.independent([
            WrenDists.Normal(1.0, 2.0),
            WrenDists.Normal(0.0, 1.0),
            WrenDists.Normal(3.0, 0.5),
        ]),
        Distributions.product_distribution([
            Distributions.Normal(1.0, 2.0),
            Distributions.Normal(0.0, 1.0),
            Distributions.Normal(3.0, 0.5),
        ]);
        test_batched=false,
        test_reactant=false, # array-of-struct problem
        other_ad_testcases=(
            let
                f(x) = logdensityof(
                    WrenCore.independent([
                        WrenDists.Normal(x[1], x[2]),
                        WrenDists.Normal(x[3], x[4]),
                        WrenDists.Normal(x[5], x[6]),
                    ]),
                    [0.5, 0.5, 0.5],
                )
                ADTestCase(f, [1.0, 2.0, 0.0, 1.0, 3.0, 0.5], (), ())
            end,
        ),
    )

    # independent() with a heterogeneous array is very problematic. For example,
    #
    # rand and logdensityof are type-unstable
    #
    # The bijectors are type unstable
    #
    # ReverseDiff: errors with the 'type does not have a definite number of fields' error
    # https://github.com/JuliaDiff/ReverseDiff.jl/issues/289
    #
    # Enzyme: Doesn't like the type instability.
    #
    # test_dist(
    #     "Independent multiple types",
    #     WrenCore.independent(WrenDists.UnivariateDist[WrenDists.Normal(0.0, 1.0), WrenDists.Exponential(2.0), WrenDists.Beta(2.0, 3.0)]),
    #     Distributions.product_distribution(Distributions.UnivariateDistribution[Distributions.Normal(0.0, 1.0), Distributions.Exponential(2.0), Distributions.Beta(2.0, 3.0)]);
    #     test_batched=false,
    # )

    # Specialized: IIDNormal
    test_dist(
        "IIDNormal",
        WrenDists.IIDNormal((3, 4), 1.0, 2.0),
        Distributions.product_distribution(fill(Distributions.Normal(1.0, 2.0), 3, 4));
        test_batched=false,
        test_reactant=false, # loop-based logdensityof is not traceable yet
        other_ad_testcases=(
            let
                f(x) = logdensityof(WrenDists.IIDNormal((3, 4), x[1], x[2]), ones(3, 4))
                ADTestCase(f, [1.0, 2.0], (), ())
            end,
        ),
    )

    # Specialized: DiagNormal
    test_dist(
        "DiagNormal",
        WrenDists.DiagNormal([1.0, 2.0, 3.0], [0.5, 1.5, 1.0]),
        Distributions.product_distribution([
            Distributions.Normal(1.0, 0.5),
            Distributions.Normal(2.0, 1.5),
            Distributions.Normal(3.0, 1.0),
        ]);
        test_batched=false,
        test_reactant=false, # loop-based logdensityof is not traceable yet
        other_ad_testcases=(
            let
                f(x) = logdensityof(WrenDists.DiagNormal(x[1:3], x[4:6]), [0.5, 0.5, 0.5])
                ADTestCase(f, [1.0, 2.0, 3.0, 0.5, 1.5, 1.0], (), ())
            end,
        ),
    )

    @testset "BatchedLogNormal Enzyme rule" begin
        using EnzymeTestUtils
        mu = [0.1, 0.5, 1.0]
        sigma = [0.3, 0.7, 1.2]
        x = [1.5, 2.0, 3.0]
        @testset "reverse" begin
            @testset for Tmu in (Const, Duplicated),
                Tsigma in (Const, Duplicated),
                Tx in (Const, Duplicated)

                test_reverse(
                    WrenDists._lognormal_loop,
                    Active,
                    (mu, Tmu),
                    (sigma, Tsigma),
                    (x, Tx),
                )
            end
        end
        @testset "forward" begin
            @testset for RT in (Const, DuplicatedNoNeed, Duplicated),
                Tmu in (Const, Duplicated),
                Tsigma in (Const, Duplicated),
                Tx in (Const, Duplicated)

                test_forward(
                    WrenDists._lognormal_loop,
                    RT,
                    (mu, Tmu),
                    (sigma, Tsigma),
                    (x, Tx),
                )
            end
        end
    end

    # Specialized: IsoNormal
    test_dist(
        "IsoNormal",
        WrenDists.IsoNormal([1.0, 2.0, 3.0], 1.0),
        Distributions.product_distribution([
            Distributions.Normal(1.0, 1.0),
            Distributions.Normal(2.0, 1.0),
            Distributions.Normal(3.0, 1.0),
        ]);
        test_batched=false,
        test_reactant=false, # loop-based logdensityof is not traceable yet
        other_ad_testcases=(
            let
                f(x) = logdensityof(WrenDists.IsoNormal(x[1:3], x[4]), [0.5, 0.5, 0.5])
                ADTestCase(f, [1.0, 2.0, 3.0, 1.0], (), ())
            end,
        ),
    )
    # Specialized: batched univariate distributions
    test_dist(
        "BatchedBernoulli",
        WrenDists.BatchedBernoulli([0.2, 0.5, 0.8]),
        Distributions.product_distribution(Distributions.Bernoulli.([0.2, 0.5, 0.8]));
        test_batched=false,
        test_reactant=false, # loop-based logdensityof is not traceable yet
        other_ad_testcases=(
            let
                f(x) = logdensityof(WrenDists.BatchedBernoulli(x), [1.0, 0.0, 1.0])
                ADTestCase(f, [0.2, 0.5, 0.8], (), ())
            end,
        ),
    )
    test_dist(
        "BatchedBinomial",
        WrenDists.BatchedBinomial([5, 10, 20], [0.2, 0.5, 0.8]),
        Distributions.product_distribution(
            Distributions.Binomial.([5, 10, 20], [0.2, 0.5, 0.8]),
        );
        test_batched=false,
        test_reactant=false, # loop-based logdensityof is not traceable yet
        other_ad_testcases=(
            let
                f(x) = logdensityof(
                    WrenDists.BatchedBinomial([5, 10, 20], x),
                    [1.0, 4.0, 15.0],
                )
                ADTestCase(f, [0.2, 0.5, 0.8], (), ())
            end,
        ),
    )
    test_dist(
        "BatchedPoisson",
        WrenDists.BatchedPoisson([1.0, 3.0, 7.0]),
        Distributions.product_distribution(Distributions.Poisson.([1.0, 3.0, 7.0]));
        test_batched=false,
        test_reactant=false, # loop-based logdensityof is not traceable yet
        other_ad_testcases=(
            let
                f(x) = logdensityof(WrenDists.BatchedPoisson(x), [1.0, 2.0, 8.0])
                ADTestCase(f, [1.0, 3.0, 7.0], (), ())
            end,
        ),
    )
    test_dist(
        "BatchedBernoulliLogit",
        WrenDists.BatchedBernoulliLogit([-1.4, 0.0, 1.4]),
        Distributions.product_distribution(Distributions.BernoulliLogit.([-1.4, 0.0, 1.4]));
        test_batched=false,
        other_ad_testcases=(
            let
                f(x) = logdensityof(WrenDists.BatchedBernoulliLogit(x), [1.0, 0.0, 1.0])
                ADTestCase(f, [-1.4, 0.0, 1.4], (), ())
            end,
        ),
    )
    test_dist(
        "BatchedBinomialLogit",
        WrenDists.BatchedBinomialLogit([5, 10, 20], [-1.4, 0.0, 1.4]),
        Distributions.product_distribution(
            Distributions.Binomial.([5, 10, 20], inv.(1 .+ exp.(-[-1.4, 0.0, 1.4]))),
        );
        test_batched=false,
        test_reactant=false, # loop-based logdensityof is not traceable yet
        other_ad_testcases=(
            let
                f(x) = logdensityof(
                    WrenDists.BatchedBinomialLogit([5, 10, 20], x),
                    [1.0, 4.0, 15.0],
                )
                ADTestCase(f, [-1.4, 0.0, 1.4], (), ())
            end,
        ),
    )
    test_dist(
        "BatchedLogPoisson",
        WrenDists.BatchedLogPoisson(log.([1.0, 3.0, 7.0])),
        Distributions.product_distribution(Distributions.Poisson.([1.0, 3.0, 7.0]));
        test_batched=false,
        other_ad_testcases=(
            let
                f(x) = logdensityof(WrenDists.BatchedLogPoisson(x), [1.0, 2.0, 8.0])
                ADTestCase(f, log.([1.0, 3.0, 7.0]), (), ())
            end,
        ),
    )
    test_dist(
        "BatchedExponential",
        WrenDists.BatchedExponential([0.5, 1.0, 2.0]),
        Distributions.product_distribution(Distributions.Exponential.([0.5, 1.0, 2.0]));
        test_batched=false,
        test_reactant=false, # loop-based logdensityof is not traceable yet
        other_ad_testcases=(
            let
                f(x) = logdensityof(WrenDists.BatchedExponential(x), [0.5, 0.5, 0.5])
                ADTestCase(f, [0.5, 1.0, 2.0], (), ())
            end,
        ),
    )
    test_dist(
        "BatchedTDist",
        WrenDists.BatchedTDist(FillArrays.Fill(4.0, 3), [1.0, 2.0, 3.0], [0.5, 1.0, 2.0]),
        Distributions.product_distribution(
            [1.0, 2.0, 3.0] .+ [0.5, 1.0, 2.0] .* Distributions.TDist(4.0),
        );
        test_batched=false,
        test_reactant=false, # loop-based logdensityof is not traceable yet
        other_ad_testcases=(
            let
                f(x) = logdensityof(
                    WrenDists.BatchedTDist(FillArrays.Fill(4.0, 3), x[1:3], x[4:6]),
                    [1.5, 2.5, 3.5],
                )
                ADTestCase(f, [1.0, 2.0, 3.0, 0.5, 1.0, 2.0], (), ())
            end,
        ),
    )
    test_dist(
        "BatchedLogNormal",
        WrenDists.BatchedLogNormal([0.1, 0.5, 1.0], [0.3, 0.7, 1.2]),
        Distributions.product_distribution(
            Distributions.LogNormal.([0.1, 0.5, 1.0], [0.3, 0.7, 1.2]),
        );
        test_batched=false,
        test_reactant=false, # loop-based logdensityof is not traceable yet
        other_ad_testcases=(
            let
                f(x) = logdensityof(
                    WrenDists.BatchedLogNormal(x[1:3], x[4:6]),
                    [1.5, 1.5, 1.5],
                )
                ADTestCase(f, [0.1, 0.5, 1.0, 0.3, 0.7, 1.2], (), ())
            end,
        ),
    )
end
