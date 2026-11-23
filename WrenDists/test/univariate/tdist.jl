using WrenDists
using WrenDists: logdensityof, test_dist, ADTestCase
import DifferentiationInterface as DI
using Distributions
using ForwardDiff
using ReverseDiff
using Mooncake
using Enzyme
using Plaice
using Test
using FillArrays
using Random: Xoshiro

const RUN_REACTANT = !Sys.iswindows()

if RUN_REACTANT
    using Pkg
    Pkg.add("Reactant")
    using Reactant
end

@testset "TDist typed defaults" begin
    d = WrenDists.TDist(4.0f0)
    explicit = WrenDists.TDist(4.0f0, 0.0f0, 1.0f0)
    @test logdensityof(d, 0.25f0) === logdensityof(explicit, 0.25f0)
    @test rand(Xoshiro(8675309), d) === rand(Xoshiro(8675309), explicit)
end

@testset "TDist finite tails" begin
    for T in (Float32, Float64)
        x = T(2) * sqrt(floatmax(T))
        reference = logdensityof(WrenDists.Cauchy(zero(T), one(T)), x)
        @test logdensityof(WrenDists.TDist(one(T)), x) ≈ reference
        iid = WrenDists.IIDTDist((1,), one(T), zero(T), one(T))
        @test logdensityof(iid, [x]) ≈ reference
        for ν in (T[1], Fill(one(T), 1))
            @test logdensityof(WrenDists.BatchedTDist(ν, T[0], T[1]), [x]) ≈ reference
        end
    end
    f(x) = logdensityof(WrenDists.TDist(1.0), only(x))
    ad = DI.AutoReverseDiff(; compile=true)
    prep = DI.prepare_gradient(f, ad, [0.0])
    for x in (1.0, -1.0, 1e155, -1e155, Inf, -Inf, 1.0)
        value, gradient = DI.value_and_gradient(f, prep, ad, [x])
        @test value ≈ logdensityof(WrenDists.Cauchy(), x)
        isfinite(x) && @test only(gradient) ≈ -2 * (x / hypot(1, x)) / hypot(1, x)
    end
    nu_score(p) = logdensityof(WrenDists.TDist(only(p)), 1e10)
    nu_tie_score(p) = logdensityof(WrenDists.TDist(only(p)), 1.0)
    for (_, adtype) in Base.get_extension(WrenDists, :WrenDistsTestUtils).TESTED_ADTYPES
        for x in (1.0, floatmax(Float64))
            @test only(DI.gradient(f, adtype, [x])) ≈ -2 * (x / hypot(1, x)) / hypot(1, x)
        end
        @test only(DI.gradient(nu_tie_score, adtype, [1.0])) ≈ log(2) / 2
        @test only(DI.gradient(nu_score, adtype, [1e-300])) ≈ 1e300
    end
    @static if RUN_REACTANT
        d = WrenDists.TDist(1.0f0)
        WrenDists.test_dist_reactant(d, Float32[2*sqrt(floatmax(Float32)), Inf])
        g(ν, x) = logdensityof(WrenDists.TDist(ν), x)
        gradient(ν, x) = Enzyme.gradient(Reverse, g, ν, x)
        ν = Reactant.to_rarray(1.0f0; track_numbers=true)
        x = Reactant.to_rarray(1.0f0; track_numbers=true)
        compiled = Reactant.@compile gradient(ν, x)
        @test collect(Float32.(compiled(ν, x))) ≈ [log(2.0f0) / 2, -1.0f0]
    end
end

test_dist(
    "TDist",
    WrenDists.TDist(4.0, 1.0, 2.0),
    1.0 + 2.0 * Distributions.TDist(4.0);
    plaice_zero_allocs=(from_vec, from_unconstrained_vec),
    other_ad_testcases=(let f(x) = logdensityof(WrenDists.TDist(x[1], x[2], x[3]), 1.0)
        ADTestCase(f, [4.0, 1.0, 2.0], (), ())
    end,),
)

@testset "t distribution function" begin
    # At `t = -1000` Distributions' log CDF is off by about 3e-11 against a `BigFloat`
    # evaluation, so the far tail is only checked between the two paths below.
    ts = (-3.0, -0.8, -1e-3, 0.0, 1e-3, 0.8, 3.0, 50.0)
    # `loggamma(ν/2) - loggamma((ν + 1)/2)` cancels, costing digits as ν grows.
    for (ν, rtol) in ((1e-2, 1e-12), (1.0, 1e-12), (3.0, 1e-12), (1e2, 1e-12), (1e4, 1e-11))
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
            @test WrenDists.tdistlowerquantile_fixed(ν, logp, logbeta_half) ≈ t rtol = 1e-10
        end
    end
end

@testset "BatchedTDist matrix sampling" begin
    d = WrenDists.BatchedTDist(fill(4.0f0, 2, 3), Float32[1 2 3; 4 5 6], fill(0.5f0, 2, 3))
    flat = WrenDists.BatchedTDist(vec(d.ν), vec(d.μ), vec(d.σ))
    @test rand(Xoshiro(8675309), d) == reshape(rand(Xoshiro(8675309), flat), size(d))
end

test_dist(
    "BatchedTDist",
    WrenDists.BatchedTDist(FillArrays.Fill(4.0, 3), [1.0, 2.0, 3.0], [0.5, 1.0, 2.0]),
    Distributions.product_distribution(
        [1.0, 2.0, 3.0] .+ [0.5, 1.0, 2.0] .* Distributions.TDist(4.0),
    );
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
