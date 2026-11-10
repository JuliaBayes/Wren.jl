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
