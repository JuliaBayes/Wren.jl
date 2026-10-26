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
using Random: Xoshiro
using WrenDists: test_halft_quantile_reactant

const RUN_REACTANT = !Sys.iswindows()

if RUN_REACTANT
    using Pkg
    Pkg.add("Reactant")
    using Reactant
end

for (ν, μ, σ) in ((3.0, 0.0, 2.0), (3.5, -1.5, 0.7), (40.0, 1.0, 2.0))
    test_dist(
        "HalfTDist($ν,$μ,$σ)",
        WrenDists.HalfTDist(ν, μ, σ),
        truncated(μ + σ * Distributions.TDist(ν); lower=0);
        plaice_zero_allocs=(from_vec, from_unconstrained_vec),
        other_ad_testcases=(
            let f(x) = logdensityof(WrenDists.HalfTDist(x[1], x[2], x[3]), 1.0)
                ADTestCase(f, [ν, μ, σ], (), ())
            end,
        ),
    )
end

@testset "HalfTDist keeps Float32" begin
    d = WrenDists.HalfTDist(4.0f0, -1.0f0, 2.0f0)
    @test rand(Xoshiro(468), d) isa Float32
    @test logdensityof(d, 1.0f0) isa Float32
    @test logdensityof(d, 1.0f0) ≈ logdensityof(WrenDists.HalfTDist(4.0, -1.0, 2.0), 1.0)
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
