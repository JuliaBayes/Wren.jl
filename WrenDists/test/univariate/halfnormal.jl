using WrenDists
using WrenDists: logdensityof, test_dist, ADTestCase
import DifferentiationInterface as DI, DifferentiationInterfaceTest
using Distributions
using ForwardDiff
using ReverseDiff
using Mooncake
using Enzyme
using Plaice
using Test
using Random: Xoshiro

const RUN_REACTANT = !Sys.iswindows()

if RUN_REACTANT
    using Pkg
    Pkg.add("Reactant")
    using Reactant
end

# Continuous positive, truncated at zero
for (μ, σ) in ((0.0, 2.0), (-1.5, 0.7), (1.0, 2.0))
    test_dist(
        "HalfNormal($μ,$σ)",
        WrenDists.HalfNormal(μ, σ),
        truncated(Distributions.Normal(μ, σ); lower=0);
        plaice_zero_allocs=(from_vec, from_unconstrained_vec),
        other_ad_testcases=(
            let f(x) = logdensityof(WrenDists.HalfNormal(x[1], x[2]), 1.0)
                ADTestCase(f, [μ, σ], (), ())
            end,
        ),
    )
end

@testset "HalfNormal far-tail sampling" begin
    # The mass above zero underflows for very negative `μ / σ`; draws must stay finite.
    for (μ, σ) in ((-40.0, 1.0), (-1e8, 3.0), (-40.0f0, 1.0f0))
        d = WrenDists.HalfNormal(μ, σ)
        xs = [rand(Xoshiro(i), d) for i in 1:10_000]
        @test all(x -> isfinite(x) && x >= 0, xs)
        # Deep in the tail the truncated normal is close to Exponential(σ^2 / -μ).
        @test isapprox(mean(xs), σ^2 / -μ; rtol=0.05)
    end
end

@testset "HalfNormal keeps Float32" begin
    for μ in (-4.0f0, 4.0f0)
        d = WrenDists.HalfNormal(μ, 1.0f0)
        @test rand(Xoshiro(468), d) isa Float32
        @test logdensityof(d, 1.0f0) isa Float32
    end
end
