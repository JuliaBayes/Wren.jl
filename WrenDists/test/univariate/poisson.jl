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

const RUN_REACTANT = !Sys.iswindows()

if RUN_REACTANT
    using Pkg
    Pkg.add("Reactant")
    using Reactant
end

test_dist(
    "Poisson",
    WrenDists.Poisson(7.0),
    Distributions.Poisson(7.0);
    plaice_zero_allocs=(from_vec, from_unconstrained_vec),
    other_ad_testcases=(let f(x) = logdensityof(WrenDists.Poisson(x[1]), 7)
        ADTestCase(f, [7.0], (), ())
    end,),
)

test_dist(
    "PoissonLog",
    WrenDists.PoissonLog(log(7.0)),
    Distributions.Poisson(7.0);
    plaice_zero_allocs=(from_vec, from_unconstrained_vec),
    other_ad_testcases=(let f(x) = logdensityof(WrenDists.PoissonLog(x[1]), 7)
        ADTestCase(f, [log(7.0)], (), ())
    end,),
)

test_dist(
    "BatchedPoisson",
    WrenDists.BatchedPoisson([1.0, 3.0, 7.0]),
    Distributions.product_distribution(Distributions.Poisson.([1.0, 3.0, 7.0]));
    other_ad_testcases=(
        let
            f(x) = logdensityof(WrenDists.BatchedPoisson(x), [1.0, 2.0, 8.0])
            ADTestCase(f, [1.0, 3.0, 7.0], (), ())
        end,
    ),
)

test_dist(
    "BatchedPoissonLog",
    WrenDists.BatchedPoissonLog(log.([1.0, 3.0, 7.0])),
    Distributions.product_distribution(Distributions.Poisson.([1.0, 3.0, 7.0]));
    other_ad_testcases=(
        let
            f(x) = logdensityof(WrenDists.BatchedPoissonLog(x), [1.0, 2.0, 8.0])
            ADTestCase(f, log.([1.0, 3.0, 7.0]), (), ())
        end,
    ),
)

@testset "Poisson sampling beyond maxintfloat" begin
    # Counts this large cannot step by one in their float type, which used to hang `rand`.
    for λ in (1.0f8, 1.0f12, 1.0e17)
        d = WrenDists.Poisson(λ)
        xs = [rand(Xoshiro(i), d) for i in 1:1_000]
        @test eltype(xs) == typeof(λ)
        @test all(x -> isinteger(x) && x >= 0, xs)
        @test isapprox(mean(xs), λ; rtol=1e-3)
    end
end
