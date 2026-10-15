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
        let
            f(x) = logdensityof(WrenDists.BatchedPoisson(x), [0.0, 0.0])
            ADTestCase(f, [0.0, 1.0], (), ())
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
        let
            f(x) = logdensityof(WrenDists.BatchedPoissonLog(x), [0.0, 0.0])
            ADTestCase(f, [-Inf, 0.0], (), ())
        end,
    ),
)

@testset "batched Poisson support (issue #158)" begin
    for (scalar, batched, rates) in (
        (WrenDists.Poisson, WrenDists.BatchedPoisson, [0.0, 1.0]),
        (WrenDists.PoissonLog, WrenDists.BatchedPoissonLog, [-Inf, 0.0]),
    )
        d = batched(rates)
        points = [[0.0, 0.0], [1.0, 0.0], [0.0, 0.5], [0.0, -2.0], [0.0, Inf]]
        for x in points
            @test logdensityof(d, x) == sum(logdensityof.(scalar.(rates), x))
        end
        RUN_REACTANT && WrenDists.test_dist_reactant(d, points)
    end
end
