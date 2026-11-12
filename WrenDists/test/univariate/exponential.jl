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

@testset "BatchedExponential support" begin
    d = WrenDists.BatchedExponential(Float32[1, 1])
    points = [Float32[0, 0], Float32[-1, 0]]
    @test logdensityof(d, points[1]) == 0
    @test logdensityof(d, points[2]) == -Inf
    if RUN_REACTANT
        WrenDists.test_dist_reactant(d, points)
    end
end

test_dist(
    "Exponential",
    WrenDists.Exponential(3.0),
    Distributions.Exponential(3.0);
    plaice_zero_allocs=(from_vec, from_unconstrained_vec),
    other_ad_testcases=(let f(x) = logdensityof(WrenDists.Exponential(x[1]), 1.0)
        ADTestCase(f, [3.0], (), ())
    end,),
)

test_dist(
    "BatchedExponential",
    WrenDists.BatchedExponential([0.5, 1.0, 2.0]),
    Distributions.product_distribution(Distributions.Exponential.([0.5, 1.0, 2.0]));
    other_ad_testcases=(
        let
            f(x) = logdensityof(WrenDists.BatchedExponential(x), [0.5, 0.5, 0.5])
            ADTestCase(f, [0.5, 1.0, 2.0], (), ())
        end,
    ),
)
