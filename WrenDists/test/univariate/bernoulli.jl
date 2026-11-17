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

const RUN_REACTANT = !Sys.iswindows()

if RUN_REACTANT
    using Pkg
    Pkg.add("Reactant")
    using Reactant
end

@testset "BatchedBernoulli support" begin
    d = WrenDists.BatchedBernoulli(Float32[0, 1])
    points = [Float32[0, 1], Float32[1, 0]]
    @test logdensityof(d, points[1]) == 0
    @test logdensityof(d, points[2]) == -Inf
    interior = WrenDists.BatchedBernoulli(Float32[0.2, 0.8])
    @test logdensityof(interior, [2.0, 1.0]) == -Inf
    f(p, x) = logdensityof(WrenDists.BatchedBernoulli(p), x)
    for (_, adtype) in Base.get_extension(WrenDists, :WrenDistsTestUtils).TESTED_ADTYPES
        @test last(DI.value_and_gradient(f, adtype, d.p, DI.Constant(points[1]))) ≈ [-1, 1]
    end
    if RUN_REACTANT
        WrenDists.test_dist_reactant(d, points)
        WrenDists.test_dist_reactant(interior, [[0.0, 1.0], [2.0, 1.0]])
        p, x = Reactant.to_rarray.((d.p, points[1]))
        grad(p, x) = Enzyme.gradient(Enzyme.Reverse, f, p, Enzyme.Const(x))[1]
        compiled = Reactant.@compile grad(p, x)
        @test Array(compiled(p, x)) ≈ [-1, 1]
    end
end

test_dist(
    "Bernoulli",
    WrenDists.Bernoulli(0.3),
    Distributions.Bernoulli(0.3);
    plaice_zero_allocs=(from_vec, from_unconstrained_vec),
    other_ad_testcases=(let f(x) = logdensityof(WrenDists.Bernoulli(x[1]), 1)
        ADTestCase(f, [0.3], (), ())
    end,),
)

test_dist(
    "BernoulliLogit",
    WrenDists.BernoulliLogit(-0.4),
    Distributions.BernoulliLogit(-0.4);
    plaice_zero_allocs=(from_vec, from_unconstrained_vec),
    other_ad_testcases=(let f(x) = logdensityof(WrenDists.BernoulliLogit(x[1]), 1)
        ADTestCase(f, [-0.4], (), ())
    end,),
)

test_dist(
    "BatchedBernoulli",
    WrenDists.BatchedBernoulli([0.2, 0.5, 0.8]),
    Distributions.product_distribution(Distributions.Bernoulli.([0.2, 0.5, 0.8]));
    test_reactant=false, # https://github.com/EnzymeAD/Reactant.jl/issues/3281
    other_ad_testcases=(
        let
            f(x) = logdensityof(WrenDists.BatchedBernoulli(x), [1.0, 0.0, 1.0])
            ADTestCase(f, [0.2, 0.5, 0.8], (), ())
        end,
    ),
)

test_dist(
    "BatchedBernoulliLogit",
    WrenDists.BatchedBernoulliLogit([-1.4, 0.0, 1.4]),
    Distributions.product_distribution(Distributions.BernoulliLogit.([-1.4, 0.0, 1.4]));
    other_ad_testcases=(
        let
            f(x) = logdensityof(WrenDists.BatchedBernoulliLogit(x), [1.0, 0.0, 1.0])
            ADTestCase(f, [-1.4, 0.0, 1.4], (), ())
        end,
    ),
)
