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

for T in (Float32, Float64),
    d in (WrenDists.Beta(T(2), T(3)), WrenDists.BatchedBeta([T(2)], [T(3)]))

    x = T(0.25)
    bijector = Plaice.to_unconstrained_vec(d)
    z, ladj = Plaice.with_logabsdet_jacobian(bijector, Plaice.from_vec(d)([x]))
    @test (only(z), ladj) === (log(x) - log1p(-x), -log(x) - log1p(-x))
end

test_dist(
    "Beta",
    WrenDists.Beta(2.0, 3.0),
    Distributions.Beta(2.0, 3.0);
    plaice_zero_allocs=(from_vec, from_unconstrained_vec),
    other_ad_testcases=(let f(x) = logdensityof(WrenDists.Beta(x[1], x[2]), 0.5)
        ADTestCase(f, [2.0, 3.0], (), ())
    end,),
)

# Additional parameter combos
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

test_dist(
    "BatchedBeta",
    WrenDists.BatchedBeta([1.0, 1.5, 2.5], [0.5, 0.7, 0.9]),
    Distributions.product_distribution(
        Distributions.Beta.([1.0, 1.5, 2.5], [0.5, 0.7, 0.9]),
    );
    other_ad_testcases=(
        let
            f(x) = logdensityof(WrenDists.BatchedBeta(x[1:3], x[4:6]), [0.5, 0.7, 0.9])
            ADTestCase(f, [1.0, 1.5, 2.5, 0.5, 0.7, 0.9], (), ())
        end,
    ),
)

@testset "BatchedBeta gradients" begin
    # The transformed samples in test_dist do not reach these finite-density endpoints.
    f(x) = logdensityof(WrenDists.BatchedBeta([1.0, 2.0], [2.0, 1.0]), x)
    @test only(Enzyme.gradient(Reverse, f, [0.0, 1.0])) ≈ [-1.0, 1.0]

    @static if RUN_REACTANT
        # The zero coefficient at shape one must retain its parameter derivative.
        g(a, b) = logdensityof(WrenDists.BatchedBeta(a, b), [0.25f0])
        parameter_gradient(a, b) = Enzyme.gradient(Reverse, g, a, b)
        a, b = Reactant.to_rarray.(([1.0f0], [1.0f0]))
        compiled = Reactant.@compile parameter_gradient(a, b)
        da, db = compiled(a, b)
        @test Array(da) ≈ [1 + log(0.25f0)]
        @test Array(db) ≈ [1 + log(0.75f0)]
    end
end
