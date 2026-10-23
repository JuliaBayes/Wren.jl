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

@testset "Gamma sampling gradients" begin
    # An integer shape terminates the fraction before its shape derivative settles.
    x = [5.0, 0.7]
    u = rand(Xoshiro(8675309))
    q = quantile(Distributions.Gamma(x[1]), u)
    h = 1e-4
    dq = (quantile(Distributions.Gamma(x[1] + h), u) -
          quantile(Distributions.Gamma(x[1] - h), u)) / (2h)
    expected = [x[2] * dq, q]
    draw(p) = rand(Xoshiro(8675309), WrenDists.Gamma(p[1], p[2]))
    backends = Base.get_extension(WrenDists, :WrenDistsTestUtils).TESTED_ADTYPES
    @testset "$name" for (name, backend) in backends
        value, gradient = DI.value_and_gradient(draw, backend, x)
        @test value ≈ x[2] * q
        @test gradient ≈ expected rtol = 1e-6
    end
end

test_dist(
    "Gamma",
    WrenDists.Gamma(2.0, 1.0),
    Distributions.Gamma(2.0, 1.0);
    plaice_zero_allocs=(from_vec, from_unconstrained_vec),
    other_ad_testcases=(let f(x) = logdensityof(WrenDists.Gamma(x[1], x[2]), 1.0)
        ADTestCase(f, [2.0, 1.0], (), ())
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
    "BatchedGamma",
    WrenDists.BatchedGamma([1.0, 1.5, 2.5], [0.2, 0.5, 0.8]),
    Distributions.product_distribution(
        Distributions.Gamma.([1.0, 1.5, 2.5], [0.2, 0.5, 0.8]),
    );
    other_ad_testcases=(
        let
            f(x) = logdensityof(WrenDists.BatchedGamma(x[1:3], x[4:6]), [0.5, 1.5, 2.5])
            ADTestCase(f, [1.0, 1.5, 2.5, 0.2, 0.5, 0.8], (), ())
        end,
    ),
)
