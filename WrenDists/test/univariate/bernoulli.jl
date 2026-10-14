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

@testset "BernoulliLogit batched tails" begin
    for magnitude in (20.0, 40.0, 50.0, 100.0)
        expected_value = -log1p(exp(-magnitude))
        expected_gradient = exp(-magnitude) / (1 + exp(-magnitude))
        for (η, y, sign) in ((magnitude, 1, 1), (-magnitude, 0, -1))
            scalar(θ) = logdensityof(WrenDists.BernoulliLogit(θ), y)
            batched(θ) = logdensityof(WrenDists.BatchedBernoulliLogit([θ]), [y])
            for f in (scalar, batched)
                (gradient,), value = Enzyme.autodiff(
                    Enzyme.ReverseWithPrimal, f, Enzyme.Active, Enzyme.Active(η)
                )
                @test value ≈ expected_value rtol = 1e-12 atol = 0
                @test gradient ≈ sign * expected_gradient rtol = 1e-12 atol = 0
            end
        end
    end

    @test logdensityof(WrenDists.BatchedBernoulliLogit([40.0, -50.0]), [1, 0]) ≈
          -log1p(exp(-40.0)) - log1p(exp(-50.0)) rtol = 1e-12 atol = 0
    for y in (-1, 2, 0.5)
        @test logdensityof(WrenDists.BatchedBernoulliLogit([0.0]), [y]) == -Inf
    end
end
