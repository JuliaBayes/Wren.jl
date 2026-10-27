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

@testset "Binomial batch support" for (D, parameter) in (
    (WrenDists.BatchedBinomial, 0.5),
    (WrenDists.BatchedBinomialLogit, 0.0),
)
    d = D([3], [parameter])
    for x in (-1.0, 1.5, 4.0)
        @test logdensityof(d, [x]) == -Inf
    end
    if RUN_REACTANT
        dr = Reactant.to_rarray(d)
        compiled = Reactant.@compile logdensityof(dr, Reactant.to_rarray([1.5]))
        for x in (-1.0, 1.5, 4.0)
            @test Float64(compiled(dr, Reactant.to_rarray([x]))) == -Inf
        end
        f(p, x) = logdensityof(D([3], p), x)
        gradient(p, x) = first(Enzyme.gradient(Reverse, f, p, x))
        p = Reactant.to_rarray([parameter])
        x = Reactant.to_rarray([NaN])
        compiled_gradient = Reactant.@compile gradient(p, x)
        @test Array(compiled_gradient(p, x)) == [0.0]
    end
end

test_dist(
    "Binomial",
    WrenDists.Binomial(20, 0.4),
    Distributions.Binomial(20, 0.4);
    plaice_zero_allocs=(from_vec, from_unconstrained_vec),
    other_ad_testcases=(let f(x) = logdensityof(WrenDists.Binomial(20, x[1]), 8)
        ADTestCase(f, [0.4], (), ())
    end,),
)

test_dist(
    "BinomialLogit",
    WrenDists.BinomialLogit(20, -0.4),
    Distributions.Binomial(20, inv(1 + exp(0.4)));
    plaice_zero_allocs=(from_vec, from_unconstrained_vec),
    other_ad_testcases=(let f(x) = logdensityof(WrenDists.BinomialLogit(20, x[1]), 8)
        ADTestCase(f, [-0.4], (), ())
    end,),
)

test_dist(
    "BatchedBinomial",
    WrenDists.BatchedBinomial([5, 10, 20], [0.2, 0.5, 0.8]),
    Distributions.product_distribution(
        Distributions.Binomial.([5, 10, 20], [0.2, 0.5, 0.8]),
    );
    other_ad_testcases=(
        let
            f(x) =
                logdensityof(WrenDists.BatchedBinomial([5, 10, 20], x), [1.0, 4.0, 15.0])
            ADTestCase(f, [0.2, 0.5, 0.8], (), ())
        end,
    ),
)

test_dist(
    "BatchedBinomialLogit",
    WrenDists.BatchedBinomialLogit([5, 10, 20], [-1.4, 0.0, 1.4]),
    Distributions.product_distribution(
        Distributions.Binomial.([5, 10, 20], inv.(1 .+ exp.(-[-1.4, 0.0, 1.4]))),
    );
    other_ad_testcases=(
        let
            f(x) = logdensityof(
                WrenDists.BatchedBinomialLogit([5, 10, 20], x),
                [1.0, 4.0, 15.0],
            )
            ADTestCase(f, [-1.4, 0.0, 1.4], (), ())
        end,
    ),
)
