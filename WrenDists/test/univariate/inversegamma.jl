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
    "InverseGamma",
    WrenDists.InverseGamma(2.5, 1.5),
    Distributions.InverseGamma(2.5, 1.5);
    plaice_zero_allocs=(from_vec, from_unconstrained_vec),
    other_ad_testcases=(let f(x) = logdensityof(WrenDists.InverseGamma(x[1], x[2]), 1.0)
        ADTestCase(f, [2.5, 1.5], (), ())
    end,),
)

test_dist(
    "BatchedInverseGamma",
    WrenDists.BatchedInverseGamma([1.0, 1.5, 2.5], [0.2, 0.5, 0.8]),
    Distributions.product_distribution(
        Distributions.InverseGamma.([1.0, 1.5, 2.5], [0.2, 0.5, 0.8]),
    );
    other_ad_testcases=(
        let
            f(x) = logdensityof(
                WrenDists.BatchedInverseGamma(x[1:3], x[4:6]),
                [0.5, 1.5, 2.5],
            )
            ADTestCase(f, [1.0, 1.5, 2.5, 0.2, 0.5, 0.8], (), ())
        end,
    ),
)
