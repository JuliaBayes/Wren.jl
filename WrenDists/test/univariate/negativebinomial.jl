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

for r in (2.5f0, 2.5)
    @test logdensityof(WrenDists.NegativeBinomial(r), 3) ===
          logdensityof(WrenDists.NegativeBinomial(r, oftype(r, 0.5)), 3)
end

test_dist(
    "NegativeBinomial",
    WrenDists.NegativeBinomial(2.5, 0.4),
    Distributions.NegativeBinomial(2.5, 0.4);
    plaice_zero_allocs=(from_vec, from_unconstrained_vec),
    other_ad_testcases=(
        let f(x) = logdensityof(WrenDists.NegativeBinomial(x[1], x[2]), 3)
            ADTestCase(f, [2.5, 0.4], (), ())
        end,
    ),
)

test_dist(
    "BatchedNegativeBinomial",
    WrenDists.BatchedNegativeBinomial([1.0, 1.5, 2.5], [0.2, 0.5, 0.8]),
    Distributions.product_distribution(
        Distributions.NegativeBinomial.([1.0, 1.5, 2.5], [0.2, 0.5, 0.8]),
    );
    other_ad_testcases=(
        let
            f(x) = logdensityof(
                WrenDists.BatchedNegativeBinomial(x[1:3], x[4:6]),
                [0.0, 3.0, 8.0],
            )
            ADTestCase(f, [1.0, 1.5, 2.5, 0.2, 0.5, 0.8], (), ())
        end,
    ),
)
