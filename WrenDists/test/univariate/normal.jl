using WrenDists
using WrenDists: logdensityof, test_dist, ADTestCase
import DifferentiationInterface as DI
using Distributions
using ForwardDiff
using ReverseDiff
using Mooncake
using Enzyme
using FillArrays: Fill
using Plaice
using Test

const RUN_REACTANT = !Sys.iswindows()

if RUN_REACTANT
    using Pkg
    Pkg.add("Reactant")
    using Reactant
end

test_dist(
    "Normal",
    WrenDists.Normal(1.0, 2.0),
    Distributions.Normal(1.0, 2.0);
    plaice_zero_allocs=(from_vec, from_unconstrained_vec),
    other_ad_testcases=(let f(x) = logdensityof(WrenDists.Normal(x[1], x[2]), 1.0)
        ADTestCase(f, [1.0, 2.0], (), ())
    end,),
)

test_dist(
    "iid Normal",
    WrenDists.Independent(WrenDists.Repeated(WrenDists.Normal(1.0, 2.0), (3, 4))),
    Distributions.product_distribution(fill(Distributions.Normal(1.0, 2.0), 3, 4));
    other_ad_testcases=(
        let
            f(x) = logdensityof(
                WrenDists.Independent(
                    WrenDists.Repeated(WrenDists.Normal(x[1], x[2]), (3, 4)),
                ),
                ones(3, 4),
            )
            ADTestCase(f, [1.0, 2.0], (), ())
        end,
    ),
)

test_dist(
    "BatchedNormal",
    WrenDists.BatchedNormal([1.0, 2.0, 3.0], [0.5, 1.5, 1.0]),
    Distributions.product_distribution([
        Distributions.Normal(1.0, 0.5),
        Distributions.Normal(2.0, 1.5),
        Distributions.Normal(3.0, 1.0),
    ]);
    other_ad_testcases=(
        let
            f(x) = logdensityof(WrenDists.BatchedNormal(x[1:3], x[4:6]), [0.5, 0.5, 0.5])
            ADTestCase(f, [1.0, 2.0, 3.0, 0.5, 1.5, 1.0], (), ())
        end,
    ),
)

test_dist(
    "BatchedNormal (constant sigma)",
    WrenDists.BatchedNormal([1.0, 2.0, 3.0], Fill(1.0, 3)),
    Distributions.product_distribution([
        Distributions.Normal(1.0, 1.0),
        Distributions.Normal(2.0, 1.0),
        Distributions.Normal(3.0, 1.0),
    ]);
    other_ad_testcases=(
        let
            f(x) = logdensityof(
                WrenDists.BatchedNormal(x[1:3], Fill(x[4], 3)),
                [0.5, 0.5, 0.5],
            )
            ADTestCase(f, [1.0, 2.0, 3.0, 1.0], (), ())
        end,
    ),
)

test_dist(
    "BatchedNormal (constant mu)",
    WrenDists.BatchedNormal(Fill(1.0, 3), [0.5, 1.5, 1.0]),
    Distributions.product_distribution([
        Distributions.Normal(1.0, 0.5),
        Distributions.Normal(1.0, 1.5),
        Distributions.Normal(1.0, 1.0),
    ]);
    other_ad_testcases=(
        let
            f(x) = logdensityof(
                WrenDists.BatchedNormal(Fill(x[1], 3), x[2:4]),
                [0.5, 0.5, 0.5],
            )
            ADTestCase(f, [1.0, 0.5, 1.5, 1.0], (), ())
        end,
    ),
)
