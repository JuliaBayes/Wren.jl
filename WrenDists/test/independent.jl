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

# Independent
test_dist(
    "Independent single type",
    WrenDists.Independent([
        WrenDists.Normal(1.0, 2.0),
        WrenDists.Normal(0.0, 1.0),
        WrenDists.Normal(3.0, 0.5),
    ]),
    Distributions.product_distribution([
        Distributions.Normal(1.0, 2.0),
        Distributions.Normal(0.0, 1.0),
        Distributions.Normal(3.0, 0.5),
    ]);
    test_reactant=RUN_REACTANT,
    other_ad_testcases=(
        let
            f(x) = logdensityof(
                WrenDists.Independent([
                    WrenDists.Normal(x[1], x[2]),
                    WrenDists.Normal(x[3], x[4]),
                    WrenDists.Normal(x[5], x[6]),
                ]),
                [0.5, 0.5, 0.5],
            )
            ADTestCase(f, [1.0, 2.0, 0.0, 1.0, 3.0, 0.5], (), ())
        end,
    ),
)

# independent() with a heterogeneous array is very problematic. For example,
#
# rand and logdensityof are type-unstable
#
# The bijectors are type unstable
#
# ReverseDiff: errors with the 'type does not have a definite number of fields' error
# https://github.com/JuliaDiff/ReverseDiff.jl/issues/289
#
# Enzyme: Doesn't like the type instability.
#
# test_dist(
#     "Independent multiple types",
#     WrenDists.Independent(WrenDists.UnivariateDist[WrenDists.Normal(0.0, 1.0), WrenDists.Exponential(2.0), WrenDists.Beta(2.0, 3.0)]),
#     Distributions.product_distribution(Distributions.UnivariateDistribution[Distributions.Normal(0.0, 1.0), Distributions.Exponential(2.0), Distributions.Beta(2.0, 3.0)]);
# )
