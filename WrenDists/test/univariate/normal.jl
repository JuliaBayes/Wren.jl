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
    "Normal",
    WrenDists.Normal(1.0, 2.0),
    Distributions.Normal(1.0, 2.0);
    plaice_zero_allocs=(from_vec, from_unconstrained_vec),
    other_ad_testcases=(let f(x) = logdensityof(WrenDists.Normal(x[1], x[2]), 1.0)
        ADTestCase(f, [1.0, 2.0], (), ())
    end,),
)

# Specialized: IIDNormal
test_dist(
    "IIDNormal",
    WrenDists.IIDNormal((3, 4), 1.0, 2.0),
    Distributions.product_distribution(fill(Distributions.Normal(1.0, 2.0), 3, 4));
    test_reactant=false, # loop-based logdensityof is not traceable yet
    other_ad_testcases=(
        let
            f(x) = logdensityof(WrenDists.IIDNormal((3, 4), x[1], x[2]), ones(3, 4))
            ADTestCase(f, [1.0, 2.0], (), ())
        end,
    ),
)

# Specialized: DiagNormal
test_dist(
    "DiagNormal",
    WrenDists.DiagNormal([1.0, 2.0, 3.0], [0.5, 1.5, 1.0]),
    Distributions.product_distribution([
        Distributions.Normal(1.0, 0.5),
        Distributions.Normal(2.0, 1.5),
        Distributions.Normal(3.0, 1.0),
    ]);
    other_ad_testcases=(
        let
            f(x) = logdensityof(WrenDists.DiagNormal(x[1:3], x[4:6]), [0.5, 0.5, 0.5])
            ADTestCase(f, [1.0, 2.0, 3.0, 0.5, 1.5, 1.0], (), ())
        end,
    ),
)

# Specialized: IsoNormal
test_dist(
    "IsoNormal",
    WrenDists.IsoNormal([1.0, 2.0, 3.0], 1.0),
    Distributions.product_distribution([
        Distributions.Normal(1.0, 1.0),
        Distributions.Normal(2.0, 1.0),
        Distributions.Normal(3.0, 1.0),
    ]);
    test_reactant=false, # loop-based logdensityof is not traceable yet
    other_ad_testcases=(
        let
            f(x) = logdensityof(WrenDists.IsoNormal(x[1:3], x[4]), [0.5, 0.5, 0.5])
            ADTestCase(f, [1.0, 2.0, 3.0, 1.0], (), ())
        end,
    ),
)
