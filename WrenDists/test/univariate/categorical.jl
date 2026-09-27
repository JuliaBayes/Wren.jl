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
    "Categorical",
    WrenDists.Categorical([0.2, 0.3, 0.5]),
    Distributions.Categorical([0.2, 0.3, 0.5]);
    plaice_zero_allocs=(from_vec, from_unconstrained_vec),
    other_ad_testcases=(let f(x) = logdensityof(WrenDists.Categorical(x), 2)
        ADTestCase(f, [0.2, 0.3, 0.5], (), ())
    end,),
)
