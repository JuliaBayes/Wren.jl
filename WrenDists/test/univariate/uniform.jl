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
    "Uniform",
    WrenDists.Uniform(1.0, 3.0),
    Distributions.Uniform(1.0, 3.0);
    plaice_zero_allocs=(from_vec, from_unconstrained_vec),
    other_ad_testcases=(let f(x) = logdensityof(WrenDists.Uniform(x[1], x[2]), 2.0)
        ADTestCase(f, [1.0, 3.0], (), ())
    end,),
)
