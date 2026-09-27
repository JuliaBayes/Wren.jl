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
    "Multinomial",
    WrenDists.Multinomial(10, [0.2, 0.3, 0.5]),
    Distributions.Multinomial(10, [0.2, 0.3, 0.5]);
    plaice_zero_allocs=(to_vec, from_vec, to_unconstrained_vec, from_unconstrained_vec),
    # https://github.com/JuliaDiff/ReverseDiff.jl/issues/289
    skipped_adtypes=["ReverseDiff", "ReverseDiffCpl"],
    other_ad_testcases=(
        let f(x) = logdensityof(WrenDists.Multinomial(10, x), [2, 3, 5])
            ADTestCase(f, [0.2, 0.3, 0.5], (), ("ReverseDiff", "ReverseDiffCpl"))
        end,
    ),
)
