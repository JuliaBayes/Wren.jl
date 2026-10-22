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
    "Dirichlet",
    WrenDists.Dirichlet([2.0, 3.0, 1.0]),
    Distributions.Dirichlet([2.0, 3.0, 1.0]);
    plaice_zero_allocs=(to_vec, from_vec),
    # https://github.com/JuliaDiff/ReverseDiff.jl/issues/289
    skipped_adtypes=["ReverseDiff", "ReverseDiffCpl"],
    # α != 1 to avoid xlogy(0, x) gradient disagreement across AD backends
    other_ad_testcases=(
        let f(x) = logdensityof(WrenDists.Dirichlet(x), [0.3, 0.5, 0.2])
            ADTestCase(f, [2.0, 3.0, 1.5], (), ("ReverseDiff", "ReverseDiffCpl"))
        end,
    ),
)
