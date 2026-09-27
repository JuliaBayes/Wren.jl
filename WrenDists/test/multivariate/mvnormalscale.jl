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
using LinearAlgebra

const RUN_REACTANT = !Sys.iswindows()

if RUN_REACTANT
    using Pkg
    Pkg.add("Reactant")
    using Reactant
end

μ = [1.0, 2.0]
L = [1.0 0.0; 0.5 1.0]
Σ = L * L'
test_dist(
    "MvNormalScale (full rank)",
    WrenDists.MvNormalScale(μ, L),
    Distributions.MvNormal(μ, Σ);
    plaice_zero_allocs=(to_vec, from_vec, to_unconstrained_vec, from_unconstrained_vec),
    # https://github.com/EnzymeAD/Enzyme.jl/issues/1991
    skipped_adtypes=["EnzymeFwd"],
    other_ad_testcases=(
        let f(x) = logdensityof(
                WrenDists.MvNormalScale(x[1:2], [x[3] 0.0; x[4] x[5]]),
                [0.5, 0.5],
            )
            ADTestCase(f, [1.0, 2.0, 1.0, 0.5, 1.0], (), ("EnzymeFwd",))
        end,
    ),
)
σ = [1.0, 2.0]
test_dist(
    "MvNormalScale (diagonal)",
    WrenDists.MvNormalScale(μ, Diagonal(σ)),
    Distributions.MvNormal(μ, Diagonal(σ .^ 2));
    plaice_zero_allocs=(to_vec, from_vec, to_unconstrained_vec, from_unconstrained_vec),
    other_ad_testcases=(
        let f(x) =
                logdensityof(WrenDists.MvNormalScale(x[1:2], Diagonal(x[3:4])), [0.5, 0.5])
            ADTestCase(f, [1.0, 2.0, 1.0, 2.0], (), ())
        end,
    ),
)
test_dist(
    "MvNormalScale (isotropic)",
    WrenDists.MvNormalScale(μ, 2.0 * I),
    Distributions.MvNormal(μ, 4.0 * I);
    plaice_zero_allocs=(to_vec, from_vec, to_unconstrained_vec, from_unconstrained_vec),
    other_ad_testcases=(
        let f(x) = logdensityof(WrenDists.MvNormalScale(x[1:2], x[3] * I), [0.5, 0.5])
            ADTestCase(f, [1.0, 2.0, 2.0], (), ())
        end,
    ),
)
