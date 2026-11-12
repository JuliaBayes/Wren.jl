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

@testset "Laplace large finite scale" begin
    for T in (Float32, Float64)
        b = T(0.75) * floatmax(T)
        d = WrenDists.Laplace(zero(T), b)
        @test logdensityof(d, zero(T)) ≈ -log(T(2)) - log(b)
        RUN_REACTANT && WrenDists.test_dist_reactant(d, T[0])
    end
end

test_dist(
    "Laplace",
    WrenDists.Laplace(1.0, 2.0),
    Distributions.Laplace(1.0, 2.0);
    plaice_zero_allocs=(from_vec, from_unconstrained_vec),
    # Sample point != μ to avoid abs(0) subgradient disagreement across AD backends
    other_ad_testcases=(let f(x) = logdensityof(WrenDists.Laplace(x[1], x[2]), 1.5)
        ADTestCase(f, [1.0, 2.0], (), ())
    end,),
)
