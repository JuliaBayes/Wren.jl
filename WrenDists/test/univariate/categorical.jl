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

@testset "BatchedCategorical labels" begin
    P = [0.25 0.75; 0.75 0.25]
    d = WrenDists.BatchedCategorical(P)
    @test logdensityof(d, [1.0, 2.0]) == logdensityof(d, [1, 2])
    for x in (0.0, 3.0, 1.5)
        @test logdensityof(d, [x, 2.0]) == -Inf
    end
    if RUN_REACTANT
        dr = WrenDists.BatchedCategorical(Reactant.to_rarray(P))
        compiled = Reactant.@compile logdensityof(dr, Reactant.to_rarray([1.0, 2.0]))
        for x in ([1.0, 2.0], [1.5, 2.0])
            @test Float64(compiled(dr, Reactant.to_rarray(x))) == logdensityof(d, x)
        end
    end
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
