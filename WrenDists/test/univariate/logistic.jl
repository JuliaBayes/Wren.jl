using WrenDists
using WrenDists: logdensityof, test_dist, ADTestCase
import DifferentiationInterface as DI, DifferentiationInterfaceTest
using Distributions
using ForwardDiff
using ReverseDiff
using Mooncake
using Enzyme
using Plaice
using Test
using Random

struct LogisticUniform{T} <: AbstractRNG
    value::T
end
Random.rand(rng::LogisticUniform{T}, ::Type{T}) where {T} = rng.value

@testset "Logistic uniform endpoints" begin
    for T in (Float32, Float64), u in (zero(T), T(0.25), prevfloat(one(T)))
        value = rand(LogisticUniform(u), WrenDists.Logistic(one(T), T(2)))
        @test isfinite(value)
        if !iszero(u)
            @test value ≈ 1 + 2 * (log(u) - log1p(-u))
        end
    end
end

const RUN_REACTANT = !Sys.iswindows()

if RUN_REACTANT
    using Pkg
    Pkg.add("Reactant")
    using Reactant
end

test_dist(
    "Logistic",
    WrenDists.Logistic(1.0, 2.0),
    Distributions.Logistic(1.0, 2.0);
    plaice_zero_allocs=(from_vec, from_unconstrained_vec),
    other_ad_testcases=(let f(x) = logdensityof(WrenDists.Logistic(x[1], x[2]), 1.5)
        ADTestCase(f, [1.0, 2.0], (), ())
    end,),
)
