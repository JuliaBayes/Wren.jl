using WrenDists
using WrenDists: logdensityof, test_dist
import DifferentiationInterface as DI
using Distributions
using ForwardDiff
using ReverseDiff
using Mooncake
using Enzyme
using Plaice
using Random: AbstractRNG
using Test

const RUN_REACTANT = !Sys.iswindows()

if RUN_REACTANT
    using Pkg
    Pkg.add("Reactant")
    using Reactant
end

# Distributions doesn't have a positive flat distribution, so this stands in as the
# reference. Its samples are just points to check the log-density at, and deliberately
# include negative points.
struct PositiveFlatReference end
Base.rand(rng::AbstractRNG, ::PositiveFlatReference) = 10 * randn(rng)
Distributions.logpdf(::PositiveFlatReference, x) = x >= 0 ? 0.0 : -Inf

test_dist(
    "PositiveFlat",
    WrenDists.PositiveFlat(),
    PositiveFlatReference();
    plaice_zero_allocs=(from_vec, from_unconstrained_vec),
)

@testset "PositiveFlat: rand is in (exp(-2), exp(2))" begin
    @test all(x -> exp(-2) < x < exp(2), [rand(WrenDists.PositiveFlat()) for _ in 1:1000])
end

@testset "PositiveFlat: sample type" begin
    @test rand(WrenDists.PositiveFlat()) isa Float64
    @test rand(WrenDists.PositiveFlat{Float32}()) isa Float32
end
