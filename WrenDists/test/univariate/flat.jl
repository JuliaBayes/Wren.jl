using WrenDists
using WrenDists: logdensityof, test_dist, ADTestCase
import DifferentiationInterface as DI, DifferentiationInterfaceTest
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

# Distributions doesn't have a flat distribution, so this stands in as the reference. Its
# samples are just points to check the log-density at.
struct FlatReference end
Base.rand(rng::AbstractRNG, ::FlatReference) = 10 * randn(rng)
Distributions.logpdf(::FlatReference, x) = 0.0

test_dist(
    "Flat",
    WrenDists.Flat(),
    FlatReference();
    plaice_zero_allocs=(from_vec, from_unconstrained_vec),
)

@testset "Flat: rand is in (-2, 2)" begin
    @test all(x -> -2 < x < 2, [rand(WrenDists.Flat()) for _ in 1:1000])
end

@testset "Flat: sample type" begin
    @test rand(WrenDists.Flat()) isa Float64
    @test rand(WrenDists.Flat{Float32}()) isa Float32
    @test rand(WrenDists.Flat{Float32}(), 3) isa Vector{Float32}
end
