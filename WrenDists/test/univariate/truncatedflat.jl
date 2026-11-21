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

# Distributions doesn't have a truncated flat distribution, so this stands in as the
# reference. Its samples are just points to check the log-density at, and deliberately
# include points outside the bounds.
struct TruncatedFlatReference
    lower::Float64
    upper::Float64
end
Base.rand(rng::AbstractRNG, ::TruncatedFlatReference) = 10 * randn(rng)
Distributions.logpdf(d::TruncatedFlatReference, x) = d.lower <= x <= d.upper ? 0.0 : -Inf

for (lower, upper) in ((0.0, Inf), (-Inf, 5.0), (-1.0, 3.0))
    test_dist(
        "TruncatedFlat($lower, $upper)",
        WrenDists.TruncatedFlat(lower, upper),
        TruncatedFlatReference(lower, upper);
        plaice_zero_allocs=(from_vec, from_unconstrained_vec),
    )
end

@testset "TruncatedFlat: sample type" begin
    @test rand(WrenDists.TruncatedFlat(0, Inf)) isa Float64
    @test rand(WrenDists.TruncatedFlat(0.0f0, Inf32)) isa Float32
end

@testset "TruncatedFlat: empty interval" begin
    # A dependent bound that rounds to `lower == upper` rejects every point instead of
    # throwing.
    @test logdensityof(WrenDists.TruncatedFlat(1.0, 1.0), 1.0) == -Inf
    @test logdensityof(WrenDists.TruncatedFlat(2.0, 1.0), 1.5) == -Inf
end

if RUN_REACTANT
    @testset "TruncatedFlat: traced, parameter-dependent bound" begin
        density(alpha, x) = logdensityof(WrenDists.TruncatedFlat(0.0, 1 - alpha), x)
        for (alpha, x) in ((0.3, 0.5), (0.3, 0.9), (1.2, 0.1))
            alpha_ra = Reactant.to_rarray(alpha; track_numbers=true)
            x_ra = Reactant.to_rarray(x; track_numbers=true)
            @test Float64(Reactant.@jit density(alpha_ra, x_ra)) == density(alpha, x)
        end
    end
end
