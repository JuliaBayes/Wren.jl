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

@testset "MvNormalScale shape checks" begin
    @testset "scale factor must match mean" begin
        @test_throws "scale factor must have size (3, 3), got (2, 2)" WrenDists.MvNormalScale(
            zeros(3),
            [1.0 0.0; 0.5 1.0],
        )
        @test_throws "scale factor must have size (2, 2), got (2, 3)" WrenDists.MvNormalScale(
            zeros(2),
            ones(2, 3),
        )
        @test_throws "scale factor must have 3 diagonal entries, got 2" WrenDists.MvNormalScale(
            zeros(3),
            Diagonal(ones(2)),
        )
    end

    @testset "logdensityof rejects $name with wrong length" for (name, d) in (
        ("full rank", WrenDists.MvNormalScale(zeros(3), Matrix(1.0I, 3, 3))),
        ("diagonal", WrenDists.MvNormalScale(zeros(3), Diagonal(ones(3)))),
        ("isotropic", WrenDists.MvNormalScale(zeros(3), I)),
    )
        @test_throws DimensionMismatch logdensityof(d, zeros(4))
        @test_throws "expected size (3,), got (4,)" logdensityof(d, zeros(4))
        @test_throws "expected size (3,), got (2,)" logdensityof(d, zeros(2))
    end

    @testset "zero-dimensional $name has log density 0" for (name, L) in (
        ("full rank", zeros(0, 0)),
        ("diagonal", Diagonal(Float64[])),
        ("isotropic", I),
    )
        @test logdensityof(WrenDists.MvNormalScale(Float64[], L), Float64[]) == 0.0
    end
end
