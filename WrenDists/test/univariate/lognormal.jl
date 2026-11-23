using WrenDists
using WrenDists: logdensityof, test_dist, ADTestCase
import DifferentiationInterface as DI
using Distributions
using ForwardDiff
using ReverseDiff
using Mooncake
using Enzyme
using Plaice
using Random: Xoshiro
using Test
using FillArrays: Fill

const RUN_REACTANT = !Sys.iswindows()

if RUN_REACTANT
    using Pkg
    Pkg.add("Reactant")
    using Reactant
end

sample_lognormal(s, rng) = rand(rng, WrenDists.BatchedLogNormal(zero(s), s))
sum_lognormal(s, rng=Xoshiro(8675309)) = sum(sample_lognormal(s, rng))

@testset "LogNormal support" begin
    d = WrenDists.LogNormal(0.0f0, 1.0f0)
    batch = WrenDists.BatchedLogNormal(Float32[0], Float32[1])
    points = [Float32[0], Float32[-1]]
    for x in points
        @test logdensityof(d, only(x)) == -Inf
        @test logdensityof(batch, x) == -Inf
    end
    f(p, x) = logdensityof(WrenDists.BatchedLogNormal(p, p), x)
    @test first(Enzyme.gradient(Enzyme.Reverse, f, Float32[1], Enzyme.Const(points[2]))) ==
          [0]
    @static if RUN_REACTANT
        WrenDists.test_dist_reactant(d, Float32[0, -1])
        WrenDists.test_dist_reactant(batch, points)
        p, x = Reactant.to_rarray.((Float32[1], points[2]))
        grad(p, x) = Enzyme.gradient(Enzyme.Reverse, f, p, Enzyme.Const(x))[1]
        compiled = Reactant.@compile grad(p, x)
        @test Array(compiled(p, x)) == [0]
    end
end

test_dist(
    "LogNormal",
    WrenDists.LogNormal(0.5, 0.3),
    Distributions.LogNormal(0.5, 0.3);
    plaice_zero_allocs=(from_vec, from_unconstrained_vec),
    other_ad_testcases=(let f(x) = logdensityof(WrenDists.LogNormal(x[1], x[2]), 1.5)
        ADTestCase(f, [0.5, 0.3], (), ())
    end,),
)

test_dist(
    "BatchedLogNormal",
    WrenDists.BatchedLogNormal([0.1, 0.5, 1.0], [0.3, 0.7, 1.2]),
    Distributions.product_distribution(
        Distributions.LogNormal.([0.1, 0.5, 1.0], [0.3, 0.7, 1.2]),
    );
    other_ad_testcases=(
        let
            f(x) =
                logdensityof(WrenDists.BatchedLogNormal(x[1:3], x[4:6]), [1.5, 1.5, 1.5])
            ADTestCase(f, [0.1, 0.5, 1.0, 0.3, 0.7, 1.2], (), ())
        end,
        ADTestCase(sum_lognormal, fill(0.7, 8), (), ()),
    ),
)

@testset "BatchedLogNormal sampling: $T" for T in (Float32, Float64, BigFloat)
    sigma = fill(T(0.7), 8)
    noise = randn(Xoshiro(8675309), T, size(sigma))
    expected = exp.(noise .* sigma)
    @test sample_lognormal(sigma, Xoshiro(8675309)) == expected
    @test ForwardDiff.gradient(sum_lognormal, sigma) ≈ expected .* noise
    @test ReverseDiff.gradient(sum_lognormal, sigma) ≈ expected .* noise
end

@testset "BatchedLogNormal compiled sampling" begin
    @static if RUN_REACTANT
        sample_and_gradient(s, rng) = (
            sample_lognormal(s, copy(rng)),
            first(Enzyme.gradient(Reverse, sum_lognormal, s, Const(copy(rng)))),
        )
        sigma = Reactant.to_rarray(fill(0.7f0, 8))
        rng = Reactant.ReactantRNG(Reactant.to_rarray(UInt64[8675309, 0]))
        compiled = Reactant.@compile sample_and_gradient(sigma, rng)
        sample, gradient = compiled(sigma, rng)
        x = Array(sample)
        @test Array(gradient) ≈ x .* log.(x) ./ 0.7f0
    end
end

@test logdensityof(WrenDists.BatchedLogNormal(Float64[], Fill(-1.0, 0)), Float64[]) == 0.0

test_dist(
    "BatchedLogNormal shared scale",
    WrenDists.BatchedLogNormal([0.1, 0.5, 1.0], Fill(0.7, 3)),
    Distributions.product_distribution(Distributions.LogNormal.([0.1, 0.5, 1.0], 0.7));
    other_ad_testcases=(
        let
            f(x) = logdensityof(
                WrenDists.BatchedLogNormal(x[1:3], Fill(x[4], 3)),
                [1.5, 2.0, 3.0],
            )
            ADTestCase(f, [0.1, 0.5, 1.0, 0.7], (), ())
        end,
    ),
)

@testset "BatchedLogNormal Enzyme rule" begin
    using EnzymeTestUtils
    mu = [0.1, 0.5, 1.0]
    sigma = [0.3, 0.7, 1.2]
    x = [1.5, 2.0, 3.0]
    @testset "strided views" begin
        f(v) = logdensityof(
            WrenDists.BatchedLogNormal(view(v, 1:2:5), view(v, 7:2:11)),
            view(v, 13:2:17),
        )
        v = repeat(vcat(mu, sigma, x); inner=2)
        g = ForwardDiff.gradient(f, v)
        @test only(Enzyme.gradient(Reverse, f, v)) ≈ g
        @test only(Enzyme.autodiff(Forward, f, Duplicated(v, ones(length(v))))) ≈ sum(g)
    end
    @testset "reverse" begin
        @testset for Tmu in (Const, Duplicated),
            Tsigma in (Const, Duplicated),
            Tx in (Const, Duplicated)

            test_reverse(
                WrenDists._lognormal_loop,
                Active,
                (mu, Tmu),
                (sigma, Tsigma),
                (x, Tx),
            )
        end
    end
    @testset "forward" begin
        @testset for RT in (Const, DuplicatedNoNeed, Duplicated),
            Tmu in (Const, Duplicated),
            Tsigma in (Const, Duplicated),
            Tx in (Const, Duplicated)

            test_forward(WrenDists._lognormal_loop, RT, (mu, Tmu), (sigma, Tsigma), (x, Tx))
        end
    end
end
