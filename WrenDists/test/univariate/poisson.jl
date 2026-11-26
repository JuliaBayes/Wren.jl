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
using Random: Xoshiro

const RUN_REACTANT = !Sys.iswindows()

if RUN_REACTANT
    using Pkg
    Pkg.add("Reactant")
    using Reactant
end

test_dist(
    "Poisson",
    WrenDists.Poisson(7.0),
    Distributions.Poisson(7.0);
    plaice_zero_allocs=(from_vec, from_unconstrained_vec),
    other_ad_testcases=(let f(x) = logdensityof(WrenDists.Poisson(x[1]), 7)
        ADTestCase(f, [7.0], (), ())
    end,),
)

test_dist(
    "PoissonLog",
    WrenDists.PoissonLog(log(7.0)),
    Distributions.Poisson(7.0);
    plaice_zero_allocs=(from_vec, from_unconstrained_vec),
    other_ad_testcases=(let f(x) = logdensityof(WrenDists.PoissonLog(x[1]), 7)
        ADTestCase(f, [log(7.0)], (), ())
    end,),
)

test_dist(
    "BatchedPoisson",
    WrenDists.BatchedPoisson([1.0, 3.0, 7.0]),
    Distributions.product_distribution(Distributions.Poisson.([1.0, 3.0, 7.0]));
    other_ad_testcases=(
        let
            f(x) = logdensityof(WrenDists.BatchedPoisson(x), [1.0, 2.0, 8.0])
            ADTestCase(f, [1.0, 3.0, 7.0], (), ())
        end,
        let
            f(x) = logdensityof(WrenDists.BatchedPoisson(x), [0.0, 0.0])
            ADTestCase(f, [0.0, 1.0], (), ())
        end,
    ),
)

test_dist(
    "BatchedPoissonLog",
    WrenDists.BatchedPoissonLog(log.([1.0, 3.0, 7.0])),
    Distributions.product_distribution(Distributions.Poisson.([1.0, 3.0, 7.0]));
    other_ad_testcases=(
        let
            f(x) = logdensityof(WrenDists.BatchedPoissonLog(x), [1.0, 2.0, 8.0])
            ADTestCase(f, log.([1.0, 3.0, 7.0]), (), ())
        end,
        let
            f(x) = logdensityof(WrenDists.BatchedPoissonLog(x), [0.0, 0.0])
            ADTestCase(f, [-Inf, 0.0], (), ())
        end,
    ),
)

@testset "Poisson sampling beyond maxintfloat" begin
    # Counts this large cannot step by one in their float type, which used to hang `rand`.
    for λ in (1.0f8, 1.0f12, 1.0e17)
        d = WrenDists.Poisson(λ)
        xs = [rand(Xoshiro(i), d) for i in 1:1_000]
        @test eltype(xs) == typeof(λ)
        @test all(x -> isinteger(x) && x >= 0, xs)
        @test isapprox(mean(xs), λ; rtol=1e-3)
    end
end

@testset "batched Poisson support (issue #158)" begin
    for (scalar, batched, rates) in (
        (WrenDists.Poisson, WrenDists.BatchedPoisson, [0.0, 1.0]),
        (WrenDists.PoissonLog, WrenDists.BatchedPoissonLog, [-Inf, 0.0]),
    )
        d = batched(rates)
        points = [[0.0, 0.0], [1.0, 0.0], [0.0, 0.5], [0.0, -2.0], [0.0, Inf]]
        for x in points
            @test logdensityof(d, x) == sum(logdensityof.(scalar.(rates), x))
        end
        RUN_REACTANT && WrenDists.test_dist_reactant(d, points)
    end
end

@testset "PoissonLog accuracy vs analytical BigFloat" begin
    ref(logλ, k) = Float64(k * big(logλ) - exp(big(logλ)) - WrenDists.loggamma(big(k) + 1))
    # Note: above 1e20 it starts to drift anyway. See #319.
    for k in (1, 2, 15, 16, 17, 100, 1e4, 1e10, 1e15, 1e20), logλ in log(k) .+ (-1, 0, 1)
        @test logdensityof(WrenDists.PoissonLog(logλ), k) ≈ ref(logλ, k) rtol = 1e-8
    end
end

@testset "PoissonLog edge cases" begin
    for (logλ, k, expected) in (
        (log(7.0), 2.5, -Inf),
        (log(7.0), -1, -Inf),
        (-Inf, 0, 0.0),
        (-Inf, 3, -Inf),
        # The true value (about -7e308) is too large for Float64.
        (0.0, 1e306, -Inf),
    )
        @test logdensityof(WrenDists.PoissonLog(logλ), k) == expected
    end
    # `log(k!)` overflows here, but the log-mass does not.
    @test isfinite(logdensityof(WrenDists.PoissonLog(log(1e306)), 1e306))
end

@testset "PoissonLog Enzyme gradient when exp(logλ) overflows" begin
    # The gradient wrt logλ is `k - exp(logλ)`. Here `exp(logλ)` overflows but the gradient
    # itself does not.
    for (logλ, k) in ((710.0, 1e308), (88.8f0, 3.0f38))
        f(x) = logdensityof(WrenDists.PoissonLog(x), k)
        grad = Enzyme.autodiff(Reverse, f, Active, Active(logλ))[1][1]
        expected = oftype(logλ, big(k) - exp(big(logλ)))
        @test isfinite(grad)
        @test grad ≈ expected rtol = 1e-4
    end
end
