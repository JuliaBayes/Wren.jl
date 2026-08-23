module WrenADTests

using DifferentiationInterface: AutoForwardDiff, AutoReverseDiff
using Distributions
using ForwardDiff
using ReverseDiff
using Random
using Test
using Wren

@wren function ad_model()
    x ~ Normal()
    observation = 0.5
    @observe observation ~ Normal(x)
    return x
end

@testset "AD testing utilities" begin
    params = [0.2]

    @testset "test_ad" begin
        result = test_ad(
            ad_model(),
            AutoReverseDiff();
            latent_source=FromPrior(Xoshiro(473)),
            verbose=false,
        )
        @test result isa ModelRunResult
        @test result.lp_actual ≈ result.lp_expected
        @test result.grad_actual ≈ result.grad_expected
        @test result.rand_bench === nothing
        @test result.logjoint_bench === nothing
        @test result.primal_bench === nothing
        @test result.grad_bench === nothing
        @test occursin("correctness : passed", sprint(show, MIME"text/plain"(), result))
    end

    @testset "warns if testing AD backend against itself" begin
        @test_logs (:warn, r"against itself") test_ad(
            ad_model(),
            AutoForwardDiff();
            test_correctness=WithBackend(AutoForwardDiff()),
            verbose=false,
        )
    end

    @testset "benchmark_ad" begin
        result = benchmark_ad(
            ad_model(),
            AutoForwardDiff();
            benchmark_seconds=0.01,
            latent_source=FromPrior(Xoshiro(474)),
            verbose=false,
        )
        @test result.lp_expected === nothing
        @test result.rand_bench.time > 0
        @test result.logjoint_bench.time > 0
        @test result.primal_bench.time > 0
        @test result.grad_bench.time > 0
    end

    @testset "benchmark_and_test" begin
        result = benchmark_and_test(
            ad_model(),
            AutoForwardDiff();
            test_correctness=WithBackend(AutoReverseDiff()),
            benchmark_seconds=0.01,
            latent_source=FromPrior(Xoshiro(476)),
            verbose=false,
        )
        @test result.lp_actual ≈ result.lp_expected
        @test result.grad_actual ≈ result.grad_expected
        @test result.rand_bench.time > 0
        @test result.logjoint_bench.time > 0
        @test result.primal_bench.time > 0
        @test result.grad_bench.time > 0
    end

    @testset "ADIncorrectException" begin
        @test_throws ADIncorrectException test_ad(
            ad_model(),
            AutoForwardDiff();
            test_correctness=WithExpectedResult(0.0, [0.0]),
            latent_source=FromPrior(Xoshiro(475)),
            verbose=false,
        )
    end
end

end # module WrenADTests
