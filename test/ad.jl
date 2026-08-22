module WrenADTests

using DifferentiationInterface: AutoForwardDiff
using Distributions
using ForwardDiff
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
        result =
            test_ad(ad_model(), AutoForwardDiff(); latent_source=FromPrior(Xoshiro(473)), verbose=false)
        @test result isa ModelRunResult
        @test result.value_actual ≈ result.value_expected
        @test result.grad_actual ≈ result.grad_expected
        @test result.rand_bench === nothing
        @test result.logjoint_bench === nothing
        @test result.primal_bench === nothing
        @test result.grad_bench === nothing
        @test occursin("correctness : passed", sprint(show, MIME"text/plain"(), result))
    end

    @testset "benchmark_ad" begin
        result = benchmark_ad(
            ad_model(),
            AutoForwardDiff();
            benchmark_seconds=0.01,
            latent_source=FromPrior(Xoshiro(474)),
            verbose=false,
        )
        @test result.value_expected === nothing
        @test result.rand_bench.time > 0
        @test result.logjoint_bench.time > 0
        @test result.primal_bench.time > 0
        @test result.grad_bench.time > 0
    end

    @testset "benchmark_and_test" begin
        result = benchmark_and_test(
            ad_model(),
            AutoForwardDiff();
            benchmark_seconds=0.01,
            latent_source=FromPrior(Xoshiro(476)),
            verbose=false,
        )
        @test result.value_actual ≈ result.value_expected
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
            test_ad=WithExpectedResult(0.0, [0.0]),
            latent_source=FromPrior(Xoshiro(475)),
            verbose=false,
        )
    end
end

end # module WrenADTests
