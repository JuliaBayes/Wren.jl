module WrenADTests

using DifferentiationInterface: AutoForwardDiff
using Distributions
using ForwardDiff
using Random
using Test
using Wren

@model function ad_model()
    x ~ Normal()
    observation = 0.5
    @observe observation ~ Normal(x)
    return x
end

@testset "AD testing utilities" begin
    params = [0.2]
    result = run_ad(ad_model(), AutoForwardDiff(); params, rng=Xoshiro(473), verbose=false)
    @test result.value_actual ≈ result.value_expected
    @test result.grad_actual ≈ result.grad_expected
    @test occursin("correctness : passed", sprint(show, MIME"text/plain"(), result))

    benchmark_result = run_ad(
        ad_model(),
        AutoForwardDiff();
        test=false,
        benchmark=true,
        benchmark_seconds=0.01,
        params,
        rng=Xoshiro(474),
        verbose=false,
    )
    @test benchmark_result.primal_time > 0
    @test benchmark_result.grad_time > 0

    @test_throws ADIncorrectException run_ad(
        ad_model(),
        AutoForwardDiff();
        test=WithExpectedResult(0.0, [0.0]),
        params,
        rng=Xoshiro(475),
        verbose=false,
    )
end

end # module WrenADTests
