module WrenInterfaceTests

@info "testing interface.jl"

using Distributions
using Random
using Test
using WrenCore

@wren function interface_model(y)
    x ~ Normal()
    @observe y ~ Normal(x)
    return x + 1
end

@wren function densify_model(x)
    for i in eachindex(x)
        x[i] ~ Normal()
    end
end

@testset "simulate / return_value / rand" begin
    model = interface_model(1.0)

    @testset "simulate" begin
        for result in
            (simulate(model), simulate(Xoshiro(468), model), simulate(FromPrior(), model))
            x = parameters(result)[@vn(x)]
            @test return_value(result) == x + 1
            @test logprior(result) ≈ logpdf(Normal(), x)
            @test loglikelihood(result) ≈ logpdf(Normal(x), 1.0)
            @test logjoint(result) ≈ logprior(result) + loglikelihood(result)
        end
    end

    @testset "simulate with given values" begin
        result = simulate(FromValues(VarNamedTuple(x=0.5)), model)
        @test parameters(result)[@vn(x)] == 0.5
        @test return_value(result) == 1.5
    end

    @testset "consistency with same rng" begin
        result = simulate(Xoshiro(468), model)
        @test rand(Xoshiro(468), model) == parameters(result)
        @test return_value(Xoshiro(468), model) == return_value(result)
        @test return_value(FromValues(VarNamedTuple(x=0.5)), model) == 1.5
        @test return_value(model) isa Float64
    end

    @testset "Draw round trip" begin
        result = simulate(Xoshiro(468), model)
        draw = Draw(result)
        @test parameters(draw) == parameters(result)
        @test logprior(draw) == logprior(result)
        @test loglikelihood(draw) == loglikelihood(result)
        @test logjoint(draw) == logjoint(result)
        @test extras(draw).return_value == return_value(result)

        result2 = WrenCore.ModelResult(draw)
        @test parameters(result2) == parameters(result)
        @test return_value(result2) == return_value(result)
        @test logprior(result2) == logprior(result)
        @test loglikelihood(result2) == loglikelihood(result)
    end

    @testset "densify!!" begin
        result = simulate(Xoshiro(468), densify_model(zeros(2)))
        xs = [parameters(result)[@vn(x[1])], parameters(result)[@vn(x[2])]]
        # Before densifying, `x` is stored as a PartialArray, so it does not compare equal
        # to a VarNamedTuple holding a plain Vector.
        @test parameters(result) != VarNamedTuple(; x=xs)
        dense = WrenCore.VarNames.densify!!(result)
        @test haskey(parameters(dense), @vn(x))
        @test parameters(dense) == VarNamedTuple(; x=xs)
        @test return_value(dense) == return_value(result)
        @test logprior(dense) == logprior(result)
        @test loglikelihood(dense) == loglikelihood(result)
    end
end

end
