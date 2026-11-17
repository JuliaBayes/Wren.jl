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

@testset "simulate / return_value / parameters" begin
    model = interface_model(1.0)

    @testset "simulate" begin
        for result in
            (simulate(model), simulate(model, Xoshiro(468)), simulate(model, FromPrior()))
            x = parameters(result)[@vn(x)]
            @test return_value(result) == x + 1
            @test logprior(result) ≈ logpdf(Normal(), x)
            @test loglikelihood(result) ≈ logpdf(Normal(x), 1.0)
            @test logjoint(result) ≈ logprior(result) + loglikelihood(result)
        end
    end

    @testset "simulate with given values" begin
        result = simulate(model, FromValues(VarNamedTuple(x=0.5)))
        @test parameters(result)[@vn(x)] == 0.5
        @test return_value(result) == 1.5
    end

    @testset "consistency with same rng" begin
        result = simulate(model, Xoshiro(468))
        @test parameters(model, Xoshiro(468)) == parameters(result)
        @test return_value(model, Xoshiro(468)) == return_value(result)
        @test return_value(model, FromValues(VarNamedTuple(x=0.5))) == 1.5
        @test return_value(model) isa Float64
    end

    @testset "parameter round trip" begin
        result = simulate(model, Xoshiro(468))
        result2 = simulate(model, parameters(result))
        @test parameters(result2) == parameters(result)
        @test return_value(result2) == return_value(result)
        @test logprior(result2) == logprior(result)
        @test loglikelihood(result2) == loglikelihood(result)
        @test logjoint(result2) == logjoint(result)
    end

    @testset "densify!!" begin
        result = simulate(densify_model(zeros(2)), Xoshiro(468))
        xs = [parameters(result)[@vn(x[1])], parameters(result)[@vn(x[2])]]
        # Before densifying, `x` is stored as a PartialArray, so it does not compare equal
        # to a VarNamedTuple holding a plain Vector.
        @test parameters(result) != VarNamedTuple(; x=xs)
        dense = WrenCore.VarNames.densify!!(parameters(result))
        @test haskey(dense, @vn(x))
        @test dense == VarNamedTuple(; x=xs)
    end
end

end
