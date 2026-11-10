module WrenEvaluateTests

@info "testing evaluate.jl"

using DifferentiationInterface: AutoForwardDiff
using Distributions
using ForwardDiff
using LinearAlgebra
using Random
using Test
using Plaice
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

@wren function stats_model(observation)
    x ~ Beta(2, 3)
    y ~ MvNormal(zeros(2), I)
    @observe observation ~ Normal(x + y[1])
end

struct TestFixedSpace{T<:VarNamedTuple} <: AbstractParameterSpace
    transforms::T
end
WrenCore.target_transform(space::TestFixedSpace, vn::VarName) = space.transforms[vn]

@wren function prickly_model(should_error)
    should_error[] && error("model was evaluated")
    x ~ Normal()
end

@testset "parameters(params, vecmodel)" begin
    model = stats_model(0.7)

    vecmodel = VecModel(model, AutoForwardDiff(); latent_source=FromPrior(Xoshiro(602)))
    expected = rand(Xoshiro(603), model)
    vector_params = vectorise_latents(FromValues(expected), vecmodel)
    pars = parameters(vector_params, vecmodel)
    @test pars[@vn(x)] ≈ expected[@vn(x)]
    @test pars[@vn(y)] ≈ expected[@vn(y)]

    @test_throws ArgumentError parameters(vector_params[1:(end-1)], vecmodel)
    @test_throws ArgumentError parameters(vcat(vector_params, 0.0), vecmodel)

    should_error = Ref(false)
    fixed_transforms = @vnt begin
        x := FixedTransform(Plaice.from_vec(Normal()))
    end
    fixed_space = TestFixedSpace(fixed_transforms)
    fixed_vecmodel = VecModel(
        prickly_model(should_error),
        nothing;
        latent_source=FromPrior(Xoshiro(604)),
        parameter_space=fixed_space,
    )
    fixed_params = copy(fixed_vecmodel.sample)
    fixed_vecmodel.model.args.should_error[] = true
    @test parameters(fixed_params, fixed_vecmodel)[@vn(x)] ≈ only(fixed_params)
end

end # module WrenEvaluateTests
