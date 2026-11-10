module WrenDrawTests

@info "testing draw.jl"

using DifferentiationInterface: AutoForwardDiff
using Distributions
using ForwardDiff
using LinearAlgebra
using Random
using Test
using Plaice
using WrenCore

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

@testset "Draw" begin
    model = stats_model(0.7)

    from_prior = Draw(FromPrior(Xoshiro(601)), model, (iteration=4,))
    pars = parameters(from_prior)
    @test Set(keys(pars)) == Set((@vn(x), @vn(y)))
    @test extras(from_prior).iteration == 4
    @test logprior(from_prior) ≈
          logpdf(Beta(2, 3), pars[@vn(x)]) + logpdf(MvNormal(zeros(2), I), pars[@vn(y)])
    @test loglikelihood(from_prior) ≈ logpdf(Normal(pars[@vn(x)] + pars[@vn(y)][1]), 0.7)
    @test logjoint(from_prior) ≈ logprior(from_prior) + loglikelihood(from_prior)

    accs = WrenCore.AccumulatorTuple(
        WrenCore.LogPriorAccumulator(),
        WrenCore.LogLikelihoodAccumulator(),
        WrenCore.ParamsAccumulator(),
    )
    _, accs =
        evaluate(model, FromValues(from_prior), OriginalSpace(), accs, EvalContext{false}())
    from_accs = Draw(accs, (chain=2,))
    @test parameters(from_accs) == parameters(from_prior)
    @test extras(from_accs).chain == 2
    @test logjoint(from_accs) ≈ logjoint(from_prior)

    vecmodel = VecModel(model, AutoForwardDiff(); latent_source=FromPrior(Xoshiro(602)))
    vector_params = rand(Xoshiro(603), vecmodel)
    from_vecmodel = Draw(vector_params, vecmodel)
    @test logjoint(from_vecmodel) ≈ logprior(from_vecmodel) + loglikelihood(from_vecmodel)
    @test parameters(vector_params, vecmodel) == parameters(from_vecmodel)

    @test_throws ArgumentError Draw(vector_params[1:(end-1)], vecmodel)
    @test_throws ArgumentError Draw(vcat(vector_params, 0.0), vecmodel)
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
    @test_throws ErrorException Draw(fixed_params, fixed_vecmodel)
end

end # module WrenDrawTests
