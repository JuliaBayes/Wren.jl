module WrenDrawTests

using DifferentiationInterface: AutoForwardDiff
using Distributions
using ForwardDiff
using LinearAlgebra
using Random
using Test
using Plaice
using Wren

@wren function stats_model(observation)
    x ~ Beta(2, 3)
    y ~ MvNormal(zeros(2), I)
    @observe observation ~ Normal(x + y[1])
end

struct TestFixedSpace{T<:VarNamedTuple} <: AbstractParameterSpace
    transforms::T
end
Wren.target_transform(space::TestFixedSpace, vn::VarName) = space.transforms[vn]

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

    without_log_probs =
        Draw(FromParams(parameters(from_prior)), model; include_log_probs=false)
    @test parameters(without_log_probs) == parameters(from_prior)
    @test isempty(extras(without_log_probs))

    accs = Wren.AccumulatorTuple(
        Wren.LogPriorAccumulator(),
        Wren.LogLikelihoodAccumulator(),
        Wren.ParamsAccumulator(),
    )
    _, accs = evaluate(model, FromParams(from_prior), OriginalSpace(), accs)
    from_accs = Draw(accs, (chain=2,))
    @test parameters(from_accs) == parameters(from_prior)
    @test extras(from_accs).chain == 2
    @test logjoint(from_accs) ≈ logjoint(from_prior)

    vecmodel = VecModel(model, AutoForwardDiff(); params=FromPrior(Xoshiro(602)))
    vector_params = rand(Xoshiro(603), vecmodel)
    from_vecmodel = Draw(vector_params, vecmodel)
    @test logjoint(from_vecmodel) ≈ logprior(from_vecmodel) + loglikelihood(from_vecmodel)
    @test parameters(Draw(vector_params, vecmodel; include_log_probs=false)) ==
          parameters(from_vecmodel)

    @test_throws ArgumentError Draw(vector_params[1:(end-1)], vecmodel)
    @test_throws ArgumentError Draw(vcat(vector_params, 0.0), vecmodel)

    should_error = Ref(false)
    fixed_transforms = @vnt begin
        x := FixedTransform(Plaice.from_vec(Normal()))
    end
    fixed_space = TestFixedSpace(fixed_transforms)
    fixed_vecmodel = VecModel(
        prickly_model(should_error),
        nothing;
        params=FromPrior(Xoshiro(604)),
        parameter_space=fixed_space,
    )
    fixed_params = copy(fixed_vecmodel.sample)
    fixed_vecmodel.model.args.should_error[] = true
    fixed_draw = Draw(fixed_params, fixed_vecmodel; include_log_probs=false)
    @test parameters(fixed_draw)[@vn(x)] ≈ only(fixed_params)
    @test_throws ErrorException Draw(fixed_params, fixed_vecmodel)
end

end # module WrenDrawTests
