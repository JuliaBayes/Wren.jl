module WrenParamsWithStatsTests

using DifferentiationInterface: AutoForwardDiff
using Distributions
using ForwardDiff
using LinearAlgebra
using Random
using Test
using VectorBijectors
using Wren

@model function stats_model(observation)
    x ~ Beta(2, 3)
    y ~ MvNormal(zeros(2), I)
    @observe observation ~ Normal(x + y[1])
end

struct TestFixedSpace{T<:VarNamedTuple} <: AbstractParameterSpace
    transforms::T
end
Wren.target_transform(space::TestFixedSpace, vn::VarName) = space.transforms[vn]

@model function prickly_model(should_error)
    should_error[] && error("model was evaluated")
    x ~ Normal()
end

@testset "ParamsWithStats" begin
    model = stats_model(0.7)

    from_prior = ParamsWithStats(FromPrior(Xoshiro(601)), model, (iteration=4,))
    @test Set(keys(from_prior.params)) == Set((@vn(x), @vn(y)))
    @test from_prior.stats.iteration == 4
    @test from_prior.stats.logprior ≈
          logpdf(Beta(2, 3), from_prior.params[@vn(x)]) +
          logpdf(MvNormal(zeros(2), I), from_prior.params[@vn(y)])
    @test from_prior.stats.loglikelihood ≈
          logpdf(Normal(from_prior.params[@vn(x)] + from_prior.params[@vn(y)][1]), 0.7)
    @test from_prior.stats.logjoint ≈
          from_prior.stats.logprior + from_prior.stats.loglikelihood

    without_log_probs =
        ParamsWithStats(FromParams(from_prior.params), model; include_log_probs=false)
    @test without_log_probs.params == from_prior.params
    @test isempty(without_log_probs.stats)

    accs = Wren.AccumulatorTuple(
        Wren.LogPriorAccumulator(),
        Wren.LogLikelihoodAccumulator(),
        Wren.RawValueAccumulator(),
    )
    _, accs = evaluate(model, FromParams(from_prior), OriginalSpace(), accs)
    from_accs = ParamsWithStats(accs, (chain=2,))
    @test from_accs.params == from_prior.params
    @test from_accs.stats.chain == 2
    @test from_accs.stats.logjoint ≈ from_prior.stats.logjoint

    vecmodel = VecModel(model, AutoForwardDiff(); params=FromPrior(Xoshiro(602)))
    vector_params = rand(Xoshiro(603), vecmodel)
    from_vecmodel = ParamsWithStats(vector_params, vecmodel)
    @test from_vecmodel.stats.logjoint ≈
          from_vecmodel.stats.logprior + from_vecmodel.stats.loglikelihood
    @test ParamsWithStats(vector_params, vecmodel; include_log_probs=false).params ==
          from_vecmodel.params

    @test_throws ArgumentError ParamsWithStats(vector_params[1:(end-1)], vecmodel)
    @test_throws ArgumentError ParamsWithStats(vcat(vector_params, 0.0), vecmodel)

    should_error = Ref(false)
    fixed_transforms = @vnt begin
        x := FixedTransform(VectorBijectors.from_vec(Normal()))
    end
    fixed_space = TestFixedSpace(fixed_transforms)
    fixed_vecmodel = VecModel(
        prickly_model(should_error),
        nothing;
        params=FromPrior(Xoshiro(604)),
        parameter_space=fixed_space,
    )
    fixed_params = copy(fixed_vecmodel.sample)
    should_error[] = true
    fixed_pws = ParamsWithStats(fixed_params, fixed_vecmodel; include_log_probs=false)
    @test fixed_pws.params[@vn(x)] ≈ only(fixed_params)
    @test_throws ErrorException ParamsWithStats(fixed_params, fixed_vecmodel)

    @test ParamsWithStats(from_prior.params, from_prior.stats) == from_prior
    @test isequal(ParamsWithStats(from_prior.params, from_prior.stats), from_prior)
    @test occursin("ParamsWithStats", sprint(show, MIME"text/plain"(), from_prior))
end

end # module WrenParamsWithStatsTests
