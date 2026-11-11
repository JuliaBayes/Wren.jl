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
        result = simulate(model, VarNamedTuple(x=0.5))
        @test parameters(result)[@vn(x)] == 0.5
        @test return_value(result) == 1.5
    end

    @testset "consistency with same rng" begin
        result = simulate(model, Xoshiro(468))
        @test parameters(model, Xoshiro(468)) == parameters(result)
        @test return_value(model, Xoshiro(468)) == return_value(result)
        @test return_value(model, VarNamedTuple(x=0.5)) == 1.5
        @test return_value(model) isa Float64
    end
end

@testset "interface" begin
    model = interface_model(1.0)
    x = 0.5
    vnt = VarNamedTuple(; x)
    expected_lp = logpdf(Normal(), x)
    expected_ll = logpdf(Normal(x), 1.0)

    @testset "with given values: $(typeof(values).name.name)" for values in (
        vnt,
        FromValues(vnt),
        # The fallback is never used because `vnt` has every latent variable.
        FromValues(vnt, FromPrior()),
    )
        result = simulate(model, values)
        @test result isa ModelResult
        @test parameters(result) == vnt
        @test return_value(result) == x + 1
        @test logprior(result) ≈ expected_lp
        @test loglikelihood(result) ≈ expected_ll
        @test logjoint(result) ≈ expected_lp + expected_ll

        @test parameters(model, values) == vnt
        @test return_value(model, values) == x + 1
        @test logprior(model, values) ≈ expected_lp
        @test loglikelihood(model, values) ≈ expected_ll
        @test logjoint(model, values) ≈ expected_lp + expected_ll
    end

    @testset "with an rng" begin
        # Passing an rng is the same as passing `FromPrior(rng)`.
        expected = simulate(model, FromPrior(Xoshiro(468)))
        result = simulate(model, Xoshiro(468))
        @test parameters(result) == parameters(expected)
        @test return_value(result) == return_value(expected)
        @test logprior(result) == logprior(expected)
        @test loglikelihood(result) == loglikelihood(expected)
        @test parameters(model, Xoshiro(468)) == parameters(expected)
        @test return_value(model, Xoshiro(468)) == return_value(expected)
    end

    @testset "with no latent values" begin
        result = simulate(model)
        drawn_x = parameters(result)[@vn(x)]
        @test return_value(result) == drawn_x + 1
        @test logjoint(result) ≈ logpdf(Normal(), drawn_x) + logpdf(Normal(drawn_x), 1.0)
        @test only(keys(parameters(model))) == @vn(x)
        @test return_value(model) isa Float64
    end

    fns = (simulate, parameters, return_value, logprior, loglikelihood, logjoint)
    @testset "missing values are an error: $f" for f in fns
        @test_throws "No latent value provided for variable x" f(model, VarNamedTuple())
    end

    @testset "VecModel" begin
        vecmodel = VecModel(model, AutoForwardDiff())
        # `x ~ Normal()` is already unconstrained, so the vector is just `[x]`.
        v = vector_parameters(vecmodel, vnt)
        @test v == [x]
        @test parameters(vecmodel, v) == vnt
        @test return_value(vecmodel, v) == x + 1
        result = simulate(vecmodel, v)
        @test result isa ModelResult
        @test parameters(result) == vnt
        @test return_value(result) == x + 1
        @test logprior(result) ≈ expected_lp
        @test loglikelihood(result) ≈ expected_ll
        @test lp(vecmodel, v) ≈ expected_lp + expected_ll
        for f in (simulate, parameters, return_value)
            @test_throws ArgumentError f(vecmodel, Float64[])
        end
        @test length(vector_parameters(vecmodel)) == 1
        @test vector_parameters(vecmodel, Xoshiro(468)) ==
              vector_parameters(vecmodel, Xoshiro(468))
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

@testset "parameters(vecmodel, params)" begin
    model = stats_model(0.7)

    vecmodel = VecModel(model, AutoForwardDiff(); latent_source=FromPrior(Xoshiro(602)))
    expected = parameters(model, Xoshiro(603))
    vector_params = vector_parameters(vecmodel, expected)
    pars = parameters(vecmodel, vector_params)
    @test pars[@vn(x)] ≈ expected[@vn(x)]
    @test pars[@vn(y)] ≈ expected[@vn(y)]
    # `x ~ Beta(2, 3)` is constrained, so this checks that `simulate` returns values and
    # log-densities in original space.
    result = simulate(vecmodel, vector_params)
    @test parameters(result) == pars
    @test logprior(result) ≈ logprior(model, expected)
    @test loglikelihood(result) ≈ loglikelihood(model, expected)

    @test_throws ArgumentError parameters(vecmodel, vector_params[1:(end-1)])
    @test_throws ArgumentError parameters(vecmodel, vcat(vector_params, 0.0))

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
    @test parameters(fixed_vecmodel, fixed_params)[@vn(x)] ≈ only(fixed_params)
end

@wren function interval_model()
    s ~ Exponential()
    p ~ Beta(2, 2)
    y ~ MvNormal(zeros(2), I)
end

@testset "FromUnconstrainedInterval" begin
    model = interval_model()

    @testset "constructors" begin
        src = FromUnconstrainedInterval()
        @test (src.lower, src.upper) === (-2.0, 2.0)
        src = FromUnconstrainedInterval(Xoshiro(468))
        @test (src.lower, src.upper) === (-2.0, 2.0)
        src = FromUnconstrainedInterval(0, 1)
        @test (src.lower, src.upper) === (0, 1)
        @test eltype(parameters(interval_model(), src)[@vn(y)]) == Float64
        @test FromUnconstrainedInterval(1.0, 1.0) isa FromUnconstrainedInterval
        @test_throws ArgumentError FromUnconstrainedInterval(1.0, 0.0)
    end

    @testset "values lie in the interval in unconstrained space" begin
        for seed in 1:50
            ps = parameters(model, FromUnconstrainedInterval(Xoshiro(seed), -0.1, 0.1))
            @test -0.1 <= log(ps[@vn(s)]) <= 0.1
            @test -0.1 <= log(ps[@vn(p)] / (1 - ps[@vn(p)])) <= 0.1
            @test all(y -> -0.1 <= y <= 0.1, ps[@vn(y)])
        end
    end

    @testset "reproducible with the same rng" begin
        p1 = parameters(model, FromUnconstrainedInterval(Xoshiro(468)))
        p2 = parameters(model, FromUnconstrainedInterval(Xoshiro(468)))
        @test p1 == p2
    end

    @testset "with VecModel" begin
        src = FromUnconstrainedInterval(Xoshiro(468), -0.5, 0.5)
        vecmodel = VecModel(model, AutoForwardDiff(); latent_source=src)
        v = vector_parameters(vecmodel, FromUnconstrainedInterval(Xoshiro(468), -0.5, 0.5))
        @test length(v) == 4
        @test all(x -> -0.5 <= x <= 0.5, v)
    end
end

end # module WrenEvaluateTests
