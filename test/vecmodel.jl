module WrenVecModelTests

using Bijectors
using DifferentiationInterface: AutoForwardDiff
using Distributions
using ForwardDiff
using LinearAlgebra
using Random
using Test
using Wren

@model function vector_model(observation)
    x ~ Beta(2, 3)
    y ~ MvNormal(zeros(2), I)
    @observe observation ~ Normal(x + y[1])
    return x + sum(y)
end

@testset "VecModel" begin
    model = vector_model(0.7)
    vecmodel = VecModel(model, AutoForwardDiff(); params=FromPrior(Xoshiro(468)))
    params = rand(Xoshiro(469), vecmodel)

    @test length(params) == vecmodel.dimension == 3
    @test lp(vecmodel, params) isa Real

    value, grad = lp_and_grad(vecmodel, params)
    @test value ≈ lp(vecmodel, params)

    step = 1e-6
    finite_diff_grad = map(eachindex(params)) do i
        delta = zeros(length(params))
        delta[i] = step
        (lp(vecmodel, params + delta) - lp(vecmodel, params - delta)) / (2 * step)
    end
    @test grad ≈ finite_diff_grad rtol = 1e-5

    grad_buffer = similar(params)
    inplace_value, inplace_grad = lp_and_grad!(grad_buffer, vecmodel, params)
    @test inplace_value ≈ value
    @test inplace_grad === grad_buffer
    @test grad_buffer ≈ grad

    _, value_accs = evaluate(
        model,
        FromPrior(Xoshiro(470)),
        UnconstrainedSpace(),
        Wren.AccumulatorTuple(VectorValueAccumulator()),
    )
    vector_values = get_vector_values(value_accs)
    from_vnt = VecModel(model, AutoForwardDiff(); params=vector_values)
    vector_params = internal_values_as_vector(vector_values)
    @test from_vnt.dimension == 3
    @test lp(from_vnt, vector_params) isa Real

    prior_vecmodel = VecModel(
        model,
        AutoForwardDiff();
        params=FromPrior(Xoshiro(471)),
        getlogp=getlogprior,
        parameter_space=OriginalSpace(),
    )
    prior_params = rand(Xoshiro(472), prior_vecmodel)
    @test 0 < prior_params[1] < 1
    @test lp(prior_vecmodel, prior_params) ≈
          logpdf(Beta(2, 3), prior_params[1]) +
          logpdf(MvNormal(zeros(2), I), prior_params[2:3])
end

end # module WrenVecModelTests
