module WrenBatchEvaluationTests

using WrenCore, WrenDists, Reactant, Test
using ADTypes: AutoReactant
using Enzyme: Enzyme
using Plaice

vecmodel(model, values; adtype=nothing) =
    VecModel(model, adtype; latent_source=FromValues(values))

function guarded_batch(container, points, enabled)
    Reactant.@trace if sum(enabled) > 0
        result = container.batch(view(points, :, :))
    else
        result = container.batch(-points)
    end
    return result
end

struct CounterAccumulator{V} <: AbstractAccumulator
    values::V
end
WrenCore.accumulator_name(::Type{<:CounterAccumulator}) = :counter
Base.copy(acc::CounterAccumulator) = CounterAccumulator(copy(acc.values))
function WrenCore.accumulate_assume!!(acc::CounterAccumulator, args...)
    acc.values[1] += 1
    return acc
end
WrenCore.accumulate_observe!!(acc::CounterAccumulator, args...) = acc

@wren function rate_model(count)
    rate ~ Exponential()
    @observe count ~ Poisson(rate)
end

const offset = 0.3
shift(x) = x + offset
@wren function branch_model(y)
    x ~ Normal()
    mu = if x > 0
        shift(x)
    else
        2x - 1
    end
    for i in eachindex(y)
        if x > 0
            @observe y[i] ~ Normal(mu)
        end
    end
    n = length(y)
    @observe y[n] ~ Normal(mu, 2.0)
end

@wren function bounded_model(count)
    upper ~ Exponential()
    rate ~ Uniform(0.0, upper)
    @observe count ~ Poisson(rate)
end

@wren function fallback_model(y)
    x ~ Normal()
    @batch for i in eachindex(y)
        @observe y[i] ~ Normal(x + i)
    end
end

@wren function bounded_array_model(count)
    upper ~ Exponential()
    rate ~ iid(Uniform(0.0, upper), 2)
    @observe count ~ Poisson(sum(rate))
end

function transform_columns(X)
    return mapslices(X; dims=1) do x
        Reactant.@allowscalar vcat(
            Plaice.with_logabsdet_jacobian(Plaice.Truncate(x[1], x[2]), x[3])...,
        )
    end
end

@wren function mutating_model(buffer, count)
    rate ~ Exponential()
    buffer[1] += rate
    @observe count ~ Poisson(buffer[1])
end

@wren function nested_model(child)
    a ~ child
end

@wren function control_model(y, flag)
    x ~ Normal()
    if flag
        return x
    end
    if x > 0
        if x > 1
            @observe y ~ Normal(x)
        end
    end
end

@wren function array_branch(y)
    beta ~ iid(Normal(), 2)
    if beta[1] > 0
        @observe y ~ Normal(sum(beta))
    else
        @observe y ~ Normal(2beta[1] - beta[2])
    end
end

@testset "prepared parameter batches" begin
    X = reshape(log.([0.5, 1.5, 3.0, 5.0]), 1, :)
    rates = exp.(vec(X))
    rate_values = VarNamedTuple(rate=1.5)
    normal_values = VarNamedTuple(x=1.0)
    adtype = AutoReactant(AutoEnzyme(; mode=Reverse))
    reverse = AutoEnzyme(; mode=Reverse)
    forward = AutoEnzyme(; mode=Enzyme.Forward)
    vm = vecmodel(rate_model(3), rate_values)
    batch = prepare_batch(vm, X)
    expected = vcat(reshape(-rates, 1, :), reshape(3vec(X) - rates .- log(6.0), 1, :), -X)
    @test Array(batch(X)) ≈ expected
    joint = prepare_batch(vm, X, logjoint_transformed; adtype=reverse)
    values, gradients = joint(X)
    @test vec(Array(values)) ≈ vec(expected[1, :] + expected[2, :] - expected[3, :])
    @test vec(Array(gradients)) ≈ 4 .- 2rates
    resident = Reactant.to_rarray(X[:, [4, 2, 1, 3]])
    @test Array(batch(resident)) ≈ expected[:, [4, 2, 1, 3]]
    @test Array(resident) == X[:, [4, 2, 1, 3]]
    @test vec(Array(last(joint(resident)))) ≈ (4 .- 2rates)[[4, 2, 1, 3]]
    container, enabled = (; batch), Reactant.to_rarray([1])
    compiled = Reactant.@compile guarded_batch(container, resident, enabled)
    @test Array(compiled(container, resident, enabled)) ≈ expected[:, [4, 2, 1, 3]]
    @test Array(compiled(container, Reactant.to_rarray(X), enabled)) ≈ expected
    @test Array(compiled(container, resident, Reactant.to_rarray([0]))) ≈
          Array(batch(-X[:, [4, 2, 1, 3]]))
    joint_container = (; batch=joint)
    compiled_joint = Reactant.@compile guarded_batch(joint_container, resident, enabled)
    values, gradients = compiled_joint(joint_container, resident, enabled)
    @test vec(Array(values)) ≈
          vec(expected[1, :] + expected[2, :] - expected[3, :])[[4, 2, 1, 3]]
    @test vec(Array(gradients)) ≈ (4 .- 2rates)[[4, 2, 1, 3]]
    GC.gc()
    @test Array(batch(X)) ≈ expected
    @test size(prepare_batch(vm, zeros(1, 0))(zeros(1, 0))) == (3, 0)
    for backend in (forward, reverse)
        empty_batch = prepare_batch(vm, zeros(1, 0), loglikelihood; adtype=backend)
        empty_values, empty_gradients = empty_batch(zeros(1, 0))
        @test (size(empty_values), size(empty_gradients)) == ((1, 0), (1, 1, 0))
    end
    fixed = vecmodel(rate_model(3) | (rate=2.0,), VarNamedTuple())
    values, gradients =
        prepare_batch(fixed, zeros(0, 3), loglikelihood; adtype=reverse)(zeros(0, 3))
    @test vec(Array(values)) ≈ fill(lp(fixed, Float64[]), 3)
    @test size(gradients) == (1, 0, 3)
    empty_outputs = prepare_batch(vm, X, accs -> Float64[]; adtype=forward)
    values, gradients = empty_outputs(X)
    @test size(values) == (0, 4)
    @test size(gradients) == (0, 1, 4)
    empty_container = (; batch=empty_outputs)
    compiled_empty = Reactant.@compile guarded_batch(empty_container, resident, enabled)
    values, gradients = compiled_empty(empty_container, resident, enabled)
    @test size(values) == (0, 4) && size(gradients) == (0, 1, 4)
    likelihood_accs = AccumulatorTuple(LogLikelihoodAccumulator())
    likelihood = prepare_batch(vm, X, loglikelihood; accs=likelihood_accs)
    @test vec(Array(likelihood(X))) ≈ vec(expected[2, :])
    pair(accs) = [logprior(accs), loglikelihood(accs)]
    @test Array(prepare_batch(vm, X, pair)(X)) ≈ expected[1:2, :]
    selected(accs) = vcat(parameters(accs)[@vn(rate)], loglikelihood(accs))
    selected_accs = AccumulatorTuple(ParamsAccumulator(), LogLikelihoodAccumulator())
    with_params = prepare_batch(vm, X, selected; accs=selected_accs)
    @test Array(with_params(X)) ≈ vcat(reshape(rates, 1, :), expected[2:2, :])
    seed = CounterAccumulator([7.0])
    counter =
        prepare_batch(vm, X, accs -> accs[:counter].values; accs=AccumulatorTuple(seed))
    seed.values[1] = 99.0
    counter_container = (; batch=counter)
    compiled_counter = Reactant.@compile guarded_batch(counter_container, resident, enabled)
    for _ in 1:2
        @test Array(counter(X)) == fill(8.0, 1, 4)
        @test Array(compiled_counter(counter_container, resident, enabled)) ==
              fill(8.0, 1, 4)
    end
    differentiated_counter = prepare_batch(
        vm,
        X,
        accs -> accs[:counter].values;
        accs=AccumulatorTuple(CounterAccumulator([7.0])),
        adtype=forward,
    )
    for _ in 1:2
        values, gradients = differentiated_counter(X)
        @test Array(values) == fill(8.0, 1, 4)
        @test Array(gradients) == zeros(1, 1, 4)
    end

    points = reshape([-2.0, -0.5, 0.5, 2.0], 1, :)
    y = [0.2, 0.4]
    branches = vecmodel(branch_model(y), normal_values)
    expected_branch = map(vec(points)) do x
        prior = logdensityof(Normal(), x)
        if x > 0
            mu = shift(x)
            prior += sum(logdensityof.(Ref(Normal(mu)), y))
        else
            mu = 2x - 1
        end
        prior + logdensityof(Normal(mu, 2.0), last(y))
    end
    @test vec(Array(prepare_batch(branches, points, logjoint_transformed)(points))) ≈
          expected_branch
    @test Array(prepare_batch(branches, points, logjac)(points)) == zeros(1, 4)
    for backend in (forward, reverse)
        constant_batch = prepare_batch(branches, points, logjac; adtype=backend)
        values, gradients = constant_batch(points)
        @test Array(values) == zeros(1, 4)
        @test Array(gradients) == zeros(1, 1, 4)
    end
    branch_ad = vecmodel(branch_model(y), normal_values; adtype)
    for i in (2, 3)
        x = points[i]
        expected_gradient = if x > 0
            -x + sum(y .- shift(x)) + (last(y) - shift(x)) / 4
        else
            -x + (last(y) - (2x - 1)) / 2
        end
        value, gradient = lp_and_grad(branch_ad, [x])
        @test value ≈ expected_branch[i]
        @test gradient ≈ [expected_gradient]
    end

    fallback = vecmodel(fallback_model(y), normal_values)
    expected_fallback = [
        logdensityof(Normal(), x) +
        sum(logdensityof(Normal(x + i), y[i]) for i in eachindex(y)) for
        x in vec(points)
    ]
    @test vec(Array(prepare_batch(fallback, points, logjoint_transformed)(points))) ≈
          expected_fallback

    bounded_values = VarNamedTuple(upper=2.0, rate=0.75)
    bounded = vecmodel(bounded_model(3), bounded_values)
    BX = [-0.7 0.2 1.0 0.5; -1.0 0.5 1.5 -0.2]
    upper = exp.(BX[1, :])
    fraction = 1 ./ (1 .+ exp.(-BX[2, :]))
    rate = upper .* fraction
    expected_bounded = vcat(
        reshape(-upper - log.(upper), 1, :),
        reshape(3log.(rate) - rate .- log(6.0), 1, :),
        reshape(-2BX[1, :] - log.(fraction) - log1p.(-fraction), 1, :),
    )
    @test Array(prepare_batch(bounded, BX)(BX)) ≈ expected_bounded
    component_batch = prepare_batch(bounded, BX; adtype=forward)
    values, gradients = component_batch(BX)
    expected_gradients = zeros(3, 2, 4)
    expected_gradients[1, 1, :] = -upper .- 1
    expected_gradients[2, 1, :] = 3 .- rate
    expected_gradients[2, 2, :] = (3 .- rate) .* (1 .- fraction)
    expected_gradients[3, 1, :] .= -2
    expected_gradients[3, 2, :] = 2fraction .- 1
    @test Array(values) ≈ expected_bounded
    @test Array(gradients) ≈ expected_gradients
    component_container = (; batch=component_batch)
    component_points = Reactant.to_rarray(BX)
    compiled_components =
        Reactant.@compile guarded_batch(component_container, component_points, enabled)
    values, gradients = compiled_components(component_container, component_points, enabled)
    @test Array(values) ≈ expected_bounded
    @test Array(gradients) ≈ expected_gradients
    bounded_ad = vecmodel(bounded_model(3), bounded_values; adtype)
    _, gradient = lp_and_grad(bounded_ad, BX[:, 2])
    @test gradient ≈
          [4 - upper[2] - rate[2], 1 - 2fraction[2] + (3 - rate[2]) * (1 - fraction[2])]

    arrays = vecmodel(bounded_array_model(3), VarNamedTuple(upper=2.0, rate=[0.5, 0.75]))
    AX = vcat(BX, 0.25 .- BX[2:2, :])
    expected_arrays = reduce(hcat, map(eachcol(AX)) do x
        _, accs = evaluate(arrays, x)
        vcat(logprior(accs), loglikelihood(accs), logjac(accs))
    end)
    array_batch = prepare_batch(arrays, AX)
    @test Array(array_batch(AX)) ≈ expected_arrays
    array_container, array_points = (; batch=array_batch), Reactant.to_rarray(AX)
    compiled_arrays =
        Reactant.@compile guarded_batch(array_container, array_points, enabled)
    @test Array(compiled_arrays(array_container, array_points, enabled)) ≈ expected_arrays

    bounds = [0.0 0.0 -Inf -Inf; 2.0 Inf 3.0 Inf; 0.5 0.5 0.5 0.5]
    expected_transforms = mapreduce(hcat, eachcol(bounds)) do x
        vcat(Plaice.with_logabsdet_jacobian(Plaice.Truncate(x[1], x[2]), x[3])...)
    end
    bounds_input = Reactant.to_rarray(bounds)
    @test Array(Reactant.@jit transform_columns(bounds_input)) ≈ expected_transforms

    # Model preparation already evaluated the += once, leaving the prepared seed at 3.5.
    expected_mutation = 3log.(3.5 .+ rates) - (3.5 .+ rates) .- log(6.0)
    for model in (mutating_model([2.0], 3), mutating_model([0.0], 3) | (buffer=[2.0],))
        mutating = vecmodel(model, rate_values)
        isolated = prepare_batch(mutating, X, loglikelihood)
        for _ in 1:2
            @test vec(Array(isolated(X))) ≈ expected_mutation
        end
        differentiated = prepare_batch(mutating, X, loglikelihood; adtype=reverse)
        for _ in 1:2
            values, gradients = differentiated(X)
            @test vec(Array(values)) ≈ expected_mutation
            @test vec(Array(gradients)) ≈ rates .* (3 ./ (3.5 .+ rates) .- 1)
        end
    end

    nested_values = @vnt a.x := 1.0
    for model in (
        nested_model(branch_model(y)),
        nested_model(branch_model([0.1])) | (child=branch_model(y),),
    )
        nested = prepare_batch(vecmodel(model, nested_values), points, logjoint_transformed)
        @test vec(Array(nested(points))) ≈ expected_branch
    end

    early = vecmodel(control_model(0.2, true), normal_values)
    normal_prior = logdensityof.(Ref(Normal()), vec(points))
    @test vec(Array(prepare_batch(early, points, logjoint_transformed)(points))) ≈
          normal_prior
    nested_if = vecmodel(control_model(0.2, false), normal_values)
    nested_expected = copy(normal_prior)
    nested_expected[end] += logdensityof(Normal(2.0), 0.2)
    @test vec(Array(prepare_batch(nested_if, points, logjoint_transformed)(points))) ≈
          nested_expected

    # A final conditional has an implicit accumulator/nothing return value.
    control_ad = vecmodel(control_model(0.2, false), normal_values; adtype)
    @test last(lp_and_grad(control_ad, [2.0])) ≈ [-3.8]
    for x in (-2.0, 0.5, 2.0)
        @test evaluate(control_ad, [x]) == evaluate(nested_if, [x])
    end

    array_vm = vecmodel(array_branch(0.2), VarNamedTuple(beta=[1.0, 0.5]))
    array_points = vcat(points, [0.3 -0.1 0.2 0.4])
    array_batch =
        prepare_batch(array_vm, array_points, logjoint_transformed; adtype=reverse)
    values, gradients = array_batch(array_points)
    @test vec(Array(values)) ≈ [lp(array_vm, x) for x in eachcol(array_points)]
    @test dropdims(Array(gradients); dims=1) ≈ [11.0 2.7 -1.0 -4.2; -4.8 -1.0 -0.7 -2.6]

    nonfinite = accs -> [-Inf, loglikelihood(accs)]
    values, gradients = prepare_batch(vm, X, nonfinite; adtype=forward)(X)
    @test all(Array(values)[1, :] .== -Inf)
    @test Array(values)[2, :] ≈ expected[2, :]
    @test Array(gradients) ≈ reshape(vcat(zeros(1, 4), (3 .- rates)'), 2, 1, 4)
end

end
