module WrenBatchEvaluationTests

using WrenCore, WrenDists, Reactant, Test
using ADTypes: AutoReactant
using Plaice

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
    return x
end

@wren function implicit_observation(y)
    x ~ Normal()
    if x > 0
        @observe y ~ Normal(x)
    end
end

@testset "prepared parameter batches" begin
    X = reshape(log.([0.5, 1.5, 3.0, 5.0]), 1, :)
    rates = exp.(vec(X))
    vm = VecModel(rate_model(3), nothing; latent_source=FromValues(VarNamedTuple(rate=1.5)))
    batch = prepare_batch(vm, X)
    expected = vcat(reshape(-rates, 1, :), reshape(3vec(X) - rates .- log(6.0), 1, :), -X)
    @test Array(batch(X)) ≈ expected
    resident = Reactant.to_rarray(X[:, [4, 2, 1, 3]])
    @test Array(batch(resident)) ≈ expected[:, [4, 2, 1, 3]]
    @test Array(batch(resident)) ≈ expected[:, [4, 2, 1, 3]]
    @test Array(resident) == X[:, [4, 2, 1, 3]]
    GC.gc()
    @test Array(batch(X)) ≈ expected
    @test size(prepare_batch(vm, zeros(1, 0))(zeros(1, 0))) == (3, 0)
    @test_throws DimensionMismatch prepare_batch(vm, zeros(2, 4))
    @test_throws DimensionMismatch batch(zeros(1, 3))
    likelihood = prepare_batch(
        vm,
        X,
        loglikelihood;
        accs=AccumulatorTuple(LogLikelihoodAccumulator()),
    )
    @test vec(Array(likelihood(X))) ≈ vec(expected[2, :])
    pair(accs) = [logprior(accs), loglikelihood(accs)]
    @test Array(prepare_batch(vm, X, pair)(X)) ≈ expected[1:2, :]
    selected(accs) = vcat(parameters(accs)[@vn(rate)], loglikelihood(accs))
    with_params = prepare_batch(
        vm,
        X,
        selected;
        accs=AccumulatorTuple(ParamsAccumulator(), LogLikelihoodAccumulator()),
    )
    @test Array(with_params(X)) ≈ vcat(reshape(rates, 1, :), expected[2:2, :])
    @test Array(with_params(X)) ≈ vcat(reshape(rates, 1, :), expected[2:2, :])
    seed = CounterAccumulator([7.0])
    counter =
        prepare_batch(vm, X, accs -> accs[:counter].values; accs=AccumulatorTuple(seed))
    seed.values[1] = 99.0
    @test Array(counter(X)) == fill(8.0, 1, 4)
    @test Array(counter(X)) == fill(8.0, 1, 4)

    points = reshape([-2.0, -0.5, 0.5, 2.0], 1, :)
    y = [0.2, 0.4]
    branches =
        VecModel(branch_model(y), nothing; latent_source=FromValues(VarNamedTuple(x=1.0)))
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
    branch_ad = VecModel(
        branch_model(y),
        AutoReactant(AutoEnzyme(; mode=Reverse));
        latent_source=FromValues(VarNamedTuple(x=1.0)),
    )
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

    fallback =
        VecModel(fallback_model(y), nothing; latent_source=FromValues(VarNamedTuple(x=1.0)))
    expected_fallback = [
        logdensityof(Normal(), x) +
        sum(logdensityof(Normal(x + i), y[i]) for i in eachindex(y)) for
        x in vec(points)
    ]
    @test vec(Array(prepare_batch(fallback, points, logjoint_transformed)(points))) ≈
          expected_fallback

    bounded = VecModel(
        bounded_model(3),
        nothing;
        latent_source=FromValues(VarNamedTuple(upper=2.0, rate=0.75)),
    )
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
    bounded_ad = VecModel(
        bounded_model(3),
        AutoReactant(AutoEnzyme(; mode=Reverse));
        latent_source=FromValues(VarNamedTuple(upper=2.0, rate=0.75)),
    )
    _, gradient = lp_and_grad(bounded_ad, BX[:, 2])
    @test gradient ≈
          [4 - upper[2] - rate[2], 1 - 2fraction[2] + (3 - rate[2]) * (1 - fraction[2])]

    arrays = VecModel(
        bounded_array_model(3),
        nothing;
        latent_source=FromValues(VarNamedTuple(upper=2.0, rate=[0.5, 0.75])),
    )
    AX = vcat(BX, 0.25 .- BX[2:2, :])
    expected_arrays = reduce(hcat, map(eachcol(AX)) do x
        _, accs = evaluate(arrays, x)
        vcat(logprior(accs), loglikelihood(accs), logjac(accs))
    end)
    @test Array(prepare_batch(arrays, AX)(AX)) ≈ expected_arrays

    bounds = [0.0 0.0 -Inf -Inf; 2.0 Inf 3.0 Inf; 0.5 0.5 0.5 0.5]
    expected_transforms = reduce(
        hcat,
        [
            vcat(Plaice.with_logabsdet_jacobian(Plaice.Truncate(x[1], x[2]), x[3])...)
            for x in eachcol(bounds)
        ],
    )
    bounds_input = Reactant.to_rarray(bounds)
    @test Array(Reactant.@jit transform_columns(bounds_input)) ≈ expected_transforms

    mutating = VecModel(
        mutating_model([2.0], 3),
        nothing;
        latent_source=FromValues(VarNamedTuple(rate=1.5)),
    )
    # Model preparation already evaluated the += once, leaving the prepared seed at 3.5.
    expected_mutation = 3log.(3.5 .+ rates) - (3.5 .+ rates) .- log(6.0)
    isolated = prepare_batch(mutating, X, loglikelihood)
    @test vec(Array(isolated(X))) ≈ expected_mutation
    @test vec(Array(isolated(X))) ≈ expected_mutation

    conditioned_vm = VecModel(
        mutating_model([0.0], 3) | (buffer=[2.0],),
        nothing;
        latent_source=FromValues(VarNamedTuple(rate=1.5)),
    )
    conditioned_batch = prepare_batch(conditioned_vm, X, loglikelihood)
    @test vec(Array(conditioned_batch(X))) ≈ expected_mutation
    @test vec(Array(conditioned_batch(X))) ≈ expected_mutation

    nested = VecModel(
        nested_model(branch_model(y)),
        nothing;
        latent_source=FromValues(@vnt begin
            a.x := 1.0
        end),
    )
    @test vec(Array(prepare_batch(nested, points, logjoint_transformed)(points))) ≈
          expected_branch
    conditioned_child = nested_model(branch_model([0.1])) | (child=branch_model(y),)
    conditioned_nested =
        VecModel(conditioned_child, nothing; latent_source=FromValues(@vnt begin
            a.x := 1.0
        end))
    @test vec(
        Array(prepare_batch(conditioned_nested, points, logjoint_transformed)(points)),
    ) ≈ expected_branch

    early = VecModel(
        control_model(0.2, true),
        nothing;
        latent_source=FromValues(VarNamedTuple(x=1.0)),
    )
    normal_prior = logdensityof.(Ref(Normal()), vec(points))
    @test vec(Array(prepare_batch(early, points, logjoint_transformed)(points))) ≈
          normal_prior
    nested_if = VecModel(
        control_model(0.2, false),
        nothing;
        latent_source=FromValues(VarNamedTuple(x=1.0)),
    )
    nested_expected = copy(normal_prior)
    nested_expected[end] += logdensityof(Normal(2.0), 0.2)
    @test vec(Array(prepare_batch(nested_if, points, logjoint_transformed)(points))) ≈
          nested_expected

    implicit_vm = VecModel(
        implicit_observation(0.2),
        nothing;
        latent_source=FromValues(VarNamedTuple(x=0.5)),
    )
    implicit_expected = copy(normal_prior)
    implicit_expected[3:4] += logdensityof.(Normal.(vec(points)[3:4]), 0.2)
    @test vec(Array(prepare_batch(implicit_vm, points, logjoint_transformed)(points))) ≈
          implicit_expected
    implicit_ad = VecModel(
        implicit_observation(0.2),
        AutoReactant(AutoEnzyme(; mode=Reverse));
        latent_source=FromValues(VarNamedTuple(x=0.5)),
    )
    @test last(lp_and_grad(implicit_ad, [1.0])) ≈ [-1.8]
    for x in (-1.0, 1.0)
        @test evaluate(implicit_ad, [x]) == evaluate(implicit_vm, [x])
    end
end

end
