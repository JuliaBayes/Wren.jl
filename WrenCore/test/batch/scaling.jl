module WrenBatchScalingTests

using Test, ForwardDiff, Enzyme
using WrenCore, WrenDists

@wren function observations(mu, y, full_size)
    offset ~ Normal(zero(eltype(y)))
    @batch size=full_size for i in eachindex(y)
        mu[i] ~ Normal(offset)
        @observe y[i] ~ Normal(mu[i])
    end
end

@wren function local_latents(z, full_size)
    @batch size=full_size for i in eachindex(z)
        z[i] ~ Exponential(one(eltype(z)))
    end
end

@wren function scaled_likelihood(y, full_size)
    offset ~ Exponential()
    @batch size=full_size for i in eachindex(y)
        @observe y[i] ~ Normal(offset)
    end
end

evaluate_accumulators(model, source, accs) =
    last(evaluate(model, source, OriginalSpace(), accs, EvalContext{false}()))

@testset "batch density scaling" begin
    y = Float32[1, 2]
    mu = Float32[0.25, 0.5]
    raw_likelihood = sum(logdensityof(Normal(m), value) for (m, value) in zip(mu, y))
    accs = AccumulatorTuple(
        LogPriorAccumulator{Float32}(),
        LogLikelihoodAccumulator{Float32}(),
    )
    observation_model = observations(zeros(Float32, 2), y, 5)
    observation_params = VarNamedTuple(mu=mu, offset=0.0f0)
    out = evaluate_accumulators(observation_model, FromValues(observation_params), accs)
    @test loglikelihood(out) ≈ 2.5f0 * raw_likelihood
    @test logprior(out) ≈
          logdensityof(Normal(0.0f0), 0.0f0) +
          2.5f0 * logdensityof(iid(Normal(0.0f0), (2,)), mu)

    z = [2.0, 3.0]
    model = local_latents(zeros(2), 5)
    source = FromValues(VarNamedTuple(z=z))
    for adtype in (AutoForwardDiff(), AutoEnzyme(; mode=Reverse))
        vm = VecModel(model, adtype; latent_source=source)
        value, gradient = lp_and_grad(vm, log.(z))
        @test value ≈ -2.5sum(z) + sum(log, z)
        @test gradient ≈ 1 .- 2.5z

        likelihood = scaled_likelihood([1.0, 2.0], 5)
        vm = VecModel(
            likelihood,
            adtype;
            latent_source=FromValues(VarNamedTuple(offset=2.0)),
        )
        value, gradient = lp_and_grad(vm, [log(2.0)])
        @test value ≈ -2 + log(2) + 2.5sum(logdensityof(Normal(2.0), y) for y in (1, 2))
        @test only(gradient) ≈ 1 - 2 + 2.5 * 2 * ((1 - 2) + (2 - 2))
    end
    values = evaluate_accumulators(
        model,
        source,
        AccumulatorTuple(WrenCore.VNTAccumulator{:raw}((val, args...) -> val)),
    )
    @test WrenCore.getacc(values, Val(:raw)).values[@vn(z)] == z
    conditioned = simulate(model | (; z))
    @test loglikelihood(conditioned) ≈ -2.5sum(z)
    @test iszero(logprior(conditioned))

    small = Float16[0.001]
    weighted = simulate(local_latents(small, 100000) | (; z=small))
    @test loglikelihood(weighted) == Float16(-100000 * Float64(only(small)))
end

end
