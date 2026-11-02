module WrenBatchScalingTests

using Test, ForwardDiff
using WrenCore, WrenDists

@wren function observations(mu, y, full_size)
    offset ~ Normal(zero(eltype(y)))
    @batch size=full_size for i in eachindex(y)
        mu[i] ~ Normal(offset)
        @observe y[i] ~ Normal(exp(mu[i]))
    end
end

@wren function local_latents(z, full_size)
    @batch size=full_size for i in eachindex(z)
        z[i] ~ Exponential(one(eltype(z)))
    end
end

evaluate_accumulators(model, source, accs) =
    last(evaluate(model, source, OriginalSpace(), accs, EvalContext{false}()))

@testset "batch density scaling" begin
    y = Float32[1, 2]
    mu = Float32[0.25, 0.5]
    raw_likelihood = sum(logdensityof(Normal(exp(m)), value) for (m, value) in zip(mu, y))
    accs = AccumulatorTuple(
        LogPriorAccumulator{Float32}(),
        LogLikelihoodAccumulator{Float32}(),
    )
    out = evaluate_accumulators(
        observations(zeros(Float32, 2), y, 5),
        FromValues(VarNamedTuple(mu=mu, offset=0.0f0)),
        accs,
    )
    @test loglikelihood(out) == 2.5f0 * raw_likelihood
    @test logprior(out) ==
          logdensityof(Normal(0.0f0), 0.0f0) +
          2.5f0 * logdensityof(iid(Normal(0.0f0), (2,)), mu)

    z = [2.0, 3.0]
    model = local_latents(zeros(2), 5)
    source = FromValues(VarNamedTuple(z=z))
    vm = VecModel(model, AutoForwardDiff(); latent_source=source)
    value, gradient = lp_and_grad(vm, log.(z))
    @test value ≈ -2.5sum(z) + sum(log, z)
    @test gradient ≈ 1 .- 2.5z
    values = evaluate_accumulators(
        model,
        source,
        AccumulatorTuple(WrenCore.VNTAccumulator{:raw}((val, args...) -> val)),
    )
    @test WrenCore.getacc(values, Val(:raw)).values[@vn(z)] == z
    conditioned = rand_with_logp(model | (z=z,))
    @test loglikelihood(conditioned) ≈ -2.5sum(z)
    @test iszero(logprior(conditioned))

    small = Float16[0.001]
    weighted = rand_with_logp(local_latents(small, 100000) | (z=small,))
    @test loglikelihood(weighted) == Float16(-100000 * Float64(only(small)))
    @test_throws ArgumentError rand_with_logp(local_latents(Float64[], 5))
end

end
