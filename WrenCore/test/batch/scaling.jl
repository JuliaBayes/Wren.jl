module WrenBatchScalingTests

using Test, Random, ForwardDiff
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

@testset "batch density scaling" begin
    Random.seed!(8675309)
    y = Float32[1, 2]
    mu = Float32[0.25, 0.5]
    raw_likelihood = sum(logdensityof(Normal(exp(m)), value) for (m, value) in zip(mu, y))
    accs = WrenCore.AccumulatorTuple(
        WrenCore.LogPriorAccumulator{Float32}(),
        WrenCore.LogLikelihoodAccumulator{Float32}(),
    )
    _, out = evaluate(
        observations(zeros(Float32, 2), y, 5),
        FromValues(VarNamedTuple(mu=mu, offset=0.0f0)),
        OriginalSpace(),
        accs,
        EvalContext{false}(),
    )
    @test loglikelihood(out) == 2.5f0 * raw_likelihood
    @test logprior(out) ==
          logdensityof(Normal(0.0f0), 0.0f0) +
          2.5f0 * logdensityof(iid(Normal(0.0f0), (2,)), mu)

    z = [2.0, 3.0]
    model = local_latents(zeros(2), 5)
    vm = VecModel(model, AutoForwardDiff())
    value, gradient = lp_and_grad(vm, log.(z))
    @test value ≈ -2.5sum(z) + sum(log, z)
    @test gradient ≈ 1 .- 2.5z
    _, values = evaluate(
        model,
        FromValues(VarNamedTuple(z=z)),
        OriginalSpace(),
        WrenCore.AccumulatorTuple(WrenCore.VNTAccumulator{:raw}((val, args...) -> val)),
        EvalContext{false}(),
    )
    @test WrenCore.getacc(values, Val(:raw)).values[@vn(z)] == z
    conditioned = rand_with_logp(model | VarNamedTuple(z=z))
    @test loglikelihood(conditioned) ≈ -2.5sum(z)
    @test iszero(logprior(conditioned))

    small = Float16[0.001]
    weighted = rand_with_logp(local_latents(small, 100000) | VarNamedTuple(z=small))
    @test loglikelihood(weighted) == Float16(-100000 * Float64(only(small)))

end

end
