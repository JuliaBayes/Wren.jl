module WrenCorePredictTests

@info "testing predict.jl"

using AbstractMCMC: AbstractMCMC
using Distributions
using FlexiChains: FlexiChains, Extra, Parameter, VNChain
using LinearAlgebra
using OffsetArrays: OffsetArray
using Random
using StableRNGs
using Test
using WrenCore

"""
Build a `niters × nchains` chain of exact posterior draws. `posterior_model` must sample
from the posterior of `cond_model`; each draw is re-evaluated in `cond_model` so that the
chain holds the same log-probabilities an MCMC sampler would record.
"""
function exact_posterior_chain(rng, posterior_model, cond_model, niters, nchains)
    draws = map(Iterators.product(1:niters, 1:nchains)) do _
        posterior_params = rand(rng, posterior_model)
        Draw(FromValues(posterior_params), cond_model, (acceptance_rate=1.0,))
    end
    return AbstractMCMC.from_samples(VNChain, draws)
end

@testset "predict" begin
    # `m` is stored in the chain as a single variable. `p` is sampled element by element,
    # so the model builds it as a partial array and the chain stores it densified; this
    # exercises the densification path in `predict`.
    @wren function f(p)
        m ~ MvNormal(zeros(2), I)
        for i in eachindex(p)
            p[i] ~ Normal()
        end
        x ~ Normal()
        y ~ Normal(x)
    end
    # With y = 4.0 observed, the posterior of x is Normal(2, √0.5).
    @wren function f_posterior(p)
        m ~ MvNormal(zeros(2), I)
        for i in eachindex(p)
            p[i] ~ Normal()
        end
        x ~ Normal(2.0, sqrt(0.5))
    end
    model = f(zeros(2)) | (; y=4.0)
    chn = exact_posterior_chain(StableRNG(468), f_posterior(zeros(2)), model, 1000, 2)

    # Sanity check
    @test isapprox(mean(chn[@vn(x)]), 2.0; atol=0.1)
    @test isapprox(mean(chn[@vn(m[1])]), 0.0; atol=0.1)
    @test isapprox(mean(chn[@vn(m[2])]), 0.0; atol=0.1)
    @test isapprox(mean(chn[@vn(p[1])]), 0.0; atol=0.1)
    @test isapprox(mean(chn[@vn(p[2])]), 0.0; atol=0.1)

    @testset "chain values are actually used" begin
        pdns = predict(StableRNG(468), f(zeros(2)), chn)
        # Sanity check.
        @test pdns[@vn(x)] == chn[@vn(x)]
        @test pdns[@vn(m)] == chn[@vn(m)]
        @test pdns[@vn(p)] == chn[@vn(p)]
        # Since the model was conditioned with y = 4.0, we should expect that the chain's
        # mean of x is approx 2.0. So the posterior predictions for y should be centred on
        # 2.0 (ish).
        @test isapprox(mean(pdns[@vn(y)]), 2.0; atol=0.1)
    end

    @testset "logp" begin
        pdns = predict(f(zeros(2)), chn)
        # Since we deconditioned `y`, there are no likelihood terms.
        @test all(iszero, pdns[Extra(:loglikelihood)])
        # The logprior should be the same as that of the original chain, but with an extra
        # term for y ~ Normal(x)
        chn_logprior = chn[Extra(:logprior)]
        pdns_logprior = pdns[Extra(:logprior)]
        expected_diff = logpdf.(Normal.(chn[@vn(x)]), pdns[@vn(y)])
        @test isapprox(pdns_logprior, chn_logprior .+ expected_diff)
        # Logjoint should be the same as logprior
        @test pdns[Extra(:logjoint)] == pdns[Extra(:logprior)]
    end

    @testset "non-parameter keys are preserved" begin
        pdns = predict(f(zeros(2)), chn)
        # Check that the only new thing added was the prediction for y.
        @test only(setdiff(Set(keys(pdns)), Set(keys(chn)))) == Parameter(@vn(y))
        # Check that no other keys originally in `chn` were removed.
        @test isempty(setdiff(Set(keys(chn)), Set(keys(pdns))))
    end

    @testset "indices are preserved" begin
        pdns = predict(f(zeros(2)), chn[iter=501:1000])
        @test FlexiChains.iter_indices(pdns) == 501:1000
        @test FlexiChains.chain_indices(pdns) == FlexiChains.chain_indices(chn)
    end

    @testset "no sampling time and sampler state" begin
        # it just doesn't really make sense for the predictions to carry those
        # information
        pdns = predict(f(zeros(2)), chn)
        @test all(ismissing, FlexiChains.sampling_time(pdns))
        @test all(ismissing, FlexiChains.last_sampler_state(pdns))
    end

    @testset "rng is respected" begin
        pdns1 = predict(Xoshiro(468), f(zeros(2)), chn)
        pdns2 = predict(Xoshiro(468), f(zeros(2)), chn)
        @test FlexiChains.has_same_data(pdns1, pdns2)
        pdns3 = predict(Xoshiro(469), f(zeros(2)), chn)
        @test !FlexiChains.has_same_data(pdns1, pdns3)
    end

    @testset "with non-standard Array variables" begin
        # This essentially tests that templates are correctly used when calling predict().
        @wren function offset2(x)
            # Don't sample all elements of `x` to prevent it from being densified, thus
            # bypassing the code that we want to check.
            x[-2] ~ Normal()
            y ~ Normal(x[-2])
        end
        # With y = 2.0 observed, the posterior of x[-2] is Normal(1, √0.5).
        @wren function offset2_posterior(x)
            x[-2] ~ Normal(1.0, sqrt(0.5))
        end
        offset_arg() = OffsetArray(zeros(2), -2:-1)
        cond_model = offset2(offset_arg()) | (; y=2.0)
        chn = exact_posterior_chain(
            StableRNG(468),
            offset2_posterior(offset_arg()),
            cond_model,
            2000,
            1,
        )
        @test mean(chn[@vn(x[-2])]) ≈ 1.0 atol = 0.05
        pdns = predict(StableRNG(468), offset2(offset_arg()), chn)
        @test pdns[@vn(x[-2])] == chn[@vn(x[-2])]
        @test mean(pdns[@vn(y)]) ≈ 1.0 atol = 0.05
    end
end

end # module WrenCorePredictTests
