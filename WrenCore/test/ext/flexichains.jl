module WrenCoreFlexiChainsExtTests

@info "testing simulate.jl"

using Distributions
using FlexiChains: FlexiChains, Extra, Parameter, VNChain
using LinearAlgebra
using OffsetArrays: OffsetArray
using Random
using StableRNGs
using Test
using VarNames: VarNames
using WrenCore

"""
Build a `niters × nchains` chain of exact posterior draws. `posterior_model` must sample
from the posterior of `cond_model`; each draw is re-evaluated in `cond_model` so that the
chain holds the same log-probabilities an MCMC sampler would record.
"""
function exact_posterior_chain(rng, posterior_model, cond_model, niters, nchains)
    results = map(Iterators.product(1:niters, 1:nchains)) do _
        posterior_params = parameters(posterior_model, rng)
        simulate(cond_model, posterior_params)
    end
    stats = fill((acceptance_rate=1.0,), niters, nchains)
    return WrenCore.to_flexichain(results, stats)
end

@testset "simulate on a chain" begin
    # `m` is stored in the chain as a single variable. `p` is sampled element by element,
    # so the model builds it as a partial array and the chain stores it densified; this
    # exercises the densification path in `simulate`.
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

    @testset "logprior / loglikelihood / logjoint / return_value on a chain" begin
        expected = map(vnt -> simulate(model, vnt), FlexiChains.parameters_at(chn))
        @test Array(logprior(model, chn)) ≈ Array(map(logprior, expected))
        @test Array(loglikelihood(model, chn)) ≈ Array(map(loglikelihood, expected))
        @test Array(logjoint(model, chn)) ≈ Array(map(logjoint, expected))
        @test Array(return_value(model, chn)) == Array(map(return_value, expected))
        # The chain was built by evaluating `model`, so re-evaluating gives the same values.
        @test Array(logjoint(model, chn)) ≈ Array(chn[Extra(:logjoint)])
        # Without conditioning, `y` is latent but not in the chain, which is an error.
        for f_chain in (logprior, loglikelihood, logjoint, return_value)
            @test_throws "No latent value provided for variable y" f_chain(f(zeros(2)), chn)
        end
    end

    @testset "parameters on a chain" begin
        @testset "complete chain" begin
            params = parameters(model, chn)
            @test params isa VNChain
            @test params[@vn(x)] == chn[@vn(x)]
            @test params[@vn(m)] == chn[@vn(m)]
            @test params[@vn(p)] == chn[@vn(p)]
        end

        @testset "only parameters are returned" begin
            params = parameters(f(zeros(2)), chn)
            expected_keys = Set(Parameter.([@vn(m), @vn(p), @vn(x), @vn(y)]))
            @test Set(keys(params)) == expected_keys
        end

        @testset "missing variables are drawn from the prior" begin
            params = parameters(f(zeros(2)), chn; rng=StableRNG(468))
            @test params[@vn(x)] == chn[@vn(x)]
            @test isapprox(mean(params[@vn(y)]), 2.0; atol=0.1)
            pdns = simulate(f(zeros(2)), chn; rng=StableRNG(468))
            @test params[@vn(y)] == pdns[@vn(y)]
        end

        @testset "rng is respected" begin
            params1 = parameters(f(zeros(2)), chn; rng=Xoshiro(468))
            params2 = parameters(f(zeros(2)), chn; rng=Xoshiro(468))
            @test FlexiChains.has_same_data(params1, params2)
            params3 = parameters(f(zeros(2)), chn; rng=Xoshiro(469))
            @test !FlexiChains.has_same_data(params1, params3)
        end

        @testset "indices are preserved" begin
            params = parameters(f(zeros(2)), chn[iter=501:1000])
            @test FlexiChains.iter_indices(params) == 501:1000
            @test FlexiChains.chain_indices(params) == FlexiChains.chain_indices(chn)
        end
    end

    @testset "chain values are actually used" begin
        pdns = simulate(f(zeros(2)), chn; rng=StableRNG(468))
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
        pdns = simulate(f(zeros(2)), chn)
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
        pdns = simulate(f(zeros(2)), chn)
        # Check that the only new thing added was the simulated y.
        @test only(setdiff(Set(keys(pdns)), Set(keys(chn)))) == Parameter(@vn(y))
        # Check that no other keys originally in `chn` were removed.
        @test isempty(setdiff(Set(keys(chn)), Set(keys(pdns))))
    end

    @testset "indices are preserved" begin
        pdns = simulate(f(zeros(2)), chn[iter=501:1000])
        @test FlexiChains.iter_indices(pdns) == 501:1000
        @test FlexiChains.chain_indices(pdns) == FlexiChains.chain_indices(chn)
    end

    @testset "no sampling time and sampler state" begin
        # it just doesn't really make sense for the simulated chain to carry those
        # information
        pdns = simulate(f(zeros(2)), chn)
        @test all(ismissing, FlexiChains.sampling_time(pdns))
        @test all(ismissing, FlexiChains.last_sampler_state(pdns))
    end

    @testset "rng is respected" begin
        pdns1 = simulate(f(zeros(2)), chn; rng=Xoshiro(468))
        pdns2 = simulate(f(zeros(2)), chn; rng=Xoshiro(468))
        @test FlexiChains.has_same_data(pdns1, pdns2)
        pdns3 = simulate(f(zeros(2)), chn; rng=Xoshiro(469))
        @test !FlexiChains.has_same_data(pdns1, pdns3)
    end

    @testset "with non-standard Array variables" begin
        # This essentially tests that templates are correctly used when calling simulate().
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
        pdns = simulate(offset2(offset_arg()), chn; rng=StableRNG(468))
        @test pdns[@vn(x[-2])] == chn[@vn(x[-2])]
        @test mean(pdns[@vn(y)]) ≈ 1.0 atol = 0.05
    end
end

end # module WrenCoreSimulateChainTests
