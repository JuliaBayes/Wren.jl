module WrenSamplingTests

@info "testing sampling.jl"

using Wren
using Wren.Dists
using Random: Xoshiro
using ForwardDiff: ForwardDiff
using Test

# Check that the mean and variance of `samples` (an iter × chain matrix) agree with the
# expected values, to within a tolerance derived from the Monte Carlo standard error.
function test_moments(samples, expected_mean, expected_var; nmcse=5)
    @test abs(mean(samples) - expected_mean) < nmcse * mcse(samples)
    @test abs(var(samples) - expected_var) < nmcse * mcse(samples; kind=var)
end

@testset "sampling.jl" begin
    @testset "nuts recovers a conjugate posterior" begin
        # With a Gamma(α, θ) prior and Poisson observations y, the posterior is
        # Gamma(α + sum(y), θ / (1 + n * θ)). λ is constrained, so NUTS samples it in
        # unconstrained space and this also tests the Jacobian correction. The prior and data
        # are deliberately weak: dropping the Jacobian would shift the posterior mean by half
        # a posterior standard deviation.
        @wren function gamma_poisson(y, α, θ)
            λ ~ Gamma(α, θ)
            for i in eachindex(y)
                @observe y[i] ~ Poisson(λ)
            end
        end
        y = [2, 1]
        α, θ = 1.0, 1.0
        shape = α + sum(y)
        scale = θ / (1 + length(y) * θ)

        chn = nuts(
            Xoshiro(468),
            gamma_poisson(y, α, θ),
            2000,
            2;
            adtype=AutoForwardDiff(),
            progress=false,
        )
        test_moments(chn[@vn(λ)], shape * scale, shape * scale^2)
    end

    @testset "prior samples from the prior" begin
        @wren function two_priors()
            μ ~ Normal(1.0, 2.0)
            λ ~ Gamma(3.0, 0.5)
        end
        chn = prior(Xoshiro(468), two_priors(), 2000, 2; progress=false)
        test_moments(chn[@vn(μ)], 1.0, 4.0)
        test_moments(chn[@vn(λ)], 1.5, 0.75)
    end

    @testset "seeded sampling is reproducible, with independent chains" begin
        @wren function normal_model()
            x ~ Normal()
        end
        model = normal_model()
        niters, nchains = 100, 2
        @testset "$name" for (name, sampler) in (
            "nuts" => (rng, m) ->
                nuts(rng, m, niters, nchains; adtype=AutoForwardDiff(), progress=false),
            "prior" => (rng, m) -> prior(rng, m, niters, nchains; progress=false),
        )
            x = sampler(Xoshiro(468), model)[@vn(x)]
            @test size(x) == (niters, nchains)
            @test sampler(Xoshiro(468), model)[@vn(x)] == x
            @test x[:, 1] != x[:, 2]
        end
    end
end

end # module WrenSamplingTests
