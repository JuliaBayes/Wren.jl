module WrenSamplingTests

@info "testing sampling.jl"

using Wren
using Wren.Dists
using FlexiChains: FlexiChains
using Random: Random, AbstractRNG, Xoshiro
using ForwardDiff: ForwardDiff
using Test

const failure_worker = Threads.Atomic{Int}(0)
const sibling_finished = Threads.Atomic{Bool}(false)
const sibling_running = Base.Event()

struct DrawOnlyPrior end
Random.rand(rng::AbstractRNG, ::DrawOnlyPrior) = rand(rng, Categorical([0.5, 0.25, 0.25]))

function fail_or_finish()
    if Threads.atomic_add!(failure_worker, 1) == 0
        wait(sibling_running)
        error("intentional worker failure")
    end
    notify(sibling_running)
    # Keep this worker alive when the other chain fails, so early cleanup is observable.
    sleep(0.1)
    sibling_finished[] = true
    return nothing
end

@wren function failing_prior_model()
    fail_or_finish()
    x ~ Normal()
end

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

    @testset "independent MH evaluates only the likelihood" begin
        @wren function categorical_posterior(likelihood)
            x ~ DrawOnlyPrior()
            @addloglikelihood log(likelihood[Int(x)])
            @track doubled = 2x
            return x
        end
        likelihood = [0.25, 0.75, 0.0]
        chn = independent_mh(
            Xoshiro(468),
            categorical_posterior(likelihood),
            2000,
            2;
            progress=false,
        )
        x = Int.(Array(chn[@vn(x)]))
        previous = x[1:(end-1), :]
        next = x[2:end, :]
        @test all(in((1, 2)), x)
        @test chn[@vn(doubled)] == 2 .* x
        @test chn[:return_value] == x
        @test chn[:loglikelihood] ≈ log.(likelihood[x])
        @test all((previous .== next) .| Array(chn[:accepted])[2:end, :])

        # P(i → j) = prior(j) * min(1, likelihood[j] / likelihood[i]).
        for (from, to, probability) in ((1, 2, 1 / 4), (2, 1, 1 / 6))
            n = count(==(from), previous)
            transitions = count((previous .== from) .& (next .== to))
            @test abs(transitions - n * probability) <
                  6 * sqrt(n * probability * (1 - probability))
        end
    end

    @testset "random-walk samplers recover a correlated posterior" begin
        @wren function correlated_lognormal(observation)
            x ~ MvNormalScale(zeros(2), I)
            y ~ LogNormal(0.85 * x[1], 0.5)
            @observe observation ~ Normal(x[1])
            @track saved_x = x
            return x
        end
        model = correlated_lognormal(1.0)
        @testset "$sampler" for sampler in (rwm, ram, am, dram)
            chn = sampler(Xoshiro(468), model, 3000, 2; warmup=500, progress=false)
            x = chn[@vn(x[1])]
            residual = log.(chn[@vn(y)]) .- 0.85 .* x
            test_moments(x, 0.5, 0.5)
            test_moments(residual, 0.0, 0.25)
            cross = (x .- 0.5) .* residual
            @test abs(mean(cross)) < 5 * mcse(cross)
            @test chn[@vn(saved_x)] == chn[:return_value] == chn[@vn(x)]
            if sampler === dram
                @test any(==(2), chn[:accepted_stage])
            end
        end
    end

    @testset "seeded sampling is reproducible, with independent chains" begin
        @wren function normal_model()
            x ~ Normal()
        end
        model = normal_model()
        niters, nchains = 100, 2
        @testset "$name" for (name, sampler) in (
            "nuts" =>
                (rng, m; progress=false) ->
                    nuts(rng, m, niters, nchains; adtype=AutoForwardDiff(), progress),
            "prior" => (rng, m; progress=false) -> prior(rng, m, niters, nchains; progress),
            "independent MH" =>
                (rng, m; progress=false) ->
                    independent_mh(rng, m, niters, nchains; progress),
            "DRAM" =>
                (rng, m; progress=false) ->
                    dram(rng, m, niters, nchains; warmup=50, progress),
        )
            chn1 = sampler(Xoshiro(468), model)
            chn2 = sampler(Xoshiro(468), model)
            @test FlexiChains.has_same_data(chn1, chn2)
            displayed = redirect_stdout(devnull) do
                redirect_stderr(devnull) do
                    sampler(Xoshiro(468), model; progress=true)
                end
            end
            @test FlexiChains.has_same_data(chn1, displayed)
            chn3 = sampler(Xoshiro(469), model)
            @test !FlexiChains.has_same_data(chn1, chn3)
            x = chn1[@vn(x)]
            @test size(x) == (niters, nchains)
            @test x[:, 1] != x[:, 2]
        end
    end

    @testset "failed chain waits for its sibling" begin
        @test_throws CompositeException redirect_stdout(devnull) do
            redirect_stderr(devnull) do
                prior(Xoshiro(468), failing_prior_model(), 1, 2; progress=true)
            end
        end
        @test sibling_finished[]
    end
end

end # module WrenSamplingTests
