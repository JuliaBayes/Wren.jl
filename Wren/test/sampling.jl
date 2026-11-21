module WrenSamplingTests

@info "testing sampling.jl"

using Wren
using Wren.Dists
using FlexiChains: FlexiChains
using Random: Xoshiro
using ForwardDiff: ForwardDiff
using Test

const failure_worker = Threads.Atomic{Int}(0)
const sibling_finished = Threads.Atomic{Bool}(false)
const sibling_running = Base.Event()

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
        )
            chn1 = sampler(Xoshiro(468), model)
            chn2 = sampler(Xoshiro(468), model)
            @test FlexiChains.has_same_data(chn1, chn2)
            @testset "$make_progress" for make_progress in (
                () -> true,
                TermBackend,
                ProgressLoggingBackend,
                PlainTextBackend,
                () -> (TermBackend(), PlainTextBackend()),
            )
                displayed = redirect_stdout(devnull) do
                    redirect_stderr(devnull) do
                        sampler(Xoshiro(468), model; progress=make_progress())
                    end
                end
                @test FlexiChains.has_same_data(chn1, displayed)
            end
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
                prior(Xoshiro(468), failing_prior_model(), 1, 2)
            end
        end
        @test sibling_finished[]
    end
end

end # module WrenSamplingTests
