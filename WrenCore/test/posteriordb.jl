module WrenPosteriorDBTests

@info "testing posteriordb.jl"

import DifferentiationInterface as DI
import ForwardDiff
using Distributions: Binomial, Cauchy, Gamma, Normal, logpdf
using Enzyme
using PosteriorDB: PosteriorDB
using StableRNGs
using Test
using WrenCore
using WrenDists: WrenDists

@testset "posteriordb.jl" begin
    @testset "$name" for name in available_posteriordb_models()
        model = posteriordb_model(name)
        vm = VecModel(model, DI.AutoForwardDiff())
        if name == "dugongs_data-dugongs_model"
            # Gamma(shape=0.0001) prior draws commonly underflow in Float64.
            v = [2.0, 1.0, 0.0, 0.0]
            @test isfinite(lp(vm, v))
        else
            @test isfinite(logjoint(model, rand(StableRNG(468), model)))
            v = rand(StableRNG(468), vm)
        end
        @test length(v) == vm.dimension
        logp, grad = lp_and_grad(vm, v)
        @test isfinite(logp)
        @test length(grad) == length(v)
        @test all(isfinite, grad)
    end

    @test_throws ArgumentError posteriordb_model("not-a-posterior")

    @testset "classical PosteriorDB densities and native Enzyme gradients" begin
        logistic(x) = inv(1 + exp(-x))
        logit_jacobian(x) = -log1p(exp(-x)) - log1p(exp(x))
        function check(name, dimension, reference, points)
            vm = VecModel(posteriordb_model(name), AutoEnzyme(; mode=Reverse))
            @test vm.dimension == dimension
            for point in points
                value, gradient = lp_and_grad(vm, point)
                @test isapprox(value, reference(point); atol=1e-6)
                @test isapprox(gradient, ForwardDiff.gradient(reference, point); atol=1e-6)
            end
        end

        seed_data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
            PosteriorDB.database(), "seeds_data-seeds_model",
        )))
        k, N, x1, x2 = seed_data["n"], seed_data["N"], seed_data["x1"], seed_data["x2"]
        nobs = length(k)
        log_choose = [logpdf(Binomial(N[i], 0.5), k[i]) + N[i] * log(2)
            for i in eachindex(k)]
        function seed_likelihood(alpha, b)
            eta = alpha[1] .+ alpha[2] .* x1 .+ alpha[4] .* x2 .+
                alpha[3] .* (x1 .* x2) .+ b
            return sum(log_choose[i] + k[i] * eta[i] - N[i] * log1p(exp(eta[i]))
                for i in eachindex(k))
        end
        function seeds_reference(v)
            alpha = view(v, 1:4)
            tau = exp(v[5])
            sigma = inv(sqrt(tau))
            b = view(v, 6:length(v))
            return sum(logpdf(Normal(0, 1000), a) for a in alpha) +
                logpdf(Gamma(0.001, 1000), tau) + v[5] +
                sum(logpdf(Normal(0, sigma), bi) for bi in b) +
                seed_likelihood(alpha, b)
        end
        check("seeds_data-seeds_model", nobs + 5, seeds_reference,
            [vcat([0.1, -0.2, 0.15, 0.3, -0.1], 0.05 .* sin.(1:nobs)),
             vcat([-0.2, 0.3, -0.1, 0.1, 0.2], 0.04 .* cos.(1:nobs))])

        function seeds_cauchy_reference(v, centered)
            alpha = view(v, 1:4)
            c = view(v, 5:(4 + nobs))
            sigma = exp(v[end])
            b = centered ? c .- sum(c) / nobs : c
            return sum(logpdf(Normal(), a) for a in alpha) +
                logpdf(Cauchy(0, 1), sigma) + log(2) + v[end] +
                sum(logpdf(Normal(0, sigma), ci) for ci in c) +
                seed_likelihood(alpha, b)
        end
        for (name, centered) in (
            ("seeds_data-seeds_centered_model", true),
            ("seeds_data-seeds_stanified_model", false),
        )
            reference(v) = seeds_cauchy_reference(v, centered)
            check(name, nobs + 5, reference,
                [vcat([0.1, -0.2, 0.15, 0.3], 0.05 .* sin.(1:nobs), -0.1),
                 vcat([-0.2, 0.3, -0.1, 0.1], 0.04 .* cos.(1:nobs), 0.2)])
        end

        dugong_data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
            PosteriorDB.database(), "dugongs_data-dugongs_model",
        )))
        function dugongs_reference(v)
            alpha, beta = v[1], v[2]
            lambda = 0.5 + 0.5 * logistic(v[3])
            tau = exp(v[4])
            sigma = inv(sqrt(tau))
            x, y = dugong_data["x"], dugong_data["Y"]
            return logpdf(Normal(0, 1000), alpha) +
                logpdf(Normal(0, 1000), beta) + logit_jacobian(v[3]) +
                logpdf(Gamma(0.0001, 10000), tau) + v[4] +
                sum(logpdf(Normal(alpha - beta * lambda^x[i], sigma), y[i])
                    for i in eachindex(y))
        end
        check("dugongs_data-dugongs_model", 4, dugongs_reference,
            [[2.0, 1.0, -0.3, -0.1], [1.5, 0.6, 0.2, 0.3]])

        mixture_data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
            PosteriorDB.database(), "normal_2-normal_mixture",
        )))
        function normal_mixture_reference(v)
            theta = logistic(v[1])
            mu1, mu2 = v[2], v[3]
            return logit_jacobian(v[1]) +
                logpdf(Normal(0, 10), mu1) + logpdf(Normal(0, 10), mu2) +
                sum(log(theta * exp(logpdf(Normal(mu1, 1), yi)) +
                        (1 - theta) * exp(logpdf(Normal(mu2, 1), yi)))
                    for yi in mixture_data["y"])
        end
        check("normal_2-normal_mixture", 3, normal_mixture_reference,
            [[0.2, -0.8, 0.9], [-0.3, 0.4, -0.5]])
    end
end

end # module WrenPosteriorDBTests
