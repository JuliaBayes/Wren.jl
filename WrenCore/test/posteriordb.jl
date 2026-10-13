module WrenPosteriorDBTests

@info "testing posteriordb.jl"

import DifferentiationInterface as DI
using Distributions: Binomial, logpdf
using Enzyme
import ForwardDiff
using PosteriorDB: PosteriorDB
using StableRNGs
using Test
using WrenCore
using WrenDists: WrenDists

@testset "posteriordb.jl" begin
    @testset "$name" for name in available_posteriordb_models()
        model = posteriordb_model(name)
        vm = VecModel(model, DI.AutoForwardDiff())
        # dogs_log uses flat coordinates with bounded likelihood factors; a
        # prior draw need not lie in their finite-density support.
        if name == "dogs-dogs_log"
            v = [-0.2, 0.3]
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

    @testset "discrete PosteriorDB densities and native Enzyme gradients" begin
        logistic(x) = inv(1 + exp(-x))
        logit_jacobian(x) = -log1p(exp(-x)) - log1p(exp(x))
        bernoulli_logit(y, eta) = y * eta - log1p(exp(eta))
        function check(name, dimension, oracle, points)
            vm = VecModel(posteriordb_model(name), AutoEnzyme(; mode=Reverse))
            @test vm.dimension == dimension
            for point in points
                value, gradient = lp_and_grad(vm, point)
                @test isapprox(value, oracle(point); atol=1e-5)
                @test isapprox(gradient, ForwardDiff.gradient(oracle, point); atol=1e-5)
            end
        end

        dog_data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
            PosteriorDB.database(), "dogs-dogs",
        )))
        dog_y = dog_data["y"]
        dog_count, trials = size(dog_y)
        dog_shocks = [sum((dog_y[j, s] for s in 1:(t - 1)); init=0)
            for t in 1:trials for j in 1:dog_count]
        dog_avoids = [t - 1 - sum((dog_y[j, s] for s in 1:(t - 1)); init=0)
            for t in 1:trials for j in 1:dog_count]
        dog_response = vec(dog_y)

        function dogs_oracle(v)
            eta = v[1] .+ v[2] .* dog_avoids .+ v[3] .* dog_shocks
            return -3log(100sqrt(2π)) - sum(abs2, v) / 20000 +
                sum(bernoulli_logit.(dog_response, eta))
        end
        check("dogs-dogs", 3, dogs_oracle, [[0.2, -0.1, 0.1], [-0.1, 0.1, -0.1]])

        function dogs_hierarchical_oracle(v)
            a = logistic(v[1])
            b = logistic(v[2])
            p = a .^ dog_shocks .* b .^ dog_avoids
            keep = findall(>(0), dog_shocks .+ dog_avoids)
            return logit_jacobian(v[1]) + logit_jacobian(v[2]) +
                sum(dog_response[i] == 1 ? log(p[i]) : log1p(-p[i]) for i in keep)
        end
        check("dogs-dogs_hierarchical", 2, dogs_hierarchical_oracle,
            [[0.2, -0.3], [-0.2, 0.1]])

        function dogs_log_oracle(v)
            eta = v[1] .* dog_avoids .+ v[2] .* dog_shocks
            return -2log(100) + sum(bernoulli_logit.(dog_response, eta))
        end
        check("dogs-dogs_log", 2, dogs_log_oracle,
            [[-0.2, 0.3], [-0.1, 0.1]])

        glmm_data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
            PosteriorDB.database(), "GLMM_data-GLMM1_model",
        )))
        nsite = glmm_data["nsite"]
        obs = glmm_data["obs"]
        obssite = glmm_data["obssite"]
        log_factorial = [sum((log(i) for i in 1:k); init=0.0) for k in obs]
        function glmm_oracle(v)
            alpha = view(v, 1:nsite)
            mu_alpha = v[nsite + 1]
            sd_alpha = 5logistic(v[nsite + 2])
            eta = alpha[obssite]
            return -nsite * log(sd_alpha * sqrt(2π)) -
                sum(abs2, alpha .- mu_alpha) / (2sd_alpha^2) -
                log(10sqrt(2π)) - mu_alpha^2 / 200 +
                logit_jacobian(v[nsite + 2]) +
                sum(obs .* eta .- exp.(eta) .- log_factorial)
        end
        check("GLMM_data-GLMM1_model", nsite + 2, glmm_oracle,
            [vcat(0.04 .* sin.(1:nsite), 0.2, -0.1),
             vcat(0.03 .* cos.(1:nsite), -0.1, 0.2)])

        nes_data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
            PosteriorDB.database(), "nes_logit_data-nes_logit_model",
        )))
        function nes_oracle(v)
            eta = v[1] .+ v[2] .* nes_data["income"]
            return sum(bernoulli_logit.(nes_data["vote"], eta))
        end
        check("nes_logit_data-nes_logit_model", 2, nes_oracle,
            [[0.2, -0.1], [-0.1, 0.1]])

        survey_data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
            PosteriorDB.database(), "Survey_data-Survey_model",
        )))
        k = survey_data["k"]
        nmax = survey_data["nmax"]
        nmin = maximum(k)
        function survey_oracle(v)
            theta = logistic(v[1])
            parts = [sum(logpdf(Binomial(n, theta), ki) for ki in k) - log(nmax)
                for n in nmin:nmax]
            mixture = first(parts)
            for part in Iterators.drop(parts, 1)
                mixture = max(mixture, part) + log1p(exp(-abs(mixture - part)))
            end
            return mixture + logit_jacobian(v[1])
        end
        check("Survey_data-Survey_model", 1, survey_oracle, [[0.2], [-0.3]])
    end
end

end # module WrenPosteriorDBTests
