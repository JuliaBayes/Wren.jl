module WrenPosteriorDBTests

@info "testing posteriordb.jl"

import DifferentiationInterface as DI
using Distributions: Binomial, Normal, logpdf
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

    @testset "capture-recapture densities and native Enzyme gradients" begin
        logistic(x) = inv(1 + exp(-x))
        logit_jacobian(x) = -log1p(exp(-x)) - log1p(exp(x))
        bernoulli_logit(y, eta) = y * eta - log1p(exp(eta))
        logsumexp2(a, b) = max(a, b) + log1p(exp(-abs(a - b)))

        function check(name, dimension, reference, points)
            vm = VecModel(posteriordb_model(name), AutoEnzyme(; mode=Reverse))
            @test vm.dimension == dimension
            for point in points
                value, gradient = lp_and_grad(vm, point)
                @test isapprox(value, reference(point); atol=1e-6)
                @test isapprox(gradient, ForwardDiff.gradient(reference, point); atol=1e-6)
            end
        end

        function capture_data(name)
            data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
                PosteriorDB.database(), name,
            )))
            y = data["y"]
            return y, vec(sum(y; dims=2))
        end

        function marginalized(omega, rowlogp, counts)
            present, absent = log(omega), log1p(-omega)
            return sum(counts[i] > 0 ? present + rowlogp[i] :
                logsumexp2(present + rowlogp[i], absent) for i in eachindex(counts))
        end

        y0, counts0 = capture_data("M0_data-M0_model")
        trials0 = size(y0, 2)
        function m0_reference(v)
            omega, p = logistic.(v)
            rowlogp = [logpdf(Binomial(trials0, p), count) for count in counts0]
            return sum(logit_jacobian, v) + marginalized(omega, rowlogp, counts0)
        end
        check("M0_data-M0_model", 2, m0_reference, [[0.1, -0.2], [-0.3, 0.2]])

        yb, countsb = capture_data("Mb_data-Mb_model")
        function mb_reference(v)
            omega, p, c = logistic.(v)
            rowlogp = [sum(begin
                prob = j == 1 || yb[i, j - 1] == 0 ? p : c
                yb[i, j] == 1 ? log(prob) : log1p(-prob)
            end for j in axes(yb, 2)) for i in axes(yb, 1)]
            return sum(logit_jacobian, v) + marginalized(omega, rowlogp, countsb)
        end
        check("Mb_data-Mb_model", 3, mb_reference,
            [[0.1, -0.2, 0.3], [-0.3, 0.2, -0.1]])

        yt, countst = capture_data("Mt_data-Mt_model")
        trials_t = size(yt, 2)
        function mt_reference(v)
            omega = logistic(v[1])
            p = logistic.(view(v, 2:length(v)))
            rowlogp = [sum(yt[i, j] == 1 ? log(p[j]) : log1p(-p[j])
                for j in axes(yt, 2)) for i in axes(yt, 1)]
            return sum(logit_jacobian, v) + marginalized(omega, rowlogp, countst)
        end
        check("Mt_data-Mt_model", trials_t + 1, mt_reference,
            [0.2 .* sin.(1:(trials_t + 1)), -0.2 .* cos.(1:(trials_t + 1))])

        yh, countsh = capture_data("Mth_data-Mth_model")
        m_h, t_h = size(yh)
        function mth_reference(v)
            omega = logistic(v[1])
            mean_lp = view(v, 2:(t_h + 1))
            sigma = 5logistic(v[t_h + 2])
            eps_raw = view(v, (t_h + 3):length(v))
            rowlogp = [sum(bernoulli_logit(yh[i, j], mean_lp[j] + sigma * eps_raw[i])
                for j in axes(yh, 2)) for i in axes(yh, 1)]
            return sum(logit_jacobian, view(v, 1:(t_h + 2))) +
                sum(logpdf(Normal(), e) for e in eps_raw) +
                marginalized(omega, rowlogp, countsh)
        end
        check("Mth_data-Mth_model", m_h + t_h + 2, mth_reference,
            [vcat(0.2 .* sin.(1:(t_h + 2)), 0.03 .* cos.(1:m_h))])

        ybh, countsbh = capture_data("Mtbh_data-Mtbh_model")
        m_bh, t_bh = size(ybh)
        function mtbh_reference(v)
            omega = logistic(v[1])
            mean_lp = view(v, 2:(t_bh + 1))
            gamma = v[t_bh + 2]
            sigma = 3logistic(v[t_bh + 3])
            eps_raw = view(v, (t_bh + 4):length(v))
            rowlogp = [sum(bernoulli_logit(ybh[i, j],
                mean_lp[j] + sigma * eps_raw[i] +
                (j == 1 ? 0.0 : gamma * ybh[i, j - 1]))
                for j in axes(ybh, 2)) for i in axes(ybh, 1)]
            return sum(logit_jacobian, view(v, 1:(t_bh + 1))) +
                logpdf(Normal(0, 10), gamma) + logit_jacobian(v[t_bh + 3]) +
                sum(logpdf(Normal(), e) for e in eps_raw) +
                marginalized(omega, rowlogp, countsbh)
        end
        check("Mtbh_data-Mtbh_model", m_bh + t_bh + 3, mtbh_reference,
            [vcat(0.2 .* cos.(1:(t_bh + 3)), 0.03 .* sin.(1:m_bh))])
    end
end

end # module WrenPosteriorDBTests
