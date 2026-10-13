module WrenPosteriorDBTests

@info "testing posteriordb.jl"

import DifferentiationInterface as DI
import ForwardDiff
using Distributions: Beta, Binomial, Cauchy, Gamma, LocationScale, LogNormal, Normal,
    Poisson, TDist, logpdf
using Enzyme
using LinearAlgebra: Symmetric, I, cholesky
using PosteriorDB: PosteriorDB
using StableRNGs
using Statistics: mean
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
        elseif name == "mcycle_splines-accel_splines"
            # An unconstrained prior draw can produce extreme spline predictors
            # with -Inf Normal likelihood in Float64. Use a finite reference point.
            v = vcat([-12.0, 0.1], zeros(38), [log(5.0), 0.5, -0.1],
                     zeros(38), [log(2.0)])
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

    @testset "additional Gaussian mixture densities and native Enzyme gradients" begin
        logistic(x) = inv(1 + exp(-x))
        logit_jacobian(x) = -log1p(exp(-x)) - log1p(exp(x))
        low_dim_name = "low_dim_gauss_mix_collapse-low_dim_gauss_mix_collapse"
        low_dim_data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
            PosteriorDB.database(), low_dim_name,
        )))
        function low_dim_reference(v)
            mu = view(v, 1:2)
            sigma = exp.(view(v, 3:4))
            theta = logistic(v[5])
            return sum(logpdf(Normal(0, 2), m) for m in mu) +
                sum(logpdf(Normal(0, 2), sigma[i]) + log(2) + v[i + 2]
                    for i in 1:2) +
                logpdf(Beta(5, 5), theta) + logit_jacobian(v[5]) +
                sum(log(theta * exp(logpdf(Normal(mu[1], sigma[1]), yi)) +
                        (1 - theta) * exp(logpdf(Normal(mu[2], sigma[2]), yi)))
                    for yi in low_dim_data["y"])
        end
        low_dim_vm = VecModel(posteriordb_model(low_dim_name), AutoEnzyme(; mode=Reverse))
        @test low_dim_vm.dimension == 5
        for point in (zeros(5), [0.2, -0.7, -0.1, 0.3, -0.4],
                      [-0.5, 0.6, 0.2, -0.2, 0.3])
            value, gradient = lp_and_grad(low_dim_vm, point)
            @test isapprox(value, low_dim_reference(point); atol=1e-6)
            @test isapprox(gradient, ForwardDiff.gradient(low_dim_reference, point);
                atol=1e-6)
        end

        ordered_name = "low_dim_gauss_mix-low_dim_gauss_mix"
        ordered_data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
            PosteriorDB.database(), ordered_name,
        )))
        function ordered_reference(v)
            mu = [v[1], v[1] + exp(v[2])]
            sigma = exp.(view(v, 3:4))
            theta = logistic(v[5])
            return sum(logpdf(Normal(0, 2), m) for m in mu) + v[2] +
                sum(logpdf(Normal(0, 2), sigma[i]) + log(2) + v[i + 2]
                    for i in 1:2) +
                logpdf(Beta(5, 5), theta) + logit_jacobian(v[5]) +
                sum(log(theta * exp(logpdf(Normal(mu[1], sigma[1]), yi)) +
                        (1 - theta) * exp(logpdf(Normal(mu[2], sigma[2]), yi)))
                    for yi in ordered_data["y"])
        end
        ordered_vm = VecModel(posteriordb_model(ordered_name), AutoEnzyme(; mode=Reverse))
        @test ordered_vm.dimension == 5
        for point in ([0.0, 0.0, -0.1, 0.2, 0.0],
                      [-0.4, 0.3, 0.1, -0.2, 0.5])
            value, gradient = lp_and_grad(ordered_vm, point)
            @test isapprox(value, ordered_reference(point); atol=1e-6)
            @test isapprox(gradient, ForwardDiff.gradient(ordered_reference, point);
                atol=1e-6)
        end

        mixture_name = "normal_5-normal_mixture_k"
        mixture_data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
            PosteriorDB.database(), mixture_name,
        )))
        extension = Base.get_extension(WrenCore, :WrenCorePosteriorDBExt)
        theta = [0.08, 0.12, 0.20, 0.25, 0.35]
        mu = [-1.1, -0.4, 0.3, 0.9, 1.5]
        sigma = [0.7, 0.8, 1.0, 1.2, 1.4]
        expected_likelihood = sum(log(sum(theta[k] *
            exp(logpdf(Normal(mu[k], sigma[k]), yi)) for k in eachindex(theta)))
            for yi in mixture_data["y"])
        @test isapprox(extension.NormalMixtureK.mixture_k_logp(
            theta, mu, sigma, mixture_data["y"]), expected_likelihood; atol=1e-6)
        model = posteriordb_model(mixture_name)
        enzyme_vm = VecModel(model, AutoEnzyme(; mode=Reverse))
        forward_vm = VecModel(model, DI.AutoForwardDiff())
        @test enzyme_vm.dimension == forward_vm.dimension == 14
        for point in (zeros(14), vcat([0.2, -0.1, 0.15, -0.2], mu,
                                    [-0.1, 0.2, -0.2, 0.1, 0.3]))
            value, gradient = lp_and_grad(enzyme_vm, point)
            @test isfinite(value)
            @test isapprox(value, lp(forward_vm, point); atol=1e-6)
            @test isapprox(gradient, ForwardDiff.gradient(
                v -> lp(forward_vm, v), point); atol=1e-6)
        end
    end

    @testset "continuous PosteriorDB densities and native Enzyme gradients" begin
        function check(name, dimension, reference, points)
            vm = VecModel(posteriordb_model(name), AutoEnzyme(; mode=Reverse))
            @test vm.dimension == dimension
            for point in points
                value, gradient = lp_and_grad(vm, point)
                @test isapprox(value, reference(point); atol=1e-6)
                @test isapprox(gradient, ForwardDiff.gradient(reference, point);
                    atol=1e-6)
            end
        end

        sblrc_data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
            PosteriorDB.database(), "sblrc-blr",
        )))
        function sblrc_reference(v)
            beta = view(v, 1:5)
            sigma = exp(v[6])
            mu = sblrc_data["X"] * beta
            return sum(logpdf(Normal(0, 10), b) for b in beta) +
                logpdf(Normal(0, 10), sigma) + log(2) + v[6] +
                sum(logpdf(Normal(mu[i], sigma), sblrc_data["y"][i])
                    for i in eachindex(mu))
        end
        @test sblrc_data["D"] == 5
        @test sblrc_data["N"] == length(sblrc_data["y"]) == 100
        check("sblrc-blr", 6, sblrc_reference,
            [[0.1, -0.2, 0.3, 0.05, -0.1, -0.2],
             [-0.1, 0.15, -0.05, 0.2, 0.1, 0.3]])

        gp_data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
            PosteriorDB.database(), "gp_pois_regr-gp_pois_regr",
        )))
        gp_dist_sq = (gp_data["x"] .- gp_data["x"]') .^ 2
        function gp_pois_reference(v)
            rho, alpha = exp(v[1]), exp(v[2])
            f_tilde = view(v, 3:length(v))
            covariance = alpha^2 .* exp.(-0.5 .* gp_dist_sq ./ rho^2) +
                1e-10I
            f = cholesky(Symmetric(covariance)).L * f_tilde
            return logpdf(Gamma(25, 1 / 4), rho) + v[1] +
                logpdf(Normal(0, 2), alpha) + log(2) + v[2] +
                sum(logpdf(Normal(), ft) for ft in f_tilde) +
                sum(logpdf(Poisson(exp(f[i])), gp_data["k"][i])
                    for i in eachindex(f))
        end
        @test gp_data["N"] == 11
        check("gp_pois_regr-gp_pois_regr", 13, gp_pois_reference,
            [vcat([log(5.0), log(2.0)], zeros(11)),
             vcat([log(6.0), log(1.5)], 0.1 .* sin.(1:11))])

        spline_data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
            PosteriorDB.database(), "mcycle_splines-accel_splines",
        )))
        function spline_reference(v)
            intercept = v[1]
            bs = view(v, 2:2)
            zs = view(v, 3:40)
            sds = exp(v[41])
            intercept_sigma = v[42]
            bs_sigma = view(v, 43:43)
            zs_sigma = view(v, 44:81)
            sds_sigma = exp(v[82])
            total = logpdf(LocationScale(-13, 36, TDist(3)), intercept) +
                sum(logpdf(Normal(), z) for z in zs) +
                logpdf(LocationScale(0, 36, TDist(3)), sds) + log(2) + v[41] +
                logpdf(LocationScale(0, 10, TDist(3)), intercept_sigma) +
                sum(logpdf(Normal(), z) for z in zs_sigma) +
                logpdf(LocationScale(0, 36, TDist(3)), sds_sigma) + log(2) + v[82]
            if spline_data["prior_only"] == 0
                mu = intercept .+ spline_data["Xs"] * bs .+
                    spline_data["Zs_1_1"] * (sds .* zs)
                log_sigma = intercept_sigma .+ spline_data["Xs_sigma"] * bs_sigma .+
                    spline_data["Zs_sigma_1_1"] * (sds_sigma .* zs_sigma)
                total += sum(logpdf(Normal(mu[i], exp(log_sigma[i])),
                    spline_data["Y"][i]) for i in eachindex(mu))
            end
            return total
        end
        @test spline_data["Ks"] == spline_data["Ks_sigma"] == 1
        @test spline_data["knots_1"] == spline_data["knots_sigma_1"] == 38
        check("mcycle_splines-accel_splines", 82, spline_reference,
            [vcat([-12.0, 0.1], zeros(38), [log(5.0), 0.5, -0.1],
                  zeros(38), [log(2.0)]),
             vcat([-10.0, -0.1], 0.01 .* sin.(1:38),
                  [log(4.0), 0.2, 0.05], 0.01 .* cos.(1:38), [log(1.5)])])

        arma_data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
            PosteriorDB.database(), "arma-arma11",
        )))
        function arma_reference(v)
            mu, phi, theta, sigma = v[1], v[2], v[3], exp(v[4])
            y = arma_data["y"]
            errors = [y[1] - (mu + phi * mu)]
            for t in 2:length(y)
                push!(errors, y[t] - (mu + phi * y[t - 1] + theta * errors[end]))
            end
            return logpdf(Normal(0, 10), mu) + logpdf(Normal(0, 2), phi) +
                logpdf(Normal(0, 2), theta) + logpdf(Cauchy(0, 2.5), sigma) +
                log(2) + v[4] + sum(logpdf(Normal(0, sigma), e) for e in errors)
        end
        check("arma-arma11", 4, arma_reference,
            [[0.2, 0.5, -0.3, -0.2], [-0.1, 0.3, 0.2, 0.1]])

        loss_data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
            PosteriorDB.database(), "loss_curves-losscurve_sislob",
        )))
        function loss_reference(v)
            omega, theta = exp(v[1]), exp(v[2])
            mu_LR, sd_LR = v[3], exp(v[4])
            LR = exp.(view(v, 5:14))
            loss_sd = exp(v[15])
            total = logpdf(LogNormal(0, 0.5), omega) + v[1] +
                logpdf(LogNormal(0, 0.5), theta) + v[2] +
                logpdf(Normal(0, 0.5), mu_LR) +
                logpdf(LogNormal(0, 0.5), sd_LR) + v[4] +
                sum(logpdf(LogNormal(mu_LR, sd_LR), LR[i]) + v[i + 4]
                    for i in eachindex(LR)) +
                logpdf(LogNormal(0, 0.7), loss_sd) + v[15]
            for i in eachindex(loss_data["loss"])
                cohort = loss_data["cohort_id"][i]
                t = loss_data["t_value"][loss_data["t_idx"][i]]
                growth = 1 - exp(-(t / theta)^omega)
                total += logpdf(Normal(LR[cohort] * loss_data["premium"][cohort] *
                    growth, loss_sd * loss_data["premium"][cohort]),
                    loss_data["loss"][i])
            end
            return total
        end
        @test loss_data["growthmodel_id"] == 1
        check("loss_curves-losscurve_sislob", 15, loss_reference,
            [vcat([0.0, 0.2, -0.1, -0.3], zeros(10), [-0.4]),
             vcat([0.1, -0.2, 0.2, 0.1], fill(-0.2, 10), [0.2])])

        diamonds_data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
            PosteriorDB.database(), "diamonds-diamonds",
        )))
        X = diamonds_data["X"]
        Xc = X[:, 2:end] .- mean(X[:, 2:end]; dims=1)
        function diamonds_reference(v)
            b = view(v, 1:24)
            intercept, sigma = v[25], exp(v[26])
            total = sum(logpdf(Normal(), bi) for bi in b) +
                logpdf(LocationScale(8, 10, TDist(3)), intercept) +
                logpdf(LocationScale(0, 10, TDist(3)), sigma) + log(2) + v[26]
            if diamonds_data["prior_only"] == 0
                total += sum(logpdf(Normal(intercept + sum(Xc[i, j] * b[j]
                    for j in eachindex(b)), sigma), diamonds_data["Y"][i])
                    for i in eachindex(diamonds_data["Y"]))
            end
            return total
        end
        @test diamonds_data["K"] == 25
        check("diamonds-diamonds", 26, diamonds_reference,
            [vcat(zeros(24), [8.0, log(10.0)])])
    end
end

end # module WrenPosteriorDBTests
