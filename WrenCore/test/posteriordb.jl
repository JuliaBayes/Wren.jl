module WrenPosteriorDBTests

@info "testing posteriordb.jl"

import DifferentiationInterface as DI
import ForwardDiff
using Distributions: Beta, Binomial, Cauchy, Exponential, Gamma, InverseGamma,
    Laplace, LocationScale, LogNormal, NegativeBinomial, Normal, Poisson,
    TDist, logpdf
using Enzyme
using LinearAlgebra: Symmetric, I, cholesky
using PosteriorDB: PosteriorDB
using StableRNGs
using Statistics: mean, std
using Test
using WrenCore
using WrenDists: WrenDists

@testset "posteriordb.jl" begin
    covid_point() = vcat(fill(log(3.28), 14), fill(log(0.1), 6), [log(0.5)],
        fill(log(10.0), 14), [log(5.0), log(10.0)], zeros(14))

    @testset "$name" for name in available_posteriordb_models()
        model = posteriordb_model(name)
        adtype = occursin("covid19imperial", name) || name in
            ("ovarian-logistic_regression_rhs",
            "prostate-logistic_regression_rhs") ?
            AutoEnzyme(; mode=Reverse) : DI.AutoForwardDiff()
        vm = occursin("covid19imperial", name) ?
            VecModel(model, adtype; reduce_copies=false) : VecModel(model, adtype)
        if occursin("covid19imperial", name)
            v = covid_point()
        elseif name == "dugongs_data-dugongs_model"
            # Gamma(shape=0.0001) prior draws commonly underflow in Float64.
            v = [2.0, 1.0, 0.0, 0.0]
            @test isfinite(lp(vm, v))
        elseif name == "surgical_data-surgical_model"
            # Gamma(shape=0.001) precision draws can underflow to zero.
            v = vcat(-3.0, 0.0, fill(-3.0, 12))
            @test isfinite(lp(vm, v))
        elseif name == "mcycle_splines-accel_splines"
            # An unconstrained prior draw can produce extreme spline predictors
            # with -Inf Normal likelihood in Float64. Use a finite reference point.
            v = vcat([-12.0, 0.1], zeros(38), [log(5.0), 0.5, -0.1],
                     zeros(38), [log(2.0)])
            @test isfinite(lp(vm, v))
        elseif name in ("ovarian-logistic_regression_rhs",
                "prostate-logistic_regression_rhs")
            posterior = PosteriorDB.posterior(PosteriorDB.database(), name)
            data = PosteriorDB.load(PosteriorDB.dataset(posterior))
            d = data["d"]
            v = vcat(0.0, zeros(d), log(2 * data["scale_global"]),
                zeros(d), 0.0)
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

    @testset "Imperial COVID-19 densities and native Enzyme gradients" begin
        for name in ("ecdc0401-covid19imperial_v2",
                "ecdc0501-covid19imperial_v3")
            data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
                PosteriorDB.database(), name,
            )))
            function covid_oracle(v)
                mu = exp.(view(v, 1:14))
                alpha_hier = exp.(view(v, 15:20))
                kappa = exp(v[21])
                infections = exp.(view(v, 22:35))
                phi, tau = exp(v[36]), exp(v[37])
                ifr_noise = exp.(view(v, 38:51))
                alpha = alpha_hier .- log(1.05) / 6
                density = logpdf(Exponential(inv(0.03)), tau) + v[37] +
                    logpdf(Normal(0, 5), phi) + v[36] +
                    logpdf(Normal(0, 0.5), kappa) + v[21]
                density += sum(logpdf(Gamma(0.1667, 1), alpha_hier[j]) +
                    v[14 + j] for j in 1:6)
                for m in 1:14
                    density += logpdf(Normal(3.28, kappa), mu[m]) + v[m] +
                        logpdf(Exponential(tau), infections[m]) + v[21 + m] +
                        logpdf(Normal(1, 0.1), ifr_noise[m]) + v[37 + m]
                    prediction = fill(infections[m], data["N0"])
                    for day in (data["N0"] + 1):data["N2"]
                        convolution = sum(prediction[previous] *
                            data["SI"][day - previous] for previous in 1:(day - 1))
                        rt = mu[m] * exp(-sum(data["X"][m, day, j] * alpha[j]
                            for j in 1:6))
                        push!(prediction, (1 - sum(prediction) / data["pop"][m]) *
                            rt * convolution)
                    end
                    for day in data["EpidemicStart"][m]:data["N"][m]
                        expected = ifr_noise[m] * sum(prediction[previous] *
                            data["f"][day - previous, m]
                            for previous in 1:(day - 1))
                        density += logpdf(NegativeBinomial(phi,
                            phi / (phi + expected)), data["deaths"][day, m])
                    end
                end
                return density
            end
            point = covid_point()
            vm = VecModel(posteriordb_model(name), AutoEnzyme(; mode=Reverse);
                reduce_copies=false)
            @test vm.dimension == 51
            value, gradient = lp_and_grad(vm, point)
            @test isapprox(value, covid_oracle(point); atol=1e-5)
            if name == "ecdc0401-covid19imperial_v2"
                selected = [1, 15, 21, 22, 36, 37, 51]
                positions = zeros(Int, length(point))
                for (k, i) in enumerate(selected)
                    positions[i] = k
                end
                function covid_selected_oracle(z)
                    full = [positions[i] == 0 ? convert(eltype(z), point[i]) :
                        z[positions[i]] for i in eachindex(point)]
                    return covid_oracle(full)
                end
                selected_gradient = ForwardDiff.gradient(covid_selected_oracle,
                    point[selected])
                @test isapprox(gradient[selected], selected_gradient;
                    atol=1e-4, rtol=1e-7)
            end
        end
    end

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

        sesame_data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
            PosteriorDB.database(), "sesame_data-sesame_one_pred_a",
        )))
        function sesame_reference(v)
            sigma = exp(v[3])
            return v[3] + sum(logpdf(Normal(v[1] + v[2] *
                sesame_data["encouraged"][i], sigma), sesame_data["watched"][i])
                for i in eachindex(sesame_data["watched"]))
        end
        check("sesame_data-sesame_one_pred_a", 3, sesame_reference,
            [[0.2, 0.3, log(0.8)], [-0.1, 0.5, log(1.2)]])

        garch_data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
            PosteriorDB.database(), "garch-garch11",
        )))
        function garch_reference(v)
            mu = v[1]
            alpha0 = exp(v[2])
            alpha1 = logistic(v[3])
            beta_ratio = logistic(v[4])
            beta1 = (1 - alpha1) * beta_ratio
            jacobian = v[2] + log(alpha1) + 2 * log1p(-alpha1) +
                log(beta_ratio) + log1p(-beta_ratio)
            y = garch_data["y"]
            sigma = garch_data["sigma1"]
            density = logpdf(Normal(mu, sigma), y[1])
            for t in 2:length(y)
                sigma = sqrt(alpha0 + alpha1 * (y[t - 1] - mu)^2 +
                    beta1 * sigma^2)
                density += logpdf(Normal(mu, sigma), y[t])
            end
            return jacobian + density
        end
        check("garch-garch11", 4, garch_reference,
            [[0.0, log(0.2), -0.2, 0.1], [0.15, log(0.5), 0.3, -0.4]])

        surgical_data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
            PosteriorDB.database(), "surgical_data-surgical_model",
        )))
        function surgical_reference(v)
            mu = v[1]
            log_precision = v[2]
            sigmasq = exp(-log_precision)
            sigma = sqrt(sigmasq)
            b = view(v, 3:length(v))
            return logpdf(Normal(0, 1000), mu) +
                logpdf(InverseGamma(0.001, 0.001), sigmasq) - log_precision +
                sum(logpdf(Normal(mu, sigma), bi) for bi in b) +
                sum(logpdf(Binomial(surgical_data["n"][i], logistic(b[i])),
                    surgical_data["r"][i]) for i in eachindex(b))
        end
        check("surgical_data-surgical_model", 14, surgical_reference,
            [vcat(-3.0, 0.0, -3.0 .+ 0.1 .* sin.(1:12)),
             vcat(-2.5, -0.5, -2.5 .+ 0.1 .* cos.(1:12))])

        prophet_data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
            PosteriorDB.database(), "rstan_downloads-prophet",
        )))
        n_changes, n_beta = prophet_data["S"], prophet_data["K"]
        function prophet_reference(v)
            k, m = v[1], v[2]
            delta = view(v, 3:(2 + n_changes))
            log_sigma = v[3 + n_changes]
            sigma = exp(log_sigma)
            beta = view(v, (4 + n_changes):length(v))
            t, t_change = prophet_data["t"], prophet_data["t_change"]
            X = prophet_data["X"]
            A = Float64.(t .>= t_change')
            trend = (k .+ A * delta) .* t .+ m .- A * (t_change .* delta)
            mu = trend .* (1 .+ X * (beta .* prophet_data["s_m"])) .+
                X * (beta .* prophet_data["s_a"])
            return logpdf(Normal(0, 5), k) + logpdf(Normal(0, 5), m) +
                sum(logpdf(Laplace(0, prophet_data["tau"]), d) for d in delta) +
                logpdf(Normal(0, 0.5), sigma) + log_sigma +
                sum(logpdf(Normal(0, prophet_data["sigmas"][j]), beta[j])
                    for j in eachindex(beta)) +
                sum(logpdf(Normal(mu[i], sigma), prophet_data["y"][i])
                    for i in eachindex(mu))
        end
        check("rstan_downloads-prophet", 3 + n_changes + n_beta,
            prophet_reference,
            [vcat(0.1, 0.5, 0.01 .* sin.(1:n_changes), log(0.3),
                0.01 .* cos.(1:n_beta)),
             vcat(-0.1, 0.6, 0.02 .* cos.(1:n_changes), log(0.5),
                -0.01 .* sin.(1:n_beta))])

        mcycle_gp_data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
            PosteriorDB.database(), "mcycle_gp-accel_gp",
        )))
        function mcycle_gp_reference(v)
            intercept, sdgp, lscale = v[1], exp(v[2]), exp(v[3])
            zgp = view(v, 4:43)
            intercept_sigma, sdgp_sigma, lscale_sigma =
                v[44], exp(v[45]), exp(v[46])
            zgp_sigma = view(v, 47:66)
            alpha, beta = 1.124909, 0.0177
            student_t(mu, scale, x) = logpdf(TDist(3), (x - mu) / scale) - log(scale)
            prior = student_t(-13, 36, intercept) +
                student_t(0, 36, sdgp) + log(2.0) + v[2] +
                logpdf(InverseGamma(alpha, beta), lscale) + v[3] +
                sum(logpdf(Normal(), zi) for zi in zgp) +
                student_t(0, 10, intercept_sigma) +
                student_t(0, 36, sdgp_sigma) + log(2.0) + v[45] +
                logpdf(InverseGamma(alpha, beta), lscale_sigma) + v[46] +
                sum(logpdf(Normal(), zi) for zi in zgp_sigma)
            function gp_values(X, frequencies, z, sd, lengthscale)
                weights = [sqrt(sd^2 * sqrt(2 * pi) * lengthscale *
                    exp(-0.5 * lengthscale^2 * frequencies[m]^2)) * z[m]
                    for m in eachindex(z)]
                return [sum(X[n, m] * weights[m] for m in eachindex(z))
                    for n in axes(X, 1)]
            end
            mean_gp = gp_values(mcycle_gp_data["Xgp_1"],
                vec(mcycle_gp_data["slambda_1"]), zgp, sdgp, lscale)
            scale_gp = gp_values(mcycle_gp_data["Xgp_sigma_1"],
                vec(mcycle_gp_data["slambda_sigma_1"]), zgp_sigma,
                sdgp_sigma, lscale_sigma)
            likelihood = sum(logpdf(Normal(intercept + mean_gp[n],
                exp(intercept_sigma + scale_gp[n])), mcycle_gp_data["Y"][n])
                for n in eachindex(mean_gp))
            return prior + likelihood
        end
        check("mcycle_gp-accel_gp", 66, mcycle_gp_reference,
            [vcat([-13.0, log(10.0), log(0.05)], 0.05 .* sin.(1:40),
                [3.0, log(2.0), log(0.05)], 0.05 .* cos.(1:20))])

        uk_data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
            PosteriorDB.database(),
            "uk_drivers-state_space_stochastic_level_stochastic_seasonal",
        )))
        function uk_drivers_reference(v)
            n = uk_data["n"]
            y, x, w = uk_data["y"], uk_data["x"], uk_data["w"]
            lower = mean(y) - 3 * std(y)
            width = 6 * std(y)
            mu = lower .+ width .* logistic.(view(v, 1:n))
            seasonal = view(v, (n + 1):(2n))
            beta, lambda = v[2n + 1], v[2n + 2]
            sigma1 = exp(v[2n + 3])
            sigma2 = sigma1 + exp(v[2n + 4])
            sigma3 = sigma2 + exp(v[2n + 5])
            prior = sum(log(width) + logit_jacobian(v[t]) for t in 1:n) +
                sum(logpdf(Normal(-sum(seasonal[(t - 11):(t - 1)]), sigma1),
                    seasonal[t]) for t in 12:n) +
                sum(logpdf(Normal(mu[t - 1], sigma2), mu[t]) for t in 2:n) +
                sum(logpdf(Normal(mu[t] + beta * x[t] + lambda * w[t] +
                    seasonal[t], sigma3), y[t]) for t in 1:n)
            prior += sum(logpdf(TDist(4), s) for s in (sigma1, sigma2, sigma3)) +
                sum(v[(2n + 3):(2n + 5)])
            return prior
        end
        n_uk = uk_data["n"]
        uk_point = vcat(0.02 .* sin.(1:n_uk), 0.01 .* cos.(1:n_uk),
            [0.0, 0.0, log(0.05), log(0.05), log(0.1)])
        check("uk_drivers-state_space_stochastic_level_stochastic_seasonal",
            2n_uk + 5, uk_drivers_reference, [uk_point])

        for name in ("ovarian-logistic_regression_rhs",
                "prostate-logistic_regression_rhs")
            data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
                PosteriorDB.database(), name,
            )))
            d, n = data["d"], data["n"]
            function rhs_reference(v)
                beta0 = v[1]
                z = view(v, 2:(d + 1))
                tau = exp(v[d + 2])
                lambda = exp.(view(v, (d + 3):(2d + 2)))
                caux = exp(v[end])
                prior = logpdf(Normal(0, data["scale_icept"]), beta0) +
                    sum(logpdf(Normal(), zj) for zj in z) +
                    logpdf(TDist(data["nu_global"]),
                        tau / (2 * data["scale_global"])) -
                    log(2 * data["scale_global"]) + v[d + 2] +
                    sum(logpdf(TDist(data["nu_local"]), lambda[j]) +
                        v[d + 2 + j] for j in 1:d) +
                    logpdf(InverseGamma(data["slab_df"] / 2,
                        data["slab_df"] / 2), caux) + v[end]
                c2 = data["slab_scale"]^2 * caux
                beta = [z[j] * tau * sqrt(c2 * lambda[j]^2 /
                    (c2 + tau^2 * lambda[j]^2)) for j in 1:d]
                likelihood = zero(prior)
                for i in 1:n
                    eta = beta0 + sum(data["x"][i, j] * beta[j] for j in 1:d)
                    likelihood += data["y"][i] * eta - log1p(exp(eta))
                end
                return prior + likelihood
            end
            point = vcat(0.1, 0.01 .* sin.(1:d),
                log(2 * data["scale_global"]),
                0.02 .* cos.(1:d), 0.0)
            vm = VecModel(posteriordb_model(name), AutoEnzyme(; mode=Reverse))
            @test vm.dimension == length(point)
            value, gradient = lp_and_grad(vm, point)
            @test isapprox(value, rhs_reference(point); atol=1e-5, rtol=1e-8)
            selected = [1, 2, d + 1, d + 2, d + 3, 2d + 2, 2d + 3]
            positions = zeros(Int, length(point))
            for (k, i) in enumerate(selected)
                positions[i] = k
            end
            function rhs_selected_reference(s)
                full = [positions[i] == 0 ? convert(eltype(s), point[i]) :
                    s[positions[i]] for i in eachindex(point)]
                return rhs_reference(full)
            end
            selected_gradient = ForwardDiff.gradient(rhs_selected_reference,
                point[selected])
            @test isapprox(gradient[selected], selected_gradient;
                atol=1e-5, rtol=1e-6)
        end
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

        soil_data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
            PosteriorDB.database(), "soil_carbon-soil_incubation",
        )))
        # Integrate the original two-pool equations independently of the
        # closed-form implementation. RK4 with a 0.05-day step resolves this
        # public 42-day fixture well below the density tolerance.
        function soil_reference(v)
            k1, k2, alpha21, alpha12 = exp.(view(v, 1:4))
            gamma, sigma = inv(1 + exp(-v[5])), exp(v[6])
            total = soil_data["totalC_t0"]
            c1, c2 = gamma * total, (1 - gamma) * total
            previous = soil_data["t0"]
            function rate(a, b)
                return (-k1 * a + alpha12 * k2 * b,
                    alpha21 * k1 * a - k2 * b)
            end
            density = sum(logpdf(Normal(), p) + raw for (p, raw) in
                zip((k1, k2, alpha21, alpha12), view(v, 1:4))) +
                logpdf(Beta(10, 1), gamma) - log1p(exp(-v[5])) -
                log1p(exp(v[5])) +
                logpdf(Cauchy(0, 1), sigma) + v[6]
            for i in eachindex(soil_data["ts"])
                current = soil_data["ts"][i]
                steps = ceil(Int, (current - previous) / 0.05)
                h = (current - previous) / steps
                for _ in 1:steps
                    a1, b1 = rate(c1, c2)
                    a2, b2 = rate(c1 + h * a1 / 2, c2 + h * b1 / 2)
                    a3, b3 = rate(c1 + h * a2 / 2, c2 + h * b2 / 2)
                    a4, b4 = rate(c1 + h * a3, c2 + h * b3)
                    c1 += h * (a1 + 2a2 + 2a3 + a4) / 6
                    c2 += h * (b1 + 2b2 + 2b3 + b4) / 6
                end
                density += logpdf(Normal(total - c1 - c2, sigma),
                    soil_data["eCO2mean"][i])
                previous = current
            end
            return density
        end
        @test soil_data["N_t"] == 25
        check("soil_carbon-soil_incubation", 6, soil_reference,
            [[log(0.2), log(0.02), log(0.3), log(0.1), 1.0, log(0.4)],
             [log(0.1), log(0.03), log(0.2), log(0.15), 0.8, log(0.6)]])
    end
end

end # module WrenPosteriorDBTests
