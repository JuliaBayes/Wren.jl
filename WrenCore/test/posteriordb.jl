module WrenPosteriorDBTests

@info "testing posteriordb.jl"

import DifferentiationInterface as DI
using Distributions: Bernoulli, Beta, Binomial, Cauchy, Exponential, Logistic,
    LogNormal, Normal, TDist, logpdf
using Enzyme
import ForwardDiff
using PosteriorDB: PosteriorDB
using Statistics: mean
using StableRNGs
using Test
using WrenCore
using WrenDists: WrenDists

@testset "posteriordb.jl" begin
    @testset "$name" for name in available_posteriordb_models()
        model = posteriordb_model(name)
        # LSAT has 1000 person effects; ForwardDiff's full Jacobian is too large
        # for the generic smoke test. Its targeted checks use native Enzyme below.
        adtype = name in ("lsat_data-lsat_model", "election88-election88_full",
            "sat-hier_2pl", "fims_Aus_Jpn_irt-2pl_latent_reg_irt",
            "science_irt-grsm_latent_reg_irt",
            "timssAusTwn_irt-gpcm_latent_reg_irt") ?
            AutoEnzyme(; mode=Reverse) :
            DI.AutoForwardDiff()
        vm = VecModel(model, adtype)
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
        logaddexp(a, b) = a > b ? a + log1p(exp(b - a)) : b + log1p(exp(a - b))
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

        election_data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
            PosteriorDB.database(), "election88-election88_full",
        )))
        na, nb, nc, nd, ne = (election_data[k] for k in
            ("n_age", "n_edu", "n_age_edu", "n_state", "n_region_full"))
        function election_oracle(v)
            beta = view(v, 1:5)
            raw = view(v, 6:10)
            scales = 100 ./ (1 .+ exp.(-raw))
            a = view(v, 11:(10 + na))
            b = view(v, (11 + na):(10 + na + nb))
            c = view(v, (11 + na + nb):(10 + na + nb + nc))
            d = view(v, (11 + na + nb + nc):(10 + na + nb + nc + nd))
            e = view(v, (11 + na + nb + nc + nd):length(v))
            eta = beta[1] .+ beta[2] .* election_data["black"] .+
                beta[3] .* election_data["female"] .+
                beta[5] .* (election_data["female"] .* election_data["black"]) .+
                beta[4] .* election_data["v_prev_full"] .+
                a[election_data["age"]] .+ b[election_data["edu"]] .+
                c[election_data["age_edu"]] .+ d[election_data["state"]] .+
                e[election_data["region_full"]]
            prior = sum(logpdf(Normal(0, 100), bi) for bi in beta) +
                5 * log(100.0) + sum(logit_jacobian(ui) for ui in raw) +
                sum(logpdf(Normal(0, scales[1]), ai) for ai in a) +
                sum(logpdf(Normal(0, scales[2]), bi) for bi in b) +
                sum(logpdf(Normal(0, scales[3]), ci) for ci in c) +
                sum(logpdf(Normal(0, scales[4]), di) for di in d) +
                sum(logpdf(Normal(0, scales[5]), ei) for ei in e)
            return prior + sum(bernoulli_logit.(election_data["y"], eta))
        end
        election_point = vcat([0.2, 0.1, -0.1, 0.2, 0.1],
            fill(log(1.0 / 99.0), 5), 0.05 .* sin.(1:(na + nb + nc + nd + ne)))
        check("election88-election88_full", 10 + na + nb + nc + nd + ne,
            election_oracle, [election_point])

        hmm_data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
            PosteriorDB.database(), "hmm_example-hmm_example",
        )))
        function hmm_oracle(v)
            p11, p21 = logistic(v[1]), logistic(v[2])
            mu1 = exp(v[3])
            mu2 = mu1 + exp(v[4])
            y = hmm_data["y"]
            g1 = logpdf(Normal(mu1, 1), y[1])
            g2 = logpdf(Normal(mu2, 1), y[1])
            for t in 2:length(y)
                next1 = logaddexp(g1 + log(p11), g2 + log(p21)) +
                    logpdf(Normal(mu1, 1), y[t])
                next2 = logaddexp(g1 + log1p(-p11), g2 + log1p(-p21)) +
                    logpdf(Normal(mu2, 1), y[t])
                g1, g2 = next1, next2
            end
            return logit_jacobian(v[1]) + logit_jacobian(v[2]) +
                v[3] + v[4] + logpdf(Normal(3, 1), mu1) +
                logpdf(Normal(10, 1), mu2) + logaddexp(g1, g2)
        end
        check("hmm_example-hmm_example", 4, hmm_oracle,
            [[0.8, -1.2, log(3.0), log(7.0)],
             [-0.3, 0.4, log(2.5), log(6.5)]])

        hmm_gaussian_data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
            PosteriorDB.database(), "hmm_gaussian_simulated-hmm_gaussian",
        )))
        function simplex3_oracle(a, b)
            z1 = logistic(a - log(2))
            z2 = logistic(b)
            probabilities = [z1, (1 - z1) * z2, (1 - z1) * (1 - z2)]
            jacobian = logit_jacobian(a - log(2)) + log1p(-z1) +
                logit_jacobian(b)
            return probabilities, jacobian
        end
        function hmm_gaussian_oracle(v)
            pi, jacobian = simplex3_oracle(v[1], v[2])
            rows = [simplex3_oracle(v[2i + 1], v[2i + 2]) for i in 1:3]
            A = [rows[i][1][j] for i in 1:3, j in 1:3]
            jacobian += sum(row[2] for row in rows)
            mu = [v[9], v[9] + exp(v[10]), v[9] + exp(v[10]) + exp(v[11])]
            sigma = exp.(v[12:14])
            jacobian += v[10] + v[11] + sum(v[12:14])
            y = hmm_gaussian_data["y"]
            alpha = [log(pi[j]) + logpdf(Normal(mu[j], sigma[j]), y[1]) for j in 1:3]
            for t in 2:length(y)
                alpha = [logpdf(Normal(mu[j], sigma[j]), y[t]) +
                    logaddexp(logaddexp(alpha[1] + log(A[1, j]),
                        alpha[2] + log(A[2, j])), alpha[3] + log(A[3, j]))
                    for j in 1:3]
            end
            return jacobian + logaddexp(logaddexp(alpha[1], alpha[2]), alpha[3])
        end
        check("hmm_gaussian_simulated-hmm_gaussian", 14, hmm_gaussian_oracle,
            [[0.2, -0.4, 0.5, -0.3, -0.2, 0.4, 0.1, -0.1,
              10.0, log(10.0), log(10.0), log(5.0), log(5.0), log(5.0)],
             [-0.2, 0.3, -0.4, 0.2, 0.3, -0.5, -0.1, 0.4,
              11.0, log(10.0), log(10.0), log(6.0), log(5.0), log(4.0)]])

        bball_data = [PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
            PosteriorDB.database(), "bball_drive_event_$(i)-hmm_drive_$(i)",
        ))) for i in 0:1]
        function bball_oracle(v, fixture, gaussian)
            p11, p21 = logistic(v[1]), logistic(v[2])
            phi1 = gaussian ? v[3] : exp(v[3])
            phi2 = phi1 + exp(v[4])
            lambda1 = gaussian ? v[5] : exp(v[5])
            lambda2 = lambda1 + exp(v[6])
            prior = logpdf(Beta(4, 2), p11) + logpdf(Beta(2, 4), p21) +
                logit_jacobian(v[1]) + logit_jacobian(v[2]) +
                v[4] + v[6] + (gaussian ? zero(v[3]) : v[3] + v[5]) +
                logpdf(Normal(0, 1), phi1) + logpdf(Normal(3, 1), phi2) +
                logpdf(Normal(0, 1), lambda1) + logpdf(Normal(3, 1), lambda2)
            function emission(t, k)
                phi, lambda = k == 1 ? (phi1, lambda1) : (phi2, lambda2)
                if gaussian
                    return logpdf(Normal(phi, fixture["tau"]), fixture["u"][t]) +
                        logpdf(Normal(lambda, fixture["rho"]), fixture["v"][t])
                end
                return logpdf(Exponential(inv(phi)), fixture["u"][t]) +
                    logpdf(Exponential(inv(lambda)), fixture["v"][t])
            end
            g1, g2 = emission(1, 1), emission(1, 2)
            for t in 2:fixture["N"]
                next1 = emission(t, 1) + logaddexp(g1 + log(p11), g2 + log(p21))
                next2 = emission(t, 2) +
                    logaddexp(g1 + log1p(-p11), g2 + log1p(-p21))
                g1, g2 = next1, next2
            end
            return prior + logaddexp(g1, g2)
        end
        check("bball_drive_event_0-hmm_drive_0", 6,
            v -> bball_oracle(v, bball_data[1], false),
            [[0.7, -0.5, log(6.0), log(2.0), log(0.02), log(0.03)],
             [-0.1, 0.3, log(5.0), log(2.5), log(0.025), log(0.02)]])
        check("bball_drive_event_1-hmm_drive_1", 6,
            v -> bball_oracle(v, bball_data[2], true),
            [[0.7, -0.5, -1.8, log(1.5), 3.9, log(1.0)],
             [-0.1, 0.3, -1.7, log(1.4), 3.8, log(1.2)]])

        iohmm_data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
            PosteriorDB.database(), "iohmm_reg_simulated-iohmm_reg",
        )))
        function iohmm_oracle(v)
            pi, jacobian = simplex3_oracle(v[1], v[2])
            w = reshape(v[3:14], 4, 3)
            b = reshape(v[15:26], 4, 3)
            sigma = exp.(v[27:29])
            jacobian += sum(v[27:29])
            prior = sum(logpdf(Normal(0, 5), vi) for vi in v[3:26]) +
                sum(logpdf(Normal(0, 3), si) for si in sigma)
            u, y = iohmm_data["u"], iohmm_data["y"]
            regress(t, coefficients, j) =
                sum(u[t, m] * coefficients[m, j] for m in 1:4)
            g = [log(pi[j]) + logpdf(Normal(regress(1, b, j), sigma[j]), y[1])
                for j in 1:3]
            for t in 2:length(y)
                logits = [regress(t, w, j) for j in 1:3]
                norm = logaddexp(logaddexp(logits[1], logits[2]), logits[3])
                # Match the Stan source's destination-independent logA[t][i].
                common = logaddexp(logaddexp(g[1] + logits[1] - norm,
                    g[2] + logits[2] - norm), g[3] + logits[3] - norm)
                g = [common + logpdf(Normal(regress(t, b, j), sigma[j]), y[t])
                    for j in 1:3]
            end
            return prior + jacobian + logaddexp(logaddexp(g[1], g[2]), g[3])
        end
        check("iohmm_reg_simulated-iohmm_reg", 29, iohmm_oracle,
            [vcat([0.2, -0.4], 0.1 .* sin.(1:24), fill(log(3.0), 3)),
             vcat([-0.3, 0.5], 0.15 .* cos.(1:24),
                 [log(2.5), log(3.5), log(4.0)])])

        butterfly_data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
            PosteriorDB.database(), "butterfly-multi_occupancy",
        )))
        function butterfly_oracle(v)
            alpha, beta = v[1], v[2]
            omega = logistic(v[3])
            rho_unit = logistic(v[4])
            rho = 2 * rho_unit - 1
            sigma1, sigma2 = exp(v[5]), exp(v[6])
            S, n, J, K = (butterfly_data[key] for key in ("S", "n", "J", "K"))
            uv1 = view(v, 7:(6 + S))
            uv2 = view(v, (7 + S):length(v))
            prior = logpdf(Cauchy(0, 2.5), alpha) +
                logpdf(Cauchy(0, 2.5), beta) +
                logpdf(Beta(2, 2), omega) + logit_jacobian(v[3]) +
                logpdf(Beta(2, 2), rho_unit) + log(2.0) +
                logit_jacobian(v[4]) +
                logpdf(Cauchy(0, 2.5), sigma1) + v[5] +
                logpdf(Cauchy(0, 2.5), sigma2) + v[6]
            likelihood = zero(prior)
            for i in 1:S
                z1 = uv1[i] / sigma1
                z2 = (uv2[i] / sigma2 - rho * z1) / sqrt(1 - rho^2)
                prior += logpdf(Normal(), z1) + logpdf(Normal(), z2) -
                    log(sigma1) - log(sigma2) - 0.5 * log1p(-rho^2)
                psi = logistic(alpha + uv1[i])
                theta = logistic(beta + uv2[i])
                never_at_site = logaddexp(log(psi) +
                    logpdf(Binomial(K, theta), 0), log1p(-psi))
                if i <= n
                    likelihood += log(omega)
                    for j in 1:J
                        count = butterfly_data["X"][i, j]
                        likelihood += count > 0 ?
                            log(psi) + logpdf(Binomial(K, theta), count) :
                            never_at_site
                    end
                else
                    likelihood += logaddexp(log1p(-omega),
                        log(omega) + J * never_at_site)
                end
            end
            return prior + likelihood
        end
        butterfly_S = butterfly_data["S"]
        butterfly_point = vcat([0.1, -0.2, 0.2, 0.1, log(0.8), log(1.2)],
            0.03 .* sin.(1:butterfly_S), 0.03 .* cos.(1:butterfly_S))
        check("butterfly-multi_occupancy", 6 + 2butterfly_S,
            butterfly_oracle, [butterfly_point])

        dog_nonhier_data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
            PosteriorDB.database(), "dogs-dogs_nonhierarchical",
        )))
        function dog_nonhier_oracle(v)
            mu1, mu2 = v[1], v[2]
            sigma1, sigma2 = exp(v[3]), exp(v[4])
            rho = tanh(v[5])
            J = dog_nonhier_data["n_dogs"]
            z1 = view(v, 6:(5 + J))
            z2 = view(v, (6 + J):length(v))
            prior = logpdf(Logistic(), mu1) + logpdf(Logistic(), mu2) +
                logpdf(Normal(), sigma1) + v[3] +
                logpdf(Normal(), sigma2) + v[4] +
                log(0.75) + 2 * log1p(-rho^2) +
                sum(logpdf(Normal(), zi) for zi in z1) +
                sum(logpdf(Normal(), zi) for zi in z2)
            likelihood = zero(prior)
            for j in 1:J
                logit_a = mu1 + sigma1 * z1[j] + sigma2 * rho * z2[j]
                logit_b = mu2 + sigma2 * sqrt(1 - rho^2) * z2[j]
                a, b = logistic(logit_a), logistic(logit_b)
                shocks = 0
                avoids = 0
                for t in 1:dog_nonhier_data["n_trials"]
                    likelihood += logpdf(Bernoulli(a^shocks * b^avoids),
                        dog_nonhier_data["y"][j, t])
                    shocks += dog_nonhier_data["y"][j, t]
                    avoids += 1 - dog_nonhier_data["y"][j, t]
                end
            end
            return prior + likelihood
        end
        dog_nonhier_J = dog_nonhier_data["n_dogs"]
        dog_nonhier_point = vcat([-0.3, 0.2, log(0.5), log(0.6), 0.1],
            0.05 .* sin.(1:dog_nonhier_J), 0.05 .* cos.(1:dog_nonhier_J))
        check("dogs-dogs_nonhierarchical", 5 + 2dog_nonhier_J,
            dog_nonhier_oracle, [dog_nonhier_point])

        sat_data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
            PosteriorDB.database(), "sat-hier_2pl",
        )))
        function sat_oracle(v)
            I, J = sat_data["I"], sat_data["J"]
            theta = view(v, 1:J)
            xi1 = view(v, (J + 1):(J + I))
            xi2 = view(v, (J + I + 1):(J + 2I))
            mu1, mu2 = v[J + 2I + 1], v[J + 2I + 2]
            tau1, tau2 = exp(v[J + 2I + 3]), exp(v[J + 2I + 4])
            rho = tanh(v[J + 2I + 5])
            prior = sum(logpdf(Normal(), tj) for tj in theta) +
                logpdf(Normal(), mu1) + logpdf(Normal(0, 5), mu2) +
                logpdf(Exponential(10), tau1) + v[J + 2I + 3] +
                logpdf(Exponential(10), tau2) + v[J + 2I + 4] +
                log(35 / 32) + 4 * log1p(-rho^2)
            for i in 1:I
                z1 = (xi1[i] - mu1) / tau1
                z2 = ((xi2[i] - mu2) / tau2 - rho * z1) / sqrt(1 - rho^2)
                prior += logpdf(Normal(), z1) + logpdf(Normal(), z2) -
                    log(tau1) - log(tau2) - 0.5 * log1p(-rho^2)
            end
            likelihood = sum(logpdf(Bernoulli(logistic(
                exp(xi1[sat_data["ii"][n]]) *
                (theta[sat_data["jj"][n]] - xi2[sat_data["ii"][n]]))),
                sat_data["y"][n]) for n in 1:sat_data["N"])
            return prior + likelihood
        end
        sat_I, sat_J = sat_data["I"], sat_data["J"]
        sat_point = vcat(0.2 .* sin.(1:sat_J),
            0.03 .* sin.(1:sat_I), 0.03 .* cos.(1:sat_I),
            [0.0, 0.0, log(0.5), log(1.0), 0.1])
        sat_vm = VecModel(posteriordb_model("sat-hier_2pl"),
            AutoEnzyme(; mode=Reverse))
        @test sat_vm.dimension == length(sat_point)
        sat_value, sat_gradient = lp_and_grad(sat_vm, sat_point)
        @test isapprox(sat_value, sat_oracle(sat_point); atol=1e-5)
        selected = [1, sat_J, sat_J + 1, sat_J + sat_I,
            sat_J + sat_I + 1, sat_J + 2sat_I,
            sat_J + 2sat_I + 1, sat_J + 2sat_I + 2,
            sat_J + 2sat_I + 3, sat_J + 2sat_I + 4,
            sat_J + 2sat_I + 5]
        positions = zeros(Int, length(sat_point))
        for (k, i) in enumerate(selected)
            positions[i] = k
        end
        function sat_selected_oracle(z)
            full = [positions[i] == 0 ? convert(eltype(z), sat_point[i]) :
                z[positions[i]] for i in eachindex(sat_point)]
            return sat_oracle(full)
        end
        sat_selected_gradient = ForwardDiff.gradient(sat_selected_oracle,
            sat_point[selected])
        @test isapprox(sat_gradient[selected], sat_selected_gradient; atol=1e-5)

        fims_data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
            PosteriorDB.database(), "fims_Aus_Jpn_irt-2pl_latent_reg_irt",
        )))
        fims_I, fims_J, fims_K = (fims_data[k] for k in ("I", "J", "K"))
        fims_W = fims_data["W"]
        @test all(fims_W[:, 1] .== 1)
        @test all(k -> all(x -> x == 0 || x == 1, fims_W[:, k]), 2:fims_K)
        fims_W_adj = Float64.(fims_W)
        for k in 2:fims_K
            fims_W_adj[:, k] .-= sum(fims_W[:, k]) / fims_J
        end
        function fims_oracle(v)
            alpha = exp.(view(v, 1:fims_I))
            beta_free = view(v, (fims_I + 1):(2fims_I - 1))
            beta = vcat(beta_free, -sum(beta_free))
            theta = view(v, (2fims_I):(2fims_I + fims_J - 1))
            lambda = view(v, (2fims_I + fims_J):length(v))
            prior = sum(logpdf(LogNormal(1, 1), alpha[i]) + v[i]
                for i in eachindex(alpha)) +
                sum(logpdf(Normal(0, 3), b) for b in beta) +
                sum(logpdf(TDist(3), l) for l in lambda) +
                sum(logpdf(Normal(sum(fims_W_adj[j, k] * lambda[k]
                    for k in 1:fims_K), 1), theta[j]) for j in 1:fims_J)
            likelihood = sum(logpdf(Bernoulli(logistic(
                alpha[fims_data["ii"][n]] * theta[fims_data["jj"][n]] -
                beta[fims_data["ii"][n]])), fims_data["y"][n])
                for n in 1:fims_data["N"])
            return prior + likelihood
        end
        fims_point = vcat(0.03 .* sin.(1:fims_I),
            0.1 .* cos.(1:(fims_I - 1)),
            0.1 .* sin.(1:fims_J), [0.1, -0.2, 0.3, -0.1])
        fims_vm = VecModel(posteriordb_model(
            "fims_Aus_Jpn_irt-2pl_latent_reg_irt"), AutoEnzyme(; mode=Reverse))
        @test fims_vm.dimension == length(fims_point)
        fims_value, fims_gradient = lp_and_grad(fims_vm, fims_point)
        @test isapprox(fims_value, fims_oracle(fims_point); atol=1e-5)
        fims_selected = [1, fims_I, fims_I + 1, 2fims_I - 1,
            2fims_I, 2fims_I + fims_J - 1,
            2fims_I + fims_J, 2fims_I + fims_J + 1,
            2fims_I + fims_J + 2, 2fims_I + fims_J + 3]
        fims_positions = zeros(Int, length(fims_point))
        for (k, i) in enumerate(fims_selected)
            fims_positions[i] = k
        end
        function fims_selected_oracle(z)
            full = [fims_positions[i] == 0 ? convert(eltype(z), fims_point[i]) :
                z[fims_positions[i]] for i in eachindex(fims_point)]
            return fims_oracle(full)
        end
        fims_selected_gradient = ForwardDiff.gradient(fims_selected_oracle,
            fims_point[fims_selected])
        @test isapprox(fims_gradient[fims_selected], fims_selected_gradient;
            atol=1e-5)

        science_data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
            PosteriorDB.database(), "science_irt-grsm_latent_reg_irt",
        )))
        science_I, science_J = science_data["I"], science_data["J"]
        science_m = maximum(science_data["y"])
        @test science_data["K"] == 1
        @test all(science_data["W"] .== 1)
        function science_oracle(v)
            alpha = exp.(view(v, 1:science_I))
            beta_free = view(v, (science_I + 1):(2science_I - 1))
            beta = vcat(beta_free, -sum(beta_free))
            kappa_free = view(v, (2science_I):(2science_I + science_m - 2))
            kappa = vcat(kappa_free, -sum(kappa_free))
            theta = view(v, (2science_I + science_m - 1):
                (2science_I + science_m + science_J - 2))
            lambda = v[end]
            prior = sum(logpdf(LogNormal(1, 1), alpha[i]) + v[i]
                for i in eachindex(alpha)) +
                sum(logpdf(Normal(0, 3), b) for b in beta) +
                sum(logpdf(Normal(0, 3), k) for k in kappa) +
                logpdf(TDist(3), lambda) +
                sum(logpdf(Normal(lambda, 1), t) for t in theta)
            likelihood = zero(prior)
            for n in 1:science_data["N"]
                i, j = science_data["ii"][n], science_data["jj"][n]
                eta = theta[j] * alpha[i] - beta[i]
                scores = cumsum(vcat(0.0, eta .- kappa))
                likelihood += scores[science_data["y"][n] + 1] -
                    log(sum(exp, scores))
            end
            return prior + likelihood
        end
        science_point = vcat(1 .+ 0.03 .* sin.(1:science_I),
            0.05 .* cos.(1:(science_I - 1)),
            0.05 .* sin.(1:(science_m - 1)),
            0.1 .* sin.(1:science_J), [0.1])
        science_vm = VecModel(posteriordb_model(
            "science_irt-grsm_latent_reg_irt"), AutoEnzyme(; mode=Reverse))
        @test science_vm.dimension == length(science_point)
        science_value, science_gradient = lp_and_grad(science_vm, science_point)
        @test isapprox(science_value, science_oracle(science_point); atol=1e-5)
        science_selected = [1, science_I, science_I + 1, 2science_I - 1,
            2science_I, 2science_I + science_m - 2,
            2science_I + science_m - 1,
            2science_I + science_m + science_J - 2, length(science_point)]
        science_positions = zeros(Int, length(science_point))
        for (k, i) in enumerate(science_selected)
            science_positions[i] = k
        end
        function science_selected_oracle(z)
            full = [science_positions[i] == 0 ?
                convert(eltype(z), science_point[i]) : z[science_positions[i]]
                for i in eachindex(science_point)]
            return science_oracle(full)
        end
        science_selected_gradient = ForwardDiff.gradient(science_selected_oracle,
            science_point[science_selected])
        @test isapprox(science_gradient[science_selected],
            science_selected_gradient; atol=1e-5)

        timss_data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
            PosteriorDB.database(), "timssAusTwn_irt-gpcm_latent_reg_irt",
        )))
        timss_I, timss_J, timss_K = (timss_data[k] for k in ("I", "J", "K"))
        timss_m = [maximum(timss_data["y"][n]
            for n in 1:timss_data["N"] if timss_data["ii"][n] == i)
            for i in 1:timss_I]
        timss_pos = cumsum([1; timss_m[1:(end - 1)]])
        timss_steps = sum(timss_m)
        @test all(timss_data["W"][:, 1] .== 1)
        @test all(k -> all(x -> x == 0 || x == 1, timss_data["W"][:, k]),
            2:timss_K)
        timss_W_adj = Float64.(timss_data["W"])
        for k in 2:timss_K
            timss_W_adj[:, k] .-= sum(timss_data["W"][:, k]) / timss_J
        end
        function timss_oracle(v)
            alpha = exp.(view(v, 1:timss_I))
            beta_free = view(v, (timss_I + 1):(timss_I + timss_steps - 1))
            beta = vcat(beta_free, -sum(beta_free))
            theta = view(v, (timss_I + timss_steps):
                (timss_I + timss_steps + timss_J - 1))
            lambda = view(v, (timss_I + timss_steps + timss_J):length(v))
            prior = sum(logpdf(LogNormal(1, 1), alpha[i]) + v[i]
                for i in eachindex(alpha)) +
                sum(logpdf(Normal(0, 3), b) for b in beta) +
                sum(logpdf(TDist(3), l) for l in lambda) +
                sum(logpdf(Normal(sum(timss_W_adj[j, k] * lambda[k]
                    for k in 1:timss_K), 1), theta[j]) for j in 1:timss_J)
            likelihood = zero(prior)
            for n in 1:timss_data["N"]
                i, j = timss_data["ii"][n], timss_data["jj"][n]
                steps = view(beta, timss_pos[i]:(timss_pos[i] + timss_m[i] - 1))
                scores = cumsum(vcat(0.0, theta[j] * alpha[i] .- steps))
                likelihood += scores[timss_data["y"][n] + 1] -
                    log(sum(exp, scores))
            end
            return prior + likelihood
        end
        timss_point = vcat(1 .+ 0.03 .* sin.(1:timss_I),
            0.05 .* cos.(1:(timss_steps - 1)),
            0.1 .* sin.(1:timss_J), 0.1 .* cos.(1:timss_K))
        timss_vm = VecModel(posteriordb_model(
            "timssAusTwn_irt-gpcm_latent_reg_irt"), AutoEnzyme(; mode=Reverse))
        @test timss_vm.dimension == length(timss_point)
        timss_value, timss_gradient = lp_and_grad(timss_vm, timss_point)
        @test isapprox(timss_value, timss_oracle(timss_point); atol=1e-5)
        timss_selected = [1, timss_I, timss_I + 1,
            timss_I + timss_steps - 1, timss_I + timss_steps,
            timss_I + timss_steps + timss_J - 1,
            timss_I + timss_steps + timss_J, length(timss_point)]
        timss_positions = zeros(Int, length(timss_point))
        for (k, i) in enumerate(timss_selected)
            timss_positions[i] = k
        end
        function timss_selected_oracle(z)
            full = [timss_positions[i] == 0 ? convert(eltype(z), timss_point[i]) :
                z[timss_positions[i]] for i in eachindex(timss_point)]
            return timss_oracle(full)
        end
        timss_selected_gradient = ForwardDiff.gradient(timss_selected_oracle,
            timss_point[timss_selected])
        @test isapprox(timss_gradient[timss_selected], timss_selected_gradient;
            atol=1e-5)

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

    @testset "item-response densities and native Enzyme gradients" begin
        logistic(x) = inv(1 + exp(-x))
        logit_jacobian(x) = -log1p(exp(-x)) - log1p(exp(x))
        bernoulli_logit(y, eta) = y * eta - log1p(exp(eta))
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

        irt_data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
            PosteriorDB.database(), "irt_2pl-irt_2pl",
        )))
        I, J = irt_data["I"], irt_data["J"]
        function irt_reference(v)
            sigma_theta = exp(v[1])
            theta = view(v, 2:(J + 1))
            sigma_a = exp(v[J + 2])
            a = exp.(view(v, (J + 3):(J + I + 2)))
            mu_b = v[J + I + 3]
            sigma_b = exp(v[J + I + 4])
            b = view(v, (J + I + 5):length(v))
            total = logpdf(Cauchy(0, 2), sigma_theta) + log(2) + v[1] +
                sum(logpdf(Normal(0, sigma_theta), x) for x in theta) +
                logpdf(Cauchy(0, 2), sigma_a) + log(2) + v[J + 2] +
                sum(logpdf(LogNormal(0, sigma_a), a[i]) + v[J + i + 2]
                    for i in eachindex(a)) +
                logpdf(Normal(0, 5), mu_b) +
                logpdf(Cauchy(0, 2), sigma_b) + log(2) + v[J + I + 4] +
                sum(logpdf(Normal(mu_b, sigma_b), x) for x in b)
            for j in 1:J, i in 1:I
                total += bernoulli_logit(irt_data["y"][i, j],
                    a[i] * (theta[j] - b[i]))
            end
            return total
        end
        irt_dimension = J + 2I + 4
        check("irt_2pl-irt_2pl", irt_dimension, irt_reference,
            [vcat([0.1], 0.1 .* sin.(1:J), [-0.2], 0.1 .* cos.(1:I),
                  [0.2, -0.1], 0.1 .* sin.(1:I))])

        bones_data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
            PosteriorDB.database(), "bones_data-bones_model",
        )))
        function bones_reference(v)
            total = sum(logpdf(Normal(0, 36), x) for x in v)
            grade, gamma, delta, ncat = (bones_data[k] for k in
                ("grade", "gamma", "delta", "ncat"))
            for j in axes(grade, 2), i in axes(grade, 1)
                category = grade[i, j]
                category == -1 && continue
                q = [logistic(delta[j] * (v[i] - gamma[j, k]))
                    for k in 1:(ncat[j] - 1)]
                probabilities = [1 - q[1]; [q[k - 1] - q[k]
                    for k in 2:length(q)]; q[end]]
                total += log(probabilities[category])
            end
            return total
        end
        check("bones_data-bones_model", bones_data["nChild"], bones_reference,
            [0.1 .* sin.(1:bones_data["nChild"]),
             0.2 .* cos.(1:bones_data["nChild"])])

        lsat_data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
            PosteriorDB.database(), "lsat_data-lsat_model",
        )))
        T, N = lsat_data["T"], lsat_data["N"]
        function lsat_reference(v)
            alpha = view(v, 1:T)
            theta = view(v, (T + 1):(T + N))
            beta = exp(v[end])
            total = sum(logpdf(Normal(0, 100), x) for x in alpha) +
                sum(logpdf(Normal(), x) for x in theta) +
                logpdf(Normal(0, 100), beta) + log(2) + v[end]
            previous = 0
            for pattern in 1:lsat_data["R"]
                for student in (previous + 1):lsat_data["culm"][pattern]
                    for question in 1:T
                        eta = beta * theta[student] - alpha[question]
                        total += bernoulli_logit(
                            lsat_data["response"][pattern, question], eta)
                    end
                end
                previous = lsat_data["culm"][pattern]
            end
            return total
        end
        lsat_vm = VecModel(posteriordb_model("lsat_data-lsat_model"),
            AutoEnzyme(; mode=Reverse))
        @test lsat_vm.dimension == T + N + 1
        point = vcat(0.1 .* sin.(1:T), 0.05 .* cos.(1:N), [0.2])
        value, gradient = lp_and_grad(lsat_vm, point)
        @test isapprox(value, lsat_reference(point); atol=1e-6)
        for index in (1, T, T + 1, T + 500, T + N, T + N + 1)
            plus, minus = copy(point), copy(point)
            plus[index] += 1e-5
            minus[index] -= 1e-5
            finite_difference = (lsat_reference(plus) - lsat_reference(minus)) / 2e-5
            @test isapprox(gradient[index], finite_difference; atol=1e-4, rtol=1e-4)
        end
    end
end

end # module WrenPosteriorDBTests
