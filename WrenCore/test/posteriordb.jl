module WrenPosteriorDBTests

@info "testing posteriordb.jl"

import DifferentiationInterface as DI
using Distributions: Binomial, Cauchy, Normal, logpdf
using Enzyme
import ForwardDiff
using PosteriorDB: PosteriorDB
using StableRNGs
using Statistics: mean, std
using Test
using WrenCore
using WrenDists: WrenDists

@testset "posteriordb.jl" begin
    @testset "$name" for name in available_posteriordb_models()
        model = posteriordb_model(name)
        @test isfinite(logjoint(model, rand(StableRNG(468), model)))

        vm = VecModel(model, DI.AutoForwardDiff())
        v = rand(StableRNG(468), vm)
        @test length(v) == vm.dimension
        logp, grad = lp_and_grad(vm, v)
        @test isfinite(logp)
        @test length(grad) == length(v)
        @test all(isfinite, grad)
    end

    @test_throws ArgumentError posteriordb_model("not-a-posterior")

    @testset "flat-prior Kidscore densities and native Enzyme gradients" begin
        data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
            PosteriorDB.database(), "kidiq_with_mom_work-kidscore_interaction_c",
        )))
        hs = data["mom_hs"]
        iq = data["mom_iq"]
        work = data["mom_work"]
        centered_hs = hs .- mean(hs)
        centered_iq = iq .- mean(iq)
        scaled_hs = centered_hs ./ (2 * std(hs))
        scaled_iq = centered_iq ./ (2 * std(iq))
        cases = (
            ("kidiq_with_mom_work-kidscore_interaction_c",
                hcat(ones(length(hs)), centered_hs, centered_iq,
                    centered_hs .* centered_iq)),
            ("kidiq_with_mom_work-kidscore_interaction_c2",
                hcat(ones(length(hs)), hs .- 0.5, iq .- 100.0,
                    (hs .- 0.5) .* (iq .- 100.0))),
            ("kidiq_with_mom_work-kidscore_interaction_z",
                hcat(ones(length(hs)), scaled_hs, scaled_iq,
                    scaled_hs .* scaled_iq)),
            ("kidiq_with_mom_work-kidscore_mom_work",
                hcat(ones(length(work)), work .== 2, work .== 3, work .== 4)),
        )
        for (name, X) in cases
            vm = VecModel(posteriordb_model(name), AutoEnzyme(; mode=Reverse))
            @test vm.dimension == 5
            function reference(v)
                mu = X * view(v, 1:4)
                sigma = exp(v[5])
                return v[5] + sum(logpdf(Normal(mu[i], sigma), data["kid_score"][i])
                    for i in eachindex(mu))
            end
            for point in ([100.0, 2.0, 0.2, 0.1, log(15.0)],
                          [95.0, -1.0, 0.3, -0.1, log(20.0)])
                value, gradient = lp_and_grad(vm, point)
                @test isapprox(value, reference(point); atol=1e-5)
                @test isapprox(gradient, ForwardDiff.gradient(reference, point);
                    atol=1e-5)
            end
        end
    end

    @testset "eight NES regression densities and native Enzyme gradients" begin
        for year in 1972:4:2000
            name = "nes$(year)-nes"
            data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
                PosteriorDB.database(), name,
            )))
            age = data["age_discrete"]
            X = hcat(ones(data["N"]), data["real_ideo"], data["race_adj"],
                age .== 2, age .== 3, age .== 4, data["educ1"], data["gender"],
                data["income"])
            vm = VecModel(posteriordb_model(name), AutoEnzyme(; mode=Reverse))
            @test vm.dimension == 10
            function reference(v)
                mu = X * view(v, 1:9)
                sigma = exp(v[10])
                return v[10] + sum(logpdf(Normal(mu[i], sigma), data["partyid7"][i])
                    for i in eachindex(mu))
            end
            for point in (
                [3.0, 0.2, 0.1, -0.1, 0.2, 0.1, -0.1, 0.2, 0.1, log(1.5)],
                [4.0, -0.1, 0.2, 0.1, -0.1, 0.2, 0.1, -0.1, 0.2, log(2.0)],
            )
                value, gradient = lp_and_grad(vm, point)
                @test isapprox(value, reference(point); atol=1e-5)
                @test isapprox(gradient, ForwardDiff.gradient(reference, point);
                    atol=1e-5)
            end
        end
    end

    @testset "earnings regression densities and native Enzyme gradients" begin
        data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
            PosteriorDB.database(), "earnings-earn_height",
        )))
        earn, height, male = data["earn"], data["height"], data["male"]
        N = data["N"]
        zheight = (height .- mean(height)) ./ std(height)
        cases = (
            ("earnings-earn_height", earn, hcat(ones(N), height)),
            ("earnings-log10earn_height", log10.(earn), hcat(ones(N), height)),
            ("earnings-logearn_height", log.(earn), hcat(ones(N), height)),
            ("earnings-logearn_height_male", log.(earn), hcat(ones(N), height, male)),
            ("earnings-logearn_interaction", log.(earn),
                hcat(ones(N), height, male, height .* male)),
            ("earnings-logearn_interaction_z", log.(earn),
                hcat(ones(N), zheight, male, zheight .* male)),
            ("earnings-logearn_logheight_male", log.(earn),
                hcat(ones(N), log.(height), male)),
        )
        for (name, y, X) in cases
            vm = VecModel(posteriordb_model(name), AutoEnzyme(; mode=Reverse))
            dimension = size(X, 2) + 1
            @test vm.dimension == dimension
            function reference(v)
                mu = X * view(v, 1:(dimension - 1))
                sigma = exp(v[end])
                return v[end] + sum(logpdf(Normal(mu[i], sigma), y[i])
                    for i in eachindex(y))
            end
            for scale in (0.1, -0.2)
                point = vcat(scale .* sin.(1:(dimension - 1)), log(20.0))
                value, gradient = lp_and_grad(vm, point)
                @test isapprox(value, reference(point); atol=1e-4)
                @test isapprox(gradient, ForwardDiff.gradient(reference, point);
                    atol=1e-4)
            end
        end
    end

    @testset "Rate posteriors: Stan density and unconstrained gradient" begin
        pdb = PosteriorDB.database()
        for (number, point) in (
            (2, [0.3, -0.4]), (2, [-1.2, 1.3]), (2, [2.0, -2.0]),
            (3, [0.3]), (3, [-1.2]), (3, [2.0]),
            (4, [0.3, -0.4]), (4, [-1.2, 1.3]), (4, [2.0, -2.0]),
            (5, [-0.4]), (5, [1.3]), (5, [-2.0]),
        )
            name = "Rate_$(number)_data-Rate_$(number)_model"
            data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(pdb, name)))
            vm = VecModel(posteriordb_model(name), AutoEnzyme(; mode=Reverse))
            @test vm.dimension == length(point)

            p = inv(1 + exp(-point[1]))
            jac = log(p) + log1p(-p)
            if number == 4
                q = inv(1 + exp(-point[2]))
                expected = logpdf(Binomial(data["n"], p), data["k"]) + jac +
                    log(q) + log1p(-q)
                expected_grad = [data["k"] + 1 - (data["n"] + 2) * p, 1 - 2q]
            elseif number == 2
                q = inv(1 + exp(-point[2]))
                expected = logpdf(Binomial(data["n1"], p), data["k1"]) + jac +
                    logpdf(Binomial(data["n2"], q), data["k2"]) + log(q) + log1p(-q)
                expected_grad = [
                    data["k1"] + 1 - (data["n1"] + 2) * p,
                    data["k2"] + 1 - (data["n2"] + 2) * q,
                ]
            else
                expected = logpdf(Binomial(data["n1"], p), data["k1"]) +
                    logpdf(Binomial(data["n2"], p), data["k2"]) + jac
                expected_grad = [
                    data["k1"] + data["k2"] + 1 -
                    (data["n1"] + data["n2"] + 2) * p,
                ]
            end
            value, gradient = lp_and_grad(vm, point)
            @test isapprox(value, expected; atol=1e-9)
            @test isapprox(gradient, expected_grad; atol=1e-9)
        end
    end

    @testset "noncentered schools: Stan density and gradient" begin
        data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
            PosteriorDB.database(), "eight_schools-eight_schools_noncentered",
        )))
        J, y, sigma = data["J"], data["y"], data["sigma"]
        vm = VecModel(posteriordb_model("eight_schools-eight_schools_noncentered"),
            AutoEnzyme(; mode=Reverse))
        @test vm.dimension == J + 2

        function reference(v)
            theta_trans, mu, logtau = view(v, 1:J), v[J+1], v[J+2]
            tau = exp(logtau)
            theta = theta_trans .* tau .+ mu
            return sum(logpdf(Normal(), x) for x in theta_trans) +
                logpdf(Normal(0, 5), mu) + logpdf(Cauchy(0, 5), tau) + log(2) +
                logtau + sum(logpdf(Normal(theta[i], sigma[i]), y[i]) for i in eachindex(y))
        end
        for point in (
            vcat(collect(range(-0.8, 0.8; length=J)), 0.4, -0.3),
            vcat(sin.(1:J), -1.1, 0.5),
        )
            value, gradient = lp_and_grad(vm, point)
            @test isapprox(value, reference(point); atol=1e-9)
            @test isapprox(gradient, ForwardDiff.gradient(reference, point); atol=1e-8)
        end
    end

    @testset "pooled radon: Stan density and gradient" begin
        pdb = PosteriorDB.database()
        for name in ("radon_all-radon_pooled", "radon_mn-radon_pooled")
            data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(pdb, name)))
            y, floor_measure = data["log_radon"], data["floor_measure"]
            vm = VecModel(posteriordb_model(name), AutoEnzyme(; mode=Reverse))
            @test vm.dimension == 3

            function reference(v)
                alpha, beta, logsigma = v
                sigma_y = exp(logsigma)
                return logpdf(Normal(0, 10), alpha) + logpdf(Normal(0, 10), beta) +
                    logpdf(Normal(), sigma_y) + log(2) + logsigma +
                    sum(logpdf(Normal(alpha + beta * floor_measure[i], sigma_y), y[i])
                        for i in eachindex(y))
            end
            for point in ([0.2, -0.5, log(0.8)], [1.1, 0.4, log(1.4)])
                value, gradient = lp_and_grad(vm, point)
                @test isapprox(value, reference(point); atol=1e-6)
                @test isapprox(gradient, ForwardDiff.gradient(reference, point); atol=1e-6)
            end
        end
    end

    @testset "wells logistic posteriors: Stan density and gradient" begin
        data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
            PosteriorDB.database(), "wells_data-wells_dist",
        )))
        y, dist, arsenic = data["switched"], data["dist"], data["arsenic"]
        educ, assoc = data["educ"], data["assoc"]
        dist100 = dist ./ 100.0
        c_dist100 = (dist .- sum(dist) / length(dist)) ./ 100.0
        c_arsenic = arsenic .- sum(arsenic) / length(arsenic)
        educ4 = educ ./ 4.0
        c_educ4 = (educ .- sum(educ) / length(educ)) ./ 4.0
        cases = (
            ("wells_data-wells_dist", hcat(ones(length(dist)), dist), false),
            ("wells_data-wells_dist100_model", hcat(dist100), true),
            ("wells_data-wells_dist100ars_model", hcat(dist100, arsenic), true),
            ("wells_data-wells_dae_model", hcat(dist100, arsenic, educ4), true),
            ("wells_data-wells_dae_c_model",
                hcat(c_dist100, c_arsenic, c_dist100 .* c_arsenic, educ4), true),
            ("wells_data-wells_dae_inter_model",
                hcat(c_dist100, c_arsenic, c_educ4, c_dist100 .* c_arsenic,
                    c_dist100 .* c_educ4, c_arsenic .* c_educ4), true),
            ("wells_data-wells_daae_c_model",
                hcat(c_dist100, c_arsenic, c_dist100 .* c_arsenic, assoc, educ4), true),
            ("wells_data-wells_interaction_model",
                hcat(dist100, arsenic, dist100 .* arsenic), true),
            ("wells_data-wells_interaction_c_model",
                hcat(c_dist100, c_arsenic, c_dist100 .* c_arsenic), true),
        )
        for (name, X, has_alpha) in cases
            vm = VecModel(posteriordb_model(name), AutoEnzyme(; mode=Reverse))
            dimension = size(X, 2) + has_alpha
            @test vm.dimension == dimension
            for scale in (0.2, -0.3)
                point = scale .* sin.(1:dimension)
                logit_p = has_alpha ? point[1] .+ X * view(point, 2:dimension) : X * point
                p = @. inv(1 + exp(-logit_p))
                expected = sum(y[i] * logit_p[i] - log1p(exp(logit_p[i])) for i in eachindex(y))
                residual = y .- p
                expected_grad = has_alpha ? vcat(sum(residual), X' * residual) : X' * residual
                value, gradient = lp_and_grad(vm, point)
                @test isapprox(value, expected; atol=1e-6)
                @test isapprox(gradient, expected_grad; atol=1e-6)
            end
        end
    end

    @testset "kidscore linear posteriors: density and gradient" begin
        data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
            PosteriorDB.database(), "kidiq-kidscore_momhs",
        )))
        y, mom_hs, mom_iq = data["kid_score"], data["mom_hs"], data["mom_iq"]
        intercept = ones(length(y))
        cases = (
            ("kidiq-kidscore_momhs", hcat(intercept, mom_hs)),
            ("kidiq-kidscore_momiq", hcat(intercept, mom_iq)),
            ("kidiq-kidscore_momhsiq", hcat(intercept, mom_hs, mom_iq)),
        )
        for (name, X) in cases
            vm = VecModel(posteriordb_model(name), AutoEnzyme(; mode=Reverse))
            k = size(X, 2)
            @test vm.dimension == k + 1
            for log_sigma in (log(50.0), log(85.0))
                beta = 0.2 .* cos.(1:k)
                point = vcat(beta, log_sigma)
                sigma = exp(log_sigma)
                residual = y - X * beta
                expected = -length(y) * log(sigma * sqrt(2π)) -
                    sum(abs2, residual) / (2sigma^2) +
                    log(2 / (π * 2.5)) - log1p((sigma / 2.5)^2) + log_sigma
                expected_grad = vcat(
                    X' * residual / sigma^2,
                    -length(y) + sum(abs2, residual) / sigma^2 -
                        2sigma^2 / (2.5^2 + sigma^2) + 1,
                )
                value, gradient = lp_and_grad(vm, point)
                @test isapprox(value, expected; atol=1e-6)
                @test isapprox(gradient, expected_grad; atol=1e-6)
            end
        end
    end
end

end # module WrenPosteriorDBTests
