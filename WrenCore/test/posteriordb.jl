module WrenPosteriorDBTests

@info "testing posteriordb.jl"

import DifferentiationInterface as DI
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
        @test isfinite(logjoint(model, parameters(model, StableRNG(468))))

        vm = VecModel(model, DI.AutoForwardDiff())
        v = vector_parameters(vm, StableRNG(468))
        @test length(v) == vm.dimension
        logp, grad = lp_and_grad(vm, v)
        @test isfinite(logp)
        @test length(grad) == length(v)
        @test all(isfinite, grad)
    end

    @test_throws ArgumentError posteriordb_model("not-a-posterior")

    @testset "radon county bounded-scale density and native Enzyme gradient" begin
        name = "radon_mod-radon_county"
        data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
            PosteriorDB.database(), name,
        )))
        J = data["J"]
        vm = VecModel(posteriordb_model(name), AutoEnzyme(; mode=Reverse))
        @test vm.dimension == J + 3
        normal_logp(x, mu, sigma) =
            -log(sigma * sqrt(2π)) - (x - mu)^2 / (2 * sigma^2)
        function reference(v)
            mu_a = v[1]
            ratio_a = inv(1 + exp(-v[2]))
            ratio_y = inv(1 + exp(-v[3]))
            sigma_a, sigma_y = 100 * ratio_a, 100 * ratio_y
            a = view(v, 4:length(v))
            jacobian = log(ratio_a) + log1p(-ratio_a) +
                log(ratio_y) + log1p(-ratio_y)
            return normal_logp(mu_a, 0.0, 1.0) + jacobian +
                sum(normal_logp(value, mu_a, sigma_a) for value in a) +
                sum(normal_logp(data["y"][i], a[data["county"][i]], sigma_y)
                    for i in eachindex(data["y"]))
        end
        for offset in (0.0, 0.1)
            point = vcat(0.2 + offset, log(1.0 / 99.0), log(0.8 / 99.2),
                0.1 .* sin.(1:J))
            value, gradient = lp_and_grad(vm, point)
            @test isapprox(value, reference(point); atol=1e-5)
            @test isapprox(gradient, ForwardDiff.gradient(reference, point); atol=1e-5)
        end
    end

    @testset "centered radon: density and native Enzyme gradients" begin
        normal_sum(x, mu, sigma) =
            -length(x) * log(sigma * sqrt(2π)) - sum(abs2, x .- mu) / (2sigma^2)
        normal_one(x, mu, sigma) =
            -log(sigma * sqrt(2π)) - (x - mu)^2 / (2sigma^2)
        half_one(sigma) = 0.5log(2 / π) - sigma^2 / 2
        function check(name, dimension, oracle, points)
            vm = VecModel(posteriordb_model(name), AutoEnzyme(; mode=Reverse))
            @test vm.dimension == dimension
            for point in points
                value, gradient = lp_and_grad(vm, point)
                @test isapprox(value, oracle(point); atol=1e-5)
                @test isapprox(gradient, ForwardDiff.gradient(oracle, point); atol=1e-5)
            end
        end

        for name in (
            "radon_all-radon_hierarchical_intercept_centered",
            "radon_mn-radon_hierarchical_intercept_centered",
        )
            data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
                PosteriorDB.database(), name,
            )))
            J = data["J"]
            county_idx = data["county_idx"]
            log_uppm = data["log_uppm"]
            floor_measure = data["floor_measure"]
            y = data["log_radon"]
            function oracle(v)
                alpha = view(v, 1:J)
                beta = view(v, J + 1:J + 2)
                mu_alpha = v[J + 3]
                sigma_alpha = exp(v[J + 4])
                sigma_y = exp(v[J + 5])
                mu = alpha[county_idx] .+ beta[1] .* log_uppm .+
                    beta[2] .* floor_measure
                return normal_sum(alpha, mu_alpha, sigma_alpha) +
                    normal_sum(beta, 0.0, 10.0) + normal_one(mu_alpha, 0.0, 10.0) +
                    half_one(sigma_alpha) + half_one(sigma_y) +
                    normal_sum(y, mu, sigma_y) + v[J + 4] + v[J + 5]
            end
            points = [
                vcat(0.04 .* sin.(1:J), [0.08, -0.11], 0.2, log(0.7), log(0.9)),
                vcat(0.03 .* cos.(1:J), [-0.05, 0.13], -0.1, log(1.1), log(1.2)),
            ]
            check(name, J + 5, oracle, points)
        end

        for name in (
            "radon_all-radon_partially_pooled_centered",
            "radon_mn-radon_partially_pooled_centered",
        )
            data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
                PosteriorDB.database(), name,
            )))
            J = data["J"]
            county_idx = data["county_idx"]
            y = data["log_radon"]
            function oracle(v)
                alpha = view(v, 1:J)
                mu_alpha = v[J + 1]
                sigma_alpha = exp(v[J + 2])
                sigma_y = exp(v[J + 3])
                return normal_sum(alpha, mu_alpha, sigma_alpha) +
                    normal_one(mu_alpha, 0.0, 10.0) +
                    half_one(sigma_alpha) + half_one(sigma_y) +
                    normal_sum(y, alpha[county_idx], sigma_y) + v[J + 2] + v[J + 3]
            end
            points = [
                vcat(0.04 .* sin.(1:J), 0.2, log(0.7), log(0.9)),
                vcat(0.03 .* cos.(1:J), -0.1, log(1.1), log(1.2)),
            ]
            check(name, J + 3, oracle, points)
        end

        for name in (
            "radon_all-radon_variable_intercept_centered",
            "radon_mn-radon_variable_intercept_centered",
        )
            data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
                PosteriorDB.database(), name,
            )))
            J = data["J"]
            county_idx = data["county_idx"]
            floor_measure = data["floor_measure"]
            y = data["log_radon"]
            function oracle(v)
                alpha = view(v, 1:J)
                beta = v[J + 1]
                mu_alpha = v[J + 2]
                sigma_alpha = exp(v[J + 3])
                sigma_y = exp(v[J + 4])
                mu = alpha[county_idx] .+ beta .* floor_measure
                return normal_sum(alpha, mu_alpha, sigma_alpha) +
                    normal_one(beta, 0.0, 10.0) + normal_one(mu_alpha, 0.0, 10.0) +
                    half_one(sigma_alpha) + half_one(sigma_y) +
                    normal_sum(y, mu, sigma_y) + v[J + 3] + v[J + 4]
            end
            points = [
                vcat(0.04 .* sin.(1:J), 0.1, 0.2, log(0.7), log(0.9)),
                vcat(0.03 .* cos.(1:J), -0.1, -0.2, log(1.1), log(1.2)),
            ]
            check(name, J + 4, oracle, points)
        end

        for name in (
            "radon_all-radon_variable_slope_centered",
            "radon_mn-radon_variable_slope_centered",
        )
            data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
                PosteriorDB.database(), name,
            )))
            J = data["J"]
            county_idx = data["county_idx"]
            floor_measure = data["floor_measure"]
            y = data["log_radon"]
            function oracle(v)
                alpha = v[1]
                beta = view(v, 2:J + 1)
                mu_beta = v[J + 2]
                sigma_beta = exp(v[J + 3])
                sigma_y = exp(v[J + 4])
                mu = alpha .+ beta[county_idx] .* floor_measure
                return normal_one(alpha, 0.0, 10.0) +
                    normal_sum(beta, mu_beta, sigma_beta) +
                    normal_one(mu_beta, 0.0, 10.0) +
                    half_one(sigma_beta) + half_one(sigma_y) +
                    normal_sum(y, mu, sigma_y) + v[J + 3] + v[J + 4]
            end
            points = [
                vcat(0.2, 0.04 .* sin.(1:J), 0.1, log(0.7), log(0.9)),
                vcat(-0.1, 0.03 .* cos.(1:J), -0.2, log(1.1), log(1.2)),
            ]
            check(name, J + 4, oracle, points)
        end

        for name in (
            "radon_all-radon_variable_intercept_slope_centered",
            "radon_mn-radon_variable_intercept_slope_centered",
        )
            data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
                PosteriorDB.database(), name,
            )))
            J = data["J"]
            county_idx = data["county_idx"]
            floor_measure = data["floor_measure"]
            y = data["log_radon"]
            function oracle(v)
                sigma_y = exp(v[1])
                sigma_alpha = exp(v[2])
                sigma_beta = exp(v[3])
                alpha = view(v, 4:J + 3)
                beta = view(v, J + 4:2J + 3)
                mu_alpha = v[2J + 4]
                mu_beta = v[2J + 5]
                mu = alpha[county_idx] .+ beta[county_idx] .* floor_measure
                return normal_sum(alpha, mu_alpha, sigma_alpha) +
                    normal_sum(beta, mu_beta, sigma_beta) +
                    normal_one(mu_alpha, 0.0, 10.0) +
                    normal_one(mu_beta, 0.0, 10.0) +
                    half_one(sigma_y) + half_one(sigma_alpha) + half_one(sigma_beta) +
                    normal_sum(y, mu, sigma_y) + v[1] + v[2] + v[3]
            end
            points = [
                vcat(log(0.9), log(0.7), log(0.8), 0.04 .* sin.(1:J),
                    0.04 .* cos.(1:J), 0.2, -0.1),
                vcat(log(1.2), log(1.1), log(0.9), 0.03 .* cos.(1:J),
                    0.03 .* sin.(1:J), -0.2, 0.1),
            ]
            check(name, 2J + 5, oracle, points)
        end
    end
end

end # module WrenPosteriorDBTests
