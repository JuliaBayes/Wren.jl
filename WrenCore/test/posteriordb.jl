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

    @testset "noncentered variable intercept and slope radon" begin
        for name in (
            "radon_all-radon_variable_intercept_slope_noncentered",
            "radon_mn-radon_variable_intercept_slope_noncentered",
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
                alpha_raw = view(v, 4:J + 3)
                beta_raw = view(v, J + 4:2J + 3)
                mu_alpha = v[2J + 4]
                mu_beta = v[2J + 5]
                alpha = mu_alpha .+ sigma_alpha .* alpha_raw
                beta = mu_beta .+ sigma_beta .* beta_raw
                residual = y - (alpha[county_idx] .+ floor_measure .* beta[county_idx])
                return -2J * log(sqrt(2π)) -
                    (sum(abs2, alpha_raw) + sum(abs2, beta_raw)) / 2 -
                    2log(10sqrt(2π)) - (mu_alpha^2 + mu_beta^2) / 200 +
                    1.5log(2 / π) - (sigma_y^2 + sigma_alpha^2 + sigma_beta^2) / 2 -
                    length(y) * log(sigma_y * sqrt(2π)) -
                    sum(abs2, residual) / (2sigma_y^2) + v[1] + v[2] + v[3]
            end
            vm = VecModel(posteriordb_model(name), AutoEnzyme(; mode=Reverse))
            @test vm.dimension == 2J + 5
            for (log_sigma_y, log_sigma_alpha, log_sigma_beta) in (
                (log(0.9), log(0.7), log(0.8)),
                (log(1.2), log(1.1), log(0.9)),
            )
                point = vcat(log_sigma_y, log_sigma_alpha, log_sigma_beta,
                    0.04 .* sin.(1:J), 0.04 .* cos.(1:J), 0.2, -0.1)
                value, gradient = lp_and_grad(vm, point)
                @test isapprox(value, oracle(point); atol=1e-5)
                @test isapprox(gradient, ForwardDiff.gradient(oracle, point); atol=1e-5)
            end
        end
    end
end

end # module WrenPosteriorDBTests
