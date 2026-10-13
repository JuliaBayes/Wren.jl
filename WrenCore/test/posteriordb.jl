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

    @testset "noncentered variable radon: density and gradient" begin
        for dataset in ("radon_all", "radon_mn")
            for variant in ("radon_variable_intercept_noncentered",
                            "radon_variable_slope_noncentered")
                name = "$dataset-$variant"
                data = PosteriorDB.load(PosteriorDB.dataset(PosteriorDB.posterior(
                    PosteriorDB.database(), name,
                )))
                J = data["J"]
                county_idx = data["county_idx"]
                floor_measure = data["floor_measure"]
                y = data["log_radon"]
                oracle = if variant == "radon_variable_intercept_noncentered"
                    v -> begin
                        alpha_raw = view(v, 1:J)
                        beta = v[J + 1]
                        mu_alpha = v[J + 2]
                        sigma_alpha = exp(v[J + 3])
                        sigma_y = exp(v[J + 4])
                        alpha = mu_alpha .+ sigma_alpha .* alpha_raw
                        residual = y - (alpha[county_idx] .+ floor_measure .* beta)
                        return -J * log(sqrt(2π)) - sum(abs2, alpha_raw) / 2 -
                            2log(10sqrt(2π)) - (beta^2 + mu_alpha^2) / 200 +
                            log(2 / π) - (sigma_alpha^2 + sigma_y^2) / 2 -
                            length(y) * log(sigma_y * sqrt(2π)) -
                            sum(abs2, residual) / (2sigma_y^2) + v[J + 3] + v[J + 4]
                    end
                else
                    v -> begin
                        alpha = v[1]
                        beta_raw = view(v, 2:J + 1)
                        mu_beta = v[J + 2]
                        sigma_beta = exp(v[J + 3])
                        sigma_y = exp(v[J + 4])
                        beta = mu_beta .+ sigma_beta .* beta_raw
                        residual = y - (alpha .+ floor_measure .* beta[county_idx])
                        return -J * log(sqrt(2π)) - sum(abs2, beta_raw) / 2 -
                            2log(10sqrt(2π)) - (alpha^2 + mu_beta^2) / 200 +
                            log(2 / π) - (sigma_beta^2 + sigma_y^2) / 2 -
                            length(y) * log(sigma_y * sqrt(2π)) -
                            sum(abs2, residual) / (2sigma_y^2) + v[J + 3] + v[J + 4]
                    end
                end
                vm = VecModel(posteriordb_model(name), AutoEnzyme(; mode=Reverse))
                @test vm.dimension == J + 4
                for (log_sigma, log_sigma_y) in ((log(0.7), log(0.9)), (log(1.1), log(1.2)))
                    point = variant == "radon_variable_intercept_noncentered" ?
                        vcat(0.05 .* sin.(1:J), 0.08, 0.2, log_sigma, log_sigma_y) :
                        vcat(0.08, 0.05 .* sin.(1:J), 0.2, log_sigma, log_sigma_y)
                    value, gradient = lp_and_grad(vm, point)
                    @test isapprox(value, oracle(point); atol=1e-5)
                    @test isapprox(gradient, ForwardDiff.gradient(oracle, point); atol=1e-5)
                end
            end
        end
    end
end

end # module WrenPosteriorDBTests
