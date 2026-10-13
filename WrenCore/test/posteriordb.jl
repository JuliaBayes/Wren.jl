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

    @testset "noncentered hierarchical radon: density and gradient" begin
        for name in (
            "radon_all-radon_hierarchical_intercept_noncentered",
            "radon_mn-radon_hierarchical_intercept_noncentered",
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
                alpha_raw = view(v, 1:J)
                beta = view(v, J + 1:J + 2)
                mu_alpha = v[J + 3]
                sigma_alpha = exp(v[J + 4])
                sigma_y = exp(v[J + 5])
                alpha = mu_alpha .+ sigma_alpha .* alpha_raw
                mu = alpha[county_idx] .+ beta[1] .* log_uppm .+
                    beta[2] .* floor_measure
                residual = y - mu
                return -J * log(sqrt(2π)) - sum(abs2, alpha_raw) / 2 -
                    2log(10sqrt(2π)) - sum(abs2, beta) / 200 -
                    log(10sqrt(2π)) - mu_alpha^2 / 200 +
                    log(2 / π) - (sigma_alpha^2 + sigma_y^2) / 2 -
                    length(y) * log(sigma_y * sqrt(2π)) -
                    sum(abs2, residual) / (2sigma_y^2) + v[J + 4] + v[J + 5]
            end
            vm = VecModel(posteriordb_model(name), AutoEnzyme(; mode=Reverse))
            @test vm.dimension == J + 5
            for (log_sigma_alpha, log_sigma_y) in ((log(0.7), log(0.9)), (log(1.1), log(1.2)))
                point = vcat(0.05 .* sin.(1:J), [0.08, -0.11], 0.2,
                    log_sigma_alpha, log_sigma_y)
                value, gradient = lp_and_grad(vm, point)
                @test isapprox(value, oracle(point); atol=1e-5)
                @test isapprox(gradient, ForwardDiff.gradient(oracle, point); atol=1e-5)
            end
        end
    end
end

end # module WrenPosteriorDBTests
