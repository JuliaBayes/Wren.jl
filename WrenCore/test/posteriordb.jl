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

    @testset "radon densities and native Enzyme gradients" begin
        # Sums of Normal log densities, and a half-normal(0, 1) scale on its log
        # coordinate with the Jacobian. The oracles below are built from these.
        normal_sum(x, mu, sigma) =
            -length(x) * log(sigma * sqrt(2π)) - sum(abs2, x .- mu) / (2sigma^2)
        half_normal_scale(log_sigma) = log(2 / π) / 2 - exp(2log_sigma) / 2 + log_sigma

        function hierarchical_intercept(data, v)
            J = data["J"]
            alpha_raw, beta, mu_alpha = view(v, 1:J), view(v, (J+1):(J+2)), v[J+3]
            sigma_alpha, sigma_y = exp(v[J+4]), exp(v[J+5])
            alpha = mu_alpha .+ sigma_alpha .* alpha_raw
            mu =
                alpha[data["county_idx"]] .+ beta[1] .* data["log_uppm"] .+
                beta[2] .* data["floor_measure"]
            return normal_sum(alpha_raw, 0, 1) +
                   normal_sum(beta, 0, 10) +
                   normal_sum(mu_alpha, 0, 10) +
                   half_normal_scale(v[J+4]) +
                   half_normal_scale(v[J+5]) +
                   normal_sum(data["log_radon"], mu, sigma_y)
        end
        function partially_pooled(data, v)
            J = data["J"]
            alpha_raw, mu_alpha = view(v, 1:J), v[J+1]
            sigma_alpha, sigma_y = exp(v[J+2]), exp(v[J+3])
            alpha = mu_alpha .+ sigma_alpha .* alpha_raw
            return normal_sum(alpha_raw, 0, 1) +
                   normal_sum(mu_alpha, 0, 10) +
                   half_normal_scale(v[J+2]) +
                   half_normal_scale(v[J+3]) +
                   normal_sum(data["log_radon"], alpha[data["county_idx"]], sigma_y)
        end
        function variable_intercept(data, v)
            J = data["J"]
            alpha_raw, beta, mu_alpha = view(v, 1:J), v[J+1], v[J+2]
            sigma_alpha, sigma_y = exp(v[J+3]), exp(v[J+4])
            alpha = mu_alpha .+ sigma_alpha .* alpha_raw
            mu = alpha[data["county_idx"]] .+ data["floor_measure"] .* beta
            return normal_sum(alpha_raw, 0, 1) +
                   normal_sum(beta, 0, 10) +
                   normal_sum(mu_alpha, 0, 10) +
                   half_normal_scale(v[J+3]) +
                   half_normal_scale(v[J+4]) +
                   normal_sum(data["log_radon"], mu, sigma_y)
        end
        function variable_slope(data, v)
            J = data["J"]
            alpha, beta_raw, mu_beta = v[1], view(v, 2:(J+1)), v[J+2]
            sigma_beta, sigma_y = exp(v[J+3]), exp(v[J+4])
            beta = mu_beta .+ sigma_beta .* beta_raw
            mu = alpha .+ data["floor_measure"] .* beta[data["county_idx"]]
            return normal_sum(beta_raw, 0, 1) +
                   normal_sum(alpha, 0, 10) +
                   normal_sum(mu_beta, 0, 10) +
                   half_normal_scale(v[J+3]) +
                   half_normal_scale(v[J+4]) +
                   normal_sum(data["log_radon"], mu, sigma_y)
        end
        function variable_intercept_slope(data, v)
            J = data["J"]
            sigma_y, sigma_alpha, sigma_beta = exp(v[1]), exp(v[2]), exp(v[3])
            alpha_raw, beta_raw = view(v, 4:(J+3)), view(v, (J+4):(2J+3))
            mu_alpha, mu_beta = v[2J+4], v[2J+5]
            alpha = mu_alpha .+ sigma_alpha .* alpha_raw
            beta = mu_beta .+ sigma_beta .* beta_raw
            mu =
                alpha[data["county_idx"]] .+
                data["floor_measure"] .* beta[data["county_idx"]]
            return normal_sum(alpha_raw, 0, 1) +
                   normal_sum(beta_raw, 0, 1) +
                   normal_sum(mu_alpha, 0, 10) +
                   normal_sum(mu_beta, 0, 10) +
                   sum(half_normal_scale, view(v, 1:3)) +
                   normal_sum(data["log_radon"], mu, sigma_y)
        end
        function county_intercept(data, v)
            J = data["J"]
            alpha, beta, sigma_y = view(v, 1:J), v[J+1], exp(v[J+2])
            mu = alpha[data["county_idx"]] .+ beta .* data["floor_measure"]
            return normal_sum(alpha, 0, 10) +
                   normal_sum(beta, 0, 10) +
                   half_normal_scale(v[J+2]) +
                   normal_sum(data["log_radon"], mu, sigma_y)
        end

        # One oracle and one point per model family.
        families = (
            (
                "radon_hierarchical_intercept_noncentered",
                hierarchical_intercept,
                J -> vcat(0.05 .* sin.(1:J), [0.08, -0.11], 0.2, log(0.7), log(0.9)),
            ),
            (
                "radon_partially_pooled_noncentered",
                partially_pooled,
                J -> vcat(0.05 .* sin.(1:J), 0.2, log(0.7), log(0.9)),
            ),
            (
                "radon_variable_intercept_noncentered",
                variable_intercept,
                J -> vcat(0.05 .* sin.(1:J), 0.08, 0.2, log(0.7), log(0.9)),
            ),
            (
                "radon_variable_slope_noncentered",
                variable_slope,
                J -> vcat(0.08, 0.05 .* sin.(1:J), 0.2, log(0.7), log(0.9)),
            ),
            (
                "radon_variable_intercept_slope_noncentered",
                variable_intercept_slope,
                J -> vcat(
                    log(0.9),
                    log(0.7),
                    log(0.8),
                    0.04 .* sin.(1:J),
                    0.04 .* cos.(1:J),
                    0.2,
                    -0.1,
                ),
            ),
            (
                "radon_county_intercept",
                county_intercept,
                J -> vcat(0.04 .* sin.(1:J), 0.2, log(0.9)),
            ),
        )
        for (variant, family, point_for) in families, dataset in ("radon_mn", "radon_all")

            name = "$dataset-$variant"
            data = PosteriorDB.load(
                PosteriorDB.dataset(PosteriorDB.posterior(PosteriorDB.database(), name)),
            )
            oracle(v) = family(data, v)
            point = point_for(data["J"])
            vm = VecModel(posteriordb_model(name), AutoEnzyme(; mode=Reverse))
            @test vm.dimension == length(point)
            value, gradient = lp_and_grad(vm, point)
            @test isapprox(value, oracle(point); atol=1e-5)
            # Both datasets bind the same model definition, so the full gradient is
            # checked on the smaller Minnesota dataset only.
            if dataset == "radon_mn"
                @test isapprox(gradient, ForwardDiff.gradient(oracle, point); atol=1e-5)
            end
        end
    end
end

end # module WrenPosteriorDBTests
