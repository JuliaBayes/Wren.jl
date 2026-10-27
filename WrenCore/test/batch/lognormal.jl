module WrenBatchLogNormalTests

@info "testing batch/lognormal.jl"

using Test
using WrenCore
using WrenCore.TestUtils: test_model_equivalence
using WrenDists

@testset "batch: WrenDists.LogNormal" begin
    mu = [1.0, 2.0, 3.0]
    sigma = [0.5, 1.0, 2.0]

    @testset "batched: LogNormal(mu[i], sigma[i]) -> BatchedLogNormal" begin
        @wren function both_batch(x, mu, sigma)
            @batch for i in 1:3
                x[i] ~ WrenDists.LogNormal(mu[i], sigma[i])
            end
        end
        @wren function both_nobatch(x, mu, sigma)
            for i in 1:3
                x[i] ~ WrenDists.LogNormal(mu[i], sigma[i])
            end
        end
        test_model_equivalence(
            both_batch(ones(3), mu, sigma),
            both_nobatch(ones(3), mu, sigma),
        )
    end

    @testset "batched: LogNormal(mu[i], const_sigma) -> BatchedLogNormal" begin
        @wren function mu_batch(x, mu)
            @batch for i in 1:3
                x[i] ~ WrenDists.LogNormal(mu[i], 2.0)
            end
        end
        @wren function mu_nobatch(x, mu)
            for i in 1:3
                x[i] ~ WrenDists.LogNormal(mu[i], 2.0)
            end
        end
        test_model_equivalence(mu_batch(ones(3), mu), mu_nobatch(ones(3), mu))
    end

    @testset "batched: LogNormal(mu[i]) -> BatchedLogNormal" begin
        @wren function mu_only_batch(x, mu)
            @batch for i in 1:3
                x[i] ~ WrenDists.LogNormal(mu[i])
            end
        end
        @wren function mu_only_nobatch(x, mu)
            for i in 1:3
                x[i] ~ WrenDists.LogNormal(mu[i])
            end
        end
        test_model_equivalence(mu_only_batch(ones(3), mu), mu_only_nobatch(ones(3), mu))
    end

    @testset "batched: LogNormal(const_mu, sigma[i]) -> BatchedLogNormal" begin
        @wren function sigma_batch(x, sigma)
            @batch for i in 1:3
                x[i] ~ WrenDists.LogNormal(0.0, sigma[i])
            end
        end
        @wren function sigma_nobatch(x, sigma)
            for i in 1:3
                x[i] ~ WrenDists.LogNormal(0.0, sigma[i])
            end
        end
        test_model_equivalence(sigma_batch(ones(3), sigma), sigma_nobatch(ones(3), sigma))
    end
end

end # module WrenBatchLogNormalTests
