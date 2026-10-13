module WrenBatchNormalTests

@info "testing batch/normal.jl"

using Test
using WrenCore
using WrenCore.TestUtils: test_model_equivalence
using WrenDists

@testset "batch: WrenDists.Normal" begin
    # Each testset exercises one batched/iid overload from WrenCoreWrenDistsExt

    @testset "iid: four-dimensional size" begin
        @wren function iid_4d()
            @iid x ~ WrenDists.Normal(1.0, 2.0) size=(2, 3, 4, 5)
        end
        @wren function independent_4d()
            x ~ WrenCore.independent(fill(WrenDists.Normal(1.0, 2.0), 2, 3, 4, 5))
        end
        @test size(rand(iid_4d())[@vn(x)]) == (2, 3, 4, 5)
        test_model_equivalence(iid_4d(), independent_4d())
    end

    @testset "iid: Normal(consts...) -> iid(Normal)" begin
        # x[i] ~ Normal() with no indexed args => iid path
        @wren function iid_batch(x)
            @batch for i in 1:3
                x[i] ~ WrenDists.Normal()
            end
        end
        @wren function iid_nobatch(x)
            for i in 1:3
                x[i] ~ WrenDists.Normal()
            end
        end
        test_model_equivalence(iid_batch(zeros(3)), iid_nobatch(zeros(3)))
    end

    @testset "batched: Normal(mu[i]) -> BatchedNormal" begin
        @wren function mu_only_batch(x, mu)
            @batch for i in 1:3
                x[i] ~ WrenDists.Normal(mu[i])
            end
        end
        @wren function mu_only_nobatch(x, mu)
            for i in 1:3
                x[i] ~ WrenDists.Normal(mu[i])
            end
        end
        mu = [1.0, 2.0, 3.0]
        test_model_equivalence(mu_only_batch(zeros(3), mu), mu_only_nobatch(zeros(3), mu))
    end

    @testset "batched: Normal(mu[i], const_sigma) -> BatchedNormal" begin
        @wren function iso_batch(x, mu)
            @batch for i in 1:3
                x[i] ~ WrenDists.Normal(mu[i], 2.0)
            end
        end
        @wren function iso_nobatch(x, mu)
            for i in 1:3
                x[i] ~ WrenDists.Normal(mu[i], 2.0)
            end
        end
        mu = [1.0, 2.0, 3.0]
        test_model_equivalence(iso_batch(zeros(3), mu), iso_nobatch(zeros(3), mu))
    end

    @testset "batched: Normal(const_mu, sigma[i]) -> BatchedNormal" begin
        @wren function sigma_indexed_batch(x, sigma)
            @batch for i in 1:3
                x[i] ~ WrenDists.Normal(0.0, sigma[i])
            end
        end
        @wren function sigma_indexed_nobatch(x, sigma)
            for i in 1:3
                x[i] ~ WrenDists.Normal(0.0, sigma[i])
            end
        end
        sigma = [0.5, 1.0, 2.0]
        test_model_equivalence(
            sigma_indexed_batch(zeros(3), sigma),
            sigma_indexed_nobatch(zeros(3), sigma),
        )
    end

    @testset "batched: Normal(mu[i], sigma[i]) -> BatchedNormal" begin
        @wren function diag_batch(x, mu, sigma)
            @batch for i in 1:3
                x[i] ~ WrenDists.Normal(mu[i], sigma[i])
            end
        end
        @wren function diag_nobatch(x, mu, sigma)
            for i in 1:3
                x[i] ~ WrenDists.Normal(mu[i], sigma[i])
            end
        end
        mu = [1.0, 2.0, 3.0]
        sigma = [0.5, 1.0, 2.0]
        test_model_equivalence(
            diag_batch(zeros(3), mu, sigma),
            diag_nobatch(zeros(3), mu, sigma),
        )
    end

    @testset "observe with batched Normal" begin
        @wren function obs_batch(x)
            a ~ WrenDists.Normal()
            @batch for i in 1:3
                @observe x[i] ~ WrenDists.Normal(a, 1.0)
            end
        end
        @wren function obs_nobatch(x)
            a ~ WrenDists.Normal()
            for i in 1:3
                @observe x[i] ~ WrenDists.Normal(a, 1.0)
            end
        end
        test_model_equivalence(obs_batch(ones(3)), obs_nobatch(ones(3)))
    end

    @testset "sample + observe with batched Normal" begin
        @wren function sample_obs_batch(x, y)
            @batch for i in 1:3
                x[i] ~ WrenDists.Normal()
                @observe y[i] ~ WrenDists.Normal(x[i], 0.5)
            end
        end
        @wren function sample_obs_nobatch(x, y)
            for i in 1:3
                x[i] ~ WrenDists.Normal()
                @observe y[i] ~ WrenDists.Normal(x[i], 0.5)
            end
        end
        test_model_equivalence(
            sample_obs_batch(zeros(3), ones(3)),
            sample_obs_nobatch(zeros(3), ones(3)),
        )
    end
end

end # module WrenBatchNormalTests
