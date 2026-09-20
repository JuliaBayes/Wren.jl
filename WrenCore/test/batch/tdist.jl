module WrenBatchTDistTests

@info "testing batch/tdist.jl"

using Test
using WrenCore
using WrenCore.TestUtils: test_model_equivalence
using WrenDists

@testset "batch: WrenDists.TDist" begin
    mu = [1.0, 2.0, 3.0]
    sigma = [0.5, 1.0, 2.0]
    nu = [3.0, 4.0, 5.0]

    # 3-arg overloads
    @testset "batched: TDist(const_nu, mu[i], sigma[i])" begin
        @wren function nms_batch(x, mu, sigma)
            @batch for i in 1:3
                x[i] ~ WrenDists.TDist(4.0, mu[i], sigma[i])
            end
        end
        @wren function nms_nobatch(x, mu, sigma)
            for i in 1:3
                x[i] ~ WrenDists.TDist(4.0, mu[i], sigma[i])
            end
        end
        test_model_equivalence(
            nms_batch(zeros(3), mu, sigma),
            nms_nobatch(zeros(3), mu, sigma),
        )
    end

    @testset "batched: TDist(const_nu, mu[i], const_sigma)" begin
        @wren function nm_batch(x, mu)
            @batch for i in 1:3
                x[i] ~ WrenDists.TDist(4.0, mu[i], 2.0)
            end
        end
        @wren function nm_nobatch(x, mu)
            for i in 1:3
                x[i] ~ WrenDists.TDist(4.0, mu[i], 2.0)
            end
        end
        test_model_equivalence(nm_batch(zeros(3), mu), nm_nobatch(zeros(3), mu))
    end

    @testset "batched: TDist(const_nu, const_mu, sigma[i])" begin
        @wren function ns_batch(x, sigma)
            @batch for i in 1:3
                x[i] ~ WrenDists.TDist(4.0, 0.0, sigma[i])
            end
        end
        @wren function ns_nobatch(x, sigma)
            for i in 1:3
                x[i] ~ WrenDists.TDist(4.0, 0.0, sigma[i])
            end
        end
        test_model_equivalence(ns_batch(zeros(3), sigma), ns_nobatch(zeros(3), sigma))
    end

    @testset "batched: TDist(nu[i], mu[i], sigma[i])" begin
        @wren function all_batch(x, nu, mu, sigma)
            @batch for i in 1:3
                x[i] ~ WrenDists.TDist(nu[i], mu[i], sigma[i])
            end
        end
        @wren function all_nobatch(x, nu, mu, sigma)
            for i in 1:3
                x[i] ~ WrenDists.TDist(nu[i], mu[i], sigma[i])
            end
        end
        test_model_equivalence(
            all_batch(zeros(3), nu, mu, sigma),
            all_nobatch(zeros(3), nu, mu, sigma),
        )
    end

    @testset "batched: TDist(nu[i], mu[i], const_sigma)" begin
        @wren function numi_batch(x, nu, mu)
            @batch for i in 1:3
                x[i] ~ WrenDists.TDist(nu[i], mu[i], 2.0)
            end
        end
        @wren function numi_nobatch(x, nu, mu)
            for i in 1:3
                x[i] ~ WrenDists.TDist(nu[i], mu[i], 2.0)
            end
        end
        test_model_equivalence(numi_batch(zeros(3), nu, mu), numi_nobatch(zeros(3), nu, mu))
    end

    @testset "batched: TDist(nu[i], const_mu, sigma[i])" begin
        @wren function nusi_batch(x, nu, sigma)
            @batch for i in 1:3
                x[i] ~ WrenDists.TDist(nu[i], 0.0, sigma[i])
            end
        end
        @wren function nusi_nobatch(x, nu, sigma)
            for i in 1:3
                x[i] ~ WrenDists.TDist(nu[i], 0.0, sigma[i])
            end
        end
        test_model_equivalence(
            nusi_batch(zeros(3), nu, sigma),
            nusi_nobatch(zeros(3), nu, sigma),
        )
    end

    @testset "batched: TDist(nu[i], const_mu, const_sigma)" begin
        @wren function nuonly_batch(x, nu)
            @batch for i in 1:3
                x[i] ~ WrenDists.TDist(nu[i], 0.0, 1.0)
            end
        end
        @wren function nuonly_nobatch(x, nu)
            for i in 1:3
                x[i] ~ WrenDists.TDist(nu[i], 0.0, 1.0)
            end
        end
        test_model_equivalence(nuonly_batch(zeros(3), nu), nuonly_nobatch(zeros(3), nu))
    end

    # 2-arg overloads
    @testset "batched: TDist(const_nu, mu[i])" begin
        @wren function twoarg_batch(x, mu)
            @batch for i in 1:3
                x[i] ~ WrenDists.TDist(4.0, mu[i])
            end
        end
        @wren function twoarg_nobatch(x, mu)
            for i in 1:3
                x[i] ~ WrenDists.TDist(4.0, mu[i])
            end
        end
        test_model_equivalence(twoarg_batch(zeros(3), mu), twoarg_nobatch(zeros(3), mu))
    end

    @testset "batched: TDist(nu[i], mu[i]) 2-arg" begin
        @wren function twoarg2_batch(x, nu, mu)
            @batch for i in 1:3
                x[i] ~ WrenDists.TDist(nu[i], mu[i])
            end
        end
        @wren function twoarg2_nobatch(x, nu, mu)
            for i in 1:3
                x[i] ~ WrenDists.TDist(nu[i], mu[i])
            end
        end
        test_model_equivalence(
            twoarg2_batch(zeros(3), nu, mu),
            twoarg2_nobatch(zeros(3), nu, mu),
        )
    end

    # 1-arg overload
    @testset "batched: TDist(nu[i])" begin
        @wren function onearg_batch(x, nu)
            @batch for i in 1:3
                x[i] ~ WrenDists.TDist(nu[i])
            end
        end
        @wren function onearg_nobatch(x, nu)
            for i in 1:3
                x[i] ~ WrenDists.TDist(nu[i])
            end
        end
        test_model_equivalence(onearg_batch(zeros(3), nu), onearg_nobatch(zeros(3), nu))
    end

    # iid
    @testset "iid tdist" begin
        @wren function iid_batch(x)
            @batch for i in 1:3
                x[i] ~ WrenDists.TDist(4.0)
            end
        end
        @wren function iid_nobatch(x)
            for i in 1:3
                x[i] ~ WrenDists.TDist(4.0)
            end
        end
        test_model_equivalence(iid_batch(zeros(3)), iid_nobatch(zeros(3)))
    end

    # Observe
    @testset "observe with batched TDist" begin
        @wren function obs_batch(x)
            a ~ WrenDists.Normal()
            @batch for i in 1:3
                @observe x[i] ~ WrenDists.TDist(4.0, a, 1.0)
            end
        end
        @wren function obs_nobatch(x)
            a ~ WrenDists.Normal()
            for i in 1:3
                @observe x[i] ~ WrenDists.TDist(4.0, a, 1.0)
            end
        end
        test_model_equivalence(obs_batch(ones(3)), obs_nobatch(ones(3)))
    end
end

end # module WrenBatchTDistTests
