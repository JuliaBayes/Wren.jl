module WrenBatchPoissonTests

@info "testing batch/poisson.jl"

using Test
using WrenCore
using WrenCore.TestUtils: test_model_equivalence
using WrenDists

@testset "batch: WrenDists.Poisson" begin
    @testset "support and boundaries (issue #158)" begin
        @wren function support_batch(lambda, y)
            @batch for i in eachindex(y)
                @observe y[i] ~ WrenDists.Poisson(lambda[i])
            end
        end
        for (lambda, y, expected) in (([0.0], [0.0], 0.0), ([1.0], [0.5], -Inf))
            @test logjoint(support_batch(lambda, y), VarNamedTuple()) == expected
        end
    end

    @testset "batched: Poisson(exp(mu[i])) -> BatchedPoisson" begin
        @wren function log_batch(y)
            mu ~ WrenCore.iid(WrenDists.Normal(), 3)
            @batch for i in 1:3
                @observe y[i] ~ WrenDists.Poisson(exp(mu[i]))
            end
        end
        @wren function log_nobatch(y)
            mu ~ WrenCore.iid(WrenDists.Normal(), 3)
            for i in 1:3
                @observe y[i] ~ WrenDists.Poisson(exp(mu[i]))
            end
        end
        y = [1.0, 2.0, 3.0]
        test_model_equivalence(log_batch(y), log_nobatch(y))
    end
end

@testset "batch: WrenDists.PoissonLog" begin
    @testset "batched: PoissonLog(mu[i]) -> BatchedPoissonLog" begin
        @wren function lograte_batch(y)
            mu ~ WrenCore.iid(WrenDists.Normal(), 3)
            @batch for i in 1:3
                @observe y[i] ~ WrenDists.PoissonLog(mu[i])
            end
        end
        @wren function lograte_nobatch(y)
            mu ~ WrenCore.iid(WrenDists.Normal(), 3)
            for i in 1:3
                @observe y[i] ~ WrenDists.PoissonLog(mu[i])
            end
        end
        y = [1.0, 2.0, 3.0]
        test_model_equivalence(lograte_batch(y), lograte_nobatch(y))
    end
end

end # module WrenBatchPoissonTests
