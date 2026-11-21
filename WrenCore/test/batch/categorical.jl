module WrenBatchCategoricalTests

@info "testing batch/categorical.jl"

using Test
using WrenCore
using WrenCore.TestUtils: test_model_equivalence
using WrenDists
import Plaice
using LinearAlgebra: I

@testset "batch: WrenDists.Categorical" begin
    @testset "iid multivariate: $dims" for dims in ((), (2, 2))
        d = iid(WrenDists.Dirichlet([1.0, 2.0, 3.0]), dims)
        n = 2 * prod(dims)
        @test Plaice.unconstrained_vec_length(d) == n
        @test Plaice.unconstrained_optic_vec(d) == fill(nothing, n)
        y = collect(range(-1.0, 1.0; length=n))
        x, logjac = Plaice.with_logabsdet_jacobian(Plaice.from_unconstrained_vec(d), y)
        @test size(x) == (3, dims...)
        @test Plaice.to_unconstrained_vec(d)(x) ≈ y
        @test logjac ≈ sum(log, x; init=0.0)
        normal = iid(WrenDists.MvNormalScale(zeros(3), I), dims)
        @test Plaice.unconstrained_optic_vec(normal) == Plaice.optic_vec(normal)
    end

    @testset "batched: Categorical(p[:, i]) -> BatchedCategorical" begin
        @wren function p_batch(y)
            p ~ iid(WrenDists.Dirichlet(ones(5)), 3)
            @batch for i in 1:3
                @observe y[i] ~ WrenDists.Categorical(p[:, i])
            end
        end
        @wren function p_nobatch(y)
            p = zeros(5, 3)
            for i in 1:3
                p[:, i] ~ WrenDists.Dirichlet(ones(5))
            end
            for i in 1:3
                @observe y[i] ~ WrenDists.Categorical(p[:, i])
            end
        end
        y = [1, 3, 4]
        test_model_equivalence(p_batch(y), p_nobatch(y))
    end

    @testset "batched: Categorical(p[i, :]) -> BatchedCategorical" begin
        @wren function p_batch(y)
            p = zeros(3, 5)
            for i in 1:3
                p[i, :] ~ WrenDists.Dirichlet(ones(5))
            end
            @batch for i in 1:3
                @observe y[i] ~ WrenDists.Categorical(p[i, :])
            end
        end
        @wren function p_nobatch(y)
            p = zeros(3, 5)
            for i in 1:3
                p[i, :] ~ WrenDists.Dirichlet(ones(5))
            end
            for i in 1:3
                @observe y[i] ~ WrenDists.Categorical(p[i, :])
            end
        end
        y = [1, 3, 4]
        test_model_equivalence(p_batch(y), p_nobatch(y))
    end

    @testset "batched: Categorical(p[i]) -> BatchedCategorical" begin
        @wren function p_batch(y)
            p = [zeros(5) for _ in 1:3]
            for i in 1:3
                p[i] ~ WrenDists.Dirichlet(ones(5))
            end
            @batch for i in 1:3
                @observe y[i] ~ WrenDists.Categorical(p[i])
            end
        end
        @wren function p_nobatch(y)
            p = [zeros(5) for _ in 1:3]
            for i in 1:3
                p[i] ~ WrenDists.Dirichlet(ones(5))
            end
            @batch for i in 1:3
                @observe y[i] ~ WrenDists.Categorical(p[i])
            end
        end
        y = [1, 3, 4]
        test_model_equivalence(p_batch(y), p_nobatch(y))
    end
end

end # module WrenBatchCategoricalTests
