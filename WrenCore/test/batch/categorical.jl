module WrenBatchCategoricalTests

@info "testing batch/categorical.jl"

using Test
using WrenCore
using WrenCore.TestUtils: test_model_equivalence
using WrenDists

@testset "batch: WrenDists.Categorical" begin
    @testset "batched: Categorical(p[:, i]) -> BatchedCategorical" begin
        @wren function p_batch(y)
            # TODO(penelopeysm): iid on Dirichlet doesn't work
            p = zeros(5, 3)
            for i in 1:3
                p[:, i] ~ WrenDists.Dirichlet(ones(5))
            end
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
