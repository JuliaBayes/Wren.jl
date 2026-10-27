module WrenBatchBernoulliTests

@info "testing batch/bernoulli.jl"

using Test
using WrenCore
using WrenCore.TestUtils: test_model_equivalence
using WrenDists

@testset "batch: WrenDists.Bernoulli" begin
    @testset "batched: Bernoulli(p[i]) -> BatchedBernoulli" begin
        @wren function p_batch(y)
            p ~ WrenCore.iid(WrenDists.Beta(2.0, 2.0), 3)
            @batch for i in 1:3
                @observe y[i] ~ WrenDists.Bernoulli(p[i])
            end
        end
        @wren function p_nobatch(y)
            p ~ WrenCore.iid(WrenDists.Beta(2.0, 2.0), 3)
            for i in 1:3
                @observe y[i] ~ WrenDists.Bernoulli(p[i])
            end
        end
        y = [1.0, 0.0, 1.0]
        test_model_equivalence(p_batch(y), p_nobatch(y))
    end

    @testset "batched: Bernoulli(f(a[i])) -> BatchedBernoulli" begin
        # The argument is a broadcast expression rather than a plain index.
        @wren function logit_batch(y)
            a ~ WrenCore.iid(WrenDists.Normal(), 3)
            @batch for i in 1:3
                @observe y[i] ~ WrenDists.Bernoulli(1 / (1 + exp(-a[i])))
            end
        end
        @wren function logit_nobatch(y)
            a ~ WrenCore.iid(WrenDists.Normal(), 3)
            for i in 1:3
                @observe y[i] ~ WrenDists.Bernoulli(1 / (1 + exp(-a[i])))
            end
        end
        y = [1.0, 0.0, 1.0]
        test_model_equivalence(logit_batch(y), logit_nobatch(y))
    end
end

@testset "batch: WrenDists.BernoulliLogit" begin
    @testset "batched: BernoulliLogit(a[i]) -> BatchedBernoulliLogit" begin
        @wren function a_batch(y)
            a ~ WrenCore.iid(WrenDists.Normal(), 3)
            @batch for i in 1:3
                @observe y[i] ~ WrenDists.BernoulliLogit(a[i])
            end
        end
        @wren function a_nobatch(y)
            a ~ WrenCore.iid(WrenDists.Normal(), 3)
            for i in 1:3
                @observe y[i] ~ WrenDists.BernoulliLogit(a[i])
            end
        end
        y = [1.0, 0.0, 1.0]
        test_model_equivalence(a_batch(y), a_nobatch(y))
    end
end

end # module WrenBatchBernoulliTests
