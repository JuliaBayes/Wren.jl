module WrenBatchBinomialTests

@info "testing batch/binomial.jl"

using Test
using WrenCore
using WrenCore.TestUtils: test_model_equivalence
using WrenDists

@testset "batch: WrenDists.Binomial" begin
    y = [1.0, 4.0, 3.0]
    n = [5, 6, 7]

    @testset "batched: Binomial(n[i], p[i]) -> BatchedBinomial" begin
        @wren function np_batch(y, n)
            p ~ WrenCore.iid(WrenDists.Beta(2.0, 2.0), 3)
            @batch for i in 1:3
                @observe y[i] ~ WrenDists.Binomial(n[i], p[i])
            end
        end
        @wren function np_nobatch(y, n)
            p ~ WrenCore.iid(WrenDists.Beta(2.0, 2.0), 3)
            for i in 1:3
                @observe y[i] ~ WrenDists.Binomial(n[i], p[i])
            end
        end
        test_model_equivalence(np_batch(y, n), np_nobatch(y, n))
    end

    @testset "batched: Binomial(n[i], const_p) -> BatchedBinomial" begin
        @wren function n_batch(y, n)
            p ~ WrenDists.Beta(2.0, 2.0)
            @batch for i in 1:3
                @observe y[i] ~ WrenDists.Binomial(n[i], p)
            end
        end
        @wren function n_nobatch(y, n)
            p ~ WrenDists.Beta(2.0, 2.0)
            for i in 1:3
                @observe y[i] ~ WrenDists.Binomial(n[i], p)
            end
        end
        test_model_equivalence(n_batch(y, n), n_nobatch(y, n))
    end

    @testset "batched: Binomial(n[i]) -> BatchedBinomial" begin
        @wren function n_only_batch(y, n)
            @batch for i in 1:3
                @observe y[i] ~ WrenDists.Binomial(n[i])
            end
        end
        @wren function n_only_nobatch(y, n)
            for i in 1:3
                @observe y[i] ~ WrenDists.Binomial(n[i])
            end
        end
        test_model_equivalence(n_only_batch(y, n), n_only_nobatch(y, n))
    end

    @testset "batched: Binomial(const_n, p[i]) -> BatchedBinomial" begin
        @wren function p_batch(y)
            p ~ WrenCore.iid(WrenDists.Beta(2.0, 2.0), 3)
            @batch for i in 1:3
                @observe y[i] ~ WrenDists.Binomial(10, p[i])
            end
        end
        @wren function p_nobatch(y)
            p ~ WrenCore.iid(WrenDists.Beta(2.0, 2.0), 3)
            for i in 1:3
                @observe y[i] ~ WrenDists.Binomial(10, p[i])
            end
        end
        test_model_equivalence(p_batch(y), p_nobatch(y))
    end
end

@testset "batch: WrenDists.BinomialLogit" begin
    y = [1.0, 4.0, 3.0]
    n = [5, 6, 7]

    @testset "batched: BinomialLogit(n[i], a[i]) -> BatchedBinomialLogit" begin
        @wren function na_batch(y, n)
            a ~ WrenCore.iid(WrenDists.Normal(), 3)
            @batch for i in 1:3
                @observe y[i] ~ WrenDists.BinomialLogit(n[i], a[i])
            end
        end
        @wren function na_nobatch(y, n)
            a ~ WrenCore.iid(WrenDists.Normal(), 3)
            for i in 1:3
                @observe y[i] ~ WrenDists.BinomialLogit(n[i], a[i])
            end
        end
        test_model_equivalence(na_batch(y, n), na_nobatch(y, n))
    end

    @testset "batched: BinomialLogit(n[i], const_a) -> BatchedBinomialLogit" begin
        @wren function n_batch(y, n)
            a ~ WrenDists.Normal()
            @batch for i in 1:3
                @observe y[i] ~ WrenDists.BinomialLogit(n[i], a)
            end
        end
        @wren function n_nobatch(y, n)
            a ~ WrenDists.Normal()
            for i in 1:3
                @observe y[i] ~ WrenDists.BinomialLogit(n[i], a)
            end
        end
        test_model_equivalence(n_batch(y, n), n_nobatch(y, n))
    end

    @testset "batched: BinomialLogit(n[i]) -> BatchedBinomialLogit" begin
        @wren function n_only_batch(y, n)
            @batch for i in 1:3
                @observe y[i] ~ WrenDists.BinomialLogit(n[i])
            end
        end
        @wren function n_only_nobatch(y, n)
            for i in 1:3
                @observe y[i] ~ WrenDists.BinomialLogit(n[i])
            end
        end
        test_model_equivalence(n_only_batch(y, n), n_only_nobatch(y, n))
    end

    @testset "batched: BinomialLogit(const_n, a[i]) -> BatchedBinomialLogit" begin
        @wren function a_batch(y)
            a ~ WrenCore.iid(WrenDists.Normal(), 3)
            @batch for i in 1:3
                @observe y[i] ~ WrenDists.BinomialLogit(10, a[i])
            end
        end
        @wren function a_nobatch(y)
            a ~ WrenCore.iid(WrenDists.Normal(), 3)
            for i in 1:3
                @observe y[i] ~ WrenDists.BinomialLogit(10, a[i])
            end
        end
        test_model_equivalence(a_batch(y), a_nobatch(y))
    end
end

end # module WrenBatchBinomialTests
