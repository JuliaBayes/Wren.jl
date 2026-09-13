module WrenBatchGeneralTests

@info "testing batch/general.jl"

using Test
using WrenCore
using WrenCore.TestUtils: test_model_equivalence
using Distributions
using Enzyme

adtype = AutoEnzyme(; mode=Reverse)

@testset "general @batch properties" begin
    @testset "semantic invariance" begin
        # `@batch for ... end` should *always* have the same behaviour as `for ... end`

        @testset "sample" begin
            @wren function sample1(x)
                @batch for i in 1:3
                    x[i] ~ Normal()
                end
            end
            @wren function sample2(x)
                for i in 1:3
                    x[i] ~ Normal()
                end
            end
            test_model_equivalence(sample1(zeros(3)), sample2(zeros(3)), adtype)
        end

        @testset "observe" begin
            @wren function observe1(x)
                a ~ Normal()
                @batch for i in 1:3
                    @observe x[i] ~ Normal(a)
                end
            end
            @wren function observe2(x)
                a ~ Normal()
                for i in 1:3
                    @observe x[i] ~ Normal(a)
                end
            end
            test_model_equivalence(observe1(zeros(3)), observe2(zeros(3)), adtype)
        end

        @testset "conditioned" begin
            @wren function conditioned1(x)
                a ~ Normal()
                @batch for i in 1:3
                    x[i] ~ Normal(a)
                end
            end
            @wren function conditioned2(x)
                a ~ Normal()
                for i in 1:3
                    x[i] ~ Normal(a)
                end
            end
            model1 = conditioned1(zeros(3)) | VarNamedTuple(x=ones(3))
            model2 = conditioned2(zeros(3)) | VarNamedTuple(x=ones(3))
            test_model_equivalence(model1, model2, adtype)
        end

        @testset "sample + observe" begin
            @wren function sample_observe1(x, y)
                @batch for i in 1:3
                    x[i] ~ Normal()
                    @observe y[i] ~ Normal(x[i])
                end
            end
            @wren function sample_observe2(x, y)
                for i in 1:3
                    x[i] ~ Normal()
                    @observe y[i] ~ Normal(x[i])
                end
            end
            test_model_equivalence(
                sample_observe1(zeros(3), ones(3)),
                sample_observe2(zeros(3), ones(3)),
                adtype,
            )
        end
    end

    @testset "keyword arguments" begin
        @wren function batch_kwargs(y, p)
            @batch for i in eachindex(y)
                y[i] ~ NegativeBinomial(2.0, p[i]; check_args=false)
            end
        end
        @wren function nobatch_kwargs(y, p)
            @batch for i in eachindex(y)
                y[i] ~ NegativeBinomial(2.0, p[i]; check_args=false)
            end
        end
        test_model_equivalence(
            batch_kwargs(ones(3), [0.1, 0.2, 0.3]),
            nobatch_kwargs(ones(3), [0.1, 0.2, 0.3]),
            nothing,
        )
    end

    @testset "different examples of batchable syntax" begin
        # could really do with more examples here

        @testset "issue #79" begin
            # https://github.com/JuliaBayes/Wren.jl/issues/79
            @wren function batch_diagonal(x, y)
                @batch for i in 1:2
                    x[i] ~ Normal(y[i, i])
                end
            end
            @wren function nobatch_diagonal(x, y)
                for i in 1:2
                    x[i] ~ Normal(y[i, i])
                end
            end
            y = [1.0 10.0; 20.0 2.0]
            test_model_equivalence(
                batch_diagonal(zeros(2), y),
                nobatch_diagonal(zeros(2), y),
                adtype,
            )
        end

        @testset "issue #72" begin
            # https://github.com/JuliaBayes/Wren.jl/issues/72
            @wren function batch_subtraction(y, n)
                μ ~ Normal(0, 1)
                m = fill(μ, n)
                @batch for k in eachindex(y)
                    @observe y[k] ~ Normal(1 - m[k], 1.0)
                end
            end
            @wren function nobatch_subtraction(y, n)
                μ ~ Normal(0, 1)
                m = fill(μ, n)
                for k in eachindex(y)
                    @observe y[k] ~ Normal(1 - m[k], 1.0)
                end
            end
            n = 5
            y = randn(n)
            test_model_equivalence(
                batch_subtraction(y, n),
                nobatch_subtraction(y, n),
                nothing,
            )
        end
    end
end

end # module WrenBatchGeneralTests
