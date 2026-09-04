module WrenBatchTests

@info "testing batch.jl"

using Test
using Wren
using Enzyme
using Distributions
using DifferentiationInterface: AbstractADType

function get_lp_components(model::Wren.Model, params)
    _, accs = evaluate(
        model,
        FromParams(params),
        OriginalSpace(),
        AccumulatorTuple(LogPriorAccumulator(), LogLikelihoodAccumulator()),
    )
    return logprior(accs), loglikelihood(accs)
end

function test_model_equivalence(
    model1::Wren.Model,
    model2::Wren.Model;
    adtype::AbstractADType=AutoEnzyme(; mode=Reverse),
)
    # Shouldn't matter which model we use to generate params, since they are equivalent
    params = rand(model1)

    lpr1, llk1 = get_lp_components(model1, params)
    lpr2, llk2 = get_lp_components(model2, params)
    @test lpr1 ≈ lpr2
    @test llk1 ≈ llk2

    # Test vectorisation
    vm1 = VecModel(model1, adtype)
    vm2 = VecModel(model2, adtype)
    @test vm1.dimension == vm2.dimension
    vparams = vectorise_latents(FromParams(params), vm1)
    lp1, grad1 = lp_and_grad(vm1, vparams)
    lp2, grad2 = lp_and_grad(vm2, vparams)
    @test lp1 ≈ lp2
    @test grad1 ≈ grad2
end

@testset "batch.jl" begin
    @testset "semantic invariance" begin
        # `@plate for ... end` should *always* have the same behaviour as `for ... end`

        @testset "sample" begin
            @wren function sample1(x)
                @plate for i in 1:3
                    x[i] ~ Normal()
                end
            end
            @wren function sample2(x)
                for i in 1:3
                    x[i] ~ Normal()
                end
            end
            test_model_equivalence(sample1(zeros(3)), sample2(zeros(3)))
        end

        @testset "observe" begin
            @wren function observe1(x)
                a ~ Normal()
                @plate for i in 1:3
                    @observe x[i] ~ Normal(a)
                end
            end
            @wren function observe2(x)
                a ~ Normal()
                for i in 1:3
                    @observe x[i] ~ Normal(a)
                end
            end
            test_model_equivalence(observe1(zeros(3)), observe2(zeros(3)))
        end

        @testset "conditioned" begin
            @wren function conditioned1(x)
                a ~ Normal()
                @plate for i in 1:3
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
            test_model_equivalence(model1, model2)
        end

        @testset "sample + observe" begin
            @wren function sample_observe1(x, y)
                @plate for i in 1:3
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
            )
        end
    end
end

end # module WrenBatchTests
