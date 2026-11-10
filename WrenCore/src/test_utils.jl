module TestUtils

using Test
using DifferentiationInterface: AbstractADType
using ..WrenCore

function get_lp_components(model::WrenCore.Model, params)
    _, accs = evaluate(
        model,
        FromValues(params),
        OriginalSpace(),
        AccumulatorTuple(LogPriorAccumulator(), LogLikelihoodAccumulator()),
        WrenCore.EvalContext{false}(),
    )
    return logprior(accs), loglikelihood(accs)
end

function test_model_equivalence(
    model1::WrenCore.Model,
    model2::WrenCore.Model,
    adtype::Union{AbstractADType,Nothing}=nothing,
)
    # Shouldn't matter which model we use to generate params, since they are equivalent
    params = parameters(model1)

    lpr1, llk1 = get_lp_components(model1, params)
    lpr2, llk2 = get_lp_components(model2, params)
    @test lpr1 ≈ lpr2
    @test llk1 ≈ llk2

    # Test vectorisation
    vm1 = VecModel(model1, adtype)
    vm2 = VecModel(model2, adtype)
    @test vm1.dimension == vm2.dimension
    vparams = vectorise_latents(FromValues(params), vm1)
    if adtype === nothing
        lp1 = lp(vm1, vparams)
        lp2 = lp(vm2, vparams)
        @test lp1 ≈ lp2
    else
        lp1, grad1 = lp_and_grad(vm1, vparams)
        lp2, grad2 = lp_and_grad(vm2, vparams)
        @test lp1 ≈ lp2
        @test grad1 ≈ grad2
    end
end

export test_model_equivalence

end # module TestUtils
