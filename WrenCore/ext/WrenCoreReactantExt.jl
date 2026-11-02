module WrenCoreReactantExt

using ADTypes: AutoEnzyme, AutoReactant
using Enzyme: Enzyme
using Reactant: Reactant
using WrenCore: WrenCore, Plaice
using WrenCore.Accessors: @set

include("reactant_bijectors.jl")
include("reactant_evaluation.jl")

_batch_output(value::Number) = convert(Reactant.TracedRArray, [value])
_batch_output(value::AbstractVector) = convert(Reactant.TracedRArray, value)
function _batch_points(f, params)
    return mapslices(params; dims=1) do point
        point_f = @set f.accs = copy(f.accs)
        _batch_output(point_f(vec(point)))
    end
end

struct BatchEvaluator{F,C}
    f::F
    compiled::C
    shape::Tuple{Int,Int}
end
_batch_input(params::Reactant.ConcreteRArray) = params
_batch_input(params::AbstractMatrix) = Reactant.to_rarray(params)
function (batch::BatchEvaluator)(params::AbstractMatrix)
    size(params) == batch.shape || throw(
        DimensionMismatch(
            "prepared batch expects size $(batch.shape), received $(size(params))",
        ),
    )
    return batch.compiled(batch.f, _batch_input(params))
end

function WrenCore.prepare_batch(
    vm::WrenCore.VecModel,
    params::AbstractMatrix,
    output_function=WrenCore._density_parts;
    accs::WrenCore.AccumulatorTuple=vm.accs,
)
    size(params, 1) == vm.dimension || throw(
        DimensionMismatch(
            "VecModel expects $(vm.dimension) rows, received $(size(params, 1))",
        ),
    )
    model = _reactant_model(vm.model)
    f = WrenCore._VecModelLP(
        _traced_evaluation_fn(model, false),
        Reactant.to_rarray(model.args),
        Reactant.to_rarray(model.conditioned),
        Val(keys(model.args)),
        WrenCore.EvalContext{true}(),
        output_function,
        vm.parameter_space,
        vm.vecinfo,
        Reactant.to_rarray(copy(accs)),
    )
    input = _batch_input(params)
    # The backend evaluator is created during preparation in the model's module.
    compiled = Base.invokelatest(Reactant.Compiler.compile, _batch_points, (f, input))
    return BatchEvaluator(f, compiled, size(params))
end

"""
    ReactantPrep

Stores the Reactant-compiled functions.
"""
struct ReactantPrep{P,G}
    primal::P
    value_and_gradient::G
end
struct ValueAndGradientCallable{F,A}
    f::F
    adtype::A
end

_with_primal(::AutoEnzyme{Nothing}) = Enzyme.ReverseWithPrimal
_with_primal(adtype::AutoEnzyme) = Enzyme.WithPrimal(adtype.mode)
_with_primal(x) = error("unreachable: AutoReactant containing non-AutoEnzyme AD type: $x")

function (f::ValueAndGradientCallable)(params)
    result = Enzyme.gradient(_with_primal(f.adtype.mode), f.f, params)
    return result.val, only(result.derivs)
end
function WrenCore.VecModel(
    model::WrenCore.Model,
    adtype::AutoReactant;
    latent_source::WrenCore.AbstractLatentSource=WrenCore.FromPrior(),
    lp_function=WrenCore.logjoint_transformed,
    parameter_space::WrenCore.AbstractParameterSpace=WrenCore.UnconstrainedSpace(),
    # Reactant doesn't use `WrenCore.reduce_copies`, but we still need this kwarg
    # to avoid a MethodError
    reduce_copies::Bool=true,
)
    vecinfo, sample = WrenCore._prepare_vecmodel_info(model, latent_source, parameter_space)
    model = _reactant_model(model)

    accs = WrenCore.AccumulatorTuple(
        WrenCore.LogPriorAccumulator(),
        WrenCore.LogJacobianAccumulator(),
        WrenCore.LogLikelihoodAccumulator(),
    )

    # TODO(penelopeysm): This **shouldn't** be necessary, in an ideal world. The model
    # arguments should be 'constant' data that Reactant doesn't need to trace -- only
    # the parameters should be traced.
    # However, Reactant gets incredibly slow when doing some matmuls with untraced arrays.
    # See e.g.
    #  - https://github.com/EnzymeAD/Reactant.jl/issues/3167
    #  - https://github.com/EnzymeAD/Reactant.jl/issues/3195
    # and there are probably more examples which we haven't found. So we just blanket-trace
    # the model args here.
    model_args = Reactant.to_rarray(model.args)

    lp_func = WrenCore._VecModelLP(
        _traced_evaluation_fn(model, false),
        model_args,
        model.conditioned,
        model.enzyme_copy_argnames,
        # TODO(penelopeysm): We'll conservatively copy the model args here just like we do
        # with Enzyme. However, I'm unsure if Enzyme-MLIR requires it.
        WrenCore.EvalContext{true}(),
        lp_function,
        parameter_space,
        vecinfo,
        accs,
    )

    reactant_sample = Reactant.to_rarray(sample)
    value_and_gradient_at = ValueAndGradientCallable(lp_func, adtype)
    compiled_lp = Base.invokelatest(Reactant.Compiler.compile, lp_func, (reactant_sample,))
    compiled_value_and_gradient = Base.invokelatest(
        Reactant.Compiler.compile,
        value_and_gradient_at,
        (reactant_sample,),
    )
    adprep = ReactantPrep(compiled_lp, compiled_value_and_gradient)

    return WrenCore.VecModel(
        model,
        adtype,
        lp_function,
        parameter_space,
        vecinfo,
        accs,
        length(sample),
        lp_func,
        adprep,
        sample,
    )
end

const ReactantWrenModel = WrenCore.VecModel{M,A} where {M<:WrenCore.Model,A<:AutoReactant}
_to_host(x::Reactant.ConcreteRNumber{T}) where {T} = T(x)
function WrenCore.lp(vecmodel::ReactantWrenModel, params::AbstractVector)
    return _to_host(vecmodel.adprep.primal(Reactant.to_rarray(params)))
end
function WrenCore.lp_and_grad(vecmodel::ReactantWrenModel, params::AbstractVector)
    value, grad = vecmodel.adprep.value_and_gradient(Reactant.to_rarray(params))
    return _to_host(value), Array(grad)
end
function WrenCore.lp_and_grad!(
    grad::AbstractVector,
    vecmodel::ReactantWrenModel,
    params::AbstractVector,
)
    length(grad) == vecmodel.dimension || throw(
        DimensionMismatch(
            "VecModel expects a gradient buffer of length $(vecmodel.dimension), but " *
            "received one of length $(length(grad)).",
        ),
    )
    value, allocating_grad = WrenCore.lp_and_grad(vecmodel, params)
    copyto!(grad, allocating_grad)
    return value, grad
end

end # module WrenCoreReactantExt
