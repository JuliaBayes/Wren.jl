module WrenCoreReactantExt

using ADTypes: AutoEnzyme, AutoReactant
using Enzyme: Enzyme
using Reactant: Reactant
using WrenCore: WrenCore

"""
    ReactantPrep

Stores the Reactant-compiled functions.
"""
struct ReactantPrep{P,G}
    primal::P
    value_and_gradient::G
    # Batched functions are compiled lazily, once per (eltype, number of columns).
    batch_primal::Dict{Tuple{DataType,Int},Any}
    batch_value_and_gradient::Dict{Tuple{DataType,Int},Any}
end
ReactantPrep(primal, value_and_gradient) = ReactantPrep(
    primal,
    value_and_gradient,
    Dict{Tuple{DataType,Int},Any}(),
    Dict{Tuple{DataType,Int},Any}(),
)
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
_grad_mode(::AutoEnzyme{Nothing}) = Enzyme.Reverse
_grad_mode(adtype::AutoEnzyme) = adtype.mode

# Columns are independent, so the gradient of the summed log-densities is the matrix of
# per-column gradients.
struct SummedLP{B}
    b::B
end
(s::SummedLP)(params) = sum(s.b(params))

struct BatchedValueAndGradientCallable{B,A}
    b::B
    adtype::A
end
function (f::BatchedValueAndGradientCallable)(params)
    grad = only(Enzyme.gradient(_grad_mode(f.adtype.mode), SummedLP(f.b), params))
    return f.b(params), grad
end

# `Ops.batch` vectorises the per-column log-density over the columns of `params`, so the
# compiled program does not grow with the number of columns.
function (b::WrenCore.BatchedLP)(params::Reactant.AnyTracedRMatrix)
    params = Reactant.TracedUtils.materialize_traced_array(params)
    return vec(Reactant.Ops.batch(b.lp_func, params, [1]))
end

function WrenCore.VecModel(
    model::WrenCore.Model,
    adtype::AutoReactant;
    latent_source::WrenCore.AbstractLatentSource=WrenCore.FromPrior(),
    lp_function=WrenCore.logjoint_transformed,
    parameter_space::WrenCore.AbstractParameterSpace=WrenCore.UnconstrainedSpace(),
)
    vecinfo, sample = WrenCore._prepare_vecmodel_info(model, latent_source, parameter_space)

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
        model.evaluation_fn,
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
    compiled_lp = Reactant.@compile lp_func(reactant_sample)
    compiled_value_and_gradient = Reactant.@compile value_and_gradient_at(reactant_sample)
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

function _compiled_batch(f, cache::Dict, params::AbstractMatrix)
    return get!(cache, (eltype(params), size(params, 2))) do
        Reactant.@compile f(Reactant.to_rarray(params))
    end
end
function WrenCore.lp_batch(vecmodel::ReactantWrenModel, params::AbstractMatrix)
    WrenCore._check_batch_size(vecmodel, params)
    b = WrenCore.batched_lp_function(vecmodel)
    compiled = _compiled_batch(b, vecmodel.adprep.batch_primal, params)
    return Array(compiled(Reactant.to_rarray(params)))
end
function WrenCore.lp_and_grad_batch(vecmodel::ReactantWrenModel, params::AbstractMatrix)
    WrenCore._check_batch_size(vecmodel, params)
    f = BatchedValueAndGradientCallable(
        WrenCore.batched_lp_function(vecmodel),
        vecmodel.adtype,
    )
    compiled = _compiled_batch(f, vecmodel.adprep.batch_value_and_gradient, params)
    values, grads = compiled(Reactant.to_rarray(params))
    return Array(values), Array(grads)
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
