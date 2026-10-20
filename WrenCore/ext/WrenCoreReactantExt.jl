module WrenCoreReactantExt

using ADTypes: AutoEnzyme, AutoReactant
using Enzyme: Enzyme
using Reactant: Reactant
using WrenCore: WrenCore

include("reactant_model.jl")

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

const _COMPILE_SCOPE = gensym(:wren_reactant_compile)
_within_wren_compile()::Bool = get(task_local_storage(), _COMPILE_SCOPE, false)
function _compile_vecmodel(f, params)
    return task_local_storage(_COMPILE_SCOPE, true) do
        Reactant.@compile f(params)
    end
end

# Numeric buffers must hold traced values, including matmul destinations (#3195).
# Integer metadata stays on the host; other Reactant callers retain native allocation.
_traced_buffer(::Type{T}) where {T} =
    T <: Reactant.TracedRNumber || (T <: AbstractFloat && T <: Reactant.ReactantPrimitive)
Reactant.@reactant_overlay function Base.similar(
    a::Array,
    ::Type{S},
    dims::Dims{N},
) where {S,N}
    if Reactant.call_with_native(_within_wren_compile) && _traced_buffer(S)
        return Reactant.call_with_native(
            similar,
            Reactant.TracedRArray{Reactant.unwrapped_eltype(S)},
            dims,
        )
    end
    return Reactant.call_with_native(similar, a, S, dims)
end
# Array's shorter forms allocate directly, so route them through the same policy.
Reactant.@reactant_overlay function Base.similar(a::Array)
    return Reactant.call_with_reactant(similar, a, eltype(a), size(a))
end
Reactant.@reactant_overlay function Base.similar(a::Array, ::Type{S}) where {S}
    return Reactant.call_with_reactant(similar, a, S, size(a))
end
Reactant.@reactant_overlay function Base.similar(a::Array, dims::Dims)
    return Reactant.call_with_reactant(similar, a, eltype(a), dims)
end
Reactant.@reactant_overlay function Base.similar(a::Array, n::Int)
    return Reactant.call_with_reactant(similar, a, eltype(a), (n,))
end
Reactant.@reactant_overlay function Base.copy(a::Array{T}) where {T}
    if Reactant.call_with_native(_within_wren_compile) && _traced_buffer(T)
        tensor = Reactant.call_with_native(Reactant.promote_to, Reactant.TracedRArray, a)
        return Reactant.call_with_native(copy, tensor)
    end
    return Reactant.call_with_native(copy, a)
end

function WrenCore.VecModel(
    model::WrenCore.Model,
    adtype::AutoReactant;
    latent_source::WrenCore.AbstractLatentSource=WrenCore.FromPrior(),
    lp_function=WrenCore.logjoint_transformed,
    parameter_space::WrenCore.AbstractParameterSpace=WrenCore.UnconstrainedSpace(),
    # Accepted for compatibility; Reactant does not use WrenCore.reduce_copies.
    reduce_copies::Bool=true,
    aliases=nothing,
)
    vecinfo, sample =
        WrenCore._prepare_vecmodel_info(model, latent_source, parameter_space; aliases)
    _reactant_check_aliases(model, vecinfo)

    accs = WrenCore.AccumulatorTuple(
        WrenCore.LogPriorAccumulator(),
        WrenCore.LogJacobianAccumulator(),
        WrenCore.LogLikelihoodAccumulator(),
    )

    lp_func = WrenCore._VecModelLP(
        _reactant_evaluation_fn(model),
        # Index vectors must stay on the host for VarName cache checks and lookup.
        model.args,
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
    compiled_lp = _compile_vecmodel(lp_func, reactant_sample)
    compiled_value_and_gradient = _compile_vecmodel(value_and_gradient_at, reactant_sample)
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
