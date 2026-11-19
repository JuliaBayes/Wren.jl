module WrenCoreReactantExt

using ADTypes: AutoEnzyme, AutoReactant
using Enzyme: Enzyme
using Reactant: Reactant
using WrenCore: WrenCore

const COMPILE_LOCK = ReentrantLock()

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
function _reactant_lp_function(model, lp_function, parameter_space, vecinfo, accs)
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

    return WrenCore._VecModelLP(
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
end

function _reactant_prep(model, lp_function, parameter_space, vecinfo, accs, sample, adtype)
    # Reactant's tracing caches are shared across threads.
    return lock(COMPILE_LOCK) do
        f = _reactant_lp_function(model, lp_function, parameter_space, vecinfo, accs)
        reactant_sample = Reactant.to_rarray(sample)
        primal = Reactant.@compile f(reactant_sample)
        gradient = if adtype === nothing
            nothing
        else
            value_and_gradient_at = ValueAndGradientCallable(f, adtype)
            Reactant.@compile value_and_gradient_at(reactant_sample)
        end
        return f, ReactantPrep(primal, gradient)
    end
end

function WrenCore.VecModel(
    model::WrenCore.Model,
    adtype::AutoReactant;
    latent_source::WrenCore.AbstractLatentSource=WrenCore.FromPrior(),
    lp_function=WrenCore.logjoint_transformed,
    parameter_space::WrenCore.AbstractParameterSpace=WrenCore.UnconstrainedSpace(),
    # Match the VecModel constructor interface.
    reduce_copies::Bool=true,
)
    vecinfo, sample = WrenCore._prepare_vecmodel_info(model, latent_source, parameter_space)
    accs = WrenCore.AccumulatorTuple(
        WrenCore.LogPriorAccumulator(),
        WrenCore.LogJacobianAccumulator(),
        WrenCore.LogLikelihoodAccumulator(),
    )
    lp_func, adprep =
        _reactant_prep(model, lp_function, parameter_space, vecinfo, accs, sample, adtype)

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
_to_host(x::Number) = x
_to_host(x::Reactant.ConcreteRNumber{T}) where {T} = T(x)

function WrenCore._compile_log_density(
    vm::WrenCore.VecModel,
    ::AutoReactant;
    lp_function=vm.lp_function,
    accs=vm.accs,
)
    _, prep = _reactant_prep(
        vm.model,
        lp_function,
        vm.parameter_space,
        vm.vecinfo,
        accs,
        vm.sample,
        nothing,
    )
    return params -> _to_host(prep.primal(Reactant.to_rarray(params)))
end

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
