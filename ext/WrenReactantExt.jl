module WrenReactantExt

using ADTypes: AutoEnzyme, AutoReactant
using Enzyme: Enzyme
using Reactant: Reactant
using Wren: Wren

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
function Wren.VecModel(
    model::Wren.Model,
    adtype::AutoReactant;
    params::Union{Wren.VarNamedTuple,Wren.AbstractLatentSource}=Wren.FromPrior(),
    getlogp=Wren.getlogjoint_transformed,
    parameter_space::Wren.AbstractParameterSpace=Wren.UnconstrainedSpace(),
)
    values = if params isa Wren.VarNamedTuple
        params
    else
        Wren._draw_vector_values(model, params, parameter_space)
    end
    Wren._validate_vector_values(values, parameter_space)
    ranges_and_transforms, sample = Wren._ranges_and_params(values)

    accs = Wren.AccumulatorTuple(
        Wren.LogPriorAccumulator(),
        Wren.LogJacobianAccumulator(),
        Wren.LogLikelihoodAccumulator(),
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

    lp_func = Wren._VecModelLP(
        model.evaluation_fn,
        model_args,
        model.conditioned,
        getlogp,
        parameter_space,
        ranges_and_transforms,
        accs,
    )

    reactant_sample = Reactant.to_rarray(sample)
    value_and_gradient_at = ValueAndGradientCallable(lp_func, adtype)
    compiled_lp = Reactant.@allowscalar Reactant.@compile lp_func(reactant_sample)
    compiled_value_and_gradient = Reactant.@allowscalar Reactant.@compile value_and_gradient_at(
        reactant_sample
    )
    adprep = ReactantPrep(compiled_lp, compiled_value_and_gradient)

    return Wren.VecModel(
        model,
        adtype,
        getlogp,
        parameter_space,
        ranges_and_transforms,
        accs,
        length(sample),
        lp_func,
        adprep,
        sample,
    )
end

const ReactantWrenModel = Wren.VecModel{M,A} where {M<:Wren.Model,A<:AutoReactant}
_to_host(x::Reactant.ConcreteRNumber{T}) where {T} = T(x)
function Wren.lp(vecmodel::ReactantWrenModel, params::AbstractVector)
    return _to_host(vecmodel.adprep.primal(Reactant.to_rarray(params)))
end
function Wren.lp_and_grad(vecmodel::ReactantWrenModel, params::AbstractVector)
    value, grad = vecmodel.adprep.value_and_gradient(Reactant.to_rarray(params))
    return _to_host(value), Array(grad)
end
function Wren.lp_and_grad!(grad::AbstractVector, vecmodel::ReactantWrenModel, params::AbstractVector)
    length(grad) == vecmodel.dimension || throw(
        DimensionMismatch(
            "VecModel expects a gradient buffer of length $(vecmodel.dimension), but " *
            "received one of length $(length(grad)).",
        ),
    )
    value, allocating_grad = Wren.lp_and_grad(vecmodel, params)
    copyto!(grad, allocating_grad)
    return value, grad
end

end # module WrenReactantExt
