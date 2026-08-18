module WrenReactantExt

using ADTypes: AutoEnzyme, AutoReactant
using Enzyme: Enzyme
using Reactant: Reactant
using Wren: Wren

"""
    ReactantPrep

Stores the Reactant-compiled functions.
"""
struct ReactantPrep{P,G,G2}
    primal::P
    value_and_gradient::G
    value_and_gradient!::G2
end
struct ValueAndGradientCallable{F,A}
    f::F
    adtype::A
end
struct ValueAndGradientCallable!{F,A}
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
function (f::ValueAndGradientCallable!)(params, grad)
    result = Enzyme.gradient!(_with_primal(f.adtype.mode), grad, f.f, params)
    return result.val, grad
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
    lp_at = Wren._VecModelLP(
        model.evaluation_fn,
        model.args,
        model.conditioned,
        getlogp,
        parameter_space,
        ranges_and_transforms,
        accs,
    )

    reactant_sample = Reactant.to_rarray(sample)
    value_and_gradient_at = ValueAndGradientCallable(lp_at, adtype)
    value_and_gradient_at! = ValueAndGradientCallable!(lp_at, adtype)
    compiled_lp = Reactant.@allowscalar Reactant.@compile lp_at(reactant_sample)
    compiled_value_and_gradient = Reactant.@allowscalar Reactant.@compile value_and_gradient_at(
        reactant_sample
    )
    compiled_value_and_gradient! = Reactant.@allowscalar Reactant.@compile value_and_gradient_at!(reactant_sample, reactant_sample)
    adprep = ReactantPrep(compiled_lp, compiled_value_and_gradient, compiled_value_and_gradient!)

    return Wren.VecModel(
        model,
        adtype,
        getlogp,
        parameter_space,
        ranges_and_transforms,
        sample,
        accs,
        lp_at,
        adprep,
    )
end

const ReactantWrenModel = Wren.VecModel{M,A} where {M<:Wren.Model,A<:AutoReactant}
_to_host(x::Reactant.ConcreteRNumber{T}) where {T} = T(x)
function Wren.lp(vecmodel::ReactantWrenModel, params::AbstractVector)
    return _to_host(vecmodel._adprep.primal(Reactant.to_rarray(params)))
end
function Wren.lp_and_grad(vecmodel::ReactantWrenModel, params::AbstractVector)
    value, grad = vecmodel._adprep.value_and_gradient(Reactant.to_rarray(params))
    return _to_host(value), Array(grad)
end
function Wren.lp_and_grad!(grad::AbstractVector, vecmodel::ReactantWrenModel, params::AbstractVector)
    # TODO(penelopeysm): This doesn't work yet
    @warn "lp_and_grad! doesn't work correctly yet"
    value, grad = vecmodel._adprep.value_and_gradient!(Reactant.to_rarray(params), Reactant.to_rarray(grad))
    return _to_host(value), Array(grad)
end

end # module WrenReactantExt
