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
function _batch_point(f, params)
    point_f = @set f.accs = copy(f.accs)
    return _batch_output(point_f(params))
end
_batch_points(f, params) = mapslices(x -> _batch_point(f, vec(x)), params; dims=1)

function _batch_derivatives(adtype, f, params)
    if isempty(params)
        values = _batch_points(f, params)
        return values, similar(values, size(values, 1), size(params)...)
    end
    mode = _with_primal(adtype)
    packed = mapslices(params; dims=1) do point
        point_f = x -> _batch_point(f, x)
        result = if mode isa Enzyme.ForwardMode
            Enzyme.jacobian(mode, point_f, vec(point); chunk=Val(1))
        else
            Enzyme.gradient(mode, x -> Reactant.@allowscalar(only(point_f(x))), vec(point))
        end
        vcat(_batch_output(result.val), vec(only(result.derivs)))
    end
    noutputs = size(packed, 1) ÷ (size(params, 1) + 1)
    if isempty(packed)
        return (
            packed,
            Reactant.materialize_traced_array(reshape(packed, noutputs, size(params)...)),
        )
    end
    values = packed[1:noutputs, :]
    derivatives = Reactant.materialize_traced_array(
        reshape(packed[(noutputs+1):end, :], noutputs, size(params)...),
    )
    return values, derivatives
end

struct BatchEvaluator{F,B,C}
    f::F
    batch_function::B
    compiled::C
    shape::Tuple{Int,Int}
end

function Reactant.traced_type_inner(
    ::Type{BatchEvaluator{F,B,C}},
    seen,
    mode::Reactant.TraceMode,
    track_numbers::Type,
    ndevices,
    runtime,
) where {F,B,C}
    traced = Reactant.traced_type_inner(F, seen, mode, track_numbers, ndevices, runtime)
    return BatchEvaluator{traced,B,C}
end

function Reactant.make_tracer(seen, batch::BatchEvaluator, path, mode; kwargs...)
    if mode == Reactant.TracedToTypes
        push!(path, typeof(batch), batch.shape)
        Reactant.make_tracer(seen, batch.f, path, mode; kwargs...)
        return nothing
    end
    f = Reactant.make_tracer(seen, batch.f, (path..., :f), mode; kwargs...)
    # The selected mode, executable, and fixed shape are preparation state.
    return BatchEvaluator(f, batch.batch_function, batch.compiled, batch.shape)
end

_batch_input(params::Reactant.ConcreteRArray) = params
_batch_input(params::AbstractMatrix) = Reactant.to_rarray(params)
function (batch::BatchEvaluator)(params::AbstractMatrix)
    size(params) == batch.shape || throw(
        DimensionMismatch(
            "prepared batch expects size $(batch.shape), received $(size(params))",
        ),
    )
    if Reactant.within_compile()
        return batch.batch_function(batch.f, params)
    end
    return batch.compiled(batch.f, _batch_input(params))
end

function WrenCore.prepare_batch(
    vm::WrenCore.VecModel,
    params::AbstractMatrix,
    output_function=WrenCore._density_parts;
    accs::WrenCore.AccumulatorTuple=vm.accs,
    adtype::Union{Nothing,AutoEnzyme}=nothing,
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
    batch_function = if adtype === nothing
        _batch_points
    else
        Base.Fix1(_batch_derivatives, adtype)
    end
    # The backend evaluator is created during preparation in the model's module.
    compiled = Base.invokelatest(Reactant.Compiler.compile, batch_function, (f, input))
    return BatchEvaluator(f, batch_function, compiled, size(params))
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
