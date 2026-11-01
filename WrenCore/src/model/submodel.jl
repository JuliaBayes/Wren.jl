"""
    is_submodel(::Type{T}) where {T}

Return true if objects of `T` should be interpreted as submodels when used on the right-hand
side of tilde expressions.

For a custom object `t` to be used as a submodel, `is_submodel(typeof(t))` must return true,
and [`evaluate_submodel`](@ref) must be overloaded for that type.`
"""
is_submodel(::Type) = false
is_submodel(::Type{<:Model}) = true

"""
    CollectOrDiscardSubmodelParams

Internal types used to denote whether the parameters of a submodel should be collected as a
top-level variable in the parent model. For example, in

```julia
@wren function f()
    retval = x ~ submodel
    y ~ x[@vn(a)]
end
```

we need `x` to contain a `VarNamedTuple` of the submodel's parameters, otherwise the next
line will fail. However, if `x` is never used again in the model body, then we can avoid
collecting the submodel's parameters. This makes performance better in hot paths that don't
use `ParamsAccumulator`, such as `VecModel` (and probably also improves type stability).
"""
abstract type CollectOrDiscardSubmodelParams end
struct CollectSubmodelParams <: CollectOrDiscardSubmodelParams end
struct DiscardSubmodelParams <: CollectOrDiscardSubmodelParams end

"""
    evaluate_submodel(
        rhs,
        latent_source::AbstractLatentSource,
        parameter_space::AbstractParameterSpace,
        accs::AccumulatorTuple,
        conditioned::NamedTuple,
        vn::VarName,
        collect_params::CollectOrDiscardSubmodelParams,
        ctx::EvalContext,
    )

Evaluate a tilde-statement of the form `vn ~ rhs`, where `rhs` is a submodel. `conditioned`
holds the parent model's conditioned values. The values under `vn` take precedence over the
submodel's own. The `collect_params` argument indicates whether the parent model might
require access to the parameters of the submodel as a top-level variable. `ctx` is the
parent model's [`EvalContext`](@ref), which should be passed on when evaluating the
submodel.
"""
function evaluate_submodel(
    rhs,
    latent_source::AbstractLatentSource,
    parameter_space::AbstractParameterSpace,
    accs::AccumulatorTuple,
    conditioned::NamedTuple,
    # TODO(penelopeysm): allow for no-prefix?
    # https://github.com/JuliaBayes/Wren.jl/issues/4
    vn::VarName,
    collect_params::CollectOrDiscardSubmodelParams,
    ctx::EvalContext,
)
    scoped_args =
        _scope_submodel(rhs, latent_source, parameter_space, accs, conditioned, vn)
    return _evaluate_scoped_submodel(rhs, scoped_args..., collect_params, ctx)
end

function _scope_submodel(rhs, latent_source, parameter_space, accs, conditioned, vn)
    child_source = scope(latent_source, vn)
    child_space = scope(parameter_space, vn)
    child_accs = scope(map(split, accs), vn)
    child_conditioned = merge(rhs.conditioned, scope(conditioned, vn))
    return child_source, child_space, child_accs, child_conditioned
end

@generated function _evaluate_scoped_submodel(
    rhs,
    latent_source::AbstractLatentSource,
    parameter_space::AbstractParameterSpace,
    accs::AccumulatorTuple{<:Any,<:NamedTuple{names}},
    conditioned::NamedTuple,
    ::CollectSubmodelParams,
    ctx::EvalContext,
) where {names}
    parent_has_params_acc = PARAMS_ACCNAME in names
    expr = Expr(:block)
    push!(expr.args, :(child_accs = accs))
    if !parent_has_params_acc
        push!(expr.args, :(child_accs = setacc!!(child_accs, ParamsAccumulator())))
    end
    push!(
        expr.args,
        :(
            (retval, child_accs) =
                evaluate(rhs, latent_source, parameter_space, child_accs, ctx, conditioned)
        ),
    )
    push!(expr.args, :(child_params = VarNames.parameters(child_accs)))
    if !parent_has_params_acc
        # If the parent wasn't collecting parameters, delete this before returning to the
        # parent, to avoid the merge
        #
        # TODO(penelopeysm): maybe we don't need to delete it here, but instead change
        # `combine_with_namespace` to just ignore unmatched accumulators?
        push!(expr.args, :(child_accs = deleteacc!!(child_accs, Val(PARAMS_ACCNAME))))
    end
    push!(expr.args, :(return (retval, child_params, child_accs)))
    return expr
end

# Sentinel value to indicate that submodel parameters were not collected as a top-level
# variable in the model. We use this instead of `nothing` so that trying to do anything with
# it yields a more informative error message.
#
# Note that this really should NOT happen because Wren should only emit
# `DiscardSubmodelParams` if it can prove that these aren't reused.
struct UncollectedParams end
function Base.getindex(::UncollectedParams, ::VarName)
    # would like to define getindex(::UncollectedParams, ::Any), but I fear method
    # ambiguities.
    _issue_error("submodel parameters are not available as a top-level variable")
end
function Base.getproperty(::UncollectedParams, ::Symbol)
    _issue_error("submodel parameters are not available as a top-level variable")
end
# in e.g. `x[1] ~ submodel`, if we don't need the params, don't needlessly set them back
# into `x` as that can trigger type instabilities.
set_lhs!!(obj, optic, ::UncollectedParams) = obj
set_namespace_lhs!!(obj, observed, optic, ::UncollectedParams) = obj

function _evaluate_scoped_submodel(
    rhs,
    latent_source::AbstractLatentSource,
    parameter_space::AbstractParameterSpace,
    accs::AccumulatorTuple,
    conditioned::NamedTuple,
    ::DiscardSubmodelParams,
    ctx::EvalContext,
)
    retval, child_accs =
        evaluate(rhs, latent_source, parameter_space, accs, ctx, conditioned)
    return retval, UncollectedParams(), child_accs
end

# The parent's conditioned values for the submodel at `vn`, keyed relative to `vn`.
@inline function scope(conditioned::NamedTuple, vn::VarName)
    if haskey(conditioned, VarNames.getsym(vn))
        supplied = getproperty(conditioned, VarNames.getsym(vn))
        child = _lookup_conditioned(supplied, VarNames.getoptic(vn), vn, supplied, true)
        if !(child isa NamedTuple)
            _conditioned_lookup_error(
                true,
                vn,
                supplied,
                ArgumentError("expected a NamedTuple, got $(typeof(child))"),
            )
        end
        return child
    else
        return NamedTuple()
    end
end

_scope_conditioned(values, ::VarNames.Iden) = values
function _scope_conditioned(values::NamedTuple, optic::VarNames.Property{sym}) where {sym}
    if haskey(values, sym)
        return _scope_conditioned(getproperty(values, sym), optic.child)
    else
        return NamedTuple()
    end
end
_scope_conditioned(values, optic::Union{VarNames.Property,VarNames.Index}) =
    _scope_conditioned(VarNames.ohead(optic)(values), VarNames.otail(optic))

# VarNames.getvalue checks canview first, which currently rejects tuple indexing.
# Direct optic access also supports e.g. (y=(1.0,),) conditioned at @vn(y[1]).
@inline function _conditioned_value(values::NamedTuple, vn::VarName{sym}) where {sym}
    supplied = getproperty(values, sym)
    return _lookup_conditioned(supplied, VarNames.getoptic(vn), vn, supplied, false)
end

_lookup_conditioned(value, ::VarNames.Iden, vn, supplied, namespace) = value

@inline function _lookup_conditioned(
    value::NamedTuple,
    optic::VarNames.Property{sym},
    vn,
    supplied,
    namespace,
) where {sym}
    if haskey(value, sym)
        return _lookup_conditioned(
            getproperty(value, sym),
            optic.child,
            vn,
            supplied,
            namespace,
        )
    elseif namespace
        return NamedTuple()
    end
    return _conditioned_lookup_failure(value, optic, vn, supplied, namespace)
end

# Catching every successful lookup inhibits inlining and loop optimisation. Keep
# native scalar indexing outside exception handling, while unusual indexers retain
# their own getindex semantics through the fallback.
@inline function _lookup_conditioned(
    value::Array,
    optic::VarNames.Index,
    vn,
    supplied,
    namespace,
)
    if isempty(optic.kw) && all(i -> i isa Union{Int,CartesianIndex}, optic.ix)
        assigned = isbitstype(eltype(value)) || isassigned(value, optic.ix...)
        if checkbounds(Bool, value, optic.ix...) && assigned
            return _lookup_conditioned(
                getindex(value, optic.ix...),
                optic.child,
                vn,
                supplied,
                namespace,
            )
        end
        _conditioned_lookup_failure(value, optic, vn, supplied, namespace)
    end
    return _lookup_conditioned_fallback(value, optic, vn, supplied, namespace)
end

@inline function _lookup_conditioned(
    value::Tuple,
    optic::VarNames.Index,
    vn,
    supplied,
    namespace,
)
    if isempty(optic.kw) && length(optic.ix) == 1 && only(optic.ix) isa Int
        i = only(optic.ix)
        if 1 <= i <= length(value)
            return _lookup_conditioned(value[i], optic.child, vn, supplied, namespace)
        end
        _conditioned_lookup_failure(value, optic, vn, supplied, namespace)
    end
    return _lookup_conditioned_fallback(value, optic, vn, supplied, namespace)
end

@inline function _lookup_conditioned(
    value::Base.RefValue,
    optic::VarNames.Index,
    vn,
    supplied,
    namespace,
)
    if isempty(optic.kw) && isempty(optic.ix)
        isassigned(value) ||
            _conditioned_lookup_failure(value, optic, vn, supplied, namespace)
        return _lookup_conditioned(value[], optic.child, vn, supplied, namespace)
    end
    return _lookup_conditioned_fallback(value, optic, vn, supplied, namespace)
end

_lookup_conditioned(value, optic, vn, supplied, namespace) =
    _lookup_conditioned_fallback(value, optic, vn, supplied, namespace)

# Preserve a non-returning bounds-failure edge for the optimiser and AD backend.
@noinline function _conditioned_lookup_failure(value, optic, vn, supplied, namespace)
    _lookup_conditioned_fallback(value, optic, vn, supplied, namespace)
    _conditioned_lookup_error(
        namespace,
        vn,
        supplied,
        ArgumentError("invalid conditioned access"),
    )
end

@noinline function _lookup_conditioned_fallback(value, optic, vn, supplied, namespace)
    try
        if namespace
            return _scope_conditioned(value, optic)
        else
            return optic(value)
        end
    catch error
        error isa InterruptException && rethrow()
        _conditioned_lookup_error(namespace, vn, supplied, error)
    end
end

@noinline function _conditioned_lookup_error(namespace, vn, supplied, error)
    kind = if namespace
        "submodel namespace"
    else
        "variable"
    end
    value = sprint(show, supplied; context=(:compact => true, :limit => true))
    cause = sprint(showerror, error)
    throw(ArgumentError("could not access conditioned $kind $vn in $value: $cause"))
end
