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
        vn::VarName,
        collect_params::CollectOrDiscardSubmodelParams,
    )

Evaluate a tilde-statement of the form `vn ~ rhs`, where `rhs` is a submodel. The
`collect_params` argument indicates whether the parent model might require access to the
parameters of the submodel as a top-level variable.
"""
@generated function evaluate_submodel(
    rhs,
    latent_source::AbstractLatentSource,
    parameter_space::AbstractParameterSpace,
    accs::AccumulatorTuple{<:Any,<:NamedTuple{names}},
    # TODO(penelopeysm): allow for no-prefix?
    # https://github.com/JuliaBayes/Wren.jl/issues/4
    vn::VarName,
    ::CollectSubmodelParams,
) where {names}
    parent_has_params_acc = PARAMS_ACCNAME in names
    expr = Expr(:block)
    push!(expr.args, :(child_source = scope(latent_source, vn)))
    push!(expr.args, :(child_space = scope(parameter_space, vn)))
    push!(expr.args, :(child_accs = scope(map(split, accs), vn)))
    if !parent_has_params_acc
        push!(expr.args, :(child_accs = setacc!!(child_accs, ParamsAccumulator())))
    end
    push!(
        expr.args,
        :((retval, child_accs) = evaluate(rhs, child_source, child_space, child_accs)),
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

function evaluate_submodel(
    rhs,
    latent_source::AbstractLatentSource,
    parameter_space::AbstractParameterSpace,
    accs::AccumulatorTuple,
    vn::VarName,
    ::DiscardSubmodelParams,
)
    child_source = scope(latent_source, vn)
    child_space = scope(parameter_space, vn)
    child_accs = scope(map(split, accs), vn)
    retval, child_accs = evaluate(rhs, child_source, child_space, child_accs)
    return retval, UncollectedParams(), child_accs
end
