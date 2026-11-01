function default_accumulators(::Type{FloatT}=LogProbType) where {FloatT}
    return AccumulatorTuple(
        LogPriorAccumulator{FloatT}(),
        LogJacobianAccumulator{FloatT}(),
        LogLikelihoodAccumulator{FloatT}(),
        ParamsAccumulator(),
    )
end

"""
    evaluate(model, latent_source, parameter_space, accs, ctx[, conditioned])

Evaluate `model` in three stages for every latent variable:

1. `latent_source` supplies a [`TransformedValue`](@ref).
2. `parameter_space` determines the model-space value, target representation, and
   forward log-Jacobian.
3. `accs` accumulate the resulting information.

All three components are required. Higher-level interfaces may choose defaults, but this
method keeps the evaluation pipeline explicit.

`ctx` is an [`EvalContext`](@ref), which says whether to make the copies of model arguments
that Enzyme needs. Unless the model is being differentiated with Enzyme, this should be
`EvalContext{false}()`.

`conditioned` defaults to `conditioned(model)`. A parent model passes other values when it
evaluates a submodel.
"""
@inline function evaluate(
    model::Model,
    latent_source::AbstractLatentSource,
    parameter_space::AbstractParameterSpace,
    accs::AccumulatorTuple,
    ctx::EvalContext,
    conditioned::NamedTuple=model.conditioned,
)
    return model.evaluation_fn(
        latent_source,
        parameter_space,
        accs,
        model.args,
        conditioned,
        model.enzyme_copy_argnames,
        ctx,
    )
end

function Base.rand(rng::AbstractRNG, model::Model)
    _, accs = evaluate(
        model,
        FromPrior(rng),
        OriginalSpace(),
        AccumulatorTuple(ParamsAccumulator()),
        EvalContext{false}(),
    )
    return VarNames.parameters(accs)
end
Base.rand(model::Model) = rand(default_rng(), model)

"""
    rand_with_logp([rng::Random.AbstractRNG,] model::Model)::AccumulatorTuple

Draw a sample from the prior distribution of `model` and return the params, log-prior,
and log-likelihood.

To extract the contents of the `AccumulatorTuple`, you can use [`parameters`](@ref),
[`logprior`](@ref), [`loglikelihood`](@ref), and [`logjoint`](@ref).
"""
function rand_with_logp(rng::AbstractRNG, model::Model)
    _, accs = evaluate(
        model,
        FromPrior(rng),
        OriginalSpace(),
        AccumulatorTuple(
            ParamsAccumulator(),
            LogPriorAccumulator(),
            LogLikelihoodAccumulator(),
        ),
        EvalContext{false}(),
    )
    return accs
end
rand_with_logp(model::Model) = rand_with_logp(default_rng(), model)

"""
    conditioned(model::Model)::NamedTuple

Return the current set of values that `model` is conditioned on.
"""
conditioned(model::Model) = model.conditioned

"""
    condition(model::Model, values::NamedTuple)::Model

Condition `model` on the values in `values`, returning a new model. The original model is
not modified. Each key conditions a whole variable, including every indexed or field
tilde-statement on that variable. Partial conditioning is not supported.

For a submodel, the value under its namespace is a `NamedTuple` of the child's
conditioned variables, for example `model | (child=(y=data,),)`.
Values replace existing keys as in `merge(::NamedTuple, ::NamedTuple)`, including a
nested submodel namespace. Nested values are not merged recursively.
Invalid lookups during evaluation report the affected variable or submodel namespace
and the supplied value.
"""
function condition(model::Model, values::NamedTuple)
    new_conditioned = merge(conditioned(model), values)
    return Model(
        model.name,
        model.args,
        model.stmts,
        model.mod;
        conditioned=new_conditioned,
        enzyme_copy_argnames=model.enzyme_copy_argnames,
        _unsafe_reuse_evaluation_fn=model.evaluation_fn,
    )
end

"""
    Base.:|(model::Model, values::NamedTuple)::Model

`model | values` is an alias for [`condition(model, values)`](@ref).
"""
Base.:|(model::Model, values::NamedTuple) = condition(model, values)

"""
    decondition(model::Model, vars::Union{Symbol,VarName}...)::Model

Remove whole variables from this model's conditioning, returning a new model.
Use symbols for top-level names, for example `decondition(model, :y)`, and qualified
`VarName`s for child variables, for example `decondition(model, @vn(child.y))` or
`decondition(model, @vn(children[1].y))`. With no selectors, all conditioning is removed.

Removing a parent override does not remove a child model's own conditioning.
Components of an ordinary observed array or record cannot be made latent individually.

Variables observed with `@observe` are part of the model's structure rather than its
conditioning, and are unaffected.
"""
function decondition(model::Model, vars::Union{Symbol,VarName}...)
    new_conditioned = if isempty(vars)
        NamedTuple()
    else
        foldl(vars; init=conditioned(model)) do values, var
            optic =
                var isa Symbol ? VarNames.Property{var}() : VarNames.varname_to_optic(var)
            _decondition(values, optic)
        end
    end
    return Model(
        model.name,
        model.args,
        model.stmts,
        model.mod;
        conditioned=new_conditioned,
        enzyme_copy_argnames=model.enzyme_copy_argnames,
        _unsafe_reuse_evaluation_fn=model.evaluation_fn,
    )
end

_decondition(values::NamedTuple, ::VarNames.Iden) = NamedTuple()
function _decondition(values::NamedTuple, optic::VarNames.Property{sym}) where {sym}
    haskey(values, sym) || return values
    if optic.child isa VarNames.Iden
        return Base.structdiff(values, NamedTuple{(sym,)})
    end
    child = _decondition(getproperty(values, sym), optic.child)
    # Keep empty parents: deleting a structured observation's root would make it latent.
    return merge(values, NamedTuple{(sym,)}((child,)))
end
function _decondition(values, optic::Union{VarNames.Property,VarNames.Index})
    head, tail = VarNames.ohead(optic), VarNames.otail(optic)
    return Accessors.set(values, head, _decondition(head(values), tail))
end
_decondition(values, ::VarNames.Iden) =
    throw(ArgumentError("decondition requires a whole variable or submodel namespace"))

"""
    logjoint(model::Model, params::VarNamedTuple)

Return the log-joint probability of `model` evaluated at `params` in original space.
"""
function logjoint(model::Model, params::VarNamedTuple)
    _, accs = evaluate(
        model,
        FromValues(params),
        OriginalSpace(),
        AccumulatorTuple(LogPriorAccumulator(), LogLikelihoodAccumulator()),
        EvalContext{false}(),
    )
    return logjoint(accs)
end
