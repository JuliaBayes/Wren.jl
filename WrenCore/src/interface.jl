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

"""
    Base.rand([rng::Random.AbstractRNG,] model::Model)::VarNamedTuple

Draw a sample from the prior distribution of `model`, returning the parameters in
original space.

This is equivalent to `parameters(simulate(model))`, but is more efficient than
[`simulate`](@ref) because it does not compute the log-density terms.
"""
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
    return_value(model::Model)
    return_value(rng::Random.AbstractRNG, model::Model)
    return_value(latents::AbstractLatentSource, model::Model)

Obtain the model's return value calculated with a given set of latent values. If no latent
values are provided, they are drawn from the prior distribution, optionally with a specified
`rng`.

This is equivalent to `return_value(simulate(model))`, but is more efficient than
[`simulate`](@ref) because it does not compute the log-density terms or store the
parameters.
"""
function return_value(latents::AbstractLatentSource, model::Model)
    retval, _ =
        evaluate(model, latents, OriginalSpace(), AccumulatorTuple(), EvalContext{false}())
    return retval
end
return_value(rng::AbstractRNG, model::Model) = return_value(FromPrior(rng), model)
return_value(model::Model) = return_value(FromPrior(), model)

"""
    ModelResult

A struct containing the parameters, return value, log-prior, and log-likelihood of a model
evaluation.

You can access these fields with [`parameters`](@ref), [`return_value`](@ref),
[`logprior`](@ref), and [`loglikelihood`](@ref). The log-joint can be obtained with
[`logjoint`](@ref).
"""
struct ModelResult{R,V<:VarNamedTuple,LP,LL}
    parameters::V
    return_value::R
    logprior::LP
    loglikelihood::LL
end

function Base.show(io::IO, ::MIME"text/plain", result::ModelResult)
    println(io, "ModelResult")
    print(io, "├─ ")
    printstyled(io, "parameters"; color=:cyan)
    print(io, " => ")
    if isempty(result.parameters)
        print(io, "VarNamedTuple()")
    else
        vnt_pretty_print(io, result.parameters, "│  ", 0)
    end
    println(io)
    fields = (
        return_value=result.return_value,
        logprior=result.logprior,
        loglikelihood=result.loglikelihood,
    )
    for (i, (name, value)) in enumerate(pairs(fields))
        print(io, i == length(fields) ? "└─ " : "├─ ")
        printstyled(io, string(name); color=:cyan)
        print(io, " => ")
        show(io, value)
        i < length(fields) && println(io)
    end
    return nothing
end

"""
    parameters(result::ModelResult)::VarNamedTuple

Get the parameters from a [`ModelResult`](@ref).
"""
VarNames.parameters(result::ModelResult) = result.parameters

"""
    return_value(result::ModelResult)::R

Get the return value from a [`ModelResult`](@ref).
"""
return_value(result::ModelResult) = result.return_value

"""
    logprior(result::ModelResult)

Get the log-prior from a [`ModelResult`](@ref).
"""
logprior(result::ModelResult) = result.logprior

"""
    loglikelihood(result::ModelResult)

Get the log-likelihood from a [`ModelResult`](@ref).
"""
StatsAPI.loglikelihood(result::ModelResult) = result.loglikelihood

"""
    logjoint(result::ModelResult)

Get the log-joint (log-prior plus log-likelihood) from a [`ModelResult`](@ref).
"""
logjoint(result::ModelResult) = logprior(result) + loglikelihood(result)

"""
    VarNames.densify!!(result::ModelResult)

Return a [`ModelResult`](@ref) with its parameters densified using
[`VarNames.densify!!`](@extref). The other fields are unchanged.
"""
function VarNames.densify!!(result::ModelResult)
    return ModelResult(
        VarNames.densify!!(result.parameters),
        result.return_value,
        result.logprior,
        result.loglikelihood,
    )
end

"""
    VarNames.Draw(::ModelResult)

Convert a [`ModelResult`](@ref) into a [`VarNames.Draw`](@extref). The log-prior,
log-likelihood, log-joint, and return value are stored as extras with the names
`logprior`, `loglikelihood`, `logjoint`, and `return_value`.
"""
VarNames.Draw(result::ModelResult) = VarNames.Draw(
    VarNames.densify!!(result.parameters),
    (
        logprior=result.logprior,
        loglikelihood=result.loglikelihood,
        logjoint=logjoint(result),
        return_value=result.return_value,
    ),
)

"""
    ModelResult(::VarNames.Draw)

Convert a [`VarNames.Draw`](@extref) back into a [`ModelResult`](@ref). The log-prior,
log-likelihood, and return value are extracted from the extras of the `Draw`. If the
corresponding keys are not present, the fields are set to `nothing`.
"""
function ModelResult(draw::VarNames.Draw)
    ModelResult(
        VarNames.parameters(draw),
        get(VarNames.extras(draw), :return_value, nothing),
        get(VarNames.extras(draw), :logprior, nothing),
        get(VarNames.extras(draw), :loglikelihood, nothing),
    )
end

"""
    simulate(model::Model)
    simulate(rng::Random.AbstractRNG, model::Model)
    simulate(latents::AbstractLatentSource, model::Model)

Draw a sample from `model`, using the given latent source. If one is not provided,
draws from the prior distribution, optionally with a specified `rng`.

Returns the params, return value, log-prior, and log-likelihood of the model packaged in a
[`ModelResult`](@ref).
"""
function simulate(latents::AbstractLatentSource, model::Model)
    retval, accs = evaluate(
        model,
        latents,
        OriginalSpace(),
        AccumulatorTuple(
            ParamsAccumulator(),
            LogPriorAccumulator(),
            LogLikelihoodAccumulator(),
        ),
        EvalContext{false}(),
    )
    return ModelResult(
        VarNames.parameters(accs),
        retval,
        logprior(accs),
        loglikelihood(accs),
    )
end
simulate(rng::AbstractRNG, model::Model) = simulate(FromPrior(rng), model)
simulate(model::Model) = simulate(FromPrior(), model)

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
