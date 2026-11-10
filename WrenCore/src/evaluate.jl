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
    ModelResult

A struct containing the parameters, return value, log-prior, and log-likelihood of a model
evaluation.

You can access these fields with [`parameters`](@ref), [`return_value`](@ref),
[`logprior`](@ref), and [`loglikelihood`](@ref). The log-joint can be obtained with
[`logjoint`](@ref).
"""
struct ModelResult{V<:VarNamedTuple,R,LP,LL}
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
    simulate(model::Model)
    simulate(model::Model, latents)

Draw a sample from `model`, using the given latent source. If one is not provided,
draws from the prior distribution.

`latents` is converted to a latent source with [`to_latent_source`](@ref), so it can also
be an `AbstractRNG` (to draw from the prior) or a `VarNamedTuple` of values.

Returns the params, return value, log-prior, and log-likelihood of the model packaged in a
[`ModelResult`](@ref).
"""
function simulate(model::Model, latents::AbstractLatentSource)
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
simulate(model::Model, latents) = simulate(model, to_latent_source(latents))
simulate(model::Model) = simulate(model, FromPrior())

"""
    simulate(vecmodel::VecModel, params::AbstractVector)::ModelResult

Evaluate the model of `vecmodel` at the parameter vector `params`, which is in the parameter
space of `vecmodel`. The log-prior and log-likelihood in the result are in original space.
"""
function simulate(vecmodel::VecModel, params::AbstractVector)
    _check_vecmodel_dimension(params, vecmodel)
    retval, accs = evaluate(
        vecmodel.model,
        FromVector(params, vecmodel.vecinfo),
        vecmodel.parameter_space,
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

############################################################################################
# parameters
############################################################################################

"""
    parameters(result::ModelResult)::VarNamedTuple

Get the parameters from a [`ModelResult`](@ref).
"""
VarNames.parameters(result::ModelResult) = result.parameters

"""
    parameters(model::Model)::VarNamedTuple
    parameters(model::Model, latents)::VarNamedTuple

Obtain the model's parameters in original space, calculated with a given set of latent
values. If no latent values are provided, they are drawn from the prior distribution.

`latents` is converted to a latent source with [`to_latent_source`](@ref), so it can also
be an `AbstractRNG` (to draw from the prior) or a `VarNamedTuple` of values.

This is equivalent to `parameters(simulate(model))`, but is more efficient than
[`simulate`](@ref) because it does not compute the log-density terms.
"""
function VarNames.parameters(model::Model, latents::AbstractLatentSource)
    _, accs = evaluate(
        model,
        latents,
        OriginalSpace(),
        AccumulatorTuple(ParamsAccumulator()),
        EvalContext{false}(),
    )
    return VarNames.parameters(accs)
end
function VarNames.parameters(model::Model, latents)
    return VarNames.parameters(model, to_latent_source(latents))
end
VarNames.parameters(model::Model) = VarNames.parameters(model, FromPrior())

function _check_vecmodel_dimension(params::AbstractVector, vecmodel::VecModel)
    actual = length(params)
    expected = vecmodel.dimension
    actual == expected || throw(
        ArgumentError(
            "The parameter vector has length $actual, but the VecModel expects length " *
            "$expected.",
        ),
    )
    return nothing
end

function _all_fixed_transforms(vecmodel::VecModel)
    return all(
        rt -> rt.transform isa FixedTransform,
        values(vecmodel.vecinfo.varname_to_rat),
    )
end

function _raw_params_from_fixed_transforms(params::AbstractVector, vecinfo::VecInfo)
    varname_to_rat = vecinfo.varname_to_rat
    raw_params = VarNamedTuple()
    for (vn, rt) in pairs(varname_to_rat)
        template = get(varname_to_rat.data, VarNames.getsym(vn), VarNames.NoTemplate())
        raw_value = rt.transform.transform(params[rt.range])
        raw_params = VarNames.templated_setindex!!(raw_params, raw_value, vn, template)
    end
    return raw_params
end

"""
    VarNames.parameters(vecmodel::VecModel, params::AbstractVector)::VarNamedTuple

Return the model-space parameters of `vecmodel` at `params`. If every transform is fixed,
the raw parameters are recovered without evaluating the model.
"""
function VarNames.parameters(vecmodel::VecModel, params::AbstractVector)
    _check_vecmodel_dimension(params, vecmodel)
    if _all_fixed_transforms(vecmodel)
        return _raw_params_from_fixed_transforms(params, vecmodel.vecinfo)
    end

    source = FromVector(params, vecmodel.vecinfo)
    accs = AccumulatorTuple(ParamsAccumulator())
    _, accs = evaluate(
        vecmodel.model,
        source,
        vecmodel.parameter_space,
        accs,
        EvalContext{false}(),
    )
    return VarNames.parameters(accs)
end

############################################################################################
# return_value
############################################################################################

"""
    return_value(result::ModelResult)::R

Get the return value from a [`ModelResult`](@ref).
"""
return_value(result::ModelResult) = result.return_value

"""
    return_value(model::Model)
    return_value(model::Model, latents)

Obtain the model's return value calculated with a given set of latent values. If no latent
values are provided, they are drawn from the prior distribution.

`latents` is converted to a latent source with [`to_latent_source`](@ref), so it can also
be an `AbstractRNG` (to draw from the prior) or a `VarNamedTuple` of values.

This is equivalent to `return_value(simulate(model))`, but is more efficient than
[`simulate`](@ref) because it does not compute the log-density terms or store the
parameters.
"""
function return_value(model::Model, latents::AbstractLatentSource)
    retval, _ =
        evaluate(model, latents, OriginalSpace(), AccumulatorTuple(), EvalContext{false}())
    return retval
end
return_value(model::Model, latents) = return_value(model, to_latent_source(latents))
return_value(model::Model) = return_value(model, FromPrior())

"""
    return_value(vecmodel::VecModel, params::AbstractVector)

Obtain the return value of the model of `vecmodel` at the parameter vector `params`, which
is in the parameter space of `vecmodel`.
"""
function return_value(vecmodel::VecModel, params::AbstractVector)
    _check_vecmodel_dimension(params, vecmodel)
    retval, _ = evaluate(
        vecmodel.model,
        FromVector(params, vecmodel.vecinfo),
        vecmodel.parameter_space,
        AccumulatorTuple(),
        EvalContext{false}(),
    )
    return retval
end

############################################################################################
# logprior
############################################################################################

"""
    logprior(result::ModelResult)

Get the log-prior from a [`ModelResult`](@ref).
"""
logprior(result::ModelResult) = result.logprior

"""
    logprior(model::Model, latents)

Return the log-prior probability of `model` evaluated at the given latent values, in
original space.

`latents` is converted to a latent source with [`to_latent_source`](@ref), so it can also
be an `AbstractRNG` (to draw from the prior) or a `VarNamedTuple` of values.
"""
function logprior(model::Model, latents::AbstractLatentSource)
    _, accs = evaluate(
        model,
        latents,
        OriginalSpace(),
        AccumulatorTuple(LogPriorAccumulator()),
        EvalContext{false}(),
    )
    return logprior(accs)
end
logprior(model::Model, latents) = logprior(model, to_latent_source(latents))

############################################################################################
# loglikelihood
############################################################################################

"""
    loglikelihood(result::ModelResult)

Get the log-likelihood from a [`ModelResult`](@ref).
"""
StatsAPI.loglikelihood(result::ModelResult) = result.loglikelihood

"""
    loglikelihood(model::Model, latents)

Return the log-likelihood of `model` evaluated at the given latent values.

`latents` is converted to a latent source with [`to_latent_source`](@ref), so it can also
be an `AbstractRNG` (to draw from the prior) or a `VarNamedTuple` of values.
"""
function StatsAPI.loglikelihood(model::Model, latents::AbstractLatentSource)
    _, accs = evaluate(
        model,
        latents,
        OriginalSpace(),
        AccumulatorTuple(LogLikelihoodAccumulator()),
        EvalContext{false}(),
    )
    return loglikelihood(accs)
end
function StatsAPI.loglikelihood(model::Model, latents)
    return StatsAPI.loglikelihood(model, to_latent_source(latents))
end

############################################################################################
# logjoint
############################################################################################

"""
    logjoint(result::ModelResult)

Get the log-joint (log-prior plus log-likelihood) from a [`ModelResult`](@ref).
"""
logjoint(result::ModelResult) = logprior(result) + loglikelihood(result)
logjoint(::ModelResult{<:VarNamedTuple,<:Any,Nothing,<:Any}) = nothing
logjoint(::ModelResult{<:VarNamedTuple,<:Any,<:Any,Nothing}) = nothing
logjoint(::ModelResult{<:VarNamedTuple,<:Any,Nothing,Nothing}) = nothing

"""
    logjoint(model::Model, latents)

Return the log-joint probability of `model` evaluated at the given latent values, in
original space.

`latents` is converted to a latent source with [`to_latent_source`](@ref), so it can also
be an `AbstractRNG` (to draw from the prior) or a `VarNamedTuple` of values.
"""
function logjoint(model::Model, latents::AbstractLatentSource)
    _, accs = evaluate(
        model,
        latents,
        OriginalSpace(),
        AccumulatorTuple(LogPriorAccumulator(), LogLikelihoodAccumulator()),
        EvalContext{false}(),
    )
    return logjoint(accs)
end
logjoint(model::Model, latents) = logjoint(model, to_latent_source(latents))
