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

############################################################################################
# parameters
############################################################################################

"""
    parameters(result::ModelResult)::VarNamedTuple

Get the parameters from a [`ModelResult`](@ref).
"""
VarNames.parameters(result::ModelResult) = result.parameters

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
    VarNames.parameters(params::AbstractVector, vecmodel::VecModel)::VarNamedTuple

Return the model-space parameters of `vecmodel` at `params`. If every transform is fixed,
the raw parameters are recovered without evaluating the model.
"""
function VarNames.parameters(params::AbstractVector, vecmodel::VecModel)
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

############################################################################################
# logprior
############################################################################################

"""
    logprior(result::ModelResult)

Get the log-prior from a [`ModelResult`](@ref).
"""
logprior(result::ModelResult) = result.logprior

############################################################################################
# loglikelihood
############################################################################################

"""
    loglikelihood(result::ModelResult)

Get the log-likelihood from a [`ModelResult`](@ref).
"""
StatsAPI.loglikelihood(result::ModelResult) = result.loglikelihood

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
