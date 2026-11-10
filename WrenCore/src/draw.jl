function _draw_accumulators()
    return AccumulatorTuple(
        LogPriorAccumulator(),
        LogLikelihoodAccumulator(),
        ParamsAccumulator(),
    )
end

function _draw(accs::AccumulatorTuple, stats::NamedTuple)
    densified_params = VarNames.densify!!(VarNames.parameters(accs))
    has_prior = haskey(accs, Val(LOG_PRIOR_ACCNAME))
    has_likelihood = haskey(accs, Val(LOG_LIKELIHOOD_ACCNAME))
    has_prior && (stats = merge(stats, (logprior=logprior(accs),)))
    has_likelihood && (stats = merge(stats, (loglikelihood=loglikelihood(accs),)))
    if has_prior && has_likelihood
        stats = merge(stats, (logjoint=logjoint(accs),))
    end
    return VarNames.Draw(densified_params, stats)
end

"""
    VarNames.Draw(
        source::AbstractLatentSource,
        model::Model,
        stats::NamedTuple=NamedTuple(),
    )

Evaluate `model` using `source` and return its model-space parameters. The log prior, log
likelihood, and log joint are added to `stats`.
"""
function VarNames.Draw(
    source::AbstractLatentSource,
    model::Model,
    stats::NamedTuple=NamedTuple(),
)
    accs = _draw_accumulators()
    _, accs = evaluate(model, source, OriginalSpace(), accs, EvalContext{false}())
    return _draw(accs, stats)
end

"""
    VarNames.Draw(
        accs::AccumulatorTuple,
        stats::NamedTuple=NamedTuple(),
    )

Extract parameters and any available log-probability statistics from an accumulator tuple.
The tuple must contain a [`ParamsAccumulator`](@ref).
"""
function VarNames.Draw(accs::AccumulatorTuple, stats::NamedTuple=NamedTuple())
    return _draw(accs, stats)
end

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
    VarNames.Draw(
        params::AbstractVector,
        vecmodel::VecModel,
        stats::NamedTuple=NamedTuple(),
    )

Re-evaluate `vecmodel` at `params` and return model-space parameters together with
log-probability statistics.
"""
function VarNames.Draw(
    params::AbstractVector,
    vecmodel::VecModel,
    stats::NamedTuple=NamedTuple(),
)
    _check_vecmodel_dimension(params, vecmodel)
    source = FromVector(params, vecmodel.vecinfo)
    accs = _draw_accumulators()
    _, accs = evaluate(
        vecmodel.model,
        source,
        vecmodel.parameter_space,
        accs,
        EvalContext{false}(),
    )
    return _draw(accs, stats)
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

"""
    FromValues(
        draw::VarNames.Draw,
        fallback::Union{AbstractLatentSource,Nothing}=nothing,
    )

Create a latent source from the parameters in `draw`. The extras are ignored, and `fallback`
is used for any absent parameters.
"""
function FromValues(
    draw::VarNames.Draw,
    fallback::Union{AbstractLatentSource,Nothing}=FromPrior(),
)
    return FromValues(parameters(draw), fallback)
end

logprior(draw::VarNames.Draw) = extras(draw).logprior
StatsAPI.loglikelihood(draw::VarNames.Draw) = extras(draw).loglikelihood
logjoint(draw::VarNames.Draw) = extras(draw).logjoint
