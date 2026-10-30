function _draw_accumulators(include_log_probs::Bool)
    return if include_log_probs
        AccumulatorTuple(
            LogPriorAccumulator(),
            LogLikelihoodAccumulator(),
            ParamsAccumulator(),
            GQAccumulator(),
        )
    else
        AccumulatorTuple(ParamsAccumulator(), GQAccumulator())
    end
end

function _params_and_gqs(accs::AccumulatorTuple)
    params = VarNames.parameters(accs)
    hasacc(accs, Val(GQ_ACCNAME)) || return params
    return merge(params, generated_quantities(accs))
end

function _draw(accs::AccumulatorTuple, stats::NamedTuple, include_log_probs::Bool)
    densified_params = VarNames.densify!!(_params_and_gqs(accs))
    if include_log_probs
        has_prior = haskey(accs, Val(LOG_PRIOR_ACCNAME))
        has_likelihood = haskey(accs, Val(LOG_LIKELIHOOD_ACCNAME))
        has_prior && (stats = merge(stats, (logprior=logprior(accs),)))
        has_likelihood && (stats = merge(stats, (loglikelihood=loglikelihood(accs),)))
        if has_prior && has_likelihood
            stats = merge(stats, (logjoint=logjoint(accs),))
        end
    end
    return VarNames.Draw(densified_params, stats)
end

"""
    VarNames.Draw(
        source::AbstractLatentSource,
        model::Model,
        stats::NamedTuple=NamedTuple();
        include_log_probs::Bool=true,
    )

Evaluate `model` using `source` and return its model-space parameters and generated
quantities. By default the log prior, log likelihood, and log joint are added to `stats`.
"""
function VarNames.Draw(
    source::AbstractLatentSource,
    model::Model,
    stats::NamedTuple=NamedTuple();
    include_log_probs::Bool=true,
)
    accs = _draw_accumulators(include_log_probs)
    _, accs = evaluate(model, source, OriginalSpace(), accs, EvalContext{false}())
    return _draw(accs, stats, include_log_probs)
end

"""
    VarNames.Draw(
        accs::AccumulatorTuple,
        stats::NamedTuple=NamedTuple();
        include_log_probs::Bool=true,
    )

Extract parameters and any available log-probability statistics from an accumulator tuple.
The tuple must contain a [`ParamsAccumulator`](@ref). If there are generated quantities, they
are merged into the parameters.
"""
function VarNames.Draw(
    accs::AccumulatorTuple,
    stats::NamedTuple=NamedTuple();
    include_log_probs::Bool=true,
)
    return _draw(accs, stats, include_log_probs)
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
    return VarNames.densify!!(raw_params)
end

"""
    VarNames.Draw(
        params::AbstractVector,
        vecmodel::VecModel,
        stats::NamedTuple=NamedTuple();
        include_log_probs::Bool=true,
    )

Re-evaluate `vecmodel` at `params` and return model-space parameters together with optional
log-probability statistics. If log probabilities are not requested and every transform is
fixed, the raw parameters are recovered without evaluating the model.
"""
function VarNames.Draw(
    params::AbstractVector,
    vecmodel::VecModel,
    stats::NamedTuple=NamedTuple();
    include_log_probs::Bool=true,
)
    _check_vecmodel_dimension(params, vecmodel)
    if !include_log_probs && _all_fixed_transforms(vecmodel)
        raw_params = _raw_params_from_fixed_transforms(params, vecmodel.vecinfo)
        return VarNames.Draw(raw_params, stats)
    end

    source = FromVector(params, vecmodel.vecinfo)
    accs = _draw_accumulators(include_log_probs)
    _, accs = evaluate(
        vecmodel.model,
        source,
        vecmodel.parameter_space,
        accs,
        EvalContext{false}(),
    )
    return _draw(accs, stats, include_log_probs)
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
