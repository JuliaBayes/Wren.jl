"""
    ParamsWithStats

Model-space parameter values together with optional sampling statistics.
"""
struct ParamsWithStats{P<:VarNamedTuple,S<:NamedTuple}
    params::P
    stats::S
end

function _params_with_stats_accumulators(include_log_probs::Bool)
    return if include_log_probs
        AccumulatorTuple(
            LogPriorAccumulator(),
            LogLikelihoodAccumulator(),
            ParamsAccumulator(),
        )
    else
        AccumulatorTuple(ParamsAccumulator())
    end
end

function _params_with_stats(
    accs::AccumulatorTuple,
    stats::NamedTuple,
    include_log_probs::Bool,
)
    densified_params = VarNames.densify!!(params(accs))
    if include_log_probs
        has_prior = haskey(accs, Val(LOG_PRIOR_ACCNAME))
        has_likelihood = haskey(accs, Val(LOG_LIKELIHOOD_ACCNAME))
        has_prior && (stats = merge(stats, (logprior=logprior(accs),)))
        has_likelihood && (stats = merge(stats, (loglikelihood=loglikelihood(accs),)))
        if has_prior && has_likelihood
            stats = merge(stats, (logjoint=logjoint(accs),))
        end
    end
    return ParamsWithStats(densified_params, stats)
end

"""
    ParamsWithStats(
        source::AbstractLatentSource,
        model::Model,
        stats::NamedTuple=NamedTuple();
        include_log_probs::Bool=true,
    )

Evaluate `model` using `source` and return its model-space parameters. By default the log
prior, log likelihood, and log joint are added to `stats`.
"""
function ParamsWithStats(
    source::AbstractLatentSource,
    model::Model,
    stats::NamedTuple=NamedTuple();
    include_log_probs::Bool=true,
)
    accs = _params_with_stats_accumulators(include_log_probs)
    _, accs = evaluate(model, source, OriginalSpace(), accs)
    return _params_with_stats(accs, stats, include_log_probs)
end

"""
    ParamsWithStats(
        accs::AccumulatorTuple,
        stats::NamedTuple=NamedTuple();
        include_log_probs::Bool=true,
    )

Extract parameters and any available log-probability statistics from an accumulator tuple.
The tuple must contain a [`ParamsAccumulator`](@ref).
"""
function ParamsWithStats(
    accs::AccumulatorTuple,
    stats::NamedTuple=NamedTuple();
    include_log_probs::Bool=true,
)
    return _params_with_stats(accs, stats, include_log_probs)
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
        values(vecmodel.ranges_and_transforms),
    )
end

function _raw_params_from_fixed_transforms(
    params::AbstractVector,
    ranges_and_transforms::VarNamedTuple,
)
    raw_params = VarNamedTuple()
    for (vn, rt) in pairs(ranges_and_transforms)
        template =
            get(ranges_and_transforms.data, VarNames.getsym(vn), VarNames.NoTemplate())
        raw_value = rt.transform.transform(params[rt.range])
        raw_params = VarNames.templated_setindex!!(raw_params, raw_value, vn, template)
    end
    return VarNames.densify!!(raw_params)
end

"""
    ParamsWithStats(
        params::AbstractVector,
        vecmodel::VecModel,
        stats::NamedTuple=NamedTuple();
        include_log_probs::Bool=true,
    )

Re-evaluate `vecmodel` at `params` and return model-space parameters together with optional
log-probability statistics. If log probabilities are not requested and every transform is
fixed, the raw parameters are recovered without evaluating the model.
"""
function ParamsWithStats(
    params::AbstractVector,
    vecmodel::VecModel,
    stats::NamedTuple=NamedTuple();
    include_log_probs::Bool=true,
)
    _check_vecmodel_dimension(params, vecmodel)
    if !include_log_probs && _all_fixed_transforms(vecmodel)
        raw_params =
            _raw_params_from_fixed_transforms(params, vecmodel.ranges_and_transforms)
        return ParamsWithStats(raw_params, stats)
    end

    source = FromVector(params, vecmodel.ranges_and_transforms)
    accs = _params_with_stats_accumulators(include_log_probs)
    _, accs = evaluate(vecmodel.model, source, vecmodel.parameter_space, accs)
    return _params_with_stats(accs, stats, include_log_probs)
end

function Base.show(io::IO, ::MIME"text/plain", pws::ParamsWithStats)
    printstyled(io, "ParamsWithStats"; bold=true)
    print(io, "\n ├─ ")
    if isempty(pws.params)
        printstyled(io, "params"; bold=true)
        println(io, " (empty)")
    else
        printstyled(io, "params"; bold=true)
        print(io, "\n │  ")
        VarNames.vnt_pretty_print(io, pws.params, " │  ", 0)
        println(io)
    end
    print(io, " └─ ")
    printstyled(io, "stats"; bold=true)
    if isempty(pws.stats)
        println(io, " (empty)")
    else
        nstats = length(pws.stats)
        for (index, (name, value)) in enumerate(pairs(pws.stats))
            print(io, index == nstats ? "\n    └─ " : "\n    ├─ ")
            printstyled(io, name; color=:blue)
            print(io, " = ")
            show(io, value)
        end
    end
    return nothing
end

function Base.:(==)(left::ParamsWithStats, right::ParamsWithStats)
    return (left.params == right.params) & (left.stats == right.stats)
end

function Base.isequal(left::ParamsWithStats, right::ParamsWithStats)
    return isequal(left.params, right.params) && isequal(left.stats, right.stats)
end

"""
    FromParams(
        pws::ParamsWithStats,
        fallback::Union{AbstractLatentSource,Nothing}=FromPrior(),
    )

Create a latent source from `pws.params`. The statistics are ignored, and `fallback` is
used for parameters absent from `pws.params`.
"""
function FromParams(
    pws::ParamsWithStats,
    fallback::Union{AbstractLatentSource,Nothing}=FromPrior(),
)
    return FromParams(pws.params, fallback)
end

StatsAPI.params(pws::ParamsWithStats) = pws.params
logprior(pws::ParamsWithStats) = pws.stats.logprior
StatsAPI.loglikelihood(pws::ParamsWithStats) = pws.stats.loglikelihood
logjoint(pws::ParamsWithStats) = pws.stats.logjoint
