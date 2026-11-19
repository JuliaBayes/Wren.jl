using Random: Random
using StatsAPI: loglikelihood
using MCMCProgress: advance!, close_phase!
using VarNames: VarNames, VarNamedTuple, parameters

"""
    independent_mh([rng], model, niters, nchains=1;
                   progress=true, track=AllQuantities(), max_init_attempts=1000,
                   adtype=nothing) -> VNChain

Sample from the posterior by independent Metropolis–Hastings with prior proposals.
Each chain starts from the first prior draw with finite log likelihood, within
`max_init_attempts` draws. The chain records `loglikelihood` and `accepted`.
Use `adtype=AutoReactant()` with traceable priors and fixed latent names and shapes.
"""
independent_mh(model::WrenCore.Model, niters::Integer, nchains::Integer=1; kwargs...) =
    independent_mh(Random.default_rng(), model, niters, nchains; kwargs...)
function independent_mh(
    rng::Random.AbstractRNG,
    model::WrenCore.Model,
    niters::Integer,
    nchains::Integer=1;
    progress::Bool=true,
    track::TrackedQuantitySpec=AllQuantities(),
    max_init_attempts::Integer=1000,
    adtype::Union{Nothing,AutoReactant}=nothing,
)
    _check_mh_arguments(niters, nchains, max_init_attempts)
    _check_mh_backend(adtype)

    results, stats = _run_mh_chains(rng, model, nchains, progress) do chain_rng, run, j
        _independent_mh_chain(
            chain_rng,
            model,
            niters,
            run,
            j,
            track,
            max_init_attempts,
            adtype,
        )
    end
    return WrenCore._to_flexichain(results, stats)
end

function _independent_mh_chain(
    rng,
    model,
    niters,
    run,
    j,
    track,
    max_init_attempts,
    ::Nothing,
)
    chain_model = WrenCore._copy_for_thread(model)
    source = WrenCore.FromPrior(rng)
    current, _ = _mh_start(max_init_attempts, "log likelihood") do _
        draw = _independent_mh_draw(source, chain_model, track)
        return draw, draw.loglikelihood
    end
    params = _recorded_parameters(current.accs)
    results = Vector{VarNamedTuple}(undef, niters)
    stats = Vector{NamedTuple{(:accepted, :return_value, :loglikelihood)}}(undef, niters)
    phase = _mh_phase(run, j, "Sampling", niters)
    try
        for i in 1:niters
            proposal = _independent_mh_draw(source, chain_model, track)
            # The proposal density is the prior, so only the likelihood ratio remains.
            accepted = log(rand(rng)) < proposal.loglikelihood - current.loglikelihood
            if accepted
                current = proposal
                params = _recorded_parameters(current.accs)
            end
            results[i] = params
            stats[i] = (; accepted, current.return_value, current.loglikelihood)
            phase === nothing || advance!(phase, i)
        end
    finally
        phase === nothing || close_phase!(phase)
    end
    return results, stats
end

_recorded_parameters(accs) = merge(WrenCore.tracked_quantities(accs), parameters(accs))

function _independent_mh_record(source, model, track, scores...)
    return WrenCore.evaluate(
        model,
        source,
        WrenCore.OriginalSpace(),
        WrenCore.AccumulatorTuple(
            WrenCore.ParamsAccumulator(),
            WrenCore.TrackedQuantityAccumulator(track),
            scores...,
        ),
        WrenCore.EvalContext{false}(),
    )
end

function _independent_mh_draw(source, model, track)
    return_value, accs =
        _independent_mh_record(source, model, track, WrenCore.LogLikelihoodAccumulator())
    ll = loglikelihood(accs)
    ll < Inf || throw(DomainError(ll, "Expected finite log likelihood or -Inf"))
    return (; accs, return_value, loglikelihood=ll)
end

function _independent_mh_chain(
    rng,
    model,
    niters,
    run,
    j,
    track,
    max_init_attempts,
    ::AutoReactant,
)
    chain_model = WrenCore._copy_for_thread(model)
    layout = parameters(WrenCore._copy_for_thread(model), WrenCore.FromPrior(copy(rng)))
    entries = Tuple(pairs(layout))
    names, sample = map(first, entries), map(last, entries)
    backend = Base.get_extension(WrenCore, :WrenCoreReactantExt)
    step, state, history = backend._prepare_device_independent_mh(
        chain_model,
        rng,
        names,
        sample,
        niters,
        max_init_attempts,
    )
    state, history =
        _device_mh_phase(backend, step, state, history, niters, run, j, "Sampling")
    values, loglikelihoods, accepted = backend._finish_device_independent_mh(state, history)
    results = Vector{VarNamedTuple}(undef, niters)
    stats = Vector{NamedTuple{(:accepted, :return_value, :loglikelihood)}}(undef, niters)
    skeleton = VarNames.skeleton(layout)
    params, return_value = VarNamedTuple(), nothing
    for i in 1:niters
        if i == 1 || accepted[i]
            source = WrenCore.FromValues(_prior_values_at(skeleton, names, values, i))
            return_value, accs = _independent_mh_record(source, chain_model, track)
            params = _recorded_parameters(accs)
        end
        results[i] = params
        stats[i] = (; accepted=accepted[i], return_value, loglikelihood=loglikelihoods[i])
    end
    return results, stats
end

function _prior_values_at(skeleton, names, values, i)
    result = VarNamedTuple()
    for (name, history) in zip(names, values)
        value =
            ndims(history) == 1 ? history[i] : copy(selectdim(history, ndims(history), i))
        template = get(skeleton.data, VarNames.getsym(name), VarNames.NoTemplate())
        result = VarNames.templated_setindex!!(result, value, name, template)
    end
    return result
end
