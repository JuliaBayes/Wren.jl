function _device_prior_draw(model, names)
    evaluation_fn = model.evaluation_fn
    copy_args = model.enzyme_copy_argnames
    args = _device_array(model.args)
    conditioned = _device_array(model.conditioned)
    return function (seed)
        rng = Reactant.ReactantRNG(seed)
        _, accs = evaluation_fn(
            WrenCore.FromPrior(rng),
            WrenCore.OriginalSpace(),
            WrenCore.AccumulatorTuple(
                WrenCore.ParamsAccumulator(),
                WrenCore.LogLikelihoodAccumulator(),
            ),
            args,
            conditioned,
            copy_args,
            WrenCore.EvalContext{true}(),
        )
        parameters = WrenCore.parameters(accs)
        Tuple(first(pair) for pair in pairs(parameters)) == names ||
            throw(ArgumentError("Reactant prior proposals require fixed latent names"))
        return rng.seed, map(name -> parameters[name], names), WrenCore.loglikelihood(accs)
    end
end

function _device_prior_start(draw, seed, max_attempts)
    seed, values, loglikelihood = draw(seed)
    i = Reactant.promote_to(Reactant.TracedRNumber{Int}, 1)
    Reactant.@trace track_numbers = false while (i < max_attempts) & (loglikelihood == -Inf)
        seed, values, loglikelihood = draw(seed)
        i += 1
    end
    return seed, values, loglikelihood
end

function _device_prior_step(draw, state, _)
    (; seed, values, loglikelihood, invalid_score) = state
    seed, proposed, proposed_score = draw(seed)
    valid = proposed_score < Inf
    rng = Reactant.ReactantRNG(seed)
    accepted = valid & (log(rand(rng)) < proposed_score - loglikelihood)
    values = map((y, x) -> ifelse.(accepted, y, x), proposed, values)
    loglikelihood = ifelse(accepted, proposed_score, loglikelihood)
    invalid_score = ifelse(!valid & (invalid_score < Inf), proposed_score, invalid_score)
    return (; seed=rng.seed, values, loglikelihood, invalid_score),
    (values, loglikelihood, accepted)
end

function _device_independent_mh(
    sample_chain,
    model,
    rng,
    names,
    sample,
    niters,
    max_init_attempts,
)
    draw = _device_prior_draw(model, names)
    seed = _device_rng(rng).seed
    initialize = seed -> _device_prior_start(draw, seed, max_init_attempts)
    seed, values, loglikelihood = _device_compile(initialize, seed)(seed)
    ll = Reactant.to_number(loglikelihood)
    ll < Inf || throw(DomainError(ll, "Expected finite log likelihood or -Inf"))
    isfinite(ll) || throw(
        ArgumentError("No finite log likelihood after $max_init_attempts prior draws"),
    )

    step = (state, i) -> _device_prior_step(draw, state, i)
    state = (;
        seed,
        values,
        loglikelihood=_device_scalar(loglikelihood),
        invalid_score=_device_scalar(zero(ll)),
    )
    history = (
        map(x -> _device_array(zeros(eltype(x), size(x)..., niters)), sample),
        _device_array(zeros(typeof(ll), niters)),
        _device_array(zeros(Bool, niters)),
    )
    state, history = sample_chain(step, state, history)
    invalid_score = Reactant.to_number(state.invalid_score)
    invalid_score < Inf ||
        throw(DomainError(invalid_score, "Expected finite log likelihood or -Inf"))
    values, loglikelihoods, accepted = history
    return map(Array, values), Array(loglikelihoods), Array(accepted)
end
