using Random: Random
using StatsAPI: loglikelihood
using MCMCProgress: advance!, close_phase!
using VarNames: VarNamedTuple, parameters

"""
    independent_mh([rng], model, niters, nchains=1;
                   progress=true, track=AllQuantities(), max_init_attempts=1000,
                   adtype=nothing) -> VNChain

Sample from the posterior by independent Metropolis–Hastings with prior proposals.
Each chain starts from the first prior draw with finite log likelihood, within
`max_init_attempts` draws. The chain records `loglikelihood` and `accepted`.
Use `adtype=AutoReactant()` to score fixed, continuous latents on the Reactant device.
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

function _independent_mh_chain(rng, model, niters, run, j, track, max_init_attempts, adtype)
    chain_model = WrenCore._copy_for_thread(model)
    source = WrenCore.FromPrior(rng)
    scorer = _prepare_prior_scorer(chain_model, source, adtype)
    current, _ = _mh_start(max_init_attempts, "log likelihood") do _
        draw = _independent_mh_draw(source, chain_model, track, scorer)
        return draw, draw.loglikelihood
    end
    params = _recorded_parameters(current.accs)
    results = Vector{VarNamedTuple}(undef, niters)
    stats = Vector{NamedTuple{(:accepted, :return_value, :loglikelihood)}}(undef, niters)
    phase = _mh_phase(run, j, "Sampling", niters)
    try
        for i in 1:niters
            proposal = _independent_mh_draw(source, chain_model, track, scorer)
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

function _independent_mh_draw(source, model, track, scorer)
    return_value, accs = WrenCore.evaluate(
        model,
        source,
        WrenCore.OriginalSpace(),
        WrenCore.AccumulatorTuple(
            WrenCore.ParamsAccumulator(),
            WrenCore.TrackedQuantityAccumulator(track),
            _prior_scoring_accumulator(scorer),
        ),
        WrenCore.EvalContext{false}(),
    )
    ll = _prior_loglikelihood(scorer, accs)
    ll < Inf || throw(DomainError(ll, "Expected finite log likelihood or -Inf"))
    return (; accs, return_value, loglikelihood=ll)
end

struct _PriorScorer{F,A}
    score::F
    accumulator::A
end

_prepare_prior_scorer(model, source, ::Nothing) = nothing

function _prepare_prior_scorer(model, source, adtype::AutoReactant)
    # Probe the layout without consuming the chain's first proposal.
    layout_source = _ContinuousPrior(WrenCore.FromPrior(copy(source.rng)))
    vm = WrenCore.VecModel(
        WrenCore._copy_for_thread(model),
        nothing;
        latent_source=layout_source,
        parameter_space=WrenCore.OriginalSpace(),
    )
    (!isempty(vm.sample) && eltype(vm.sample) <: AbstractFloat) ||
        throw(ArgumentError("Reactant prior proposals require floating-point latents"))
    score = WrenCore._compile_log_density(
        vm,
        adtype;
        lp_function=loglikelihood,
        accs=WrenCore.AccumulatorTuple(WrenCore.LogLikelihoodAccumulator()),
    )
    accumulator = WrenCore.VecSampleAccumulator(similar(vm.sample), vm.vecinfo, Val(true))
    return _PriorScorer(score, accumulator)
end

_prior_scoring_accumulator(::Nothing) = WrenCore.LogLikelihoodAccumulator()
_prior_loglikelihood(::Nothing, accs) = loglikelihood(accs)

function _prior_scoring_accumulator(scorer::_PriorScorer)
    fill!(scorer.accumulator.vals, NaN)
    return scorer.accumulator
end

function _prior_loglikelihood(scorer::_PriorScorer, accs)
    params = WrenCore.vector_parameters(accs)
    all(isfinite, params) || throw(
        ArgumentError("Reactant prior proposals require a fixed, finite latent layout"),
    )
    return scorer.score(params)
end
