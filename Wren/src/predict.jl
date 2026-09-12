using AbstractMCMC: AbstractMCMC
using FlexiChains: VNChain
using Random: AbstractRNG, default_rng
using StatsAPI: StatsAPI
using VarNames: VarNames, Draw, VarNamedTuple, parameters

_chain_params(chain::VNChain) = AbstractMCMC.to_samples(VarNamedTuple, chain)

"""
Return the variables in `new_params` that are absent from `old_params`.
"""
function _newly_sampled(new_params::VarNamedTuple, old_params::VarNamedTuple)
    fresh = [vn for (vn, _) in pairs(new_params) if !haskey(old_params, vn)]
    return VarNames.subset(new_params, fresh)
end

"""
    predict(
        [rng::Random.AbstractRNG=Random.default_rng(),]
        model::WrenCore.Model,
        chain::FlexiChains.VNChain;
        include_all::Bool=true,
    ) -> FlexiChains.VNChain

Draw predictions from `model` for every iteration of `chain`.

`model` is re-evaluated once per iteration, taking each variable's value from that iteration
of `chain` and sampling from the prior any variable that `chain` does not contain. The
variables to be predicted are therefore the ones `model` leaves unconditioned; use
[`decondition`](@ref) to release the variables that were conditioned on for sampling.

The returned chain holds the predicted variables alongside those already in `chain`; pass
`include_all=false` for the predicted variables only. It also carries `logprior`,
`loglikelihood` and `logjoint` extras from the re-evaluation, in which the predicted
variables count as prior terms. `include_all` does not affect these.
"""
function predict(
    rng::AbstractRNG,
    model::WrenCore.Model,
    chain::VNChain;
    include_all::Bool=true,
)
    accs = WrenCore.AccumulatorTuple(
        WrenCore.ParamsAccumulator(),
        WrenCore.LogPriorAccumulator(),
        WrenCore.LogLikelihoodAccumulator(),
    )
    params = _chain_params(chain)
    results = WrenCore.reevaluate(model, params, accs, WrenCore.FromPrior(rng))
    new_draws = map(params, results) do old_params, (_, new_accs)
        new_params = parameters(new_accs)
        lprior = WrenCore.logprior(new_accs)
        llik = WrenCore.loglikelihood(new_accs)
        Draw(
            include_all ? new_params : _newly_sampled(new_params, old_params),
            (; logprior=lprior, loglikelihood=llik, logjoint=lprior + llik),
        )
    end
    return AbstractMCMC.from_samples(VNChain, new_draws)
end
function predict(model::WrenCore.Model, chain::VNChain; include_all::Bool=true)
    return predict(default_rng(), model, chain; include_all)
end

"""
Re-evaluate `model` over `chain` with `accs`, and read one log-probability off each result.
"""
function _pointwise_logp(get_logp, model::WrenCore.Model, chain::VNChain, accs)
    results = WrenCore.reevaluate(model, _chain_params(chain), accs)
    return map(result -> get_logp(last(result)), results)
end

"""
    logprior(model::WrenCore.Model, chain::FlexiChains.VNChain) -> DimMatrix

Return the log-prior of `model` at each iteration of `chain`.

Unlike [`predict`](@ref), this errors on a variable that `model` requires and `chain` does
not hold, rather than sampling it.
"""
function WrenCore.logprior(model::WrenCore.Model, chain::VNChain)
    accs = WrenCore.AccumulatorTuple(WrenCore.LogPriorAccumulator())
    return _pointwise_logp(WrenCore.logprior, model, chain, accs)
end

"""
    loglikelihood(model::WrenCore.Model, chain::FlexiChains.VNChain) -> DimMatrix

Return the log-likelihood of `model` at each iteration of `chain`.

Unlike [`predict`](@ref), this errors on a variable that `model` requires and `chain` does
not hold, rather than sampling it.
"""
function StatsAPI.loglikelihood(model::WrenCore.Model, chain::VNChain)
    accs = WrenCore.AccumulatorTuple(WrenCore.LogLikelihoodAccumulator())
    return _pointwise_logp(WrenCore.loglikelihood, model, chain, accs)
end

"""
    logjoint(model::WrenCore.Model, chain::FlexiChains.VNChain) -> DimMatrix

Return the log-joint density of `model` at each iteration of `chain`, in original
coordinates.

Unlike [`predict`](@ref), this errors on a variable that `model` requires and `chain` does
not hold, rather than sampling it.
"""
function WrenCore.logjoint(model::WrenCore.Model, chain::VNChain)
    accs = WrenCore.AccumulatorTuple(
        WrenCore.LogPriorAccumulator(),
        WrenCore.LogLikelihoodAccumulator(),
    )
    return _pointwise_logp(WrenCore.logjoint, model, chain, accs)
end
