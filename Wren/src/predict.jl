using AbstractMCMC: AbstractMCMC
using FlexiChains: FlexiChains, VNChain
using Random: AbstractRNG, default_rng
using StatsAPI: StatsAPI
using VarNames: VarNames, Draw, extras, parameters

"""
    predict(
        [rng::Random.AbstractRNG=Random.default_rng(),]
        model::WrenCore.Model,
        chain::FlexiChains.VNChain;
        include_all::Bool=true,
    ) -> FlexiChains.VNChain

Draw predictions from `model` for every iteration of `chain`.

The returned `FlexiChain` by default will contain all the predicted variables, as well as the
variables already present in the input `chain`. If you only want the predicted variables,
set `include_all=false`.

The returned chain will also contain log-probabilities corresponding to the re-evaluation of
the model. In particular, the log probability for the newly predicted variables are
now considered as prior terms. However, note that the log-prior of the returned chain will
also contain the log-prior terms of the parameters already present in the input `chain`.
Thus, if you want to obtain the log-probability of the predicted variables only, you can
subtract the two log-prior terms. The `include_all` keyword argument has no effect on the
log-probability fields.
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
    param_values = FlexiChains.parameters_at(chain)
    existing_vns = Set(parameters(chain))
    fallback = WrenCore.FromPrior(rng)
    results = WrenCore.reevaluate(model, param_values, accs, fallback)
    draws = map(results) do retval_and_accs
        draw = Draw(last(retval_and_accs))
        include_all && return draw
        reeval_params = parameters(draw)
        pred_vns = [vn for (vn, _) in pairs(reeval_params) if !(vn in existing_vns)]
        return Draw(VarNames.subset(reeval_params, pred_vns), extras(draw))
    end
    preds = AbstractMCMC.from_samples(VNChain, draws)
    preds = FlexiChains.renumber_iters(preds, FlexiChains.iter_indices(chain))
    preds = FlexiChains.renumber_chains(preds, FlexiChains.chain_indices(chain))
    return merge(FlexiChains.subset_extras(chain), preds)
end
function predict(model::WrenCore.Model, chain::VNChain; include_all::Bool=true)
    return predict(default_rng(), model, chain; include_all)
end

# Internal helper function
function _pointwise_logp(get_logp, model::WrenCore.Model, chain::VNChain, accs)
    results = WrenCore.reevaluate(model, FlexiChains.parameters_at(chain), accs)
    return map(retval_and_accs -> get_logp(last(retval_and_accs)), results)
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
    return _pointwise_logp(StatsAPI.loglikelihood, model, chain, accs)
end

"""
    logjoint(model::WrenCore.Model, chain::FlexiChains.VNChain) -> DimMatrix

Returns a `DimMatrix` of the log-joint probabilities, re-evaluated using the parameters at
each iteration of the chain.
"""
function WrenCore.logjoint(model::WrenCore.Model, chain::VNChain)
    accs = WrenCore.AccumulatorTuple(
        WrenCore.LogPriorAccumulator(),
        WrenCore.LogLikelihoodAccumulator(),
    )
    return _pointwise_logp(WrenCore.logjoint, model, chain, accs)
end
