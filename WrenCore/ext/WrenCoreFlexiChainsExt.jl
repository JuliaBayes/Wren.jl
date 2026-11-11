module WrenCoreFlexiChainsExt

using AbstractMCMC: AbstractMCMC
using FlexiChains: FlexiChains, VNChain
using Random: AbstractRNG, default_rng
using StatsAPI: StatsAPI
using VarNames: VarNames, Draw, extras, parameters
using WrenCore: WrenCore

"""
    predict(
        [rng::Random.AbstractRNG=Random.default_rng(),]
        model::WrenCore.Model,
        chain::FlexiChains.VNChain;
        gqs::GeneratedQuantitySpec=AllGQs(),
    ) -> FlexiChains.VNChain

Draw predictions from `model` for all iterations in `chain`.

The returned `FlexiChain` will contain all the predicted variables, the generated quantities
selected by `gqs`, as well as the variables already present in the input `chain`, which
implies that:

- variables that are in the `chain` are fixed to their respective values;
- variables that arent in the `chain` are resampled according to the model;
- variables that are `@observe`d are never included.

To move a variable from the first into the second category, subset `chain` to exclude that
variable (more about subsetting chains [here](https://juliabayes.org/FlexiChains.jl/stable/api#Subsetting-chains)).

The returned chain will also contain log-probabilities corresponding to the re-evaluation of
the model. In particular, the log probability for the newly predicted variables are
now considered as prior terms. However, note that the log-prior of the returned chain will
also contain the log-prior terms of the parameters already present in the input `chain`.
Thus, if you want to obtain the log-probability of the predicted variables only, you can
subtract the two log-prior terms.
"""
function StatsAPI.predict(
    rng::AbstractRNG,
    model::WrenCore.Model,
    chain::VNChain;
    gqs::WrenCore.GeneratedQuantitySpec=WrenCore.AllGQs(),
)
    accs = WrenCore.AccumulatorTuple(
        WrenCore.ParamsAccumulator(),
        WrenCore.LogPriorAccumulator(),
        WrenCore.LogLikelihoodAccumulator(),
        WrenCore.GeneratedQuantityAccumulator(gqs),
    )
    fallback = WrenCore.FromPrior(rng)
    results = _reevaluate(model, chain, accs, fallback)
    draws = Draw.(last.(results))
    preds_chain = AbstractMCMC.from_samples(VNChain, draws)
    preds_chain = FlexiChains.renumber_iters(preds_chain, FlexiChains.iter_indices(chain))
    preds_chain = FlexiChains.renumber_chains(preds_chain, FlexiChains.chain_indices(chain))
    return merge(FlexiChains.subset_extras(chain), preds_chain)
end

function StatsAPI.predict(model::WrenCore.Model, chain::VNChain; kwargs...)
    return StatsAPI.predict(default_rng(), model, chain; kwargs...)
end

"""
    generated_quantities(
        model::WrenCore.Model,
        chain::FlexiChains.VNChain,
        gqs::GeneratedQuantitySpec=AllGQs(),
    ) -> FlexiChains.VNChain

Re-evaluate `model` at each iteration of `chain` and return a chain holding only the
generated quantities (`@gq` statements) selected by `gqs`, with the same iteration and chain
indices as `chain`. Every parameter that `model` requires must be in `chain`.

Generated quantities that are not selected are not evaluated, unless a selected one
depends on them.
"""
function WrenCore.generated_quantities(
    model::WrenCore.Model,
    chain::VNChain,
    gqs::WrenCore.GeneratedQuantitySpec=WrenCore.AllGQs(),
)
    gq_acc = WrenCore.GeneratedQuantityAccumulator(gqs)
    results = _reevaluate(model, chain, WrenCore.AccumulatorTuple(gq_acc))
    draws = map(results) do retval_and_accs
        gqs = WrenCore.generated_quantities(last(retval_and_accs))
        Draw(VarNames.densify!!(gqs), NamedTuple())
    end
    gqs_chain = AbstractMCMC.from_samples(VNChain, draws)
    gqs_chain = FlexiChains.renumber_iters(gqs_chain, FlexiChains.iter_indices(chain))
    return FlexiChains.renumber_chains(gqs_chain, FlexiChains.chain_indices(chain))
end

function _reevaluate(model::WrenCore.Model, chain::VNChain, accs, fallback=WrenCore.Void())
    return map(FlexiChains.parameters_at(chain)) do vnt
        WrenCore.evaluate(
            model,
            WrenCore.FromValues(vnt, fallback),
            WrenCore.OriginalSpace(),
            map(WrenCore.reset, accs),
            WrenCore.EvalContext{false}(),
        )
    end
end

function _pointwise_logp(get_logp, model::WrenCore.Model, chain::VNChain, accs)
    results = _reevaluate(model, chain, accs)
    return map(retval_and_accs -> get_logp(last(retval_and_accs)), results)
end

"""
    logprior(model::WrenCore.Model, chain::FlexiChains.VNChain) -> DimMatrix

Return the log-prior of `model` at each iteration of `chain`.

Unlike [`predict`](@ref StatsAPI.predict), this errors on a variable that `model` requires
and `chain` does not hold, rather than sampling it.
"""
function WrenCore.logprior(model::WrenCore.Model, chain::VNChain)
    accs = WrenCore.AccumulatorTuple(WrenCore.LogPriorAccumulator())
    return _pointwise_logp(WrenCore.logprior, model, chain, accs)
end

"""
    loglikelihood(model::WrenCore.Model, chain::FlexiChains.VNChain) -> DimMatrix

Return the log-likelihood of `model` at each iteration of `chain`.

Unlike [`predict`](@ref StatsAPI.predict), this errors on a variable that `model` requires
and `chain` does not hold, rather than sampling it.
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

end # module WrenCoreFlexiChainsExt
