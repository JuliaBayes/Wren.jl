"""
    pathfinder([rng::Random.AbstractRNG], model::Model, ndraws::Integer;
               nruns=4, adtype=DEFAULT_AD, latent_source=FromPrior(rng),
               reduce_copies=true, kwargs...)

Approximate a Wren posterior with multi-path Pathfinder. Requires Pathfinder.jl.
Run `nruns` Gaussian fits in unconstrained coordinates and, by default, use Pareto-smoothed
importance resampling to obtain `ndraws` approximate posterior draws.

Return the native Pathfinder result, including its ELBO estimates and
`psis_result.pareto_shape` diagnostic. `draws_transformed` is a model-space
`FlexiChains.VNChain`, ready for plotting, summaries, and prediction.
Fits are mixture components, not independent MCMC chains. All draws share one chain index.

`adtype` and `reduce_copies` have the same meaning as for [`nuts`](@ref).
`latent_source` prepares the parameter mapping. Pathfinder's `init` and `init_sampler`
select optimiser starts in unconstrained coordinates; other keywords pass to Pathfinder.
Only `ntasks=1` and `ntasks_per_run=1` are supported because model and AD state can be mutable.
"""
function pathfinder end

function pathfinder(model::WrenCore.Model, ndraws::Integer; kwargs...)
    return pathfinder(Random.default_rng(), model, ndraws; kwargs...)
end
