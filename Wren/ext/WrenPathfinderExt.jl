module WrenPathfinderExt

using Wren: Wren, Model, DEFAULT_AD, FromPrior, logdensity, Draw
using AbstractMCMC: AbstractMCMC
using Accessors: @set, setproperties
using FlexiChains: VNChain
using Random: Random, AbstractRNG
import Pathfinder

"""
    Pathfinder.pathfinder(model::Model; rng=Random.default_rng(), adtype=DEFAULT_AD,
                          latent_source=FromPrior(rng), reduce_copies=true, kwargs...)

Run single-path Pathfinder on a Wren posterior in unconstrained coordinates, including
the change-of-variables Jacobian. Return a native `PathfinderResult` with model-space
draws in `draws_transformed`, a `FlexiChains.VNChain` with one chain index.

`adtype` and `reduce_copies` have the same meaning as for [`nuts`](@ref).
`latent_source` prepares the parameter mapping. Pathfinder's `init` and `init_sampler`
select optimiser starts in unconstrained coordinates. Other keywords pass to Pathfinder.
Only `ntasks=1` is supported because model and AD state can be mutable.
"""
function Pathfinder.pathfinder(model::Model; kwargs...)
    return _pathfinder(Pathfinder.pathfinder, model; kwargs...)
end

"""
    Pathfinder.multipathfinder(model::Model, ndraws::Integer; nruns=4, kwargs...)

Run multi-path Pathfinder on a Wren posterior. Fit `nruns` Gaussian mixture components
and, by default, use Pareto-smoothed importance resampling to obtain `ndraws` draws.
When `init` is supplied, its length determines the number of fits instead of `nruns`.

Return a native `MultiPathfinderResult`. Its `draws_transformed` and each single-path
result's `draws_transformed` are model-space `FlexiChains.VNChain`s with one chain index.
Fits are mixture components, not independent MCMC chains. All other options follow
[`Pathfinder.pathfinder`](@ref). Only `ntasks=1` and `ntasks_per_run=1` are supported.
"""
function Pathfinder.multipathfinder(model::Model, ndraws::Integer; kwargs...)
    return Pathfinder.multipathfinder(model, Int(ndraws); kwargs...)
end

# Match Pathfinder's (Any, Int) method to avoid ambiguous dispatch for Model/Int calls.
function Pathfinder.multipathfinder(
    model::Model,
    ndraws::Int;
    nruns::Integer=4,
    init=nothing,
    ntasks_per_run::Integer=1,
    kwargs...,
)
    ntasks_per_run == 1 || throw(ArgumentError("Wren models require ntasks_per_run=1."))
    return _pathfinder(
        Pathfinder.multipathfinder,
        model,
        ndraws;
        nruns=Int(nruns),
        init,
        kwargs...,
    )
end

function _pathfinder(
    f,
    model::Model,
    args...;
    rng::AbstractRNG=Random.default_rng(),
    adtype=DEFAULT_AD,
    latent_source=FromPrior(rng),
    reduce_copies=true,
    ntasks::Integer=1,
    kwargs...,
)
    ntasks == 1 || throw(ArgumentError("Wren models require ntasks=1."))
    density = logdensity(
        Wren.WrenCore._copy_for_thread(model),
        adtype;
        latent_source,
        reduce_copies,
    )
    result = f(density, args...; rng, input=model, kwargs...)
    return _transform_result(result, density)
end

function _pathfinder_draws(draws, density)
    params = map(eachcol(draws)) do point
        Draw(point, density)
    end
    return AbstractMCMC.from_samples(VNChain, reshape(params, :, 1))
end

function _transform_result(result, density)
    chain = _pathfinder_draws(result.draws, density)
    return @set result.draws_transformed = chain
end

function _transform_result(result::Pathfinder.MultiPathfinderResult, density)
    paths = map(result.pathfinder_results) do path
        _transform_result(path, density)
    end
    chain = _pathfinder_draws(result.draws, density)
    return setproperties(result, (; draws_transformed=chain, pathfinder_results=paths))
end

end
