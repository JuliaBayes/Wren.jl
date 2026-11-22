module WrenPathfinderExt

using Wren
using WrenCore
using Accessors: @set, setproperties
using FlexiChains: VNChain
using Random: Random, AbstractRNG
import Pathfinder

"""
    Pathfinder.pathfinder(
        model::Model;
        rng=Random.default_rng(),
        init_scale=2,
        init_sampler=FromUnconstrainedInterval(rng, -init_scale, init_scale),
        init=init_sampler,
        adtype=DEFAULT_AD,
        pathfinder_kwargs...
    )

Run single-path Pathfinder on a Wren posterior in unconstrained coordinates.

## Arguments

All arguments are passed to [`Pathfinder.pathfinder`](@extref), but Wren overrides
Pathfinder's `init` and `init_sampler` arguments to use Wren types instead.

In Pathfinder's original API, `init` is used to provide the initial point of the
optimisation, and `init_sampler` is used to provide new starting points when an optimisation
run fails. These must both be vectors of unconstrained parameters; however, this is
difficult to generate with Wren as you need to use low-level APIs.

Wren instead allows you to pass an [`AbstractLatentSource`](@ref) to `init` and
`init_sampler`, which is then converted into the appropriate vectors internally.

As in upstream Pathfinder, `init_sampler` defaults to drawing each unconstrained coordinate
uniformly from `[-init_scale, init_scale]`, with `init_scale=2`. `init_scale` must be
positive, and cannot be combined with an explicit `init_sampler`.

Finally, if `init` is not given, the starting point is also drawn from `init_sampler`.

Note that only  `ntasks=1` is allowed due to [Pathfinder not supporting thread-unsafe
log-densities](https://github.com/mlcolab/Pathfinder.jl/issues/439).

## Returns

Returns a [`Pathfinder.PathfinderResult`](@extref). This struct contains several fields
which are documented in Pathfinder's own docs (see previous link), but in particular
we note here that:

- each column of the `draws` field contains one vectorised draw in unconstrained
  coordinates, and

- the `draws_transformed` field contains a [`FlexiChains.VNChain`](@extref) of draws
  which have been transformed back to model space.

For example, in

```julia
using Wren, Pathfinder, Wren.Dists, Enzyme
@wren f() = x ~ Beta(2, 2); model = f()
result = pathfinder(model; ndraws=1000)
```

`result.draws` will be a `1×1000` Matrix of _unconstrained_ draws, and
`result.draws_transformed[@vn(x)]` will be a `1000×1` `DimArray` of _constrained_ draws.
"""
function Pathfinder.pathfinder(
    model::Model;
    rng::AbstractRNG=Random.default_rng(),
    init=nothing,
    init_scale=nothing,
    init_sampler=nothing,
    ntasks::Integer=1,
    kwargs...,
)
    ntasks == 1 || throw(ArgumentError("Wren models require ntasks=1."))
    retry_source = _retry_source(rng, init_scale, init_sampler)
    source = init === nothing ? retry_source : WrenCore.to_latent_source(init)
    return _pathfinder_wrapper(
        Pathfinder.pathfinder,
        model;
        rng,
        sources=source,
        retry_source,
        kwargs...,
    )
end

"""
    Pathfinder.multipathfinder(
        model::Model,
        ndraws::Integer;
        rng=Random.default_rng(),
        init_scale=2,
        init_sampler=FromUnconstrainedInterval(rng, -init_scale, init_scale),
        init=nothing,
        nruns=4,
        multipathfinder_kwargs...
    )

Run multi-path Pathfinder on a Wren posterior.

Fit `nruns` Gaussian mixture components and, by default, use Pareto-smoothed importance
resampling to obtain `ndraws` draws.

## Arguments

`init_sampler` specifies where new starting points come from when an optimisation run fails
and Pathfinder retries. It is a single value that is shared by all runs (this mimics
Pathfinder's API), and accepts the same types as `init` in [`Pathfinder.pathfinder`](@ref).

`init_scale` works as in Pathfinder itself.

`init` sets the optimisers' starting points. If it is `nothing`, each of the `nruns` fits
starts from a draw from `init_sampler`. Otherwise, it must be a collection with one entry per
fit, and its length determines the number of fits instead of `nruns`. Each entry can be
anything accepted by `init` in [`Pathfinder.pathfinder`](@ref).

Only `ntasks=1` and `ntasks_per_run=1` are allowed due to [Pathfinder not supporting
thread-unsafe log-densities](https://github.com/mlcolab/Pathfinder.jl/issues/439).

All other options are forwarded to [`Pathfinder.multipathfinder`](@extref).

## Return value

Returns a [`Pathfinder.MultiPathfinderResult`](@extref) struct. As with single-path
Pathfinder, the `draws` field contains unconstrained draws, and `draws_transformed` contains
a [`FlexiChains.VNChain`](@extref) of draws which have been transformed back to model space.
For the other fields, please see the Pathfinder docs.
"""
function Pathfinder.multipathfinder(model::Model, ndraws::Integer; kwargs...)
    return Pathfinder.multipathfinder(model, Int(ndraws); kwargs...)
end

# Match Pathfinder's (Any, Int) method to avoid ambiguous dispatch for Model/Int calls.
function Pathfinder.multipathfinder(
    model::Model,
    ndraws::Int;
    rng::AbstractRNG=Random.default_rng(),
    init=nothing,
    init_scale=nothing,
    init_sampler=nothing,
    nruns::Integer=4,
    ntasks::Integer=1,
    ntasks_per_run::Integer=1,
    kwargs...,
)
    (ntasks == 1 && ntasks_per_run == 1) ||
        throw(ArgumentError("Wren models require ntasks=1 and ntasks_per_run=1."))
    retry_source = _retry_source(rng, init_scale, init_sampler)
    sources = if init === nothing
        fill(retry_source, Int(nruns))
    else
        map(WrenCore.to_latent_source, collect(init))
    end
    isempty(sources) && throw(ArgumentError("multipathfinder needs at least one run."))
    return _pathfinder_wrapper(
        Pathfinder.multipathfinder,
        model,
        ndraws;
        rng,
        sources,
        retry_source,
        nruns=Int(nruns),
        kwargs...,
    )
end

# Matches Pathfinder's default.
const DEFAULT_INIT_SCALE = 2

# The latent source that Pathfinder's `init_sampler` draws from. Also used for the first
# starting point when `init` is not given.
function _retry_source(rng::AbstractRNG, init_scale, init_sampler)
    if init_scale !== nothing
        init_sampler === nothing || throw(
            ArgumentError("`init_scale` and `init_sampler` cannot both be specified."),
        )
        init_scale > 0 ||
            throw(ArgumentError("`init_scale` must be positive, got $init_scale."))
        return FromUnconstrainedInterval(rng, -init_scale, init_scale)
    elseif init_sampler === nothing
        return FromUnconstrainedInterval(rng, -DEFAULT_INIT_SCALE, DEFAULT_INIT_SCALE)
    else
        return WrenCore.to_latent_source(init_sampler)
    end
end

# Used to construct the VecModel.
_prep_source(source::AbstractLatentSource) = source
_prep_source(sources::AbstractVector) = first(sources)

# Starting point(s) in the form Pathfinder expects: one vector for `pathfinder`, a vector
# of vectors for `multipathfinder`.
_init_vector(vecmodel::VecModel, source::AbstractLatentSource) =
    vector_parameters(vecmodel, source)
_init_vector(vecmodel::VecModel, sources::AbstractVector) =
    [vector_parameters(vecmodel, s) for s in sources]

function _pathfinder_wrapper(
    f,
    model::Model,
    args...;
    rng::AbstractRNG,
    # Either one latent source (`pathfinder`) or a vector of them (`multipathfinder`).
    sources,
    retry_source::AbstractLatentSource,
    adtype=DEFAULT_AD,
    reduce_copies=true,
    kwargs...,
)
    haskey(kwargs, :latent_source) && throw(
        ArgumentError(
            "the `latent_source` keyword argument is controlled by `init`; please pass that instead.",
        ),
    )
    vecmodel = VecModel(
        WrenCore._copy_for_thread(model),
        adtype;
        latent_source=_prep_source(sources),
        reduce_copies,
    )
    # Pathfinder uses `init_sampler` to pick a new starting point whenever an optimisation
    # run fails. Pathfinder's version works in unconstrained space and takes an RNG; ours
    # draws from `retry_source` instead, so the RNG that Pathfinder passes in is unused.
    result = f(
        vecmodel,
        args...;
        rng,
        init=_init_vector(vecmodel, sources),
        init_sampler=(_, x) -> copyto!(x, vector_parameters(vecmodel, retry_source)),
        input=model,
        kwargs...,
    )
    return _transform_result(result, vecmodel)
end

function _pathfinder_chain(draws, vecmodel::VecModel)
    results = map(eachcol(draws)) do draw
        simulate(vecmodel, draw)
    end # is a Vector
    return to_flexichain(hcat(results))
end

function _transform_result(result, vecmodel::VecModel)
    chain = _pathfinder_chain(result.draws, vecmodel)
    return @set result.draws_transformed = chain
end

function _transform_result(result::Pathfinder.MultiPathfinderResult, vecmodel::VecModel)
    paths = map(result.pathfinder_results) do path
        _transform_result(path, vecmodel)
    end
    chain = _pathfinder_chain(result.draws, vecmodel)
    return setproperties(result, (; draws_transformed=chain, pathfinder_results=paths))
end

end
