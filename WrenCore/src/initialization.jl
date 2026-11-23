"""
    AbstractInitialization

An abstract type for ways to initialise MCMC samplers and optimisers.

Currently, only [`FindFiniteDensity`](@ref) is provided.
"""
abstract type AbstractInitialization end

"""
    FindFiniteDensity(source; max_attempts::Integer=1)

Initialise a chain (or an optimisation) by drawing from `source` until its unconstrained
coordinates and log-density are finite, trying at most `max_attempts` candidates per chain.

`source` can either be a [`WrenCore.AbstractLatentSource`](@ref), or something that can
be converted to it. In particular:

- `source::Random.AbstractRNG` is shorthand for `FromPrior(source)`, i.e., draw from the
  prior with the given RNG.
- `source::VarNamedTuple` is shorthand for `FromValues(source)`, i.e., use the given values
  *(in model space)* as the initial point.

To use Stan-style initialisation from a uniform interval, use
[`source=FromUnconstrainedInterval([rng, ]lower, upper)`](@ref FromUnconstrainedInterval).

If initialisation fails after `max_attempts`, an error is thrown.
"""
struct FindFiniteDensity{S<:AbstractLatentSource} <: AbstractInitialization
    source::S
    max_attempts::Int

    function FindFiniteDensity(
        source::S,
        max_attempts::Integer,
    ) where {S<:AbstractLatentSource}
        max_attempts > 0 || throw(ArgumentError("`max_attempts` must be positive."))
        return new{S}(source, Int(max_attempts))
    end
end

FindFiniteDensity(source; max_attempts::Integer=1) =
    FindFiniteDensity(to_latent_source(source), max_attempts)

"""
    to_initialization(x)::AbstractInitialization

Convert `x` to an [`AbstractInitialization`](@ref). An `AbstractInitialization` is
returned unchanged. Anything else is treated as a latent source and wrapped in
[`FindFiniteDensity`](@ref), so that e.g. a `VarNamedTuple` or an `AbstractRNG` can be
passed directly as `init`.
"""
to_initialization(init::AbstractInitialization) = init
to_initialization(source) = FindFiniteDensity(source)

_initial_source(init::FindFiniteDensity) = init.source

function _initial_position(vm::VecModel, init::FindFiniteDensity)
    for _ in 1:init.max_attempts
        q = vector_parameters(vm, init.source)
        all(isfinite, q) && isfinite(lp(vm, q)) && return q
    end
    throw(
        ArgumentError(
            "No finite-density initial point found after $(init.max_attempts) attempts.",
        ),
    )
end
