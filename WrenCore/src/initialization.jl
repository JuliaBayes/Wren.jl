"""
    AbstractInitialization

An abstract type for ways to initialise MCMC samplers and optimisers.

Currently, only [`FindFiniteDensity`](@ref) is provided.

For convenience, other objects may be implicitly converted by Wren into an
`AbstractInitialization`.

Specifically, `AbstractLatentSource`s, or things that can be converted to them (such as
`Random.AbstractRNG` or `VarNamedTuple`), are wrapped in [`FindFiniteDensity`](@ref). The
`max_attempts` keyword argument depends on what is passed in.
  
For sets of initial values without fallbacks, we use `max_attempts=1` since it doesn't make
sense to try the same parameters multiple times. For other sources, we use
`max_attempts=100` by default.
"""
abstract type AbstractInitialization end

"""
    FindFiniteDensity(source; max_attempts::Union{Nothing,Integer}=nothing)

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

If `max_attempts` is not given, it is 1 for fixed values without a random fallback (since
trying the same parameters again won't help), and 100 for other sources.

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
function FindFiniteDensity(source; max_attempts::Union{Nothing,Integer}=nothing)
    source = to_latent_source(source)
    max_attempts = max_attempts === nothing ? _default_max_attempts(source) : max_attempts
    return FindFiniteDensity(source, max_attempts)
end

# Sources that always give the same values only need one attempt, since trying the same
# parameters again won't help. Random sources get more attempts.
_default_max_attempts(::AbstractLatentSource) = 100
_default_max_attempts(::FromVector) = 1
# `Void` throws an error rather than giving values, so it adds no randomness.
_default_max_attempts(::Void) = 1
# Fixed values are only random if their fallback is.
_default_max_attempts(source::FromValues) = _default_max_attempts(source.fallback)

"""
    to_initialization(x)::AbstractInitialization

Convert `x` to an [`AbstractInitialization`](@ref). See its string for the details.
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
