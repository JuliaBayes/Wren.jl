"""
    FromUnconstrainedInterval(
        [rng::AbstractRNG=Random.default_rng()],
        [lower::Real=-2.0, upper::Real=2.0]
    ) <: AbstractLatentSource

Generate latent values by first transforming the prior distribution of each variable to
unconstrained space, and then sampling each element of the unconstrained vector uniformly
from the interval `[lower, upper]`.

If `lower` and `upper` are not given, they default to `-2` and `2`, which mimics Stan's
default initialisation strategy (see the [Stan reference manual page on
initialisation](https://mc-stan.org/docs/reference-manual/execution.html#initialization) for
more details).

Requires that `lower <= upper`.
"""
struct FromUnconstrainedInterval{R<:AbstractRNG,T<:Real} <: AbstractLatentSource
    rng::R
    lower::T
    upper::T
    function FromUnconstrainedInterval(rng::R, lower::T, upper::T) where {R,T}
        lower > upper &&
            throw(ArgumentError("`lower` must be less than or equal to `upper`"))
        return new{R,T}(rng, lower, upper)
    end
end
FromUnconstrainedInterval(rng::AbstractRNG) = FromUnconstrainedInterval(rng, -2.0, 2.0)
FromUnconstrainedInterval(lower::Real, upper::Real) =
    FromUnconstrainedInterval(default_rng(), lower, upper)
FromUnconstrainedInterval() = FromUnconstrainedInterval(default_rng())

function get_latent(source::FromUnconstrainedInterval, ::VarName, right, ::Val)
    len = Plaice.unconstrained_vec_length(right)
    (; rng, lower, upper) = source
    y = lower .+ (upper - lower) .* rand(rng, float(typeof(lower)), len)
    return TransformedValue(y, AsUnconstrainedVec())
end
scope(source::FromUnconstrainedInterval, ::VarName) = source
