"""
    TruncatedFlat(lower, upper)

Improper flat distribution restricted to `[lower, upper]`. Its log-density is zero
inside the bounds and `-Inf` outside; no normalising constant is added. Either bound may
be infinite: in particular, `TruncatedFlat(0, Inf)` represents a flat density on a
positive parameter, as in a Stan `real<lower=0>` declaration without a prior.

Since this isn't a proper distribution, `rand` can't draw from it. Instead it draws a
value uniformly from (-2, 2) in the unconstrained space and maps it into the bounds, which
is the same way that Stan initialises constrained parameters.

The bounds may depend on other parameters, as in `TruncatedFlat(0, 1 - alpha)`, and may
be traced numbers when the model is compiled with Reactant. As for other distributions
with invalid parameters, bounds with `lower >= upper` give a log-density of `-Inf`
everywhere rather than an error, so a dependent bound that rounds to an empty interval
rejects that point instead of stopping the sampler.

**Customised batched implementations**: Not yet (will fall back to a product of scalar
distributions)
"""
struct TruncatedFlat{L<:Number,U<:Number} <: UnivariateDist
    lower::L
    upper::U
end

@inline function DensityInterface.logdensityof(d::TruncatedFlat, x::Number)
    value = zero(float(x))
    # `&` rather than `&&`, so the check also compiles for traced bounds.
    valid = (d.lower < d.upper) & (x >= d.lower) & (x <= d.upper)
    return select(valid, () -> value, () -> oftype(value, -Inf))
end

@inline function Base.rand(rng::AbstractRNG, d::TruncatedFlat)
    T = noisetype(d)
    u = rand(rng, Uniform(-2 * one(T), 2 * one(T)))
    return Plaice.inverse(Plaice.scalar_to_scalar_bijector(d))(u)
end
