"""
    TruncatedFlat(lower, upper)

Flat distribution on the finite interval `[lower, upper]`, i.e. a log-density of zero
inside the bounds and `-Inf` outside. No normalising constant is added, so unlike
`Uniform(lower, upper)`, the log-density does not depend on the bounds. This is like a
Stan `real<lower=lower, upper=upper>` parameter without a prior.

The bounds may depend on other parameters, as in `TruncatedFlat(0, 1 - alpha)`. If
`lower >= upper`, the log-density is `-Inf` everywhere (rather than an error).

Both bounds must be finite. For a flat distribution on `[0, ∞)`, use
[`PositiveFlat`](@ref).

`rand` draws from `Uniform(lower, upper)`.

The type of samples is determined by the types of `lower` and `upper`.

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

@inline Base.rand(rng::AbstractRNG, d::TruncatedFlat) = rand(rng, Uniform(d.lower, d.upper))
