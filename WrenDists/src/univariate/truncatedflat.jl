struct TruncatedFlat{T<:AbstractFloat,L<:Real,U<:Real} <: UnivariateDist
    lower::L
    upper::U
end

"""
    truncated(d::Flat; lower=-Inf, upper=Inf)

Restrict an improper flat distribution to `[lower, upper]`. Its log-density remains
zero inside the bounds and `-Inf` outside; no normalising constant is added. In
particular, `truncated(Flat(); lower=0)` represents a flat density on a positive
parameter, as in a Stan `real<lower=0>` declaration without a prior.

`rand` draws an initial value by sampling uniformly from `(-2, 2)` in the
unconstrained coordinate and mapping it into the bounds. It is not a draw from
the improper distribution. The bounds must satisfy `lower < upper`.

Only `Flat` supports this method. Truncating a proper distribution also requires
its probability mass inside the bounds to normalise the density.

**Customised batched implementations**: Not yet (will fall back to a product of
scalar distributions).
"""
function truncated(::Flat{T}; lower::Real=T(-Inf), upper::Real=T(Inf)) where {T}
    lower < upper || throw(ArgumentError("lower must be less than upper"))
    return TruncatedFlat{T,typeof(lower),typeof(upper)}(lower, upper)
end

@inline _promoted_paramtype(::Type{TruncatedFlat{T,L,U}}) where {T,L,U} =
    promote_type(T, L, U)

@inline function DensityInterface.logdensityof(d::TruncatedFlat, x::Number)
    value = zero(float(x))
    return select((x >= d.lower) & (x <= d.upper), () -> value,
        () -> oftype(value, -Inf))
end

@inline function Base.rand(rng::AbstractRNG, d::TruncatedFlat)
    u = 4 * rand(rng, noisetype(d)) - 2
    return Plaice.inverse(Plaice.scalar_to_scalar_bijector(d))(u)
end
