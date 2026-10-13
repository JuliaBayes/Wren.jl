"""
    HalfNormal(μ, σ)
    HalfNormal()

Normal distribution with mean `μ` (default 0.0) and standard deviation `σ` (default 1.0),
truncated to `[0, ∞)`.

The log normalising constant, including the log probability `log Φ(μ/σ)` that the
untruncated normal assigns to `[0, ∞)`, is computed on construction and stored in the
struct.

**Customised batched implementations**: Not yet (will fall back to a product of scalar
distributions)
"""
struct HalfNormal{M<:Number,S<:Number,L<:Number} <: PositiveUnivariateDist
    μ::M
    σ::S
    lognorm::L
    function HalfNormal(μ::Number, σ::Number)
        lognorm = log(σ) + log2π / 2 + normlogcdf(μ / σ)
        return new{typeof(μ),typeof(σ),typeof(lognorm)}(μ, σ, lognorm)
    end
end

HalfNormal() = HalfNormal(0.0, 1.0)

@inline normlogcdf(z::Number) = logerfc(-z * invsqrt2) - logtwo

@inline zval(d::HalfNormal, x::Number) = (x - d.μ) / d.σ

@inline function DensityInterface.logdensityof(d::HalfNormal, x::Number)
    value = -zval(d, x)^2 / 2 - d.lognorm
    return select(x >= zero(x), () -> value, () -> oftype(value, -Inf))
end

#=
  Inverse-CDF draw through the upper tail: `P(X > x) = erfc((x - μ) / (σ√2)) / 2`, and
  the mass above zero is `erfc(-μ / (σ√2)) / 2`. Rounding can land the draw a hair below
  zero, so it is clamped there.
=#
@inline function Base.rand(rng::AbstractRNG, d::HalfNormal)
    u = one(noisetype(d)) - rand(rng, noisetype(d))
    mass = erfc(-(d.μ / d.σ) * invsqrt2)
    x = muladd(d.σ * sqrt2, erfcinv(u * mass), d.μ)
    return max(x, zero(x))
end
