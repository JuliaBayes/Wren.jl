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
        # `log2π / 2` alone would be a `Float64`, promoting `Float32` parameters.
        lognorm = (log2π + 2 * log(σ)) / 2 + normlogcdf(μ / σ)
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

  When `μ < 0`, that mass underflows once `μ / σ` is very negative, and the draw is then
  small next to `μ`. So that case solves for `s = x / (σ√2)` directly instead.
=#
@inline function Base.rand(rng::AbstractRNG, d::HalfNormal)
    r = rand(rng, noisetype(d))
    a = -(d.μ / d.σ) * invsqrt2
    a > zero(a) && return d.σ * sqrt2 * _halfnormal_tail_offset(a, log1p(-r))
    mass = erfc(a)
    x = muladd(d.σ * sqrt2, erfcinv((one(r) - r) * mass), d.μ)
    return max(x, zero(x))
end

#=
  Solve `log(erfc(a + s) / erfc(a)) = logu` for `s ≥ 0`. Writing `erfc(t)` as
  `erfcx(t) * exp(-t²)` turns the left side into
  `-s * (2a + s) + log(erfcx(a + s) / erfcx(a))`, which has no cancellation however large
  `a` is. Dropping the `erfcx` ratio gives the starting point. The left side is concave in
  `s`, so Newton's method converges monotonically after its first step.
=#
function _halfnormal_tail_offset(a::Number, logu::Number)
    s = -logu / (a + sqrt(muladd(a, a, -logu)))
    erfcx_a = erfcx(a)
    for _ in 1:50
        t = a + s
        g = -s * (a + t) + log(erfcx(t) / erfcx_a) - logu
        step = g * erfcx(t) * sqrtπ / 2
        s = max(s + step, zero(s))
        abs(step) <= eps(one(s)) * s && break
    end
    return s
end
