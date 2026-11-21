"""
    HalfCauchy(μ, σ)
    HalfCauchy()

Cauchy distribution with location `μ` (default 0.0) and scale `σ` (default 1.0),
truncated to `[0, ∞)`.

The log normalising constant, including the log probability that the untruncated Cauchy
assigns to `[0, ∞)`, is computed on construction and stored in the struct.

**Customised batched implementations**: Not yet (will fall back to a product of scalar
distributions)
"""
struct HalfCauchy{M<:Number,S<:Number,L<:Number} <: PositiveUnivariateDist
    μ::M
    σ::S
    lognorm::L
    function HalfCauchy(μ::Number, σ::Number)
        lognorm = log(σ) + log(halfcauchy_angle(μ, σ))
        return new{typeof(μ),typeof(σ),typeof(lognorm)}(μ, σ, lognorm)
    end
end

HalfCauchy() = HalfCauchy(0.0, 1.0)

#=
  The untruncated Cauchy puts mass `(π/2 + atan(μ/σ)) / π` on `[0, ∞)`. The numerator is
  the angle of `(-μ/σ, 1)`, which `atan(y, x)` returns without the cancellation
  `π/2 + atan(μ/σ)` suffers when `μ/σ` is large and negative. The `π` cancels against
  the one in the Cauchy density.
=#
@inline halfcauchy_angle(μ, σ) = atan(one(μ / σ), -(μ / σ))

@inline zval(d::HalfCauchy, x::Number) = (x - d.μ) / d.σ

@inline function DensityInterface.logdensityof(d::HalfCauchy, x::Number)
    value = -log1psq(zval(d, x)) - d.lognorm
    return select(x >= zero(x), () -> value, () -> oftype(value, -Inf))
end

# A Cauchy draw is `μ + σ tan(φ)` for uniform `φ`; above zero, `φ` spans an interval of
# length `halfcauchy_angle` ending at `π/2`, and `tan(π/2 - t) = cot(t)`.
@inline function Base.rand(rng::AbstractRNG, d::HalfCauchy)
    u = one(noisetype(d)) - rand(rng, noisetype(d))
    t = u * halfcauchy_angle(d.μ, d.σ)
    x = muladd(d.σ, cos(t) / sin(t), d.μ)
    return max(x, zero(x))
end
